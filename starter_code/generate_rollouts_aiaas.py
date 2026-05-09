"""Generate rollouts via EPFL AIaaS — same regime as generate_rollouts.py
but hits the AIaaS HTTP endpoint instead of running model.generate locally.

Used to reproduce corpus rollouts WITHOUT a local GPU. Empirically:
  - Gemma 4-31B-it: 14/14 = 100% verdict alignment with corpus (with the
    flags below), ~120 s/sample
  - Qwen 3.6-27B:   ~80% alignment (BF16 nondeterminism on borderline samples,
    same as cluster's direct model.generate)

Critical flags that produced 100% Gemma alignment in our experiments:
  - reasoning.enabled=true, effort=high
  - chat_template_kwargs.enable_thinking=true   ← AIaaS doesn't apply this
                                                   for Gemma without it
  - temperature=0
  - max_tokens=16000

Usage:
    export AIAAS_KEY=sk--...
    python generate_rollouts_aiaas.py \
        --model_key gemma4_31b \
        --attacks  /path/to/attacks.jsonl \
        --out      /path/to/rollouts.jsonl \
        --concurrency 8

JSONL in/out, resumable (skips sample_ids already in --out).
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import httpx


AIAAS_URL = "https://inference.rcp.epfl.ch/v1/chat/completions"
AIAAS_MODELS = {
    "gemma4_31b": "google/gemma-4-31B-it",
    "qwen36":     "Qwen/Qwen3.6-27B",
}


def strip_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, post_thinking_answer) given a raw model output.
    Handles both Gemma's <|channel|>analysis...<|channel|>final and Qwen's
    <think>...</think> formats."""
    if not text: return "", ""
    # Gemma Harmony — extract last `final` channel
    m = re.search(r"<\|channel\|>\s*final\s*<\|message\|>\s*(.*?)\s*(?:<\|return\|>|<\|end\|>|$)",
                  text, re.DOTALL)
    if m:
        # the analysis channel content (if any) is the thinking
        am = re.search(r"<\|channel\|>\s*analysis\s*<\|message\|>(.*?)<\|end\|>", text, re.DOTALL)
        thinking = (am.group(1).strip() if am else "")
        return thinking, m.group(1).strip()
    # Qwen-style <think>
    tm = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if tm:
        return tm.group(1).strip(), text[tm.end():].strip()
    end = text.rfind("</think>")
    if end >= 0:
        return text[:end].lstrip("<think>").strip(), text[end+len("</think>"):].strip()
    # No thinking markers — assume whole text is the answer
    cleaned = re.sub(r"<\|[^|<>]{1,40}\|>", "", text).strip()
    return "", cleaned


def query(model_id: str, prompt: str, max_tokens: int, key: str, timeout: float = 600.0) -> dict:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        # The two reasoning knobs that AIaaS expects.
        "reasoning": {"enabled": True, "effort": "high"},
        # The flag we found was missing — without this Gemma drops Harmony
        # channels entirely on AIaaS.
        "chat_template_kwargs": {"enable_thinking": True},
    }
    t0 = time.time()
    r = httpx.Client(timeout=timeout).post(
        AIAAS_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=body,
    )
    dt = time.time() - t0
    if r.status_code != 200:
        return {"err": f"HTTP {r.status_code}: {r.text[:200]}", "dt": dt}
    msg = r.json()["choices"][0]["message"]
    raw = (msg.get("content") or "").strip()
    rc_field = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if isinstance(rc_field, dict): rc_field = json.dumps(rc_field)
    rc_field = str(rc_field)
    thinking_inline, answer = strip_thinking(raw)
    return {
        "raw": raw,
        "thinking": thinking_inline or rc_field,
        "rollout": answer,
        "reasoning_chars": len(rc_field) + len(thinking_inline),
        "dt": dt,
    }


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists(): return set()
    done = set()
    for line in out_path.open():
        try: done.add(json.loads(line)["sample_id"])
        except Exception: pass
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", choices=list(AIAAS_MODELS.keys()), required=True)
    ap.add_argument("--attacks", required=True, help="JSONL with sample_id + attack_prompt per row.")
    ap.add_argument("--out", required=True, help="Output JSONL (resumable).")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=16000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--key", default=os.environ.get("AIAAS_KEY"))
    args = ap.parse_args()

    if not args.key:
        sys.exit("set $AIAAS_KEY or pass --key (get one at https://portal.rcp.epfl.ch/aiaas/keys)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)

    rows = [json.loads(l) for l in open(args.attacks) if l.strip()]
    todo = [r for r in rows if r["sample_id"] not in done]
    if args.limit > 0: todo = todo[:args.limit]

    print(f"=== generate_rollouts_aiaas model={args.model_key} ===", flush=True)
    print(f"  rows: {len(rows)} | done: {len(done)} | to do: {len(todo)} | conc: {args.concurrency}", flush=True)
    if not todo: print("  nothing to do."); return

    model_id = AIAAS_MODELS[args.model_key]
    fout = out_path.open("a")
    t_start = time.time()

    def _one(atk):
        try:
            r = query(model_id, atk["attack_prompt"], args.max_new_tokens, args.key)
            return atk, r, None
        except Exception as e:
            return atk, None, f"{type(e).__name__}: {str(e)[:200]}"

    n_done = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_one, atk): atk for atk in todo}
            for fut in as_completed(futs):
                atk, r, err = fut.result()
                n_done += 1
                if err or (r and r.get("err")):
                    msg = err or r["err"]
                    print(f"  [{n_done}/{len(todo)}] ✗ {atk['sample_id']}: {msg[:120]}", flush=True)
                    continue
                row = {**atk,
                       "rollout":          r["rollout"],
                       "thinking":         r["thinking"],
                       "reasoning_chars":  r["reasoning_chars"],
                       "gen_seconds":      round(r["dt"], 2),
                       "target_model":     model_id,
                       "via":              "aiaas"}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed else 0
                eta = (len(todo)-n_done) / rate / 60 if rate else 0
                print(f"  [{n_done}/{len(todo)}] ✓ {atk['sample_id']}: reasoning={r['reasoning_chars']:>5} ({r['dt']:.0f}s) | rate={rate:.2f}/s eta={eta:.0f}min", flush=True)
    finally:
        fout.close()
    print(f"\n=== done in {(time.time()-t_start)/60:.1f} min — wrote {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
