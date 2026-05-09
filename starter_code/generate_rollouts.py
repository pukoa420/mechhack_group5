"""Generate rollouts from Gemma 4-31B-it / Qwen 3.6-27B against a list of
attack prompts, with the EXACT same decoding regime that produced the
hackathon's reference corpus (sweep_1k_shard.py).

Why mimic the original recipe matters
-------------------------------------
The corpus's stored `is_refusal` labels were derived from these specific
parameters. If you generate fresh rollouts with different decoding (different
chat-template flags, different sampling, AIaaS/OpenRouter instead of local
generate), the labels you get will NOT match the corpus. See
docs/onboarding.md for the why.

Identical to the corpus pipeline:
  - Gemma:  apply_chat_template(..., enable_thinking=True) + parse_response()
  - Qwen:   apply_chat_template(...), no extra flags (template handles thinking)
  - Tokenizer: truncation=True, max_length=8192
  - Decoding: do_sample=False (greedy), max_new_tokens=16384, use_cache=True
  - Gemma uses chunked_sdpa_scope() (head_dim=512 needs it)

Input format (JSONL, one row per prompt — minimum required fields):
    {"sample_id": "EVO_0001", "attack_prompt": "..."}
You can include any other fields; they're passed through to the output.

Output format (JSONL): same fields plus rollout, thinking (Gemma only),
in_tokens, out_tokens_actual, gen_seconds, hit_token_cap.

Usage:
    python generate_rollouts.py \
        --model_key gemma4_31b \
        --attacks /path/to/attacks.jsonl \
        --out      /path/to/rollouts.jsonl
    # auto-resolves /data/Gemma-4-31B-it; resumes from existing rows in --out

Resume: rows whose sample_id already exists in --out are skipped.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import torch

# Local chunked-sdpa shim for Gemma's head_dim=512.
sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope


HF_REPOS = {
    "gemma4_31b": "google/gemma-4-31B-it",
    "qwen36":     "Qwen/Qwen3.6-27B",
}


def resolve_model_path(model_key: str, model_path: str | None) -> str:
    """Lookup order:  --model_path > $HACKATHON_MODELS_DIR/<repo>
    > /data/<repo>  > <repo-root>/models/<repo>  > HF cache."""
    if model_path and Path(model_path).exists():
        return str(Path(model_path).resolve())
    repo_id = HF_REPOS.get(model_key)
    if not repo_id:
        raise ValueError(f"unknown model_key {model_key!r}")
    name = repo_id.split("/")[-1]
    candidates = []
    if env := os.environ.get("HACKATHON_MODELS_DIR"):
        candidates.append(Path(env) / name)
    candidates += [Path("/data") / name,
                   Path(__file__).resolve().parent.parent / "models" / name]
    for c in candidates:
        if c.exists(): return str(c)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, local_files_only=True)


def load_model(path: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def build_prompt(tok, prompt_text: str, is_gemma: bool) -> str:
    """Apply chat template the SAME way the corpus generator did."""
    msgs = [{"role": "user", "content": prompt_text}]
    if is_gemma:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_gemma_response(tok, raw_decode: str) -> tuple[str, str]:
    """Use the model's processor to split <|channel|>thought from <|channel|>final."""
    try:
        parsed = tok._processor.parse_response(raw_decode)
    except Exception:
        return "", raw_decode
    if isinstance(parsed, dict):
        thinking = parsed.get("thinking") or parsed.get("thought") or ""
        response = parsed.get("response") or parsed.get("content") or raw_decode
    else:
        thinking = getattr(parsed, "thinking", "") or getattr(parsed, "thought", "")
        response = (getattr(parsed, "response", None)
                    or getattr(parsed, "content", None)
                    or raw_decode)
    return thinking or "", response or raw_decode


def parse_qwen_response(raw_decode: str) -> tuple[str, str]:
    """Split Qwen's `<think>...</think>` block from the final answer.

    Matches the corpus's hackathon attacks_full.jsonl schema, which stores
    `thinking` and `response` as separate fields for Qwen rows.
    """
    import re
    m = re.search(r"<think>(.*?)</think>\s*", raw_decode, re.DOTALL)
    if m:
        return m.group(1).strip(), raw_decode[m.end():].strip()
    end = raw_decode.rfind("</think>")
    if end >= 0:
        return raw_decode[:end].lstrip("<think>").strip(), raw_decode[end+len("</think>"):].strip()
    return "", raw_decode.strip()


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
    ap.add_argument("--model_key", choices=list(HF_REPOS.keys()), required=True)
    ap.add_argument("--model_path", default=None,
                    help="Local HF snapshot dir; defaults to /data/<repo-name>.")
    ap.add_argument("--attacks", required=True,
                    help="JSONL with at least {sample_id, attack_prompt} per row.")
    ap.add_argument("--out", required=True, help="Output JSONL path (resumable).")
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--max_input_tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N rows (0 = all).")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)

    rows = [json.loads(l) for l in open(args.attacks) if l.strip()]
    todo = [r for r in rows if r["sample_id"] not in done]
    if args.limit > 0: todo = todo[:args.limit]
    is_gemma = (args.model_key == "gemma4_31b")

    print(f"=== generate_rollouts model={args.model_key} ===", flush=True)
    print(f"  total rows: {len(rows)} | already done: {len(done)} | to do: {len(todo)}", flush=True)
    if not todo:
        print("  nothing to do."); return

    model_path = resolve_model_path(args.model_key, args.model_path)
    print(f"  model path: {model_path}", flush=True)
    print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB", flush=True)

    t0 = time.time()
    model, tok = load_model(model_path)
    print(f"  loaded in {time.time()-t0:.0f}s | params={sum(p.numel() for p in model.parameters())/1e9:.1f}B", flush=True)

    cm = chunked_sdpa_scope() if is_gemma else None
    if cm is not None: cm.__enter__()

    fout = out_path.open("a")
    t_start = time.time()
    try:
        for i, atk in enumerate(todo):
            sid = atk["sample_id"]
            prompt = atk["attack_prompt"]
            text = build_prompt(tok, prompt, is_gemma)
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=args.max_input_tokens).to("cuda:0")
            seq_in = int(inputs.input_ids.shape[1])

            t_g = time.time()
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            gen_s = time.time() - t_g
            comp_ids = gen[0, seq_in:]
            out_tokens = int(comp_ids.shape[0])
            hit_cap = out_tokens >= args.max_new_tokens

            # Decode raw with special tokens preserved so we can split thinking
            # the same way the corpus pipeline did.
            raw = tok.decode(comp_ids, skip_special_tokens=False)
            if is_gemma:
                thinking_text, response = parse_gemma_response(tok, raw)
            else:
                # Qwen: split <think>...</think> block from the final answer.
                thinking_text, response = parse_qwen_response(raw)

            row = {**atk,
                   "rollout": response,
                   "thinking": thinking_text,
                   "in_tokens": seq_in,
                   "out_tokens_actual": out_tokens,
                   "out_tokens_max": args.max_new_tokens,
                   "hit_token_cap": hit_cap,
                   "gen_seconds": round(gen_s, 2),
                   "target_model": HF_REPOS[args.model_key].split("/")[-1]}

            fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
            elapsed = time.time() - t_start
            rate = (i+1) / elapsed if elapsed else 0
            eta = (len(todo)-i-1) / rate / 60 if rate else 0
            print(f"  [{i+1}/{len(todo)}] {sid}: in={seq_in} out={out_tokens} cap={hit_cap} ({gen_s:.0f}s) | rate={rate:.2f}/s eta={eta:.0f}min", flush=True)

            del gen, inputs
            torch.cuda.empty_cache()
    finally:
        fout.close()
        if cm is not None: cm.__exit__(None, None, None)
    print(f"\n=== done in {(time.time()-t_start)/60:.1f} min — wrote {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
