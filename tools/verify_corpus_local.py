"""Cluster-hosted version of verify_corpus.py — runs on the GPU pod with the
real model.generate() against /data/ snapshots, so we can compare:

  corpus stored response   vs.   AIaaS re-roll   vs.   cluster model.generate

Reasoning is whatever the chat template + model produce natively (Harmony for
Gemma, <think> for Qwen). Greedy decode (do_sample=False).

Usage on the pod:
    python verify_corpus_local.py --model_key gemma4_31b
    python verify_corpus_local.py --model_key qwen36
"""
import argparse, json, re, sys, time
from pathlib import Path
import torch

REPO_DEFAULT = "/scratch/test/mechhack"  # path inside the pod
PATHS = {
    "gemma4_31b": "/data/Gemma-4-31B-it",
    "qwen36":     "/data/Qwen3.6-27B",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", choices=["gemma4_31b", "qwen36"], required=True)
    ap.add_argument("--repo", default=REPO_DEFAULT,
                    help="Hackathon repo root (for the dataset jsonl + chunked_sdpa)")
    ap.add_argument("--sample_ids_file", default=None,
                    help="Optional: path to a JSON list of sample_ids to re-roll. "
                         "Default: stratified sample 5 refusal + 5 compliance.")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    return ap.parse_args()


def stratified(rows, n=5, seed=42):
    import random
    rng = random.Random(seed)
    refs  = [r for r in rows if r.get("is_refusal") is True]
    comps = [r for r in rows if r.get("is_refusal") is False]
    rng.shuffle(refs); rng.shuffle(comps)
    return refs[:n] + comps[:n]


_HARMONY_FINAL = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>\s*(.*?)\s*(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>\s*analysis\s*<\|message\|>", re.DOTALL)
_LEFTOVER_SPECIAL = re.compile(r"<\|[^|<>]{1,40}\|>|<bos>|<eos>|<pad>|<unk>|<start_of_turn>|<end_of_turn>")


def extract_answer(text):
    if _HARMONY_ANALYSIS.search(text):
        m = _HARMONY_FINAL.search(text)
        if m: text = m.group(1)
    end = text.rfind("</think>")
    if end >= 0:
        text = text[end + len("</think>"):]
    return _LEFTOVER_SPECIAL.sub("", text).strip()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(args.repo) / "starter_code"))
    from chunked_sdpa import chunked_sdpa_scope

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = PATHS[args.model_key]
    print(f"=== loading {model_path} ===", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="cuda:0", trust_remote_code=True)
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s | layers={getattr(model.config,'num_hidden_layers','?')}", flush=True)

    # Sample selection
    jsonl = Path(args.repo) / f"datasets/refusal_probes/{args.model_key}/attacks_full.jsonl"
    rows = [json.loads(l) for l in open(jsonl)]
    if args.sample_ids_file:
        ids = set(json.load(open(args.sample_ids_file)))
        sel = [r for r in rows if r["sample_id"] in ids]
    else:
        sel = stratified(rows, n=5)
    print(f"=== running {len(sel)} samples ===", flush=True)

    use_chunked = (args.model_key == "gemma4_31b")
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm: cm.__enter__()

    out = []
    try:
        for i, row in enumerate(sel):
            sid = row["sample_id"]
            prompt = row["attack_prompt"]
            stored_refusal = bool(row["is_refusal"])

            txt = tok.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(txt, return_tensors="pt").to("cuda:0")
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=False, temperature=1.0,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            torch.cuda.synchronize()
            dt = time.time() - t0
            comp_ids = gen[0, enc.input_ids.shape[1]:]
            raw = tok.decode(comp_ids, skip_special_tokens=False)
            answer = extract_answer(raw)

            rec = {
                "model_key": args.model_key, "sample_id": sid,
                "stored_refusal": stored_refusal,
                "stored_response": (row.get("response") or "")[:600],
                "raw_response": raw[:1200],          # keep markers visible
                "answer_only": answer[:600],
                "n_new_tokens": int(comp_ids.shape[0]),
                "dt_gen_s": round(dt, 1),
            }
            out.append(rec)
            print(f"  [{i+1}/{len(sel)}] {sid}: {dt:.1f}s | new_tokens={int(comp_ids.shape[0])} | answer[:80]={answer[:80]!r}", flush=True)
    finally:
        if cm: cm.__exit__(None, None, None)

    out_json = Path(args.out_json) if args.out_json else Path(f"/scratch/verify_corpus_local_{args.model_key}.json")
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
