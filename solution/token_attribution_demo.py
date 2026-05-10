"""Token attribution demo — shows which tokens drive the harm/refusal signal.

Usage:
    python solution/token_attribution_demo.py <sample_id> [--task cyber3_prohibited_vs_rest]

Examples:
    python solution/token_attribution_demo.py abc123
    python solution/token_attribution_demo.py abc123 --task refusal_gemma4_31b
"""
import sys, json, argparse, math
from pathlib import Path
import torch

PROBE_DIR    = Path("/scratch/outputs/probes/dim")
EXTRACTS     = {
    "cyber1_dual_use_vs_benign":        Path("/scratch/outputs/extracts/cyber/gemma4_31b/gemma4_31b"),
    "cyber2_high_risk_vs_dual_benign":  Path("/scratch/outputs/extracts/cyber/gemma4_31b/gemma4_31b"),
    "cyber3_prohibited_vs_rest":        Path("/scratch/outputs/extracts/cyber/gemma4_31b/gemma4_31b"),
    "refusal_gemma4_31b":               Path("/scratch/outputs/extracts/gemma4_31b/gemma4_31b"),
    "refusal_qwen36":                   Path("/scratch/outputs/extracts/qwen36/qwen36"),
}
DATASETS     = {
    "cyber1_dual_use_vs_benign":        Path("datasets/cyber_probes/test.jsonl"),
    "cyber2_high_risk_vs_dual_benign":  Path("datasets/cyber_probes/test.jsonl"),
    "cyber3_prohibited_vs_rest":        Path("datasets/cyber_probes/test.jsonl"),
    "refusal_gemma4_31b":               Path("datasets/refusal_probes/gemma4_31b/test_split.jsonl"),
    "refusal_qwen36":                   Path("datasets/refusal_probes/qwen36/test_split.jsonl"),
}

RED   = "\033[91m"
GREEN = "\033[92m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

def bar(v, lo, hi, width=12):
    if hi == lo: return "░" * width
    t = (v - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    filled = round(t * width)
    return "█" * filled + "░" * (width - filled)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_id")
    ap.add_argument("--task", default="cyber3_prohibited_vs_rest")
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    probe_file = PROBE_DIR / f"{args.task}_dim_probe.pt"
    if not probe_file.exists():
        sys.exit(f"Probe not found: {probe_file}")

    ckpt  = torch.load(str(probe_file), weights_only=False, map_location="cpu")
    r_hat = ckpt["r_hat"].to(torch.float32)
    layer = ckpt.get("layer_idx", 0)

    ext_path = EXTRACTS[args.task] / f"{args.sample_id}.pt"
    if not ext_path.exists():
        sys.exit(f"Extract not found: {ext_path}")

    ex       = torch.load(str(ext_path), weights_only=False, map_location="cpu")
    residuals = ex["residuals"]
    mask      = ex["attention_mask"].bool()
    input_ids = ex.get("input_ids")

    # pick layer
    if residuals.dim() == 3:
        full = residuals[min(layer, residuals.shape[0]-1)].to(torch.float32)
    else:
        full = residuals.to(torch.float32)

    # per-token dot product with r_hat
    scores = (full @ r_hat).numpy()          # (n_tokens,)
    valid  = mask.numpy().astype(bool)
    scores = scores[valid]
    n      = len(scores)

    # overall score = last token
    overall_raw  = float(scores[-1])
    overall_prob = 1.0 / (1.0 + math.exp(-overall_raw))

    # decode tokens if possible
    tokens = None
    if input_ids is not None:
        ids = input_ids[valid] if input_ids.shape[0] == mask.shape[0] else input_ids
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "/data/Gemma-4-31B-it" if "qwen" not in args.task else "/data/Qwen3.6-27B",
                use_fast=True)
            tokens = [tok.decode([int(i)]) for i in ids]
        except Exception:
            tokens = [f"<{int(i)}>" for i in ids]

    if tokens is None:
        tokens = [f"tok_{i}" for i in range(n)]

    # lookup original prompt text
    prompt_text = None
    ds_path = DATASETS[args.task]
    if ds_path.exists():
        for line in open(ds_path):
            row = json.loads(line)
            if row.get("sample_id") == args.sample_id:
                prompt_text = row.get("attack_prompt") or row.get("prompt", "")
                break

    # ── print ────────────────────────────────────────────────────────────────
    label = f"{RED}{BOLD}HARMFUL{RESET}" if overall_prob > 0.5 else f"{GREEN}{BOLD}BENIGN{RESET}"
    print(f"\n{BOLD}Sample:{RESET} {args.sample_id}   {BOLD}Task:{RESET} {args.task}")
    if prompt_text:
        preview = prompt_text[:120].replace("\n", " ")
        print(f"{DIM}Prompt: {preview}{'…' if len(prompt_text)>120 else ''}{RESET}")
    print(f"\n{BOLD}Overall score:{RESET} {overall_raw:+.3f}  (prob={overall_prob:.3f})  → {label}\n")

    lo, hi = scores.min(), scores.max()
    pairs = sorted(enumerate(zip(tokens, scores)), key=lambda x: x[1][1])
    n_show = min(args.top_n, n)

    # bottom N (lowest = most benign / suppressing signal)
    print(f"{GREEN}{BOLD}Top {n_show//2} lowest-scoring tokens (benign signal):{RESET}")
    for _, (tok, sc) in pairs[:n_show//2]:
        b = bar(sc, lo, hi)
        print(f"  {GREEN}{sc:+6.3f}{RESET}  {b}  {repr(tok)}")

    print()
    # top N (highest = driving harm/refusal signal)
    print(f"{RED}{BOLD}Top {n_show//2} highest-scoring tokens (harm signal):{RESET}")
    for _, (tok, sc) in pairs[-(n_show//2):]:
        b = bar(sc, lo, hi)
        print(f"  {RED}{sc:+6.3f}{RESET}  {b}  {repr(tok)}")

    print()

if __name__ == "__main__":
    main()
