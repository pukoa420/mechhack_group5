"""End-to-end input-embedding gradient attribution.

For each rollout, run model forward+backward to compute:
    attrib[n] = (∂probe_logit / ∂input_embed[n]) · input_embed[n]

This gives TRUE per-input-token attribution (vs the residual-stream gradient
which only attributes at one layer). Reference baseline for Level 2.

Usage:
    python grad_input_baseline.py \
        --model_path ./models/Gemma-4-31B-it \
        --probe_weights ./probes/refusal_gemma4_31b_attention.pt \
        --extracts_dir ./extracts/gemma4_31b \
        --out_dir ./edit_eval \
        --model_key gemma4_31b --variant middle

The probe-weights file should expose `state_dict` and a list of test rows in
`rows_test` (sample_id, prob, label, split, n_tokens). Extracts are produced
by extract_residuals.py.
"""
import os, sys, json, time, math, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

# Per-model layer index for "early/middle/late" — into the (n_layers+1)
# hidden_states tuple. Matches what extract_residuals.py would pick with
# layers="early" / "middle" / "late" for these specific architectures.
LAYER_IDX_DEFAULT = {
    "gemma4_31b": {"early": 10, "middle": 30, "late": 50},
    "qwen36":     {"early": 10, "middle": 32, "late": 54},
}

HF_REPOS = {
    "gemma4_31b": "google/gemma-4-31B-it",
    "qwen36":     "Qwen/Qwen3.6-27B",
}

DEVICE, DTYPE = "cuda:0", torch.bfloat16


def resolve_model_path(model_key: str, model_path: str | None) -> str:
    """Lookup order (first hit wins):
      1. explicit --model_path
      2. $HACKATHON_MODELS_DIR/<repo-name>
      3. /data/<repo-name>                    (cluster RO mount)
      4. <repo-root>/models/<repo-name>
      5. HF cache via snapshot_download(local_files_only=True)
    """
    from huggingface_hub import snapshot_download
    if model_path and Path(model_path).exists():
        return str(Path(model_path).resolve())
    repo_id = HF_REPOS.get(model_key)
    if not repo_id:
        raise ValueError(f"Unknown model_key {model_key!r}; pass --model_path.")
    repo_basename = repo_id.split("/")[-1]
    candidates = []
    if env := os.environ.get("HACKATHON_MODELS_DIR"):
        candidates.append(Path(env) / repo_basename)
    candidates.append(Path("/data") / repo_basename)
    candidates.append(Path(__file__).resolve().parent.parent / "models" / repo_basename)
    for c in candidates:
        if c.exists():
            return str(c)
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate {model_key} ({repo_id}). Tried:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + f"\nPass --model_path, set HACKATHON_MODELS_DIR, or run download_models.py first."
        ) from e


# ---------- probe ----------
class AttentionProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)

    def forward(self, x_full, mask):
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(logits, dim=-1)
        pooled = (alpha.unsqueeze(-1) * x_full).sum(dim=1)
        return self.head(pooled).squeeze(-1), alpha


def load_probe(weights_path):
    ckpt = torch.load(str(weights_path), weights_only=False, map_location="cpu")
    sd = ckpt["state_dict"]
    d = sd["q"].shape[0]
    probe = AttentionProbe(d).to(DEVICE).to(torch.float32)
    probe.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    probe.eval()
    for p in probe.parameters(): p.requires_grad_(False)
    return probe, ckpt


def get_functional_ids(tok):
    ids = set(tok.all_special_ids or [])
    if hasattr(tok, "added_tokens_encoder"):
        ids |= set(tok.added_tokens_encoder.values())
    if hasattr(tok, "added_tokens_decoder"):
        ids |= set(tok.added_tokens_decoder.keys())
    for s in ["<bos>", "<eos>", "<|turn>", "<turn|>", "<channel|>", "<|channel>",
              "<start_of_turn>", "<end_of_turn>", "<pad>", "<unk>",
              "<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        try:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                ids.add(int(tid))
        except: pass
    return ids


def topk_editable(scores, n_tokens, ids, functional_ids, k=TOPK):
    s = np.asarray(scores)[:n_tokens]
    editable_mask = np.array([int(ids[i]) not in functional_ids for i in range(n_tokens)])
    s_masked = np.where(editable_mask, s, -np.inf)
    order = np.argsort(-s_masked)
    out = []
    for i in order:
        if not editable_mask[i]: break
        out.append((int(i), float(s[i])))
        if len(out) >= k: break
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", choices=["gemma4_31b", "qwen36"], default="gemma4_31b")
    ap.add_argument("--model_path", default=None,
                    help="Local HF snapshot dir. Falls back to ./models/<repo-name> or HF cache.")
    ap.add_argument("--probe_weights", required=True,
                    help="Path to .pt file with `state_dict` + `rows_test` list.")
    ap.add_argument("--extracts_dir", required=True,
                    help="Directory of per-sample .pt extracts (input_ids, n_tokens_total).")
    ap.add_argument("--out_dir", default="./edit_eval",
                    help="Where to write the JSON output.")
    ap.add_argument("--variant", choices=["early", "middle", "late"], default="middle")
    ap.add_argument("--layer_idx", type=int, default=None,
                    help="Override the early/middle/late default for this model.")
    ap.add_argument("--limit_tokens", type=int, default=2048,
                    help="Skip rollouts longer than this (autograd-budget constraint).")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--sample_limit", type=int, default=0,
                    help="0 = all eligible refusals, else first N.")
    return ap.parse_args()


def main():
    args = parse_args()
    layer_idx = (args.layer_idx if args.layer_idx is not None
                 else LAYER_IDX_DEFAULT[args.model_key][args.variant])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    extracts_dir = Path(args.extracts_dir)

    print(f"=== grad_input_embeds  model={args.model_key}  variant={args.variant}  layer={layer_idx} ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    probe, probe_ckpt = load_probe(args.probe_weights)
    auc = probe_ckpt.get("metrics", {}).get("auc")
    print(f"loaded probe ({args.probe_weights})" + (f" AUC={auc:.3f}" if auc else ""), flush=True)
    rows = probe_ckpt["rows_test"]

    model_path = resolve_model_path(args.model_key, args.model_path)
    print(f"loading model from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    # Enable gradient checkpointing — recompute activations during backward
    # to fit the autograd graph in 80GB VRAM.
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # ensures grads flow to inputs even with frozen params
    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else None
    print(f"model loaded | layers={n_layers}", flush=True)

    functional_ids = get_functional_ids(tok)

    # Filter to refusals within the autograd-friendly token budget
    refs = []; skipped = 0
    for r in rows:
        if r["split"] != "test" or r["label"] != 1: continue
        ex_path = extracts_dir / f"{r['sample_id']}.pt"
        if not ex_path.exists(): continue
        ex = torch.load(str(ex_path), weights_only=False, map_location="cpu")
        n_tot = ex.get("n_tokens_total") or ex.get("n_tokens")
        if n_tot and n_tot > args.limit_tokens: skipped += 1; continue
        refs.append((r, ex))
    refs.sort(key=lambda t: -t[0]["prob"])
    if args.sample_limit > 0: refs = refs[:args.sample_limit]
    print(f"eligible refusals: {len(refs)} (skipped {skipped} > {args.limit_tokens} tokens)", flush=True)

    use_chunked = (args.model_key == "gemma4_31b")
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm is not None: cm.__enter__()

    out_records = []
    t_start = time.time()
    try:
        for i, (r, ex) in enumerate(refs):
            sid = r["sample_id"]
            n = r["n_tokens"]
            ids = ex["input_ids"][:n].to(DEVICE).long().unsqueeze(0)  # (1, N)
            attn = torch.ones_like(ids, dtype=torch.long, device=DEVICE)

            # Get input embeddings, mark requires_grad
            embed_layer = model.get_input_embeddings()
            with torch.enable_grad():
                inputs_embeds = embed_layer(ids).detach().clone()  # (1, N, d_emb)
                inputs_embeds.requires_grad_(True)

                torch.cuda.synchronize(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()

                out = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # residual at chosen layer (layer_idx is for hidden_states tuple — index of post-layer i)
                # hidden_states[L+1] = output of transformer block L
                # Match Phase 4 indexing: phase4 used hidden_states[layer_idx_map[VARIANT]] from (n_layers+1) tuple
                residual = out.hidden_states[layer_idx][0].to(torch.float32)  # (N, d) cast to fp32 for probe
                mask = torch.ones(n, dtype=torch.bool, device=DEVICE)
                probe_logit, alpha = probe(residual.unsqueeze(0), mask.unsqueeze(0))
                probe_logit_scalar = probe_logit.squeeze()  # scalar

                probe_logit_scalar.backward()

                # input_embeds.grad shape: (1, N, d_emb)
                g = inputs_embeds.grad[0]                                       # (N, d_emb)
                e = inputs_embeds[0].detach()                                   # (N, d_emb)
                attrib = (g * e).sum(-1).detach().cpu().to(torch.float32).numpy()  # (N,)

                fwd = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1024**3

            # Free graph
            del out, inputs_embeds, residual, alpha, probe_logit, probe_logit_scalar, g, e
            torch.cuda.empty_cache()

            ids_list = ex["input_ids"][:n].tolist()
            pieces = tok.convert_ids_to_tokens(ids_list)
            pieces = [p.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n") for p in pieces]
            text_full = tok.decode(ids_list, skip_special_tokens=False)

            top = topk_editable(attrib, n, ids_list, functional_ids, k=args.topk)
            if not top:
                print(f"  [{i+1}/{len(refs)}] {sid}: no editable tokens after mask", flush=True); continue

            out_records.append({
                "sample_id": sid,
                "method": f"gradinput_embeds_{args.variant}",
                "model_key": args.model_key,
                "probe_prob": float(r["prob"]),
                "label": int(r["label"]),
                "n_tokens": n,
                "n_tokens_total": ex.get("n_tokens_total") or ex.get("n_tokens"),
                "prompt_text_decoded": text_full,
                "top_k_tokens": [
                    {"position": int(p), "token_text": pieces[p] if p < len(pieces) else "",
                     "alpha": float(s), "share_pct": float(s)}
                    for p, s in top
                ],
                "all_token_attrib": [round(float(x), 5) for x in attrib.tolist()],
                "all_token_text":   pieces,
                "_diag": {"fwd_seconds": round(fwd, 2), "peak_vram_gb": round(peak, 1),
                          "attrib_max": float(attrib.max()), "attrib_min": float(attrib.min())},
            })

            print(f"  [{i+1}/{len(refs)}] {sid}: N={n} fwd={fwd:.2f}s peak={peak:.1f}GB top1={out_records[-1]['top_k_tokens'][0]['token_text'][:30]!r}", flush=True)
    finally:
        if cm is not None: cm.__exit__(None, None, None)

    out_path = out_dir / f"eval_inputs_{args.model_key}_gradinput_embeds_{args.variant}.json"
    out_path.write_text(json.dumps(out_records, indent=2, ensure_ascii=False))
    print(f"\nDONE: wrote {out_path} ({len(out_records)} records, total {(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
