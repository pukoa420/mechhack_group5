"""Refusal-direction token attribution (Arditi et al. DFA method).

Two attribution methods:
  Method A (residual projection, fast, no model needed):
      attrib[t] = dot(residual[layer, t, :], r̂)
      Measures per-token contribution to the refusal direction in residual stream.

  Method B (DFA — Direct Feature Attribution, richer, needs model forward pass):
      For each attention head h, compute dfa[h] = attention_output[h] @ r̂
      Then token_attrib[t] = sum over top-8 heads of: dfa[h] * attn_weight[h, last_pos, t]
      Identifies which SOURCE tokens head h is pulling refusal signal from.

Method A works offline from pre-extracted residuals.
Method B requires loading the transformer model (use on cluster only).

Usage (Method A, offline):
    python solution/refusal_direction_attribution.py \\
        --probe_path /scratch/probes/dim/refusal_gemma4_31b_dim_probe.pt \\
        --extracts_dir /scratch/extracts/gemma4_31b \\
        --method A \\
        --out_dir /scratch/attribution

Usage (Method B, needs model on GPU):
    python solution/refusal_direction_attribution.py \\
        --probe_path /scratch/probes/dim/refusal_gemma4_31b_dim_probe.pt \\
        --model_key gemma4_31b \\
        --samples_jsonl /scratch/datasets/refusal_probes/gemma4_31b/test_split.jsonl \\
        --method B \\
        --out_dir /scratch/attribution
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))
sys.path.insert(0, str(Path(__file__).parent))

from diff_in_means_probe import load_probe, get_layer_tensor, get_last_token


# ─────────────────────────────────────────────────────────────────────────────
# Method A: residual-stream per-token projection
# ─────────────────────────────────────────────────────────────────────────────

def attrib_method_a(
    residuals: torch.Tensor,  # (n_layers, n_tokens, d) or (n_tokens, d)
    mask: torch.Tensor,       # (n_tokens,) bool
    r_hat: torch.Tensor,      # (d,) unit vector
    layer_idx: int = 0,
) -> np.ndarray:
    """Per-token dot product with refusal direction. Returns (n_tokens,) float32."""
    if residuals.dim() == 3:
        n_lay = residuals.shape[0]
        idx = min(layer_idx, n_lay - 1)
        full = residuals[idx].to(torch.float32)
    else:
        full = residuals.to(torch.float32)
    r = r_hat.to(torch.float32)
    # (n_tokens,) — dot each token vector with r̂
    scores = torch.mv(full, r)
    # Zero out padding positions
    if mask is not None:
        m = mask.bool() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)
        scores = scores * m.float()
    return scores.numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Method B: DFA via attention head outputs (needs live model)
# ─────────────────────────────────────────────────────────────────────────────

def _register_attn_hooks(model, layer_idx: int, n_heads: int):
    """Register forward hooks to capture:
      - attention weights: (1, n_heads, n_q, n_kv)
      - per-head value projections: (1, n_heads, n_q, d_head)
    Returns (handles, storage_dict).
    """
    storage = {"attn_weights": None, "head_outputs": None}
    handles = []

    def _get_block(model, li: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[li - 1]  # hidden_states index li → block li-1
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[li - 1]
        raise RuntimeError("Cannot find transformer blocks — unsupported architecture")

    # We hook the self-attention module inside the block.
    # Different HF model names use different attribute names.
    block = _get_block(model, layer_idx)
    attn_module = None
    for attr in ["self_attn", "attn", "attention", "self_attention"]:
        if hasattr(block, attr):
            attn_module = getattr(block, attr)
            break
    if attn_module is None:
        raise RuntimeError(f"Cannot find attention module in block {layer_idx}")

    def _hook(module, inp, out):
        # `out` is typically (hidden_state, attn_weights, ...) or just hidden_state
        if isinstance(out, tuple):
            hidden = out[0]  # (1, n_q, d_model)
            attn_w = out[1] if len(out) > 1 and out[1] is not None else None
        else:
            hidden = out
            attn_w = None
        storage["attn_weights"] = attn_w
        storage["head_outputs"] = hidden  # shape (1, n_q, d_model)

    h = attn_module.register_forward_hook(_hook)
    handles.append(h)
    return handles, storage


def attrib_method_b(
    input_ids: torch.Tensor,   # (1, n_tokens) or (n_tokens,)
    r_hat: torch.Tensor,        # (d,) unit vector
    model,
    layer_idx: int,
    top_k_heads: int = 8,
) -> np.ndarray:
    """DFA attribution: project head outputs onto r̂, weight by attention.

    Returns (n_tokens,) float32 attribution scores.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device)
    n_tokens = input_ids.shape[1]

    # Count heads from model config
    n_heads = getattr(model.config, "num_attention_heads", 32)

    handles, storage = _register_attn_hooks(model, layer_idx, n_heads)
    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_attentions=True,  # request attention weights from HF
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for h in handles:
            h.remove()

    r = r_hat.to(torch.float32).to(model.device)

    # HF returns attentions as a tuple of (1, n_heads, n_q, n_kv) per layer
    # index layer_idx-1 since layer 0 = embedding output
    attn_weights_all = out.get("attentions") if hasattr(out, "get") else getattr(out, "attentions", None)

    # Fall back to Method A if attention weights not available
    if attn_weights_all is None:
        # Grab residual at this layer from hidden_states
        hs = out.get("hidden_states") if hasattr(out, "get") else getattr(out, "hidden_states", None)
        if hs is not None:
            residual = hs[layer_idx][0].to(torch.float32)  # (n_tokens, d)
            scores = torch.mv(residual, r).cpu().numpy().astype(np.float32)
            return scores
        return np.zeros(n_tokens, dtype=np.float32)

    # layer_idx-1 because attentions[0] = block 0 = hidden_states index 1
    li_attn = layer_idx - 1
    if li_attn < 0 or li_attn >= len(attn_weights_all):
        li_attn = min(max(0, li_attn), len(attn_weights_all) - 1)

    attn_w = attn_weights_all[li_attn]  # (1, n_heads, n_q, n_kv) bfloat16
    if attn_w is None:
        return np.zeros(n_tokens, dtype=np.float32)

    attn_w = attn_w[0].to(torch.float32)  # (n_heads, n_q, n_kv)

    # Get per-head hidden state projections
    # hidden_states[layer_idx] has shape (1, n_tokens, d_model)
    hs = getattr(out, "hidden_states", None)
    if hs is not None:
        residual = hs[layer_idx][0].to(torch.float32)  # (n_tokens, d)
    elif storage["head_outputs"] is not None:
        residual = storage["head_outputs"][0].to(torch.float32)
    else:
        return np.zeros(n_tokens, dtype=np.float32)

    # DFA: project residual onto r̂ per token
    token_proj = torch.mv(residual, r)  # (n_tokens,)

    # Weight each source token by how much the last-position query attends to it
    # averaged across top-K heads by their DFA score
    last_pos = n_tokens - 1

    # Compute DFA score for each head: how much does this head's attention at
    # last position project onto r̂?  We approximate by dot(residual[last], r̂)
    # weighted by each head's attention entropy / max.
    # Simpler: use attn_w[:, last_pos, :] as redistribution weights per head.
    attn_from_last = attn_w[:, last_pos, :]  # (n_heads, n_kv)

    # Score each head by its DFA: sum_t attn[h, last, t] * residual_proj[t]
    head_dfa = torch.mv(attn_from_last, token_proj)  # (n_heads,)

    # Select top-k heads by absolute DFA
    k = min(top_k_heads, n_heads)
    top_heads = head_dfa.abs().topk(k).indices  # (k,)

    # Token attributions: for each top head, distribute its DFA to source tokens
    token_attrib = torch.zeros(n_tokens, device=model.device)
    for hi in top_heads:
        head_score = head_dfa[hi]  # scalar
        attn_dist  = attn_from_last[hi]  # (n_kv,)
        # Clamp n_kv to n_tokens (KV cache may differ for grouped-query attn)
        n_kv = attn_dist.shape[0]
        n = min(n_kv, n_tokens)
        token_attrib[:n] += head_score * attn_dist[:n]

    return token_attrib.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# High-level per-sample attribution (offline, Method A)
# ─────────────────────────────────────────────────────────────────────────────

def compute_attribution_offline(
    probe: dict,
    extracts_dir: Path,
    sample_id: str,
) -> dict:
    """Load extract and compute Method A attribution. Returns result dict."""
    from diff_in_means_probe import load_extract

    ex = load_extract(extracts_dir, sample_id)
    residuals = ex["residuals"]
    mask = ex["attention_mask"]
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool)

    r_hat     = probe["r_hat"]
    layer_idx = probe.get("layer_idx", 0)

    attrib = attrib_method_a(residuals, mask, r_hat, layer_idx=layer_idx)

    input_ids = ex.get("input_ids", None)
    n_tokens  = int(mask.sum().item())

    return {
        "sample_id":  sample_id,
        "method":     "dim_method_a",
        "layer_idx":  layer_idx,
        "label":      int(ex.get("label", -1)),
        "probe_score": float(attrib[mask.bool().nonzero().max().item()]),
        "attribution": attrib.tolist(),
        "n_tokens":   n_tokens,
        "input_ids":  input_ids.tolist() if input_ids is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility with iterative_edit_agent.py interface
# ─────────────────────────────────────────────────────────────────────────────

def get_token_attribution(
    probe: dict,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    method: str = "A",
    model=None,
    input_ids: Optional[torch.Tensor] = None,
    top_k_heads: int = 8,
) -> np.ndarray:
    """Unified interface returning (n_tokens,) float32 attribution.

    method='A': residual projection (no model needed)
    method='B': DFA via attention heads (requires `model` and `input_ids`)
    """
    r_hat     = probe["r_hat"].to(torch.float32)
    layer_idx = probe.get("layer_idx", 0)

    if method == "A" or model is None:
        return attrib_method_a(residuals, mask, r_hat, layer_idx=layer_idx)
    else:
        if input_ids is None:
            raise ValueError("Method B requires input_ids")
        return attrib_method_b(input_ids, r_hat, model, layer_idx, top_k_heads=top_k_heads)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe_path",    required=True,
                    help="Path to DIM probe .pt file")
    ap.add_argument("--extracts_dir",  default=None,
                    help="Directory of pre-extracted .pt residuals (Method A)")
    ap.add_argument("--samples_jsonl", default=None,
                    help="JSONL with sample_id fields to process")
    ap.add_argument("--method",        choices=["A", "B"], default="A")
    ap.add_argument("--model_key",     choices=["gemma4_31b", "qwen36"], default=None,
                    help="Required for Method B")
    ap.add_argument("--model_path",    default=None)
    ap.add_argument("--top_k_heads",   type=int, default=8)
    ap.add_argument("--sample_limit",  type=int, default=0)
    ap.add_argument("--out_dir",       default="./attribution")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = load_probe(args.probe_path)
    print(f"Loaded probe: task={probe.get('task')}  layer={probe.get('layer_idx')}  "
          f"test_auc={probe.get('metrics', {}).get('test_auc', '?'):.4f}")

    # Collect sample IDs from probe rows_test or from samples_jsonl
    if args.samples_jsonl:
        samples = [json.loads(l) for l in open(args.samples_jsonl) if l.strip()]
    elif "rows_test" in probe:
        samples = probe["rows_test"]
    else:
        raise SystemExit("Provide --samples_jsonl or use a probe with rows_test")

    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]

    if args.method == "B":
        # Load model
        sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))
        from extract_residuals import resolve_model_path
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from chunked_sdpa import chunked_sdpa_scope

        model_path = resolve_model_path(args.model_key, args.model_path)
        print(f"Loading model from {model_path}…")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager",   # need eager for output_attentions
            device_map=device, trust_remote_code=True)
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        use_chunked = (args.model_key == "gemma4_31b")
        cm = chunked_sdpa_scope() if use_chunked else None
        if cm is not None: cm.__enter__()
    else:
        tok = None; model = None; cm = None
        if not args.extracts_dir:
            raise SystemExit("Method A requires --extracts_dir")
        extracts_dir = Path(args.extracts_dir)

    records = []
    try:
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            try:
                if args.method == "A":
                    rec = compute_attribution_offline(probe, extracts_dir, sid)
                else:
                    # Method B: need to re-tokenize original prompt
                    prompt = s.get("attack_prompt") or s.get("prompt", "")
                    txt = tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)
                    enc = tok(txt, return_tensors="pt")
                    ids = enc.input_ids.to(model.device)
                    mask = enc.attention_mask[0].bool()
                    r_hat     = probe["r_hat"].to(torch.float32)
                    layer_idx = probe.get("layer_idx", 0)
                    attrib = attrib_method_b(ids, r_hat, model, layer_idx, args.top_k_heads)
                    rec = {
                        "sample_id": sid,
                        "method": "dim_method_b",
                        "layer_idx": layer_idx,
                        "label": int(s.get("is_refusal", s.get("label", -1))),
                        "attribution": attrib.tolist(),
                        "n_tokens": int(ids.shape[1]),
                        "input_ids": ids[0].cpu().tolist(),
                    }
                records.append(rec)
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [{i+1}/{len(samples)}] {sid}")
            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] {sid}: FAIL {type(e).__name__}: {e}")
    finally:
        if cm is not None: cm.__exit__(None, None, None)

    task = probe.get("task", "unknown")
    out_path = out_dir / f"{task}_attribution_method{args.method}.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nDone: {len(records)} records → {out_path}")


if __name__ == "__main__":
    main()
