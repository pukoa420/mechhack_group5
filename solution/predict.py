"""Level-1 submission: Diff-in-Means probe bundle.

load_probe() returns a dict of all task-specific DIM probes.
predict() routes to the right probe based on residual d_model
(Qwen has a different hidden dim than Gemma).

For Gemma-sourced residuals the scorer may call us for both cyber
tasks and refusal_gemma.  We use the task-specific r_hat when we
can disambiguate, falling back to the refusal direction (which
correlates with harm level due to RLHF).
"""
from __future__ import annotations
import math, os
from pathlib import Path
from typing import Any

import torch
import numpy as np

# ── probe locations ───────────────────────────────────────────────────────────
# Adjust PROBE_DIR to wherever the probes were saved on the scoring machine.
PROBE_DIR = Path(os.environ.get(
    "PROBE_DIR",
    "/scratch/outputs/probes/dim"
))

PROBE_FILES = {
    "cyber1":          "cyber1_dual_use_vs_benign_dim_probe.pt",
    "cyber2":          "cyber2_high_risk_vs_dual_benign_dim_probe.pt",
    "cyber3":          "cyber3_prohibited_vs_rest_dim_probe.pt",
    "refusal_gemma":   "refusal_gemma4_31b_dim_probe.pt",
    "refusal_qwen":    "refusal_qwen36_dim_probe.pt",
}


def load_probe() -> dict:
    """Load all DIM probe checkpoints into a bundle dict."""
    bundle = {}
    for key, fname in PROBE_FILES.items():
        path = PROBE_DIR / fname
        if path.exists():
            ckpt = torch.load(str(path), weights_only=False, map_location="cpu")
            bundle[key] = {
                "r_hat":     ckpt["r_hat"].to(torch.float32),
                "layer_idx": ckpt.get("layer_idx", 0),
                "d_model":   ckpt.get("d_model"),
                "n_layers":  ckpt.get("n_layers", 1),
                "metrics":   ckpt.get("metrics", {}),
            }
        else:
            print(f"[warn] probe not found: {path}")
    if not bundle:
        raise RuntimeError(f"No probes found in {PROBE_DIR}")
    return bundle


def _get_vec(residuals: torch.Tensor, mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Extract last-token vector at the given layer."""
    r = residuals
    if r.dim() == 3:
        n_layers = r.shape[0]
        idx = min(layer_idx, n_layers - 1)
        r = r[idx]
    r = r.to(torch.float32)
    valid = mask.nonzero()
    last = int(valid.max().item()) if len(valid) > 0 else r.shape[0] - 1
    return r[last]


def predict(
    probe_bundle: dict,
    residuals,
    attention_mask=None,
) -> float:
    """Score a single sample against the most appropriate DIM probe.

    Routing logic:
    - d_model of the residuals identifies the model family (Qwen ≠ Gemma).
    - For Qwen residuals → refusal_qwen probe.
    - For Gemma residuals → cyber3 probe as general harm direction
      (cyber3 = prohibited vs all, which spans the full harm ladder;
       the refusal direction also correlates but cyber3 is more
       directly calibrated to the harm taxonomy used for all 3
       cyber tasks, and also generalises to refusal since the model
       refuses harmful prompts).
    """
    if isinstance(residuals, np.ndarray):
        residuals = torch.from_numpy(residuals)
    residuals = residuals.to(torch.float32)

    if attention_mask is None:
        attention_mask = torch.ones(residuals.shape[-2], dtype=torch.bool)
    elif isinstance(attention_mask, np.ndarray):
        attention_mask = torch.from_numpy(attention_mask).bool()
    else:
        attention_mask = attention_mask.bool()

    # Identify d_model
    d = residuals.shape[-1]

    # Pick probe by model family
    qwen_probe = probe_bundle.get("refusal_qwen")
    gemma_cyber3 = probe_bundle.get("cyber3")
    gemma_refusal = probe_bundle.get("refusal_gemma")

    if qwen_probe is not None and d == qwen_probe["r_hat"].shape[0]:
        # Qwen residuals
        probe = qwen_probe
    elif gemma_cyber3 is not None and d == gemma_cyber3["r_hat"].shape[0]:
        # Gemma residuals — use cyber3 (general harm direction)
        # It works for all 3 cyber tasks and correlates with refusal
        probe = gemma_cyber3
    elif gemma_refusal is not None:
        # Fallback
        probe = gemma_refusal
    else:
        # Last resort: first available probe
        probe = next(iter(probe_bundle.values()))

    vec = _get_vec(residuals, attention_mask, probe["layer_idx"])
    r_hat = probe["r_hat"]
    raw = float(torch.dot(vec, r_hat).item())
    # Sigmoid to map to [0, 1]
    return float(1.0 / (1.0 + math.exp(-raw)))


# ── quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bundle = load_probe()
    print("Loaded probes:")
    for k, v in bundle.items():
        m = v.get("metrics", {})
        print(f"  {k:25s}  layer={v['layer_idx']}  d_model={v['d_model']}"
              f"  test_auc={m.get('test_auc', float('nan')):.4f}")
    aucs = [v["metrics"].get("test_auc", float("nan"))
            for v in bundle.values()
            if not math.isnan(v["metrics"].get("test_auc", float("nan")))]
    if aucs:
        import numpy as np
        print(f"\nMean test AUC across {len(aucs)} tasks: {np.mean(aucs):.4f}")
