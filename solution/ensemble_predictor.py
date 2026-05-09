"""Ensemble predictor: DIM probe (0.6) + Attention probe (0.4).

Based on Cunningham et al. (Constitutional Classifiers++) insight that
ensembling distinct probe types dramatically improves robustness.

Ensemble score:
    score = 0.6 * sigmoid(dim_score) + 0.4 * sigmoid(attn_score)

The DIM probe needs only the last-token residual vector.
The Attention probe needs the full per-token residual sequence + mask.

Interface:
    predictor = load_predictor(dim_probe_path, attn_probe_path)
    score = predict(predictor, residuals, mask)   # float in [0, 1]
    auc   = evaluate_ensemble(predictor, samples, extracts_dir)

Usage:
    python solution/ensemble_predictor.py \\
        --dim_probe  /scratch/probes/dim/refusal_gemma4_31b_dim_probe.pt \\
        --attn_probe /scratch/probes/weights/refusal_gemma4_31b_attention.pt \\
        --test_jsonl /scratch/datasets/refusal_probes/gemma4_31b/test_split.jsonl \\
        --extracts_dir /scratch/extracts/gemma4_31b \\
        --out_dir /scratch/probes/ensemble

    # Evaluate all 5 tasks (provide DIM summary + matching attn probe dir):
    python solution/ensemble_predictor.py \\
        --all_tasks \\
        --dim_summary /scratch/probes/dim/dim_probe_summary.json \\
        --dim_probe_dir /scratch/probes/dim \\
        --attn_probe_dir /scratch/probes/weights \\
        --extracts_dir_cyber  /scratch/extracts/cyber \\
        --extracts_dir_gemma  /scratch/extracts/gemma4_31b \\
        --extracts_dir_qwen   /scratch/extracts/qwen36 \\
        --test_cyber  /scratch/datasets/cyber_probes/test.jsonl \\
        --test_gemma  /scratch/datasets/refusal_probes/gemma4_31b/test_split.jsonl \\
        --test_qwen   /scratch/datasets/refusal_probes/qwen36/test_split.jsonl \\
        --out_dir /scratch/probes/ensemble
"""
from __future__ import annotations
import os, sys, json, argparse, math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))
sys.path.insert(0, str(Path(__file__).parent))

from diff_in_means_probe import (
    load_probe as load_dim_probe,
    load_extract,
    get_layer_tensor,
    get_last_token,
    CYBER_LABEL_FNS,
    REFUSAL_LABEL_FN,
    filter_nonnull,
)

DEVICE = "cpu"
DTYPE  = torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Re-create AttentionProbe class (mirrors train_probe.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionProbe(nn.Module):
    """Single learned-query attention over tokens — mirrors train_probe.py."""
    def __init__(self, d: int):
        super().__init__()
        self.q    = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)

    def forward(
        self,
        x_final: Optional[torch.Tensor],
        x_full:  torch.Tensor,
        mask:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = x_full.shape[-1]
        logits = (x_full @ self.q) / math.sqrt(d)
        logits = logits.masked_fill(~mask, float("-inf"))
        alpha  = F.softmax(logits, dim=-1)               # (B, N)
        pooled = torch.einsum("bn,bnd->bd", alpha, x_full)
        return self.head(pooled).squeeze(-1), alpha       # (B,), (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor bundle
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """Holds both probes and ensemble weights."""

    def __init__(
        self,
        dim_probe: dict,
        attn_probe: Optional[nn.Module],
        attn_d: int,
        alpha_dim: float = 0.6,
        alpha_attn: float = 0.4,
    ):
        self.dim_probe  = dim_probe
        self.attn_probe = attn_probe  # may be None → degrade to DIM-only
        self.attn_d     = attn_d
        self.alpha_dim  = alpha_dim
        self.alpha_attn = alpha_attn if attn_probe is not None else 0.0
        # Renormalise weights if attn probe missing
        total = self.alpha_dim + self.alpha_attn
        self.alpha_dim  /= total
        self.alpha_attn /= total

    def score(
        self,
        residuals: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Return ensemble probability in [0, 1]."""
        return predict(self, residuals, mask)


def load_predictor(
    dim_probe_path: str | Path,
    attn_probe_path: Optional[str | Path] = None,
    alpha_dim: float = 0.6,
    alpha_attn: float = 0.4,
) -> EnsemblePredictor:
    """Load DIM + optional attention probe into an EnsemblePredictor.

    If attn_probe_path is None or the file doesn't exist, the ensemble
    degrades to DIM-only (alpha_dim=1.0).
    """
    dim = load_dim_probe(str(dim_probe_path))

    attn_probe = None
    attn_d     = dim.get("d_model", 2048)

    if attn_probe_path is not None:
        p = Path(attn_probe_path)
        if p.exists():
            ckpt = torch.load(str(p), weights_only=False, map_location="cpu")
            sd   = ckpt.get("state", ckpt.get("state_dict", {}))
            d    = sd["q"].shape[0]
            attn_d = d
            probe = AttentionProbe(d).to(DEVICE).to(DTYPE)
            probe.load_state_dict({k: v.to(DTYPE) for k, v in sd.items()})
            probe.eval()
            for param in probe.parameters():
                param.requires_grad_(False)
            attn_probe = probe
        else:
            print(f"[warn] attn_probe not found at {attn_probe_path} — using DIM-only")

    return EnsemblePredictor(dim, attn_probe, attn_d, alpha_dim, alpha_attn)


def predict(predictor: EnsemblePredictor, residuals: torch.Tensor, mask: torch.Tensor) -> float:
    """Score a single sample. Returns probability in [0, 1].

    Args:
        predictor : EnsemblePredictor
        residuals : (n_layers, n_tokens, d) or (n_tokens, d)
        mask      : (n_tokens,) bool

    Returns float in [0, 1].
    """
    dim   = predictor.dim_probe
    r_hat = dim["r_hat"].to(DTYPE)
    layer_idx = dim.get("layer_idx", 0)

    # ── DIM score ─────────────────────────────────────────────────────────────
    if residuals.dim() == 3:
        n_lay = residuals.shape[0]
        idx   = min(layer_idx, n_lay - 1)
        full  = residuals[idx].to(DTYPE)
    else:
        full = residuals.to(DTYPE)

    m   = mask.bool() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.bool)
    vec = get_last_token(full, m)
    dim_score = float(torch.dot(vec, r_hat).item())
    dim_prob  = 1.0 / (1.0 + math.exp(-dim_score))

    if predictor.attn_probe is None or predictor.alpha_attn == 0.0:
        return dim_prob

    # ── Attention probe score ─────────────────────────────────────────────────
    # Attention probe uses full token sequence
    x_full = full.unsqueeze(0)      # (1, N, d)
    x_mask = m.unsqueeze(0)         # (1, N)

    with torch.no_grad():
        out = predictor.attn_probe(None, x_full, x_mask)
        logit_attn = out[0] if isinstance(out, tuple) else out
        logit_attn = float(logit_attn.item())
    attn_prob = 1.0 / (1.0 + math.exp(-logit_attn))

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ensemble_prob = predictor.alpha_dim * dim_prob + predictor.alpha_attn * attn_prob
    return float(ensemble_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ensemble(
    predictor: EnsemblePredictor,
    samples: list[dict],
    label_fn,
    extracts_dir: Path,
    id_key: str = "sample_id",
) -> dict:
    """Compute AUC + ACC of ensemble on a list of samples."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    probs, labels, ids = [], [], []
    skipped = 0
    for s in samples:
        sid = s[id_key]
        lbl = label_fn(s)
        if lbl is None:
            continue
        path = extracts_dir / f"{sid}.pt"
        if not path.exists():
            skipped += 1
            continue
        try:
            ex        = load_extract(extracts_dir, sid)
            residuals = ex["residuals"]
            mask      = ex["attention_mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)
            p = predict(predictor, residuals, mask)
            probs.append(p)
            labels.append(int(lbl))
            ids.append(sid)
        except Exception as e:
            print(f"  [warn] {sid}: {e}")
            skipped += 1

    if skipped:
        print(f"  skipped {skipped} samples")

    if not probs:
        return {"auc": float("nan"), "acc": float("nan"), "n": 0}

    y_true = np.array(labels, dtype=int)
    y_prob = np.array(probs,  dtype=float)
    y_pred = (y_prob > 0.5).astype(int)

    auc = float("nan")
    acc = float("nan")
    if len(set(y_true.tolist())) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        acc = float(accuracy_score(y_true, y_pred))

    return {
        "auc":  auc,
        "acc":  acc,
        "n":    len(probs),
        "n_pos":  int(y_true.sum()),
        "n_neg":  int((1 - y_true).sum()),
        "ids":  ids,
        "probs": [round(float(p), 5) for p in y_prob.tolist()],
        "labels": y_true.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all_tasks", action="store_true")

    # Single-task mode
    ap.add_argument("--dim_probe",   default=None, help="DIM probe .pt")
    ap.add_argument("--attn_probe",  default=None, help="Attention probe .pt (optional)")
    ap.add_argument("--test_jsonl",  default=None, help="Test split JSONL")
    ap.add_argument("--extracts_dir",default=None)
    ap.add_argument("--task",        default=None,
                    help="Task name (e.g. refusal_gemma4_31b) for label resolution")

    # All-tasks mode
    ap.add_argument("--dim_summary",   default=None, help="dim_probe_summary.json from diff_in_means_probe.py")
    ap.add_argument("--dim_probe_dir", default=None, help="Directory of DIM probe .pt files")
    ap.add_argument("--attn_probe_dir",default=None, help="Directory of attention probe .pt files")
    ap.add_argument("--extracts_dir_cyber", default=None)
    ap.add_argument("--extracts_dir_gemma", default=None)
    ap.add_argument("--extracts_dir_qwen",  default=None)
    ap.add_argument("--test_cyber",  default=None)
    ap.add_argument("--test_gemma",  default=None)
    ap.add_argument("--test_qwen",   default=None)

    # Shared
    ap.add_argument("--alpha_dim",   type=float, default=0.6)
    ap.add_argument("--alpha_attn",  type=float, default=0.4)
    ap.add_argument("--out_dir",     default="./probes/ensemble")
    return ap.parse_args()


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def task_label_fn(task_name: str):
    if task_name in CYBER_LABEL_FNS:
        return CYBER_LABEL_FNS[task_name]
    if task_name.startswith("refusal"):
        return REFUSAL_LABEL_FN
    raise ValueError(f"Unknown task: {task_name}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    if args.all_tasks:
        if not (args.dim_summary and args.dim_probe_dir):
            raise SystemExit("--all_tasks requires --dim_summary and --dim_probe_dir")

        summary = json.load(open(args.dim_summary))
        dim_probe_dir  = Path(args.dim_probe_dir)
        attn_probe_dir = Path(args.attn_probe_dir) if args.attn_probe_dir else None

        # Map task → extracts dir and test JSONL
        task_config = {
            "cyber1_dual_use_vs_benign":     (args.extracts_dir_cyber, args.test_cyber,
                                               CYBER_LABEL_FNS["cyber1_dual_use_vs_benign"]),
            "cyber2_high_risk_vs_dual_benign":(args.extracts_dir_cyber, args.test_cyber,
                                               CYBER_LABEL_FNS["cyber2_high_risk_vs_dual_benign"]),
            "cyber3_prohibited_vs_rest":      (args.extracts_dir_cyber, args.test_cyber,
                                               CYBER_LABEL_FNS["cyber3_prohibited_vs_rest"]),
            "refusal_gemma4_31b":             (args.extracts_dir_gemma, args.test_gemma,
                                               REFUSAL_LABEL_FN),
            "refusal_qwen36":                 (args.extracts_dir_qwen,  args.test_qwen,
                                               REFUSAL_LABEL_FN),
        }

        for task_name, (ext_dir_str, test_path, label_fn) in task_config.items():
            if not (ext_dir_str and test_path):
                print(f"[skip] {task_name} — missing paths")
                continue

            dim_path  = dim_probe_dir / f"{task_name}_dim_probe.pt"
            if not dim_path.exists():
                print(f"[skip] {task_name} — DIM probe not found at {dim_path}")
                continue

            attn_path = None
            if attn_probe_dir:
                # train_probe.py saves as {task_name}_attention.pt
                candidate = attn_probe_dir / f"{task_name}_attention.pt"
                if candidate.exists():
                    attn_path = candidate

            predictor = load_predictor(dim_path, attn_path, args.alpha_dim, args.alpha_attn)

            test_samples = load_jsonl(test_path)
            if label_fn in CYBER_LABEL_FNS.values():
                test_samples = filter_nonnull(test_samples, label_fn)

            ext_dir = Path(ext_dir_str)
            print(f"\n{'='*60}")
            print(f"TASK: {task_name}  n_test={len(test_samples)}  "
                  f"alpha=(dim={predictor.alpha_dim:.2f}, attn={predictor.alpha_attn:.2f})")

            result = evaluate_ensemble(predictor, test_samples, label_fn, ext_dir)
            result["task"] = task_name
            all_metrics.append(result)
            print(f"  ensemble_auc={result['auc']:.4f}  acc={result['acc']:.4f}  n={result['n']}")

            # Save per-task scores
            scores_path = out_dir / f"{task_name}_ensemble_scores.json"
            scores_path.write_text(json.dumps({
                "task": task_name,
                "auc": result["auc"],
                "acc": result["acc"],
                "n": result["n"],
                "alpha_dim": predictor.alpha_dim,
                "alpha_attn": predictor.alpha_attn,
                "sample_scores": [
                    {"sample_id": sid, "prob": p, "label": lbl}
                    for sid, p, lbl in zip(result["ids"], result["probs"], result["labels"])
                ],
            }, indent=2))

    else:
        # Single-task mode
        if not (args.dim_probe and args.test_jsonl and args.extracts_dir):
            raise SystemExit("Provide --dim_probe, --test_jsonl, --extracts_dir "
                             "(or use --all_tasks)")

        predictor = load_predictor(args.dim_probe, args.attn_probe, args.alpha_dim, args.alpha_attn)

        # Determine label function
        task_name = args.task or Path(args.dim_probe).stem.replace("_dim_probe", "")
        label_fn  = task_label_fn(task_name)

        test_samples = load_jsonl(args.test_jsonl)
        if task_name in CYBER_LABEL_FNS:
            test_samples = filter_nonnull(test_samples, label_fn)

        print(f"Task: {task_name}  n_test={len(test_samples)}  "
              f"alpha=(dim={predictor.alpha_dim:.2f}, attn={predictor.alpha_attn:.2f})")
        result = evaluate_ensemble(predictor, test_samples, label_fn, Path(args.extracts_dir))
        result["task"] = task_name
        all_metrics.append(result)
        print(f"ensemble_auc={result['auc']:.4f}  acc={result['acc']:.4f}  n={result['n']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENSEMBLE SUMMARY")
    print(f"{'task':<42} {'auc':>8} {'acc':>8} {'n':>6}")
    print("-" * 66)
    aucs = []
    for m in all_metrics:
        auc = m["auc"]
        acc = m.get("acc", float("nan"))
        print(f"  {m['task']:<40} {auc:>8.4f} {acc:>8.4f} {m['n']:>6}")
        if not math.isnan(auc):
            aucs.append(auc)
    if aucs:
        print(f"\n  Mean ensemble AUC: {np.mean(aucs):.4f}")

    summary_path = out_dir / "ensemble_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "metrics": all_metrics,
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "alpha_dim": args.alpha_dim,
            "alpha_attn": args.alpha_attn,
        }, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
