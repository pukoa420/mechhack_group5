"""Diff-in-means (Arditi et al.) refusal direction probe.

Computes r̂ = (μ_pos - μ_neg) / ||μ_pos - μ_neg|| from training activations,
then uses dot(activation, r̂) as the probe score.  No gradient, no training —
just the diff-in-means direction.

Works for all 5 hackathon tasks:
  cyber-1 : dual_use vs benign
  cyber-2 : high_risk_dual_use vs {dual_use, benign}
  cyber-3 : prohibited vs {high_risk_dual_use, dual_use, benign}
  refusal-gemma : refusal vs compliance on gemma4_31b
  refusal-qwen  : refusal vs compliance on qwen36

Usage (single task):
    python solution/diff_in_means_probe.py \\
        --task refusal_gemma4_31b \\
        --extracts_dir /scratch/extracts/gemma4_31b \\
        --train_manifest /scratch/datasets/refusal_probes/gemma4_31b/train_split.jsonl \\
        --test_manifest  /scratch/datasets/refusal_probes/gemma4_31b/test_split.jsonl \\
        --out_dir /scratch/probes/dim

Usage (all 5 tasks via combined manifest):
    python solution/diff_in_means_probe.py \\
        --all_tasks \\
        --cyber_train  /scratch/datasets/cyber_probes/train.jsonl \\
        --cyber_test   /scratch/datasets/cyber_probes/test.jsonl \\
        --refusal_gemma_train /scratch/datasets/refusal_probes/gemma4_31b/train_split.jsonl \\
        --refusal_gemma_test  /scratch/datasets/refusal_probes/gemma4_31b/test_split.jsonl \\
        --refusal_qwen_train  /scratch/datasets/refusal_probes/qwen36/train_split.jsonl \\
        --refusal_qwen_test   /scratch/datasets/refusal_probes/qwen36/test_split.jsonl \\
        --cyber_extracts_dir  /scratch/extracts/cyber \\
        --gemma_extracts_dir  /scratch/extracts/gemma4_31b \\
        --qwen_extracts_dir   /scratch/extracts/qwen36 \\
        --out_dir /scratch/probes/dim
"""
from __future__ import annotations
import os, sys, json, argparse, math
from pathlib import Path
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "starter_code"))

DEVICE = "cpu"  # Level 1 never needs GPU — only pre-extracted residuals


# ─────────────────────────────────────────────────────────────────────────────
# Residual loading helpers (mirrors train_probe.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_extract(extracts_dir: Path, sample_id: str) -> dict:
    path = extracts_dir / f"{sample_id}.pt"
    return torch.load(str(path), weights_only=False, map_location="cpu")


def get_layer_tensor(ex: dict, layer_idx: Optional[int] = None) -> torch.Tensor:
    """Return (n_tokens, d_model) float32 tensor for one layer.

    If `residuals` has shape (n_layers, n_tokens, d_model) and layer_idx is
    given, picks that layer index into the first dimension.  Falls back to
    first layer if layer_idx is out of range.
    """
    r = ex["residuals"]
    if r.dim() == 2:
        return r.to(torch.float32)
    # (n_layers, n_tokens, d_model)
    n_layers = r.shape[0]
    if layer_idx is None:
        layer_idx = 0
    idx = min(layer_idx, n_layers - 1)
    return r[idx].to(torch.float32)


def get_last_token(full: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the last real-token vector (before padding)."""
    valid = mask.nonzero()
    if len(valid) == 0:
        return full[-1]
    last = int(valid.max().item())
    return full[last]


# ─────────────────────────────────────────────────────────────────────────────
# Core diff-in-means logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_dim_direction(
    pos_vecs: torch.Tensor,  # (n_pos, d)
    neg_vecs: torch.Tensor,  # (n_neg, d)
) -> torch.Tensor:
    """Return unit-normalised diff-in-means direction r̂."""
    mu_pos = pos_vecs.mean(dim=0)
    mu_neg = neg_vecs.mean(dim=0)
    r = mu_pos - mu_neg
    norm = r.norm()
    if norm < 1e-8:
        raise RuntimeError("Diff-in-means direction has near-zero norm — check labels.")
    return r / norm  # (d,)


def score_vec(vec: torch.Tensor, r_hat: torch.Tensor) -> float:
    """Dot product of a single activation vector with r̂."""
    return float(torch.dot(vec.to(r_hat), r_hat).item())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def compute_dim_streaming(
    samples: list[dict],
    label_fn,
    extracts_dir: Path,
    layer_idx: Optional[int],
    id_key: str = "sample_id",
) -> tuple[torch.Tensor, int, int]:
    """Compute DIM direction without loading all vectors into memory at once.

    Accumulates running sums for pos/neg classes, frees each .pt file immediately.
    Returns (r_hat, n_pos, n_neg).
    """
    sum_pos = None; count_pos = 0
    sum_neg = None; count_neg = 0
    skipped = 0
    for s in samples:
        sid = s[id_key]
        try:
            ex = load_extract(extracts_dir, sid)
        except FileNotFoundError:
            skipped += 1; continue
        full = get_layer_tensor(ex, layer_idx)
        mask = ex["attention_mask"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        vec = get_last_token(full, mask.bool()).to(torch.float32)
        del ex, full  # free large tensor immediately
        lbl = label_fn(s)
        if lbl is None:
            skipped += 1; continue
        if lbl == 1.0:
            sum_pos = vec.clone() if sum_pos is None else sum_pos + vec
            count_pos += 1
        else:
            sum_neg = vec.clone() if sum_neg is None else sum_neg + vec
            count_neg += 1
    if skipped:
        print(f"    [warn] skipped {skipped}/{len(samples)} samples (missing extracts or label)")
    if count_pos == 0 or count_neg == 0:
        raise RuntimeError(f"Training set missing one class: pos={count_pos} neg={count_neg}")
    print(f"  train loaded (streaming): pos={count_pos}  neg={count_neg}")
    r = (sum_pos / count_pos) - (sum_neg / count_neg)
    norm = r.norm()
    if norm < 1e-8:
        raise RuntimeError("DIM direction has near-zero norm — check labels.")
    return r / norm, count_pos, count_neg


def build_vecs_from_samples(
    samples: list[dict],
    label_fn,
    extracts_dir: Path,
    layer_idx: Optional[int] = None,
    id_key: str = "sample_id",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load last-token residuals for each sample.

    Returns:
        vecs : (N, d) float32
        labels : (N,) float32  (1 = positive class, 0 = negative)
        ids    : list[str]
    """
    vecs, labels, ids = [], [], []
    skipped = 0
    for s in samples:
        sid = s[id_key]
        try:
            ex = load_extract(extracts_dir, sid)
        except FileNotFoundError:
            skipped += 1
            continue
        full = get_layer_tensor(ex, layer_idx)  # (N, d)
        mask = ex["attention_mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = torch.tensor(mask, dtype=torch.bool)
        vec = get_last_token(full, mask)  # (d,)
        lbl = label_fn(s)
        if lbl is None:
            skipped += 1
            continue
        vecs.append(vec)
        labels.append(float(lbl))
        ids.append(sid)
    if skipped:
        print(f"    [warn] skipped {skipped}/{len(samples)} samples (missing extracts or label)")
    if not vecs:
        raise RuntimeError("No vectors loaded — check extracts_dir and manifest paths.")
    return (torch.stack(vecs, dim=0),
            torch.tensor(labels, dtype=torch.float32),
            ids)


# ─────────────────────────────────────────────────────────────────────────────
# Layer sweep helper
# ─────────────────────────────────────────────────────────────────────────────

def sweep_layers(
    train_samples: list[dict],
    label_fn,
    val_samples: list[dict],
    extracts_dir: Path,
    id_key: str = "sample_id",
    candidate_layers: Optional[list[int]] = None,
) -> tuple[int, float]:
    """Try each candidate layer index; pick the one with best val AUC.

    Returns (best_layer_idx, best_auc).
    """
    from sklearn.metrics import roc_auc_score

    # Detect how many layers are available from first sample
    first_sid = None
    for s in train_samples:
        p = extracts_dir / f"{s[id_key]}.pt"
        if p.exists():
            first_sid = s[id_key]
            break
    if first_sid is None:
        raise RuntimeError("No extract found for any training sample.")
    first_ex = load_extract(extracts_dir, first_sid)
    r = first_ex["residuals"]
    n_layers = r.shape[0] if r.dim() == 3 else 1

    if candidate_layers is None:
        # Default sweep: every 5 layers across middle range
        if n_layers == 1:
            candidate_layers = [0]
        else:
            lo = max(0, n_layers // 10)
            hi = min(n_layers - 1, int(0.85 * n_layers))
            step = max(1, (hi - lo) // 8)
            candidate_layers = list(range(lo, hi + 1, step))

    best_layer, best_auc = candidate_layers[0], -1.0
    for li in candidate_layers:
        try:
            tr_vecs, tr_labels, _ = build_vecs_from_samples(
                train_samples, label_fn, extracts_dir, layer_idx=li, id_key=id_key)
            val_vecs, val_labels, _ = build_vecs_from_samples(
                val_samples, label_fn, extracts_dir, layer_idx=li, id_key=id_key)
        except RuntimeError:
            continue
        pos_mask = tr_labels == 1
        neg_mask = tr_labels == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        r_hat = compute_dim_direction(tr_vecs[pos_mask], tr_vecs[neg_mask])
        scores = torch.mv(val_vecs, r_hat).numpy()
        y_true = val_labels.numpy().astype(int)
        if len(set(y_true.tolist())) < 2:
            continue
        auc = roc_auc_score(y_true, scores)
        print(f"      layer={li:3d}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_layer = li

    return best_layer, best_auc


# ─────────────────────────────────────────────────────────────────────────────
# Public predict interface (used by score_probes.py and ensemble_predictor.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_probe(probe_path: str | Path) -> dict:
    """Load a saved DIM probe checkpoint.

    Returns a dict with keys: r_hat (d,), layer_idx, task, metrics.
    """
    ckpt = torch.load(str(probe_path), weights_only=False, map_location="cpu")
    return ckpt


def predict(probe: dict, residuals: torch.Tensor, mask: torch.Tensor) -> float:
    """Score a single sample.

    Args:
        probe     : checkpoint dict as returned by load_probe()
        residuals : (n_layers, n_tokens, d_model) or (n_tokens, d_model) float tensor
        mask      : (n_tokens,) bool tensor

    Returns:
        float in (-inf, +inf) — higher = more like positive class
        Caller can apply sigmoid for a [0,1] probability.
    """
    r_hat = probe["r_hat"].to(torch.float32)
    layer_idx = probe.get("layer_idx", 0)
    if residuals.dim() == 3:
        n_layers = residuals.shape[0]
        idx = min(layer_idx, n_layers - 1)
        full = residuals[idx].to(torch.float32)
    else:
        full = residuals.to(torch.float32)
    vec = get_last_token(full, mask.bool())
    return score_vec(vec, r_hat)


def predict_prob(probe: dict, residuals: torch.Tensor, mask: torch.Tensor) -> float:
    """Sigmoid-squashed probe score → [0, 1]."""
    raw = predict(probe, residuals, mask)
    return float(1.0 / (1.0 + math.exp(-raw)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-task training + evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval_dim(
    task_name: str,
    train_samples: list[dict],
    test_samples: list[dict],
    label_fn,
    extracts_dir: Path,
    out_dir: Path,
    id_key: str = "sample_id",
    do_layer_sweep: bool = True,
    fixed_layer_idx: Optional[int] = None,
) -> dict:
    """Full train + eval for one task. Returns metrics dict."""
    from sklearn.metrics import roc_auc_score, accuracy_score

    # ---- resume: skip if probe already saved ----
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / f"{task_name}_dim_probe.pt"
    if probe_path.exists():
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}  [SKIPPING — probe already exists at {probe_path}]")
        ckpt = torch.load(str(probe_path), weights_only=False, map_location="cpu")
        m = ckpt.get("metrics", {"task": task_name, "test_auc": float("nan"), "train_auc": float("nan")})
        print(f"  test_auc={m.get('test_auc', float('nan')):.4f}")
        return m

    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"  train={len(train_samples)}  test={len(test_samples)}  extracts={extracts_dir}")

    # ---- detect number of available layers ----
    first_sid = None
    for s in train_samples:
        p = extracts_dir / f"{s[id_key]}.pt"
        if p.exists():
            first_sid = s[id_key]
            break
    if first_sid is None:
        raise RuntimeError(f"[{task_name}] No extract found for any training sample in {extracts_dir}")

    first_ex = load_extract(extracts_dir, first_sid)
    r_tensor = first_ex["residuals"]
    n_layers = r_tensor.shape[0] if r_tensor.dim() == 3 else 1
    d_model  = r_tensor.shape[-1]
    print(f"  n_layers={n_layers}  d_model={d_model}")

    # ---- choose layer ----
    if fixed_layer_idx is not None:
        best_layer = fixed_layer_idx
        print(f"  using fixed layer_idx={best_layer}")
    elif n_layers == 1 or not do_layer_sweep:
        best_layer = 0
        print(f"  single-layer extract → layer_idx=0")
    else:
        # Split train into tr/val (80/20) for layer sweep
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(train_samples))
        cut = max(1, int(0.2 * len(train_samples)))
        val_samples_sweep = [train_samples[i] for i in idx[:cut]]
        tr_samples_sweep  = [train_samples[i] for i in idx[cut:]]
        print(f"  sweeping layers (val={len(val_samples_sweep)})…")
        best_layer, sweep_auc = sweep_layers(
            tr_samples_sweep, label_fn, val_samples_sweep,
            extracts_dir, id_key=id_key)
        print(f"  best layer_idx={best_layer}  sweep_val_auc={sweep_auc:.4f}")

    # ---- compute DIM direction on full training set (streaming to avoid OOM) ----
    r_hat, n_pos, n_neg = compute_dim_streaming(
        train_samples, label_fn, extracts_dir, best_layer, id_key=id_key)

    # ---- evaluate on test set ----
    te_vecs, te_labels, te_ids = build_vecs_from_samples(
        test_samples, label_fn, extracts_dir, layer_idx=best_layer, id_key=id_key)
    te_scores = torch.mv(te_vecs, r_hat).numpy()
    y_true    = te_labels.numpy().astype(int)

    auc = float("nan")
    acc = float("nan")
    if len(set(y_true.tolist())) > 1:
        auc = float(roc_auc_score(y_true, te_scores))
        preds = (te_scores > 0.0).astype(int)
        acc   = float(accuracy_score(y_true, preds))

    tr_auc = float("nan")  # skipped — streaming train load doesn't retain all vecs

    metrics = {
        "task": task_name,
        "layer_idx": best_layer,
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "n_test": int(len(te_ids)),
        "test_auc": auc,
        "test_acc": acc,
        "train_auc": tr_auc,
        "d_model": d_model,
        "n_layers": n_layers,
    }
    print(f"  test_auc={auc:.4f}  test_acc={acc:.4f}  train_auc={tr_auc:.4f}")

    # ---- save probe + per-sample scores ----
    # (train records omitted — streaming load doesn't retain per-sample scores)
    te_records = [{"sample_id": sid, "split": "test",  "score": float(s), "label": int(l),
                   "prob": float(1.0 / (1.0 + math.exp(-float(s))))}
                  for sid, s, l in zip(te_ids, te_scores, y_true)]

    torch.save({
        "r_hat":      r_hat.cpu(),
        "layer_idx":  best_layer,
        "task":       task_name,
        "metrics":    metrics,
        "d_model":    d_model,
        "n_layers":   n_layers,
        "rows_train": [],
        "rows_test":  te_records,
        "rows_all":   te_records,
    }, str(probe_path))
    print(f"  saved → {probe_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Task definitions
# ─────────────────────────────────────────────────────────────────────────────

CYBER_LABEL_FNS = {
    # cyber-1: dual_use (1) vs benign (0)
    "cyber1_dual_use_vs_benign": (
        lambda s: 1.0 if s.get("category") == "dual_use" else
                  0.0 if s.get("category") == "benign"    else None
    ),
    # cyber-2: high_risk_dual_use (1) vs {dual_use, benign} (0)
    "cyber2_high_risk_vs_dual_benign": (
        lambda s: 1.0 if s.get("category") == "high_risk_dual_use" else
                  0.0 if s.get("category") in {"dual_use", "benign"} else None
    ),
    # cyber-3: prohibited (1) vs rest (0)
    "cyber3_prohibited_vs_rest": (
        lambda s: 1.0 if s.get("category") == "prohibited" else
                  0.0 if s.get("category") in {"high_risk_dual_use", "dual_use", "benign"} else None
    ),
}

REFUSAL_LABEL_FN = lambda s: 1.0 if s.get("is_refusal") else 0.0


def filter_nonnull(samples: list[dict], label_fn) -> list[dict]:
    return [s for s in samples if label_fn(s) is not None]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all_tasks", action="store_true",
                    help="Run all 5 hackathon tasks (requires all dataset / extracts flags)")

    # Single-task mode
    ap.add_argument("--task", default=None,
                    help="Task name (e.g. refusal_gemma4_31b, cyber1_dual_use_vs_benign)")
    ap.add_argument("--extracts_dir", default=None)
    ap.add_argument("--train_manifest", default=None,
                    help="JSONL of training samples with sample_id + label fields")
    ap.add_argument("--test_manifest",  default=None,
                    help="JSONL of test samples")
    ap.add_argument("--id_key", default="sample_id")

    # All-tasks mode — dataset paths
    ap.add_argument("--cyber_train",  default=None)
    ap.add_argument("--cyber_test",   default=None)
    ap.add_argument("--refusal_gemma_train", default=None)
    ap.add_argument("--refusal_gemma_test",  default=None)
    ap.add_argument("--refusal_qwen_train",  default=None)
    ap.add_argument("--refusal_qwen_test",   default=None)
    ap.add_argument("--cyber_extracts_dir",  default=None)
    ap.add_argument("--gemma_extracts_dir",  default=None)
    ap.add_argument("--qwen_extracts_dir",   default=None)

    # Common
    ap.add_argument("--out_dir",  default="./probes/dim")
    ap.add_argument("--no_layer_sweep", action="store_true",
                    help="Skip layer sweep, use middle layer (faster but may miss best layer)")
    ap.add_argument("--fixed_layer_idx", type=int, default=None,
                    help="Use this exact layer index (overrides sweep)")
    return ap.parse_args()


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    do_sweep = not args.no_layer_sweep

    all_metrics = []

    if args.all_tasks:
        # ── Cyber tasks ──────────────────────────────────────────────────────
        if not (args.cyber_train and args.cyber_test and args.cyber_extracts_dir):
            raise SystemExit("--all_tasks requires --cyber_train, --cyber_test, --cyber_extracts_dir")
        cyber_train = load_jsonl(args.cyber_train)
        cyber_test  = load_jsonl(args.cyber_test)
        cyber_extracts = Path(args.cyber_extracts_dir)

        for task_name, label_fn in CYBER_LABEL_FNS.items():
            tr = filter_nonnull(cyber_train, label_fn)
            te = filter_nonnull(cyber_test,  label_fn)
            m = train_and_eval_dim(task_name, tr, te, label_fn, cyber_extracts,
                                    out_dir, id_key="sample_id",
                                    do_layer_sweep=do_sweep,
                                    fixed_layer_idx=args.fixed_layer_idx)
            all_metrics.append(m)

        # ── Refusal tasks ─────────────────────────────────────────────────────
        for model_key, train_arg, test_arg, extracts_arg in [
            ("gemma4_31b", args.refusal_gemma_train, args.refusal_gemma_test, args.gemma_extracts_dir),
            ("qwen36",     args.refusal_qwen_train,  args.refusal_qwen_test,  args.qwen_extracts_dir),
        ]:
            if not (train_arg and test_arg and extracts_arg):
                print(f"[warn] Skipping refusal_{model_key} — missing paths")
                continue
            task_name = f"refusal_{model_key}"
            tr = load_jsonl(train_arg)
            te = load_jsonl(test_arg)
            m = train_and_eval_dim(task_name, tr, te, REFUSAL_LABEL_FN,
                                    Path(extracts_arg), out_dir,
                                    id_key="sample_id",
                                    do_layer_sweep=do_sweep,
                                    fixed_layer_idx=args.fixed_layer_idx)
            all_metrics.append(m)

    else:
        # ── Single-task mode ──────────────────────────────────────────────────
        if not (args.task and args.train_manifest and args.test_manifest and args.extracts_dir):
            raise SystemExit("Provide --task, --train_manifest, --test_manifest, --extracts_dir "
                             "(or use --all_tasks)")
        tr = load_jsonl(args.train_manifest)
        te = load_jsonl(args.test_manifest)

        # Determine label function
        if args.task in CYBER_LABEL_FNS:
            label_fn = CYBER_LABEL_FNS[args.task]
            tr = filter_nonnull(tr, label_fn)
            te = filter_nonnull(te, label_fn)
        elif args.task.startswith("refusal"):
            label_fn = REFUSAL_LABEL_FN
        else:
            raise SystemExit(f"Unknown task {args.task!r}. "
                             f"Known cyber tasks: {list(CYBER_LABEL_FNS.keys())}; "
                             f"or refusal_gemma4_31b / refusal_qwen36")

        m = train_and_eval_dim(args.task, tr, te, label_fn,
                                Path(args.extracts_dir), out_dir,
                                id_key=args.id_key,
                                do_layer_sweep=do_sweep,
                                fixed_layer_idx=args.fixed_layer_idx)
        all_metrics.append(m)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"{'task':<40} {'test_auc':>10} {'train_auc':>10}")
    print("-" * 62)
    aucs = []
    for m in all_metrics:
        auc = m["test_auc"]
        print(f"  {m['task']:<38} {auc:>10.4f}  {m['train_auc']:>10.4f}")
        if not math.isnan(auc):
            aucs.append(auc)
    if aucs:
        print(f"\n  Mean AUC across {len(aucs)} tasks: {np.mean(aucs):.4f}")

    # Save summary JSON
    summary_path = out_dir / "dim_probe_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"metrics": all_metrics,
                   "mean_auc": float(np.mean(aucs)) if aucs else None}, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
