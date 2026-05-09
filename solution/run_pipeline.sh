#!/usr/bin/env bash
# =============================================================================
# Hackathon full pipeline: Level 1 (predict) + Level 2 (flip)
# =============================================================================
# Runs on the cluster pod at /scratch.
# Models mounted read-only at /data/Gemma-4-31B-it and /data/Qwen3.6-27B.
# All outputs go under /scratch/outputs/.
#
# Usage:
#   bash solution/run_pipeline.sh [--skip-extract] [--skip-level1] [--level2-only]
#   bash solution/run_pipeline.sh --limit-samples 200   # fast dev run
#
# Environment requirements:
#   AIAAS_KEY  — API key for AIaaS LLM calls (Level 2)
# =============================================================================
set -euo pipefail

# ── Configurable paths ───────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${SCRATCH:-/scratch}"
DATA_DIR="${DATA_DIR:-/data}"

OUT_ROOT="$SCRATCH/outputs"
EXTRACTS_ROOT="$OUT_ROOT/extracts"
PROBES_ROOT="$OUT_ROOT/probes"
RESULTS_ROOT="$OUT_ROOT/results"

DATASETS="$REPO_ROOT/datasets"
STARTER="$REPO_ROOT/starter_code"
SOLUTION="$REPO_ROOT/solution"

# Layer specs — sweep all candidate layers for best AUC
# For gemma4_31b (60 layers): middle = 30, try 14:50:4 range
# For qwen36     (64 layers): middle = 32, try 14:54:4 range
LAYERS_GEMMA="${LAYERS_GEMMA:-14:52:4}"   # ~10 candidate layers
LAYERS_QWEN="${LAYERS_QWEN:-14:54:4}"
LAYERS_CYBER="${LAYERS_CYBER:-middle}"    # cyber: one model runs both — use middle by default

# Model paths
GEMMA_PATH="$DATA_DIR/Gemma-4-31B-it"
QWEN_PATH="$DATA_DIR/Qwen3.6-27B"

# Level 2 settings
EDITOR="${EDITOR_MODEL:-minimax-m2.7}"
JUDGE="${JUDGE_MODEL:-minimax-m2.7}"
MAX_ITERS=5
K_REWRITES=5
LEVEL2_LIMIT="${LEVEL2_LIMIT:-0}"   # 0 = all
LEVEL2_MODE="${LEVEL2_MODE:-both}"  # span | rewrite | both

# ── CLI flags ─────────────────────────────────────────────────────────────────
SKIP_EXTRACT=0
SKIP_LEVEL1=0
LEVEL2_ONLY=0
SAMPLE_LIMIT=0   # 0 = all samples

for arg in "$@"; do
  case "$arg" in
    --skip-extract)  SKIP_EXTRACT=1 ;;
    --skip-level1)   SKIP_LEVEL1=1  ;;
    --level2-only)   LEVEL2_ONLY=1; SKIP_EXTRACT=1; SKIP_LEVEL1=1 ;;
    --limit-samples=*) SAMPLE_LIMIT="${arg#*=}" ;;
    --help|-h)
      echo "Usage: $0 [--skip-extract] [--skip-level1] [--level2-only] [--limit-samples=N]"
      exit 0
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m>>> $*\033[0m" >&2; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m" >&2; }
die()  { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; exit 1; }

check_gpu() {
  python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
  log "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
}

mkdir -p "$EXTRACTS_ROOT" "$PROBES_ROOT/dim" "$PROBES_ROOT/weights" \
         "$PROBES_ROOT/ensemble" "$RESULTS_ROOT"

# =============================================================================
# PHASE 0 — Sanity checks
# =============================================================================
log "Sanity checks"
[[ -d "$GEMMA_PATH" ]] || warn "Gemma not found at $GEMMA_PATH"
[[ -d "$QWEN_PATH"  ]] || warn "Qwen not found at $QWEN_PATH"
[[ -n "${AIAAS_KEY:-}" ]] || warn "AIAAS_KEY not set — Level 2 LLM calls will fail"
python3 -c "import torch, numpy, sklearn, httpx, transformers" \
  || die "Missing Python dependencies"

if [[ $SKIP_EXTRACT -eq 0 && $LEVEL2_ONLY -eq 0 ]]; then
  check_gpu
fi

# =============================================================================
# PHASE 1 — Extract residuals
# =============================================================================

if [[ $SKIP_EXTRACT -eq 0 ]]; then

  # ── 1a. Gemma — refusal probes ──────────────────────────────────────────────
  log "1a. Extracting residuals: gemma4_31b (refusal)"
  python3 "$STARTER/extract_residuals.py" \
    --model_key gemma4_31b \
    --model_path "$GEMMA_PATH" \
    --samples_file "$DATASETS/refusal_probes/gemma4_31b/attacks_full.jsonl" \
    --out_dir "$EXTRACTS_ROOT/gemma4_31b" \
    --layers "$LAYERS_GEMMA" \
    --dtype fp16 \
    ${SAMPLE_LIMIT:+--sample_limit "$SAMPLE_LIMIT"}

  # ── 1b. Qwen — refusal probes ──────────────────────────────────────────────
  log "1b. Extracting residuals: qwen36 (refusal)"
  python3 "$STARTER/extract_residuals.py" \
    --model_key qwen36 \
    --model_path "$QWEN_PATH" \
    --samples_file "$DATASETS/refusal_probes/qwen36/attacks_full.jsonl" \
    --out_dir "$EXTRACTS_ROOT/qwen36" \
    --layers "$LAYERS_QWEN" \
    --dtype fp16 \
    ${SAMPLE_LIMIT:+--sample_limit "$SAMPLE_LIMIT"}

  # ── 1c. Cyber probes (model-agnostic prompts) ───────────────────────────────
  # Cyber probes use "prompt" field; extract_residuals.py expects "attack_prompt".
  # Preprocess: add attack_prompt = prompt for compatibility.
  log "1c. Preprocessing cyber probe JSONL (prompt → attack_prompt)"
  CYBER_TRAIN_TMP="$EXTRACTS_ROOT/cyber_train_fixed.jsonl"
  CYBER_TEST_TMP="$EXTRACTS_ROOT/cyber_test_fixed.jsonl"
  export CYBER_TRAIN_SRC="$DATASETS/cyber_probes/train.jsonl" CYBER_TRAIN_DST="$CYBER_TRAIN_TMP"
  export CYBER_TEST_SRC="$DATASETS/cyber_probes/test.jsonl"   CYBER_TEST_DST="$CYBER_TEST_TMP"
  python3 - <<'PYEOF'
import json, os
for src, dst in [
    (os.environ["CYBER_TRAIN_SRC"], os.environ["CYBER_TRAIN_DST"]),
    (os.environ["CYBER_TEST_SRC"],  os.environ["CYBER_TEST_DST"]),
]:
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            s = json.loads(line)
            s.setdefault("attack_prompt", s.get("prompt", ""))
            s.setdefault("is_refusal", -1)
            fout.write(json.dumps(s) + "\n")
print("Cyber JSONL preprocessing done.", flush=True)
PYEOF

  log "1c. Extracting residuals: gemma4_31b (cyber)"
  python3 "$STARTER/extract_residuals.py" \
    --model_key gemma4_31b \
    --model_path "$GEMMA_PATH" \
    --samples_file "$CYBER_TRAIN_TMP" \
    --out_dir "$EXTRACTS_ROOT/cyber/gemma4_31b" \
    --layers "$LAYERS_GEMMA" \
    --dtype fp16 \
    ${SAMPLE_LIMIT:+--sample_limit "$SAMPLE_LIMIT"}

  # Extract cyber test set too
  python3 "$STARTER/extract_residuals.py" \
    --model_key gemma4_31b \
    --model_path "$GEMMA_PATH" \
    --samples_file "$CYBER_TEST_TMP" \
    --out_dir "$EXTRACTS_ROOT/cyber/gemma4_31b" \
    --layers "$LAYERS_GEMMA" \
    --dtype fp16 \
    ${SAMPLE_LIMIT:+--sample_limit "$SAMPLE_LIMIT"}

  log "Phase 1 (extraction) complete"
fi

# =============================================================================
# PHASE 2 — Level 1: Train probes and evaluate
# =============================================================================

if [[ $SKIP_LEVEL1 -eq 0 ]]; then

  # ── 2a. DIM probes for all 5 tasks ─────────────────────────────────────────
  log "2a. Training diff-in-means probes (all 5 tasks)"
  python3 "$SOLUTION/diff_in_means_probe.py" \
    --all_tasks \
    --cyber_train  "$DATASETS/cyber_probes/train.jsonl" \
    --cyber_test   "$DATASETS/cyber_probes/test.jsonl" \
    --refusal_gemma_train "$DATASETS/refusal_probes/gemma4_31b/train_split.jsonl" \
    --refusal_gemma_test  "$DATASETS/refusal_probes/gemma4_31b/test_split.jsonl" \
    --refusal_qwen_train  "$DATASETS/refusal_probes/qwen36/train_split.jsonl" \
    --refusal_qwen_test   "$DATASETS/refusal_probes/qwen36/test_split.jsonl" \
    --cyber_extracts_dir  "$EXTRACTS_ROOT/cyber/gemma4_31b" \
    --gemma_extracts_dir  "$EXTRACTS_ROOT/gemma4_31b" \
    --qwen_extracts_dir   "$EXTRACTS_ROOT/qwen36" \
    --out_dir "$PROBES_ROOT/dim"

  # ── 2b. Attention probes (train_probe.py) ──────────────────────────────────
  # train_probe.py needs a manifest with cyber_samples and refusal_samples keys.
  # Generate it from the JSONL files + extraction metadata.
  log "2b. Generating train_probe manifest"
  MANIFEST_PATH="$PROBES_ROOT/combined_manifest.json"
  python3 - <<PYEOF
import json, os
from pathlib import Path

datasets = "$DATASETS"
extracts_gemma = "$EXTRACTS_ROOT/gemma4_31b"
extracts_qwen  = "$EXTRACTS_ROOT/qwen36"
extracts_cyber = "$EXTRACTS_ROOT/cyber/gemma4_31b"
out = "$MANIFEST_PATH"

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def get_extracted_ids(extracts_dir):
    """IDs that have a .pt file."""
    p = Path(extracts_dir)
    return {f.stem for f in p.glob("*.pt")}

cyber_ids = get_extracted_ids(extracts_cyber)
gemma_ids = get_extracted_ids(extracts_gemma)
qwen_ids  = get_extracted_ids(extracts_qwen)

cyber_train = load_jsonl(f"{datasets}/cyber_probes/train.jsonl")
cyber_test  = load_jsonl(f"{datasets}/cyber_probes/test.jsonl")
cyber_all   = [s for s in cyber_train + cyber_test if s["sample_id"] in cyber_ids]

def cyber_to_probe(s):
    return {"sample_id": s["sample_id"], "label": s["category"],
            "split": s["split"], "n_tokens": s.get("n_tokens", 0)}

gemma_train = load_jsonl(f"{datasets}/refusal_probes/gemma4_31b/train_split.jsonl")
gemma_test  = load_jsonl(f"{datasets}/refusal_probes/gemma4_31b/test_split.jsonl")
gemma_all   = [s for s in gemma_train + gemma_test if s["sample_id"] in gemma_ids]

qwen_train  = load_jsonl(f"{datasets}/refusal_probes/qwen36/train_split.jsonl")
qwen_test   = load_jsonl(f"{datasets}/refusal_probes/qwen36/test_split.jsonl")
qwen_all    = [s for s in qwen_train + qwen_test if s["sample_id"] in qwen_ids]

def ref_to_probe(s):
    return {"sample_id": s["sample_id"], "is_refusal": bool(s.get("is_refusal", False)),
            "split": s["split"], "n_tokens": s.get("n_tokens", 0)}

manifest = {
    "cyber_samples": [cyber_to_probe(s) for s in cyber_all],
    "refusal_samples": {
        "gemma4_31b": [ref_to_probe(s) for s in gemma_all],
        "qwen36":     [ref_to_probe(s) for s in qwen_all],
    }
}
with open(out, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written: {out} — "
      f"cyber={len(manifest['cyber_samples'])} "
      f"gemma={len(manifest['refusal_samples']['gemma4_31b'])} "
      f"qwen={len(manifest['refusal_samples']['qwen36'])}")
PYEOF

  log "2b. Training attention probes: refusal-gemma"
  python3 "$STARTER/train_probe.py" \
    --extracts_dir "$EXTRACTS_ROOT/gemma4_31b" \
    --manifest "$MANIFEST_PATH" \
    --out_dir "$PROBES_ROOT" \
    --task refusal_gemma4_31b

  log "2b. Training attention probes: refusal-qwen"
  python3 "$STARTER/train_probe.py" \
    --extracts_dir "$EXTRACTS_ROOT/qwen36" \
    --manifest "$MANIFEST_PATH" \
    --out_dir "$PROBES_ROOT" \
    --task refusal_qwen36

  log "2b. Training attention probes: cyber tasks (gemma extracts)"
  for cyber_task in cyber_prohib_vs_rest_gemma4_31b cyber_hdu_vs_rest_gemma4_31b cyber_du_vs_ben_gemma4_31b; do
    python3 "$STARTER/train_probe.py" \
      --extracts_dir "$EXTRACTS_ROOT/cyber/gemma4_31b" \
      --manifest "$MANIFEST_PATH" \
      --out_dir "$PROBES_ROOT" \
      --task "$cyber_task" || warn "Task $cyber_task failed (may not match task_specs naming)"
  done

  # ── 2c. Ensemble evaluation ──────────────────────────────────────────────────
  log "2c. Evaluating ensemble (DIM + Attention)"
  python3 "$SOLUTION/ensemble_predictor.py" \
    --all_tasks \
    --dim_summary   "$PROBES_ROOT/dim/dim_probe_summary.json" \
    --dim_probe_dir "$PROBES_ROOT/dim" \
    --attn_probe_dir "$PROBES_ROOT/weights" \
    --extracts_dir_cyber "$EXTRACTS_ROOT/cyber/gemma4_31b" \
    --extracts_dir_gemma "$EXTRACTS_ROOT/gemma4_31b" \
    --extracts_dir_qwen  "$EXTRACTS_ROOT/qwen36" \
    --test_cyber  "$DATASETS/cyber_probes/test.jsonl" \
    --test_gemma  "$DATASETS/refusal_probes/gemma4_31b/test_split.jsonl" \
    --test_qwen   "$DATASETS/refusal_probes/qwen36/test_split.jsonl" \
    --alpha_dim  0.6 \
    --alpha_attn 0.4 \
    --out_dir "$PROBES_ROOT/ensemble"

  log "Phase 2 (Level 1 probes) complete"
fi

# =============================================================================
# PHASE 3 — Compute refusal-direction attribution
# =============================================================================
if [[ $LEVEL2_ONLY -eq 0 || $SKIP_LEVEL1 -eq 0 ]]; then
  log "3. Computing token attribution (Method A offline)"
  for model_key in gemma4_31b qwen36; do
    probe_path="$PROBES_ROOT/dim/refusal_${model_key}_dim_probe.pt"
    [[ -f "$probe_path" ]] || { warn "DIM probe not found: $probe_path"; continue; }
    python3 "$SOLUTION/refusal_direction_attribution.py" \
      --probe_path   "$probe_path" \
      --extracts_dir "$EXTRACTS_ROOT/$model_key" \
      --samples_jsonl "$DATASETS/refusal_probes/$model_key/test_split.jsonl" \
      --method A \
      --out_dir "$RESULTS_ROOT/attribution"
  done
fi

# =============================================================================
# PHASE 4 — Level 2: Flip
# =============================================================================
log "4. Level 2 flip"

[[ -n "${AIAAS_KEY:-}" ]] || die "AIAAS_KEY not set — cannot run Level 2"

for model_key in gemma4_31b qwen36; do
  probe_path="$PROBES_ROOT/dim/refusal_${model_key}_dim_probe.pt"
  [[ -f "$probe_path" ]] || { warn "DIM probe not found: $probe_path"; continue; }

  log "4. Level 2: $model_key (mode=$LEVEL2_MODE)"
  python3 "$SOLUTION/level2_flip.py" \
    --probe_path  "$probe_path" \
    --eval_jsonl  "$DATASETS/refusal_probes/$model_key/test_split.jsonl" \
    --extracts_dir "$EXTRACTS_ROOT/$model_key" \
    --out_path    "$RESULTS_ROOT/level2_${model_key}_${LEVEL2_MODE}.jsonl" \
    --mode "$LEVEL2_MODE" \
    --editor "$EDITOR" \
    --judge  "$JUDGE" \
    --max_iters "$MAX_ITERS" \
    --k_rewrites "$K_REWRITES" \
    --model_key "$model_key" \
    --refusal_only \
    ${LEVEL2_LIMIT:+--limit "$LEVEL2_LIMIT"} \
    --verify_behavior \
    --resume
done

# =============================================================================
# PHASE 5 — Print final summary
# =============================================================================
log "5. Final summary"
echo ""
echo "============================================================"
echo "LEVEL 1 — Probe AUCs"
echo "============================================================"
if [[ -f "$PROBES_ROOT/ensemble/ensemble_summary.json" ]]; then
  python3 - <<'EOF'
import json, math, sys
p = "/scratch/outputs/probes/ensemble/ensemble_summary.json"
try:
    d = json.load(open(p))
    print(f"{'Task':<45} {'Ensemble AUC':>12}")
    print("-" * 59)
    aucs = []
    for m in d["metrics"]:
        auc = m["auc"]
        flag = "" if math.isnan(auc) else (" *" if auc > 0.9 else "")
        print(f"  {m['task']:<43} {auc:>12.4f}{flag}")
        if not math.isnan(auc): aucs.append(auc)
    if aucs:
        print(f"\n  Mean AUC: {sum(aucs)/len(aucs):.4f}")
except Exception as e:
    print(f"  (could not read: {e})")
EOF
fi

echo ""
echo "============================================================"
echo "LEVEL 2 — Flip rates"
echo "============================================================"
for model_key in gemma4_31b qwen36; do
  metrics_path="$RESULTS_ROOT/level2_${model_key}_${LEVEL2_MODE}.metrics.json"
  if [[ -f "$metrics_path" ]]; then
    echo "  $model_key:"
    python3 - <<EOF
import json
p = "$metrics_path"
d = json.load(open(p))
print(f"    probe_flip_rate@5:  {d.get('probe_flip_rate', 'N/A')}")
print(f"    behavior_flip_rate: {d.get('behavior_flip_rate', 'N/A')}")
print(f"    causal_concordance: {d.get('causal_concordance', 'N/A')}")
print(f"    n_total:            {d.get('n_total', '?')}")
EOF
  else
    echo "  $model_key: metrics not found ($metrics_path)"
  fi
done

echo ""
echo "All outputs in: $OUT_ROOT"
log "Pipeline complete!"
