# Starter Code

Minimal baselines and helpers. Floors to build past, not answers.

## Files at a glance

### Setup / shared

| File | What it does |
|---|---|
| [`download_models.py`](download_models.py) | Pulls `google/gemma-4-31B-it` and `Qwen/Qwen3.6-27B` from HF Hub into `./models/`. Both are gated — needs `HF_TOKEN`. ~62 + ~55 GB. |
| [`chunked_sdpa.py`](chunked_sdpa.py) | SDPA monkey-patch needed for Gemma 4-31B's `head_dim=512` — flash-attn doesn't support it and the math backend OOMs on long sequences. Imported as a context manager around the model forward. |

### Level 1 — extract residuals, train a predictor

| File | Consumes | Produces |
|---|---|---|
| [`extract_residuals.py`](extract_residuals.py) | model + a JSONL of attack prompts (`datasets/refusal_probes/...` or `datasets/cyber_probes/...`) | per-sample `.pt` files with `residuals (n_selected_layers, n_tokens, d_model)` at fp16, plus `input_ids`, `attention_mask`, `layer_idxs`, `label`. Layer-spec controlled by `extract_config.json` or `--layers` (default: one middle layer). |
| [`extract_config.json`](extract_config.json) | — | default config (model_key, layer-spec, dtype, output dir). CLI args > env vars > this file. |
| [`train_probe.py`](train_probe.py) | extracts dir + manifest from `extract_residuals.py` | trained probe weights + per-task AUC metrics. Implements linear / MLP / single- and 4-head attention probes; 5 seeds per arch. |

This is the *probe-shaped* starter path. If your Level-1 method is different (SAE, circuit, lens, etc.), `extract_residuals.py` still gives you the residuals to work from — you'd write your own training script and produce a `predict.py` matching the contract in [`rules/predict.md`](../rules/predict.md).

### Level 2 — attribute and edit

| File | Consumes | Produces |
|---|---|---|
| [`grad_input_baseline.py`](grad_input_baseline.py) | model + probe weights + extracts dir | per-rollout JSON with input-token attribution scores (`∂probe_logit / ∂input_embed[n] · input_embed[n]`) and top-K editable positions. **Reference attribution method.** |
| [`iterative_edit_agent.py`](iterative_edit_agent.py) | probe + an attribution method + an edit-LLM | per-rollout edit trajectory: 5 iterations of `attribute → propose edits → verify → apply → re-score probe + intent judge`. **Reference Level-2 baseline.** |
| [`behavior_verifier.py`](behavior_verifier.py) | model + tokenizer + an LLM judge | for each edited prompt: re-rolls the model, asks the judge whether it's now refusal or compliance. Used by `iterative_edit_agent.py --verify_behavior` and `scoring/score_disrupt.py --verify_behavior` to compute `behavior_flip_rate` and `concordance`. |
| [`llm_clients.py`](llm_clients.py) | — | wrappers around OpenRouter + EPFL AIaaS for editor / judge LLM calls. Schema-forced JSON output, reasoning-disable presets per model class. `make_editor("qwen3-30b" / "qwen3-235b" / "deepseek-v4-pro" / "kimi-k2.6" / "minimax-m2.7")`. |

## Quick start

```bash
# Pod from registry.rcp.epfl.ch/mlo-protsenk/redteam-mechinterp:v9 has all
# Python deps preinstalled — no pip install needed.
# Models are pre-staged at /data/{Gemma-4-31B-it,Qwen3.6-27B} (read-only).
# See ../README.md "Setup" for AIaaS key + Claude Code.

# 1. Extract residuals — auto-resolves /data/Gemma-4-31B-it
python extract_residuals.py --model_key gemma4_31b --out_dir ./extracts/gemma4_31b

# Other layer-spec options:
#   --layers "middle"        single middle layer (default)
#   --layers "10,30,50"      explicit list
#   --layers "0:65:8"        python-range (start:stop:step)
#   --layers "all"           every layer (~65× the disk)

# 2. Train a probe (consumes the extracts above)
python train_probe.py \
    --extracts_dir ./extracts/gemma4_31b \
    --manifest    ./extracts/gemma4_31b/extraction_metadata.json \
    --out_dir     ./probes \
    --task        refusal_gemma4_31b

# 3. Reference attribution baseline
python grad_input_baseline.py \
    --model_key     gemma4_31b \
    --probe_weights ./probes/weights/refusal_gemma4_31b_attention.pt \
    --extracts_dir  ./extracts/gemma4_31b \
    --out_dir       ./edit_eval

# Off-cluster fallback: if /data isn't mounted, download yourself:
# export HF_TOKEN=hf_...
# python download_models.py --out_dir ../models

# 5. Iterative edit agent (Level 2 baseline)
export OPENROUTER_KEY=sk-or-v1-...        # or AIAAS_KEY for EPFL AIaaS
python iterative_edit_agent.py \
    --probe_path  ./probes/weights/refusal_gemma4_31b_attention.pt \
    --eval_set    ../datasets/refusal_probes/gemma4_31b/attribution_eval.jsonl \
    --max_iters   5 \
    --editor_endpoint openrouter \
    --editor_model    deepseek/deepseek-v4-pro \
    --output      ./baseline_results.jsonl
```

## A100-80GB recipe

```python
import os, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, ...)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
model.gradient_checkpointing_enable()        # halves activation memory
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()       # required for grads-on-input-embeddings

# For Gemma 4-31B, needed:
from chunked_sdpa import chunked_sdpa_scope
with chunked_sdpa_scope():
    out = model(inputs_embeds=embs, attention_mask=attn,
                output_hidden_states=True, return_dict=True)
```

## Ideas your baseline doesn't try

- Probe ensembles
- Distilled probes
- Counterfactual editing (Wachter et al.)
- Beam search across edit trajectories
- RL fine-tuning of the edit agent against the probe-flip reward (careful — must keep intent judge as a hard gate)

Make us beat by 30%.
