# Onboarding — first 15 minutes on the cluster

This walks you through getting from "I just got added to a hackathon group" to "I'm running `extract_residuals.py` against the pre-staged models." If you're already comfortable with RunAI + the EPFL cluster, the [README quickstart](../README.md#setup) is enough.

The pod image is `registry.rcp.epfl.ch/mlo-protsenk/redteam-mechinterp:v9` — all Python deps are baked in (no `pip install` needed). Working directory is `/scratch` (your group's RW PVC mount), so anything you write lands there.

---

## 0. Launch your pod

Once you've been added to your hackathon group (admin handles this), submit a long-running interactive pod:

```bash
runai submit dev-pod \
    --image registry.rcp.epfl.ch/mlo-protsenk/redteam-mechinterp:v9 \
    --gpu 1 \
    --pvc hackathon-mechhack-scratch-gNN:/scratch \
    --pvc hackathon-mechhack-shared-ro:/data \
    --command -- sleep infinity
```

Replace `gNN` with your group number (e.g., `g03`). Two PVCs:
- `…-scratch-gNN` → `/scratch` — your group's writable workspace, where the container starts
- `…-shared-ro` → `/data` — read-only mount with `Gemma-4-31B-it/` and `Qwen3.6-27B/` pre-staged

Then exec into it:

```bash
runai exec -it dev-pod -- bash      # lands in /scratch
```

> **Need an H100?** Add `--node-pools h100`. The default pool works for most things; pick H100 explicitly only if your method actually needs it.

---

## 1. Get your AIaaS API key (most important)

[**portal.rcp.epfl.ch/aiaas/models**](https://portal.rcp.epfl.ch/aiaas/models) — sign in with EPFL credentials, browse available models (MiniMax-M2.7, Kimi-K2.6, Qwen3-30B, etc.), and grab a key from [`/aiaas/keys`](https://portal.rcp.epfl.ch/aiaas/keys). The key is shown **once** — copy it immediately.

```bash
export AIAAS_KEY=sk--...
```

This single key powers:
- the edit-LLM wrappers in [`starter_code/llm_clients.py`](../starter_code/llm_clients.py) (Level 2)
- the refusal / intent judges in [`starter_code/behavior_verifier.py`](../starter_code/behavior_verifier.py)
- Claude Code itself (next step)

---

## 2. Set up Claude Code routed through AIaaS

Use [Claude Code](https://claude.com/claude-code) as your IDE / agentic coding assistant, but with **all model traffic going to RCP-hosted AIaaS** — no external API keys, no data leaving the cluster.

```bash
git clone https://github.com/Monketoo/mechhack.git    # or whatever your URL is
cd mechhack/tools/claude-code-aiaas
./setup.sh         # installs Node + Claude Code + aiohttp into ./vendor/
./aiaas-claude.sh  # menu-pick a model, launches claude
```

The model picker is constrained to **MiniMax-M2.7** and **Kimi-K2.6** — both are strong coding models and both are hosted on AIaaS. MiniMax-M2.7 is the recommended default (reasoning ON, ~8-10 s/call; first call after a quiet period is a cold start ~3 min).

Full walkthrough — proxy internals, how to debug a hung request — in [`tools/claude-code-aiaas/README.md`](../tools/claude-code-aiaas/README.md).

---

## 3. Target models — already pre-staged on `/data/`

The pod has a **read-only `/data/`** mount with both target models. Nothing to download:

```
/data/
├── Gemma-4-31B-it/      ← google/gemma-4-31B-it, full snapshot (59 GB)
└── Qwen3.6-27B/         ← Qwen/Qwen3.6-27B, full snapshot (52 GB)
```

Starter scripts auto-resolve `/data/<repo>` — no env vars, no `--model_path`. Smoke test:

```bash
python starter_code/extract_residuals.py --model_key gemma4_31b --sample_limit 2
```

The full resolver lookup order (first hit wins): `--model_path` flag → `$HACKATHON_MODELS_DIR/<repo>` → `/data/<repo>` → `<repo-root>/models/<repo>` → HF cache.

---

## Working off-cluster?

If you're on a laptop / Colab / your own pod without the RO mount:

```bash
# Both models are gated on HF — accept the licenses first:
#   https://huggingface.co/google/gemma-4-31B-it
#   https://huggingface.co/Qwen/Qwen3.6-27B
export HF_TOKEN=hf_...
python starter_code/download_models.py --out_dir ./models   # ~117 GB
```

You'll also need to pip-install the deps (the pod image has them baked in): `torch transformers numpy httpx scikit-learn huggingface-hub hf_transfer`. We don't ship a pinned `requirements.txt` — versions match the v9 image, drift breaks things.

---

## Compute

A100-80GB primary, H200-141GB backup. Suggested target ~1 GPU-hour per Level-2 run (loose, not enforced).

---

## Common gotchas

- **`runai login` succeeds but you see `dial tcp: lookup caas-test.rcp.epfl.ch on …: no such host`.** Your `~/.kube/config` (on the jumphost or laptop) has a stale cluster entry pointing at a decommissioned endpoint. Replace it with the current one from the wiki:
  ```bash
  wget https://wiki.rcp.epfl.ch/public/files/kube-config.yaml -O ~/.kube/config && chmod 600 ~/.kube/config
  runai login    # re-auth into the new cluster URL
  runai config project hackathon-mechhack-protsenk    # restore default project
  ```
- **`fatal: detected dubious ownership in repository`** when you `git pull` or `git status` on `/scratch`. Fix: run `tools/claude-code-aiaas/setup.sh` once (it registers the repo root as a git `safe.directory`), or `git config --global --add safe.directory /scratch/<your-repo-path>`.
- **`--dangerously-skip-permissions cannot be used with root`** when running Claude Code. Cluster pods run as root by design. Fix: `export IS_SANDBOX=1` (the documented bypass for sandboxed container environments). `aiaas-claude.sh` does this for you.
- **Claude Code hangs ~3 min on first prompt.** The AIaaS model is cold-loading on first request after a quiet period. Subsequent calls are fast (<1 s). If it never returns, run `tools/claude-code-aiaas/check_model.sh <model-id>` to triage.
- **`tar: Cannot change ownership to uid 1000`** when running setup.sh. Already fixed (we use `tar --no-same-owner`); if you see this, you're on a stale checkout — `git pull` and re-run.
- **AIaaS Qwen 3.6-27B drifts toward compliance vs the corpus on borderline samples (~30% of refusals flip).** AIaaS appears to silently route to `Qwen3.6-27B-fp8` for performance, and fp8 quantization shifts argmax on borderline tokens. **Gemma 4-31B-it on AIaaS is faithful** (~93% corpus alignment); Qwen needs the cluster (`generate_rollouts.py`) for label-faithful Level-2 evaluation. See [`rules/flip.md`](../rules/flip.md) for the full reproducibility data.
