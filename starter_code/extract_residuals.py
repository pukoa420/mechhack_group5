"""Residual-stream extraction with configurable layer selection.

Per sample saves shape (n_selected_layers, n_tokens, d_model) at fp16, plus
input_ids and attention_mask. By default extracts ONE layer (middle) — tweak
`extract_config.json`, pass --layers, or set LAYERS env var for more.

Usage:
    python extract_residuals.py \
        --model_path ./models/Gemma-4-31B-it \
        --samples_file ../datasets/refusal_probes/gemma4_31b/attacks_full.jsonl \
        --out_dir ./extracts/gemma4_31b \
        --layers middle
    # or just rely on extract_config.json defaults

Layer-spec syntax:
  "middle"             one layer at n_layers // 2
  "early" / "late"     n_layers // 4 / 3 * n_layers // 4
  "32"                 single index
  "10,30,50"           explicit list
  "0:65:8"             python-range (start:stop:step)
  "all"                every layer (embedding output + each block output, n_layers+1 total)

Indices are into `output_hidden_states`: index 0 is the embedding output,
indices 1..n_layers are post-block residuals.

Datasets are pre-filtered to ≤8192 tokens, so no token-level truncation.
For Gemma uses chunked-sdpa scope (head_dim=512 needs it on H100/A100).
"""
import os, sys, json, time, argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from chunked_sdpa import chunked_sdpa_scope

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def load_config(cli_args=None) -> dict:
    """Load extract_config.json next to this script; CLI args > env > config-file."""
    cfg_path = Path(__file__).parent / "extract_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    # Env overrides
    for env_key, cfg_key in [("MODEL_KEY", "model_key"), ("LAYERS", "layers"),
                              ("USE_CT", "use_chat_template"), ("DTYPE", "dtype"),
                              ("OUT_DIR", "out_dir"), ("MODEL_PATH", "model_path"),
                              ("SAMPLES_FILE", "samples_file")]:
        if env_key in os.environ:
            v = os.environ[env_key]
            cfg[cfg_key] = (v == "1") if cfg_key == "use_chat_template" else v
    # CLI overrides
    if cli_args:
        for k in ("model_key", "layers", "dtype", "out_dir", "model_path", "samples_file"):
            v = getattr(cli_args, k, None)
            if v is not None: cfg[k] = v
        if cli_args.no_chat_template: cfg["use_chat_template"] = False
    return cfg


def parse_layers(spec, n_layers: int) -> list[int]:
    """Resolve layer-spec to a sorted list of indices into hidden_states.

    hidden_states from `output_hidden_states=True` has length n_layers+1
    (0 = embeddings, 1..n_layers = block outputs). So valid range is [0, n_layers].
    """
    max_idx = n_layers  # inclusive
    if isinstance(spec, list):
        idxs = [int(x) for x in spec]
    elif isinstance(spec, int):
        idxs = [spec]
    elif isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            idxs = list(range(max_idx + 1))
        elif s == "middle":
            idxs = [n_layers // 2]
        elif s == "early":
            idxs = [n_layers // 4]
        elif s == "late":
            idxs = [3 * n_layers // 4]
        elif ":" in s:
            parts = s.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"bad range spec {spec!r}, expected start:stop[:step]")
            start = int(parts[0]); stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 else 1
            idxs = list(range(start, stop, step))
        elif "," in s:
            idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
        else:
            idxs = [int(s)]
    else:
        raise ValueError(f"unsupported layers spec type {type(spec).__name__}: {spec!r}")
    bad = [i for i in idxs if not (0 <= i <= max_idx)]
    if bad:
        raise ValueError(f"layer indices {bad} out of range [0, {max_idx}] for n_layers={n_layers}")
    return sorted(set(idxs))


HF_REPOS = {
    "gemma4_31b": "google/gemma-4-31B-it",
    "qwen36":     "Qwen/Qwen3.6-27B",
}


def resolve_model_path(model_key: str, model_path: str | None) -> str:
    """Resolve to a local directory containing the HF snapshot.

    Lookup order (first hit wins):
      1. explicit --model_path / $MODEL_PATH
      2. $HACKATHON_MODELS_DIR/<repo-name>            (custom override)
      3. /data/<repo-name>                            (cluster RO mount)
      4. <repo-root>/models/<repo-name>               (download_models.py default)
      5. HF cache via snapshot_download(local_files_only=True)
    """
    from huggingface_hub import snapshot_download
    if model_path and Path(model_path).exists():
        return str(Path(model_path).resolve())
    repo_id = HF_REPOS.get(model_key)
    if not repo_id:
        raise ValueError(f"Unknown model_key {model_key!r}; pass --model_path explicitly.")
    repo_basename = repo_id.split("/")[-1]
    candidates = []
    if env := os.environ.get("HACKATHON_MODELS_DIR"):
        candidates.append(Path(env) / repo_basename)
    candidates.append(Path("/data") / repo_basename)
    candidates.append(Path(__file__).resolve().parent.parent / "models" / repo_basename)
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to HF cache
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate {model_key} ({repo_id}). Tried:\n  "
            + "\n  ".join(str(c) for c in candidates)
            + f"\nEither pass --model_path, set MODEL_PATH / HACKATHON_MODELS_DIR, "
            f"or run download_models.py first."
        ) from e


def load_model(model_key: str, model_path: str | None, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    p = resolve_model_path(model_key, model_path)
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        p, torch_dtype=dtype, attn_implementation="sdpa",
        device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, tok


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", choices=list(HF_REPOS.keys()), default=None,
                    help="gemma4_31b or qwen36 (default from extract_config.json)")
    ap.add_argument("--model_path", default=None,
                    help="Local path to HF snapshot. If not given, falls back to "
                         "./models/<repo-name> next to this repo, then HF cache.")
    ap.add_argument("--samples_file", default=None,
                    help="Path to JSONL of attack rollouts. Default: "
                         "../datasets/refusal_probes/<model_key>/attacks_full.jsonl")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write per-sample .pt extracts (default: ./extracts)")
    ap.add_argument("--layers", default=None,
                    help="Layer-spec (see top of file). Default 'middle'.")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default=None,
                    help="Storage dtype on disk (default fp16)")
    ap.add_argument("--no_chat_template", action="store_true",
                    help="Skip applying the model's chat template (raw prompt only)")
    ap.add_argument("--sample_limit", type=int, default=0,
                    help="Process only the first N samples (0 = all)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args)
    model_key = cfg.get("model_key", "gemma4_31b")
    model_path = cfg.get("model_path")
    use_ct = bool(cfg.get("use_chat_template", True))
    layers_spec = cfg.get("layers", "middle")
    dtype_name = cfg.get("dtype", "fp16")
    out_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    out_root = Path(cfg.get("out_dir", "./extracts")) / model_key
    out_root.mkdir(parents=True, exist_ok=True)

    samples_file = cfg.get("samples_file")
    if not samples_file:
        # Default: refusal-probe attacks for the chosen model
        samples_file = str(Path(__file__).resolve().parent.parent /
                           f"datasets/refusal_probes/{model_key}/attacks_full.jsonl")
    samples = [json.loads(l) for l in open(samples_file)]
    if args.sample_limit > 0: samples = samples[:args.sample_limit]

    print(f"=== extract_residuals model={model_key} use_ct={use_ct} layers={layers_spec!r} dtype={dtype_name} ===", flush=True)
    print(f"samples_file={samples_file} (n={len(samples)})", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model, tok = load_model(model_key, model_path, torch.bfloat16)
    n_layers = len(model.model.layers) if hasattr(model.model, "layers") else 64
    layer_idxs = parse_layers(layers_spec, n_layers)
    print(f"{model_key} loaded | n_layers={n_layers} | extracting {len(layer_idxs)} layer(s): {layer_idxs}", flush=True)

    metadata = {"model_key": model_key, "use_chat_template": use_ct,
                "n_layers_model": n_layers, "layer_idxs": layer_idxs,
                "dtype": dtype_name, "samples": []}
    t_start = time.time()
    layer_idx_t = torch.tensor(layer_idxs, dtype=torch.long, device="cuda:0")

    use_chunked = (model_key == "gemma4_31b")
    cm = chunked_sdpa_scope() if use_chunked else None
    if cm is not None: cm.__enter__()
    try:
        for i, s in enumerate(samples):
            sid = s["sample_id"]
            out_path = out_root / f"{sid}.pt"
            if out_path.exists(): continue

            prompt = s["attack_prompt"]
            txt = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
                   if use_ct else prompt)
            enc = tok(txt, return_tensors="pt").to("cuda:0")
            ids, attn = enc.input_ids, enc.attention_mask

            torch.cuda.synchronize(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=attn,
                                output_hidden_states=True, return_dict=True)
                torch.cuda.synchronize()
                fwd = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1024**3
                hs = out.hidden_states  # tuple len = n_layers+1, each (1, N, d)
                # Stack only the requested layers — saves disk + memory
                stacked = torch.stack([hs[k][0] for k in layer_idxs], dim=0)
                N = stacked.shape[1]
                stacked = stacked.to("cpu", dtype=out_dtype).contiguous()
                extract = {
                    "residuals": stacked,  # (n_selected_layers, N, d) fp16/bf16
                    "input_ids": ids[0].to("cpu", dtype=torch.int32),
                    "attention_mask": attn[0].to("cpu", dtype=torch.bool),
                    "n_tokens": int(N),
                    "layer_idxs": layer_idxs,
                    "fwd_seconds": round(fwd, 3),
                    "peak_vram_gb": round(peak, 2),
                    "use_chat_template": use_ct,
                    "label": int(s.get("is_refusal", -1)),
                }
                del out, hs, stacked
                torch.cuda.empty_cache()
                torch.save(extract, str(out_path))

                if (i+1) % 10 == 0 or (i+1) == len(samples):
                    elapsed = time.time() - t_start
                    rate = (i+1) / elapsed
                    eta = (len(samples) - (i+1)) / max(rate, 1e-3) / 60
                    sz = out_path.stat().st_size / 1024**2
                    print(f"  [{i+1}/{len(samples)}] {sid}: N={extract['n_tokens']} fwd={fwd:.2f}s peak={peak:.1f}GB sz={sz:.0f}MB | {rate:.2f}/s eta={eta:.1f}min", flush=True)
                metadata["samples"].append({"sample_id": sid, "n_tokens": extract["n_tokens"], "fwd_seconds": fwd})
            except Exception as e:
                print(f"  [{i+1}/{len(samples)}] {sid}: FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
                metadata["samples"].append({"sample_id": sid, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                torch.cuda.empty_cache()
    finally:
        if cm is not None: cm.__exit__(None, None, None)

    with open(out_root / "extraction_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    n_ok = sum(1 for s in metadata["samples"] if "error" not in s)
    n_err = sum(1 for s in metadata["samples"] if "error" in s)
    print(f"\n=== DONE ok={n_ok} err={n_err} | total {(time.time()-t_start)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()
