"""Download Gemma 4-31B-it and Qwen 3.6-27B from HF Hub.

Usage:
    python download_models.py [--out_dir ./models] [--only gemma|qwen] [--token HF_TOKEN]

Both models are gated on HF — accept their license at:
  Gemma  : https://huggingface.co/google/gemma-4-31B-it
  Qwen   : https://huggingface.co/Qwen/Qwen3.6-27B

Then either set HF_TOKEN env var or pass --token. Disk: ~62 GB Gemma + ~55 GB Qwen.
"""
import argparse, os, sys
from pathlib import Path

REPOS = {
    "gemma": ("google/gemma-4-31B-it",   "Gemma-4-31B-it"),
    "qwen":  ("Qwen/Qwen3.6-27B",        "Qwen3.6-27B"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./models",
                    help="Local directory to download into (will be created)")
    ap.add_argument("--only", choices=["gemma", "qwen"], default=None,
                    help="Skip the other model")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (or set $HF_TOKEN). Required — both repos are gated.")
    args = ap.parse_args()

    if not args.token:
        print("[error] HF_TOKEN not set and --token not given.", file=sys.stderr)
        print("        Both Gemma and Qwen are gated on HF — accept the license then", file=sys.stderr)
        print("        provide your read token from https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(2)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] pip install huggingface_hub", file=sys.stderr); sys.exit(2)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    todo = [k for k in REPOS if not args.only or k == args.only]
    for key in todo:
        repo, dir_name = REPOS[key]
        target = out_root / dir_name
        print(f"\n=== {repo} -> {target} ===")
        snapshot_download(repo_id=repo, local_dir=str(target), token=args.token,
                          local_dir_use_symlinks=False)
        size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1024**3
        print(f"  done. {size_gb:.1f} GB on disk")

    print(f"\nAll downloaded under {out_root}.")
    print("Pass this path to the extraction / attribution scripts via --model_path "
          "(or MODEL_PATH env var):")
    for key in todo:
        _, dir_name = REPOS[key]
        print(f"  {key:5s}: {out_root / dir_name}")


if __name__ == "__main__":
    main()
