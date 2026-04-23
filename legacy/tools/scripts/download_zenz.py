"""Download zenz-v1 GGUF from HuggingFace for evaluation.

Usage:
    uv run python scripts/download_zenz.py --out ~/models/zenz-v1.gguf
    uv run python scripts/download_zenz.py --out ~/models/zenz-v1.gguf --filename zenz-v1-Q8_0.gguf

License note: zenz-v1 is distributed under CC-BY-SA 4.0 by Miwa-Keita.
Do not redistribute the downloaded GGUF file. Link to the HuggingFace repo instead.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_REPO = "Miwa-Keita/zenz-v1"
DEFAULT_FILENAME = "ggml-model-Q8_0.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download zenz-v1 GGUF")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--filename", default=DEFAULT_FILENAME,
        help="GGUF filename inside the repo (run `huggingface-cli repo ls %s` to check)" % DEFAULT_REPO,
    )
    parser.add_argument("--out", required=True, help="Target path for the GGUF file")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface-hub is required. Install: uv sync --group dev")

    print(f"Downloading {args.repo}/{args.filename} ...")
    cached = hf_hub_download(repo_id=args.repo, filename=args.filename)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, out_path)
    print(f"Saved to {out_path}")
    print()
    print("License: CC-BY-SA 4.0 (Miwa-Keita). Do not redistribute the file.")
    print(f"Use with: uv run python -m src.eval.run_eval --backend zenz --model-path {out_path} ...")


if __name__ == "__main__":
    main()
