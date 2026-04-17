"""Download FineWeb-2 jpn_Jpan train shards.

FineWeb-2 (``HuggingFaceFW/fineweb-2``) publishes Japanese data under
``data/jpn_Jpan/train/*.parquet``. Each train shard is ~4.6 GB. License is
ODC-BY 1.0 (see ``docs/dataset_candidates.md`` row #3).

Chosen over CulturaX (uonlp/CulturaX) because the latter is gated and
requires per-user access approval.

Usage:
    uv run python scripts/download_fineweb2_ja.py \
        --out datasets/src/fineweb2_ja/ \
        --num-shards 2

Default downloads 2 train shards (~9 GB total).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "HuggingFaceFW/fineweb-2"
SHARD_PREFIX = "data/jpn_Jpan/train/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/src/fineweb2_ja/", help="Output directory")
    parser.add_argument(
        "--num-shards", type=int, default=2, help="Number of train shards to download"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Index of the first shard to download (0-based)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.start_index, args.start_index + args.num_shards):
        filename = f"{SHARD_PREFIX}000_{i:05d}.parquet"
        print(f"Downloading {filename}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(out_dir),
        )
        size_gb = Path(path).stat().st_size / 1024**3
        print(f"  -> {path} ({size_gb:.2f} GB)")

    (out_dir / "ATTRIBUTION.md").write_text(
        "Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2\n"
        "License: ODC-By 1.0 (FineWeb-2 dataset card).\n"
        "Attribution required on redistribution of derivatives. Common Crawl\n"
        "terms of use also apply (source is CC-derived).\n",
        encoding="utf-8",
    )
    print(f"Attribution note: {out_dir / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
