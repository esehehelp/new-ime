"""Download HPLT v3 Japanese shards (jpn_Jpan).

HPLT 3.0 publishes the Japanese subset as 38 zstd-compressed JSONL shards on
``data.hplt-project.org``. Shards are named ``<tier>_<index>.jsonl.zst``,
where ``tier`` reflects a quality bucket (10 = top quality, smallest;
lower tiers have more data but lower filtering). The whole jpn_Jpan set is
~1.6 TB compressed. We download only the high-quality tiers by default.

Sizes observed (2026-04-18):
- 10_1.jsonl.zst: 3.1 GB   (top quality)
- 5_1.jsonl.zst:  37.8 GB

Usage:
    uv run python scripts/download_hplt3_ja.py \
        --out datasets/src/hplt3_ja/ \
        --tiers 10 \
        [--max-shards N] [--max-bytes BYTES]

The license is CC0-1.0 for the whole HPLT 3.0 dataset, but Common Crawl's
terms of use still apply to derivative redistributions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


MAP_URL = "https://data.hplt-project.org/three/sorted/jpn_Jpan.map"
CHUNK_BYTES = 1 << 20  # 1 MiB
USER_AGENT = "new-ime/dataset-fetch (+https://github.com/esehehelp/new-ime)"
SHARD_NAME_RE = re.compile(r"(?P<tier>\d+)_(?P<idx>\d+)\.jsonl\.zst$")


def fetch_map(url: str = MAP_URL) -> list[str]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return [line.strip() for line in r.text.splitlines() if line.strip()]


def select_shards(
    urls: list[str],
    tiers: list[int] | None,
    max_shards: int,
) -> list[str]:
    selected = []
    for u in urls:
        name = Path(urlparse(u).path).name
        m = SHARD_NAME_RE.search(name)
        if not m:
            continue
        tier = int(m.group("tier"))
        if tiers and tier not in tiers:
            continue
        selected.append(u)
    # Prefer higher tiers first (we're selecting quality top-down).
    selected.sort(
        key=lambda u: -int(SHARD_NAME_RE.search(Path(urlparse(u).path).name).group("tier"))
    )
    if max_shards > 0:
        selected = selected[:max_shards]
    return selected


def download_shard(url: str, out_dir: Path, max_bytes: int = 0) -> tuple[Path, int]:
    out_path = out_dir / Path(urlparse(url).path).name
    existing = out_path.stat().st_size if out_path.exists() else 0

    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if existing and r.status_code == 200:
            # Server ignored our Range header; restart.
            existing = 0
        r.raise_for_status()
        total_remaining = int(r.headers.get("Content-Length", 0))
        total_size = total_remaining + existing
        print(
            f"{out_path.name}: {total_size / 1024**3:.2f} GB total, "
            f"resuming from {existing / 1024**3:.2f} GB"
            if existing
            else f"{out_path.name}: {total_size / 1024**3:.2f} GB"
        )

        mode = "ab" if existing else "wb"
        bytes_this_run = 0
        with out_path.open(mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                if not chunk:
                    continue
                f.write(chunk)
                bytes_this_run += len(chunk)
                if max_bytes and (existing + bytes_this_run) >= max_bytes:
                    print(f"  hit max_bytes cap @ {(existing + bytes_this_run) / 1024**3:.2f} GB")
                    break
    return out_path, existing + bytes_this_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/src/hplt3_ja/", help="Output directory")
    parser.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[10],
        help="Quality tiers to download (10=top). Default: [10]",
    )
    parser.add_argument(
        "--max-shards", type=int, default=0, help="Cap number of shards downloaded"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Cap total compressed bytes across the run (0 = no cap)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching shard map: {MAP_URL}")
    urls = fetch_map()
    print(f"  {len(urls)} total shards")

    selected = select_shards(urls, args.tiers, args.max_shards)
    print(f"Selected {len(selected)} shard(s) from tiers {args.tiers}:")
    for u in selected:
        print(f"  {u}")
    if not selected:
        print("No shards match the tier filter.", file=sys.stderr)
        sys.exit(1)

    remaining_budget = args.max_bytes
    for u in selected:
        per_shard_cap = remaining_budget if remaining_budget > 0 else 0
        path, written = download_shard(u, out_dir, max_bytes=per_shard_cap)
        print(f"  wrote {written / 1024**3:.2f} GB → {path}")
        if args.max_bytes:
            remaining_budget -= written
            if remaining_budget <= 0:
                print("Global max-bytes budget exhausted, stopping.")
                break

    (out_dir / "ATTRIBUTION.md").write_text(
        "Source: HPLT 3.0 — https://data.hplt-project.org/\n"
        "License: CC0-1.0 (HF dataset card). Common Crawl terms of use apply\n"
        "to derivative redistribution.\n",
        encoding="utf-8",
    )
    print(f"Attribution note: {out_dir / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
