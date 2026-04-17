"""Report statistics on processed JSONL dataset files.

Usage:
    uv run python scripts/dataset_stats.py datasets/aozora_sentences.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def report_stats(path: str) -> None:
    total = 0
    reading_lens: list[int] = []
    surface_lens: list[int] = []
    has_context = 0
    ratio_sum = 0.0

    with open(path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            r_len = len(data["reading"])
            s_len = len(data["surface"])
            reading_lens.append(r_len)
            surface_lens.append(s_len)
            if data.get("context"):
                has_context += 1
            if r_len > 0:
                ratio_sum += s_len / r_len

    if total == 0:
        print(f"{path}: empty")
        return

    reading_lens.sort()
    surface_lens.sort()

    def percentile(data: list[int], p: float) -> int:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    print(f"=== {path} ===")
    print(f"Total pairs: {total:,}")
    print(f"With context: {has_context:,} ({has_context/total*100:.1f}%)")
    print(f"\nReading length (chars):")
    print(f"  min={reading_lens[0]} p25={percentile(reading_lens, 25)} "
          f"p50={percentile(reading_lens, 50)} p75={percentile(reading_lens, 75)} "
          f"p95={percentile(reading_lens, 95)} max={reading_lens[-1]}")
    print(f"\nSurface length (chars):")
    print(f"  min={surface_lens[0]} p25={percentile(surface_lens, 25)} "
          f"p50={percentile(surface_lens, 50)} p75={percentile(surface_lens, 75)} "
          f"p95={percentile(surface_lens, 95)} max={surface_lens[-1]}")
    print(f"\nSurface/Reading ratio: {ratio_sum/total:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="JSONL files to analyze")
    args = parser.parse_args()

    for path in args.files:
        report_stats(path)


if __name__ == "__main__":
    main()
