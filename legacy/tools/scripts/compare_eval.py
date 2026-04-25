"""Merge multiple eval result JSONs into a single markdown comparison table.

Usage:
    uv run python scripts/compare_eval.py results/*.json > results/comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare eval results")
    parser.add_argument("files", nargs="+", help="Result JSON files")
    parser.add_argument("--sort", default="char_acc_top1", help="Field to sort by (desc)")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.files:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue
        row = {
            "backend": data.get("backend", Path(path).stem),
            "total": data.get("total", 0),
            "char_acc_top1": data.get("char_acc_top1", 0.0),
            "char_acc_top5": data.get("char_acc_top5", 0.0),
            "char_acc_top10": data.get("char_acc_top10", 0.0),
            "exact_match_top1": data.get("exact_match_top1", 0.0),
            "exact_match_top5": data.get("exact_match_top5", 0.0),
            "p50_ms": data.get("latency", {}).get("p50_ms", 0.0),
            "p95_ms": data.get("latency", {}).get("p95_ms", 0.0),
            "file": path,
        }
        rows.append(row)

    rows.sort(key=lambda r: r.get(args.sort, 0), reverse=True)

    print("| Backend | N | CharAcc@1 | CharAcc@5 | CharAcc@10 | Exact@1 | Exact@5 | p50 ms | p95 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['backend']} | {r['total']} "
            f"| {r['char_acc_top1']:.4f} | {r['char_acc_top5']:.4f} | {r['char_acc_top10']:.4f} "
            f"| {r['exact_match_top1']:.4f} | {r['exact_match_top5']:.4f} "
            f"| {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |"
        )


if __name__ == "__main__":
    main()
