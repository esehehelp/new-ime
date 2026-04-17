"""Audit SharedCharTokenizer byte fallback and invalid decode behavior.

Usage:
    uv run python scripts/audit_tokenizer.py \
        --input datasets/eval_v3/train.jsonl \
        --tokenizer datasets/shared_tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tokenizer import INVALID_BYTE_TOKEN, SharedCharTokenizer


def iter_texts(path: str, fields: list[str]):
    with open(path, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            for field in fields:
                value = sample.get(field, "")
                if value:
                    yield field, value


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit tokenizer fallback behavior")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--tokenizer", required=True, help="Saved SharedCharTokenizer JSON")
    parser.add_argument("--fields", nargs="+", default=["reading", "surface", "context"])
    parser.add_argument("--limit", type=int, default=0, help="Max number of JSONL rows to inspect")
    args = parser.parse_args()

    tokenizer = SharedCharTokenizer.load(args.tokenizer)
    per_field = {
        field: {"fallback_chars": 0, "total_chars": 0, "examples": []}
        for field in args.fields
    }
    rows = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            rows += 1
            for field in args.fields:
                text = sample.get(field, "")
                fallback_chars, total_chars = tokenizer.count_byte_fallbacks(text)
                info = per_field[field]
                info["fallback_chars"] += fallback_chars
                info["total_chars"] += total_chars
                if fallback_chars > 0 and len(info["examples"]) < 5:
                    info["examples"].append(text[:120])
            if args.limit and rows >= args.limit:
                break

    print(f"rows={rows}")
    print(f"invalid_decode_sentinel={INVALID_BYTE_TOKEN}")
    for field in args.fields:
        info = per_field[field]
        total = max(info["total_chars"], 1)
        ratio = info["fallback_chars"] / total
        print(
            f"{field}: fallback_chars={info['fallback_chars']} "
            f"total_chars={info['total_chars']} ratio={ratio:.6f}"
        )
        for idx, example in enumerate(info["examples"], start=1):
            print(f"  example{idx}: {example}")


if __name__ == "__main__":
    main()
