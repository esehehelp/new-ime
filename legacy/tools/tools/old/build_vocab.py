"""Build shared tokenizer vocabulary from training data character frequencies.

Reads JSONL files (surface field) and counts character frequencies.
Outputs a saved SharedCharTokenizer vocabulary sorted by frequency.

Usage:
    uv run python scripts/build_vocab.py \
        --input datasets/aozora_sentences.jsonl datasets/wiki_sentences.jsonl \
        --output datasets/vocab.json \
        --max-kanji 4000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.data.tokenizer import SharedCharTokenizer


def count_characters(input_paths: list[str], fields: list[str]) -> Counter:
    """Count character frequencies across JSONL fields."""
    counter: Counter = Counter()
    total_lines = 0

    for path in input_paths:
        print(f"Counting characters in {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                for field in fields:
                    text = data.get(field, "")
                    for char in text:
                        counter[char] += 1
                total_lines += 1

    print(f"Total: {total_lines} sentences, {len(counter)} unique characters")
    return counter


def write_frequency_file(counter: Counter, path: Path) -> None:
    lines = [f"{char}\t{count}" for char, count in counter.most_common()]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build shared tokenizer vocabulary")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output vocab JSON path")
    parser.add_argument("--max-kanji", type=int, default=4000, help="Max kanji in vocab")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["reading", "surface", "context"],
        help="JSON fields to count",
    )
    parser.add_argument("--stats", default="", help="Output stats JSON path")
    args = parser.parse_args()

    counter = count_characters(args.input, args.fields)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    freq_path = output_path.with_suffix(".freq.tsv")
    write_frequency_file(counter, freq_path)
    tokenizer = SharedCharTokenizer.from_frequency_file(freq_path, max_kanji=args.max_kanji)

    tokenizer.save(output_path)
    print(f"\nVocabulary saved to {args.output}")
    print(f"Frequency list saved to {freq_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Optional stats
    if args.stats:
        stats = {
            "fields": args.fields,
            "total_characters": sum(counter.values()),
            "unique_characters": len(counter),
            "vocab_size": tokenizer.vocab_size,
            "top_100_chars": [
                {"char": c, "count": n}
                for c, n in counter.most_common(100)
            ],
        }
        stats_path = Path(args.stats)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Stats saved to {args.stats}")


if __name__ == "__main__":
    main()
