"""Build output tokenizer vocabulary from training data character frequencies.

Reads JSONL files (surface field) and counts character frequencies.
Outputs a vocabulary file sorted by frequency.

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

from src.data.tokenizer import OutputTokenizer, SPECIAL_TOKENS, BLANK_TOKEN, MASK_TOKEN


def count_characters(input_paths: list[str]) -> Counter:
    """Count character frequencies across all JSONL files."""
    counter: Counter = Counter()
    total_lines = 0

    for path in input_paths:
        print(f"Counting characters in {path}...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                surface = data.get("surface", "")
                for char in surface:
                    counter[char] += 1
                total_lines += 1

    print(f"Total: {total_lines} sentences, {len(counter)} unique characters")
    return counter


def build_vocab(counter: Counter, max_kanji: int = 4000) -> dict[str, int]:
    """Build vocabulary from character frequencies."""
    vocab: dict[str, int] = {}
    idx = 0

    # Special tokens first
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    # Byte fallback tokens
    for b in range(256):
        token = f"<0x{b:02X}>"
        vocab[token] = idx
        idx += 1

    # Sort characters by frequency (most common first)
    sorted_chars = counter.most_common()

    # Separate by category
    hiragana = []
    katakana = []
    ascii_chars = []
    kanji = []
    symbols = []

    for char, freq in sorted_chars:
        cp = ord(char)
        if 0x3041 <= cp <= 0x3096:
            hiragana.append((char, freq))
        elif 0x30A1 <= cp <= 0x30FA:
            katakana.append((char, freq))
        elif 0x20 <= cp <= 0x7E:
            ascii_chars.append((char, freq))
        elif 0x4E00 <= cp <= 0x9FFF:
            kanji.append((char, freq))
        else:
            symbols.append((char, freq))

    # Add in order: hiragana, katakana, ASCII, symbols, then kanji (limited)
    for char, _ in hiragana:
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    for char, _ in katakana:
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    for char, _ in ascii_chars:
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    # Fullwidth ASCII
    for cp in range(0xFF01, 0xFF5F):
        char = chr(cp)
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    # Common symbols (from data, by frequency)
    for char, _ in symbols:
        if char not in vocab:
            vocab[char] = idx
            idx += 1

    # Kanji: limited to max_kanji most frequent
    kanji_added = 0
    for char, freq in kanji:
        if kanji_added >= max_kanji:
            break
        if char not in vocab:
            vocab[char] = idx
            idx += 1
            kanji_added += 1

    print(f"\nVocabulary summary:")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Byte fallback: 256")
    print(f"  Hiragana: {len(hiragana)}")
    print(f"  Katakana: {len(katakana)}")
    print(f"  ASCII: {len(ascii_chars)}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Kanji: {kanji_added} (of {len(kanji)} total, capped at {max_kanji})")
    print(f"  Total vocab size: {len(vocab)}")

    return vocab


def main():
    parser = argparse.ArgumentParser(description="Build output tokenizer vocabulary")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output vocab JSON path")
    parser.add_argument("--max-kanji", type=int, default=4000, help="Max kanji in vocab")
    parser.add_argument("--stats", default="", help="Output stats JSON path")
    args = parser.parse_args()

    counter = count_characters(args.input)
    vocab = build_vocab(counter, args.max_kanji)

    # Save as OutputTokenizer format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"type": "output", "token_to_id": vocab}
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVocabulary saved to {args.output}")

    # Optional stats
    if args.stats:
        stats = {
            "total_characters": sum(counter.values()),
            "unique_characters": len(counter),
            "vocab_size": len(vocab),
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
