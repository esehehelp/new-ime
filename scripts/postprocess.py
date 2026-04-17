"""Post-process JSONL sentence pairs to improve data quality.

Filters:
    1. Reading must be pure hiragana + allowed symbols (ー、punctuation)
    2. No POS tag leakage (hyphen followed by kanji/ASCII in reading)
    3. No kanji in reading
    4. Surface/reading length ratio within reasonable bounds
    5. Minimum sentence length (exclude headings, single words)
    6. Deduplicate by surface text

Usage:
    uv run python scripts/postprocess.py \
        --input datasets/wiki_sentences.jsonl \
        --output datasets/wiki_clean.jsonl \
        --stats
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Allowed characters in reading field
# Hiragana: U+3040-309F, prolonged sound mark: ー, punctuation, whitespace
RE_VALID_READING = re.compile(
    r"^[\u3040-\u309f\u30fc、。！？「」『』（）・…―─　\s]+$"
)

# POS tag leakage: hyphen followed by kanji, katakana, or ASCII word
RE_POS_LEAK = re.compile(r"-[a-zA-Z\u4e00-\u9fff\u30a0-\u30ff]+")

# Kanji in reading (should not be there)
RE_KANJI = re.compile(r"[\u4e00-\u9fff]")

# Katakana in reading (should have been converted to hiragana)
RE_KATAKANA = re.compile(r"[\u30a1-\u30fa]")

# Numeric patterns that shouldn't be in reading
RE_ASCII_ALPHA = re.compile(r"[a-zA-Z]")

# Repeated characters (sign of bad data)
RE_REPEATED = re.compile(r"(.)\1{4,}")


class QualityFilter:
    """Filter and clean sentence pairs."""

    def __init__(
        self,
        min_surface_len: int = 5,
        max_surface_len: int = 100,
        min_reading_len: int = 3,
        min_ratio: float = 0.3,
        max_ratio: float = 1.5,
    ):
        self.min_surface_len = min_surface_len
        self.max_surface_len = max_surface_len
        self.min_reading_len = min_reading_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.reject_counts: Counter = Counter()

    def check(self, pair: dict) -> dict | None:
        """Check and optionally clean a pair. Returns None if rejected."""
        reading = pair.get("reading", "")
        surface = pair.get("surface", "")
        context = pair.get("context", "")

        # --- Cleaning ---

        # Remove POS tag leakage from reading
        if RE_POS_LEAK.search(reading):
            cleaned = RE_POS_LEAK.sub("", reading)
            if cleaned and RE_VALID_READING.match(cleaned):
                reading = cleaned
            else:
                self.reject_counts["pos_leak"] += 1
                return None

        # --- Rejection filters ---

        # Surface length
        if len(surface) < self.min_surface_len:
            self.reject_counts["surface_too_short"] += 1
            return None
        if len(surface) > self.max_surface_len:
            self.reject_counts["surface_too_long"] += 1
            return None

        # Reading length
        if len(reading) < self.min_reading_len:
            self.reject_counts["reading_too_short"] += 1
            return None

        # Reading must be valid hiragana + allowed symbols
        if not RE_VALID_READING.match(reading):
            self.reject_counts["invalid_reading_chars"] += 1
            return None

        # No kanji in reading
        if RE_KANJI.search(reading):
            self.reject_counts["kanji_in_reading"] += 1
            return None

        # No katakana in reading (should be converted)
        if RE_KATAKANA.search(reading):
            self.reject_counts["katakana_in_reading"] += 1
            return None

        # No ASCII letters in reading
        if RE_ASCII_ALPHA.search(reading):
            self.reject_counts["ascii_in_reading"] += 1
            return None

        # Surface/reading length ratio
        if len(reading) > 0:
            ratio = len(surface) / len(reading)
            if ratio < self.min_ratio or ratio > self.max_ratio:
                self.reject_counts["bad_ratio"] += 1
                return None

        # Repeated characters (corrupted data)
        if RE_REPEATED.search(reading) or RE_REPEATED.search(surface):
            self.reject_counts["repeated_chars"] += 1
            return None

        # Reading should not be identical to surface (trivial pair)
        # Exception: all-hiragana surface is expected to match
        has_non_hiragana = any(
            not ("\u3040" <= c <= "\u309f" or c in "ー、。！？「」『』（）・…―─　 \n")
            for c in surface
        )
        if not has_non_hiragana and reading == surface:
            # All-hiragana, reading == surface: keep (it's valid)
            pass
        elif reading == surface and has_non_hiragana:
            # Non-trivial surface identical to reading: MeCab failed
            self.reject_counts["reading_equals_surface"] += 1
            return None

        return {"reading": reading, "surface": surface, "context": context}

    def report(self) -> str:
        """Return rejection statistics report."""
        lines = ["Rejection reasons:"]
        total = sum(self.reject_counts.values())
        for reason, count in self.reject_counts.most_common():
            lines.append(f"  {reason}: {count:,}")
        lines.append(f"  TOTAL rejected: {total:,}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post-process JSONL sentence pairs")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dedup", action="store_true", default=True, help="Deduplicate")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false")
    parser.add_argument("--min-len", type=int, default=5, help="Min surface length")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filt = QualityFilter(min_surface_len=args.min_len)
    seen_surfaces: set[str] = set() if args.dedup else None
    total_in = 0
    total_out = 0
    dedup_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for input_path in args.input:
            print(f"Processing {input_path}...")
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    total_in += 1
                    pair = json.loads(line)

                    cleaned = filt.check(pair)
                    if cleaned is None:
                        continue

                    # Dedup
                    if seen_surfaces is not None:
                        if cleaned["surface"] in seen_surfaces:
                            dedup_count += 1
                            continue
                        seen_surfaces.add(cleaned["surface"])

                    out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    total_out += 1

                    if total_in % 500000 == 0:
                        print(f"  {total_in:,} in, {total_out:,} out...", flush=True)

    print(f"\nTotal: {total_in:,} in → {total_out:,} out "
          f"({total_out / max(total_in, 1) * 100:.1f}% kept)")
    if args.dedup:
        print(f"Deduplicated: {dedup_count:,}")

    if args.stats:
        print(f"\n{filt.report()}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
