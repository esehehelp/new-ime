"""Post-process JSONL sentence pairs to improve data quality.

Filters:
    1. Reading must be pure hiragana + allowed symbols (ー、punctuation)
    2. No POS tag leakage (hyphen followed by kanji/ASCII in reading)
    3. No kanji/katakana/ASCII in reading
    4. Surface/reading length ratio within reasonable bounds
    5. Minimum sentence length (exclude headings, single words)
    6. Deduplicate by surface text
    7. Reject old orthography (旧仮名遣い)
    8. Strip leading/trailing whitespace
    9. Reject stage directions (戯曲ト書き)
   10. Reject chapter numbers / bare titles

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

# ASCII letters in reading
RE_ASCII_ALPHA = re.compile(r"[a-zA-Z]")

# Repeated characters (sign of bad data)
RE_REPEATED = re.compile(r"(.)\1{4,}")

# --- Old orthography detection ---
# 旧仮名遣い indicators in surface text
# ゐ (wi), ゑ (we) — archaic kana
RE_OLD_KANA_SURFACE = re.compile(r"[ゐゑヰヱ]")

# 旧仮名遣い verb endings in surface: つた(=った), つて(=って), ふ(=う at word boundary)
# Also ゝ ゞ (iteration marks)
RE_OLD_ITERATION = re.compile(r"[ゝゞヽヾ]")

# Historical kana: surface contains っ written as つ before た/て/たり etc.
# e.g. 眠つた、振り返つて、思つた — common in pre-war writing
# Detect: non-っ つ followed by た/て/たら/たり
RE_OLD_SOKUON = re.compile(r"[^っ]つ[たてだで]")

# 旧仮名遣い ふ/ひ used as auxiliary verb う/い
# e.g. 思ふ(=思う), 違ふ(=違う), 云ふ(=言う), ないらしい→ないらしひ
RE_OLD_FU = re.compile(r"[はかさたなまらわがざだば]ふ[。、）」\s]|ふ[。」]$")

# 旧仮名遣い せう/ませう (=しょう/ましょう)
RE_OLD_SYOU = re.compile(r"[でせ]せう|ませう")

# Surface starts with digits or digit readings (章番号・箇条書き)
RE_STARTS_WITH_NUMBER = re.compile(r"^\d+[　\s]|^[０-９]+[　\s]")

# --- Stage direction / script detection ---
# Pattern: character name + full-width spaces + dialogue
# e.g. "ステラ　　いけません" or "お妙　（ニコニコしながら）"
RE_STAGE_DIRECTION = re.compile(
    r"^[\u3040-\u9fff\u30a0-\u30ffA-Za-z]{1,10}[　\s]{2,}"  # Name + wide spaces
    r"|^[\u3040-\u9fff\u30a0-\u30ff]{1,10}　[（\(]"  # Name + parenthetical
)

# --- Chapter/section number ---
# e.g. "その六十七", "第三章", bare numbers with prefix
RE_CHAPTER = re.compile(
    r"^[　\s]*(その|第)[一二三四五六七八九十百千\d]+[章節回話]?[　\s]*$"
)

# --- Bare quoted title ---
# e.g. "「黄色い顔」" — just a title in quotes, no sentence structure
RE_BARE_TITLE = re.compile(r"^[「『](.{1,20})[」』]$")

# --- Whitespace normalization ---
RE_LEADING_TRAILING_WS = re.compile(r"^[　\s]+|[　\s]+$")
RE_MULTI_WS = re.compile(r"[　]{2,}")


class QualityFilter:
    """Filter and clean sentence pairs."""

    def __init__(
        self,
        min_surface_len: int = 8,
        max_surface_len: int = 100,
        min_reading_len: int = 5,
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

        # Strip leading/trailing whitespace (including full-width)
        reading = RE_LEADING_TRAILING_WS.sub("", reading)
        surface = RE_LEADING_TRAILING_WS.sub("", surface)
        context = RE_LEADING_TRAILING_WS.sub("", context)

        # Collapse multiple full-width spaces
        surface = RE_MULTI_WS.sub("　", surface)
        reading = RE_MULTI_WS.sub("　", reading)

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

        # --- Old orthography rejection ---

        # Archaic kana in surface: ゐ, ゑ, ヰ, ヱ
        if RE_OLD_KANA_SURFACE.search(surface):
            self.reject_counts["old_kana"] += 1
            return None

        # Iteration marks: ゝ ゞ ヽ ヾ
        if RE_OLD_ITERATION.search(surface):
            self.reject_counts["old_iteration_mark"] += 1
            return None

        # Historical sokuon: つた/つて instead of った/って
        if RE_OLD_SOKUON.search(surface):
            self.reject_counts["old_sokuon"] += 1
            return None

        # Historical ふ as う: 思ふ, 云ふ, etc.
        if RE_OLD_FU.search(surface):
            self.reject_counts["old_fu"] += 1
            return None

        # Historical せう/ませう = しょう/ましょう
        if RE_OLD_SYOU.search(surface):
            self.reject_counts["old_syou"] += 1
            return None

        # Starts with number (list item, reference number)
        if RE_STARTS_WITH_NUMBER.search(surface):
            self.reject_counts["starts_with_number"] += 1
            return None

        # --- Structural filters ---

        # Stage directions (play scripts)
        if RE_STAGE_DIRECTION.search(surface):
            self.reject_counts["stage_direction"] += 1
            return None

        # Chapter/section numbers
        if RE_CHAPTER.match(surface):
            self.reject_counts["chapter_number"] += 1
            return None

        # Bare quoted titles (「title」 with no sentence structure)
        m = RE_BARE_TITLE.match(surface)
        if m and len(m.group(1)) < 15:
            self.reject_counts["bare_title"] += 1
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
            pass  # All-hiragana, reading == surface: valid
        elif reading == surface and has_non_hiragana:
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
    parser.add_argument("--min-len", type=int, default=8, help="Min surface length")
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

    print(f"\nTotal: {total_in:,} in -> {total_out:,} out "
          f"({total_out / max(total_in, 1) * 100:.1f}% kept)")
    if args.dedup:
        print(f"Deduplicated: {dedup_count:,}")

    if args.stats:
        print(f"\n{filt.report()}")

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
