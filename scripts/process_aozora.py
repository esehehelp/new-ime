"""Process Aozora Bunko morphological analysis data into sentence pairs.

Input: CSV.gz files from Aozora Bunko morphological analysis corpus.
Format: filename,sentence_id,morpheme_id,surface,pos,pos_detail,...,lemma,reading_katakana,pronunciation

Output: JSONL with {reading, surface, context} triples.

Usage:
    uv run python scripts/process_aozora.py \
        --input datasets/src/utf8_all.csv.gz \
        --output datasets/aozora_sentences.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

import jaconv


# Filter patterns
RE_MOSTLY_PUNCT = re.compile(r"^[\s、。！？「」『』（）・…―─　]+$")


def is_valid_sentence(surface: str, reading: str) -> bool:
    """Check if sentence is suitable for training."""
    if len(surface) < 3 or len(surface) > 100:
        return False
    if RE_MOSTLY_PUNCT.match(surface):
        return False
    # Must have at least some kanji or katakana (not pure hiragana/punct)
    has_content = any(
        ("\u4e00" <= c <= "\u9fff") or ("\u30a0" <= c <= "\u30ff")
        for c in surface
    )
    if not has_content:
        return False
    # Reading should not be empty
    if not reading.strip():
        return False
    return True


def process_aozora_csv(
    input_path: str,
    output_path: str,
    context_window: int = 40,
) -> tuple[int, int]:
    """Process Aozora CSV into sentence pairs.

    Returns (total_sentences, valid_pairs).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_sentences = 0
    valid_pairs = 0
    prev_surface = ""

    current_file = ""
    current_sent_id = ""
    surface_parts: list[str] = []
    reading_parts: list[str] = []
    has_unknown_reading = False

    def flush_sentence():
        nonlocal total_sentences, valid_pairs, prev_surface
        nonlocal surface_parts, reading_parts, has_unknown_reading

        if not surface_parts:
            return None

        surface = "".join(surface_parts)
        reading = "".join(reading_parts)
        total_sentences += 1

        surface_parts = []
        reading_parts = []

        if has_unknown_reading:
            has_unknown_reading = False
            prev_surface = ""
            return None

        if not is_valid_sentence(surface, reading):
            prev_surface = ""
            return None

        context = prev_surface[-context_window:] if prev_surface else ""
        prev_surface = surface
        valid_pairs += 1

        return {
            "reading": reading,
            "surface": surface,
            "context": context,
        }

    open_fn = gzip.open if input_path.endswith(".gz") else open

    with open_fn(input_path, "rt", encoding="utf-8", errors="replace") as f, \
         open(output, "w", encoding="utf-8") as out:

        reader = csv.reader(f)
        for row in reader:
            if len(row) < 12:
                continue

            filename = row[0]
            sent_id = row[1]
            surface = row[3]
            reading_kata = row[11] if len(row) > 11 else row[10]

            # New file → reset context
            if filename != current_file:
                result = flush_sentence()
                if result:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                current_file = filename
                current_sent_id = sent_id
                prev_surface = ""
                surface_parts = []
                reading_parts = []
                has_unknown_reading = False

            # New sentence in same file
            elif sent_id != current_sent_id:
                result = flush_sentence()
                if result:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                current_sent_id = sent_id
                has_unknown_reading = False

            # Accumulate morphemes
            surface_parts.append(surface)

            if reading_kata and reading_kata != "*":
                reading_parts.append(jaconv.kata2hira(reading_kata))
            elif all("\u3040" <= c <= "\u309f" or not c.strip() for c in surface):
                # Surface is already hiragana
                reading_parts.append(surface)
            elif all(c in "、。！？「」『』（）・…―─　「」\n\r\t " for c in surface):
                # Punctuation — keep as-is in reading
                reading_parts.append(surface)
            else:
                has_unknown_reading = True
                reading_parts.append(surface)

        # Flush last sentence
        result = flush_sentence()
        if result:
            with open(output, "a", encoding="utf-8") as out:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")

    return total_sentences, valid_pairs


def main():
    parser = argparse.ArgumentParser(description="Process Aozora Bunko data")
    parser.add_argument("--input", required=True, help="Input CSV.gz path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    print(f"Processing {args.input}...")
    total, valid = process_aozora_csv(args.input, args.output)
    print(f"Done: {total} sentences → {valid} valid pairs ({valid/max(total,1)*100:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
