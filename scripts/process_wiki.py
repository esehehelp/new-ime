"""Process Wikipedia dump into sentence pairs for kana-kanji conversion training.

Usage:
    uv run python scripts/process_wiki.py \
        --input datasets/src/jawiki-latest-pages-articles.xml.bz2 \
        --output datasets/wiki_sentences.jsonl

Pipeline:
    1. Extract text from XML dump using wikiextractor
    2. Split into sentences
    3. Assign readings via MeCab
    4. Filter bad sentences
    5. Output (reading, surface, context) triples as JSONL
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import MeCab
import jaconv


def extract_wiki_text(xml_bz2_path: str, output_dir: str) -> None:
    """Run wikiextractor to extract plain text from Wikipedia dump."""
    print(f"Extracting Wikipedia text from {xml_bz2_path}...")
    subprocess.run(
        [
            sys.executable, "-m", "wikiextractor.WikiExtractor",
            xml_bz2_path,
            "--output", output_dir,
            "--json",
            "--no-templates",
            "--min_text_length", "50",
            "--processes", "4",
        ],
        check=True,
    )
    print(f"Extraction complete: {output_dir}")


def iter_wiki_articles(extracted_dir: str):
    """Iterate over extracted Wikipedia articles (JSON lines)."""
    extracted = Path(extracted_dir)
    for subdir in sorted(extracted.iterdir()):
        if not subdir.is_dir():
            continue
        for fpath in sorted(subdir.iterdir()):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        article = json.loads(line)
                        yield article.get("text", "")
                    except json.JSONDecodeError:
                        continue


# Sentence splitting regex
SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")

# Filter patterns
RE_NON_JA = re.compile(r"[a-zA-Z]{10,}")  # Long ASCII runs
RE_URL = re.compile(r"https?://")
RE_MARKUP = re.compile(r"[{}\[\]|<>]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Remove section headers (== title ==)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    # Split
    sentences = SENTENCE_SPLIT.split(text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    return result


def is_valid_sentence(text: str) -> bool:
    """Check if sentence is suitable for training."""
    # Length check (character count)
    if len(text) < 5 or len(text) > 100:
        return False
    # No URLs
    if RE_URL.search(text):
        return False
    # No remaining markup
    if RE_MARKUP.search(text):
        return False
    # Not too much ASCII (>30% of characters)
    ascii_count = sum(1 for c in text if ord(c) < 0x80)
    if ascii_count / len(text) > 0.3:
        return False
    return True


class ReadingAssigner:
    """Assign hiragana readings to Japanese text using MeCab."""

    def __init__(self) -> None:
        self.tagger = MeCab.Tagger()
        # Force UTF-8 output
        self.tagger.parse("")  # Initialize

    def get_reading(self, text: str) -> str | None:
        """Get full hiragana reading for text. Returns None if reading is unreliable."""
        node = self.tagger.parseToNode(text)
        readings = []
        has_unknown = False

        while node:
            surface = node.surface
            if not surface:
                node = node.next
                continue

            features = node.feature.split(",")

            # Check for unknown words
            if features[0] == "未知語":
                has_unknown = True

            # Get reading (last feature field in ipadic/unidic-lite)
            reading = None
            if len(features) >= 8 and features[7] != "*":
                reading = features[7]
            elif len(features) >= 7 and features[6] != "*":
                reading = features[6]

            if reading:
                # Convert katakana reading to hiragana
                readings.append(jaconv.kata2hira(reading))
            else:
                # No reading available — use surface if it's already hiragana
                if all("\u3040" <= c <= "\u309f" or not c.strip() for c in surface):
                    readings.append(surface)
                else:
                    has_unknown = True
                    readings.append(surface)

            node = node.next

        if has_unknown:
            return None

        return "".join(readings)


def process_sentences(
    sentences: list[str],
    assigner: ReadingAssigner,
    context_window: int = 40,
) -> list[dict]:
    """Process sentences into (reading, surface, context) triples."""
    results = []
    prev_surface = ""

    for sent in sentences:
        if not is_valid_sentence(sent):
            prev_surface = ""
            continue

        reading = assigner.get_reading(sent)
        if reading is None:
            prev_surface = ""
            continue

        # Trim context to last N characters
        context = prev_surface[-context_window:] if prev_surface else ""

        results.append({
            "reading": reading,
            "surface": sent,
            "context": context,
        })

        prev_surface = sent

    return results


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia dump")
    parser.add_argument("--input", required=True, help="Path to jawiki XML bz2 dump")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-articles", type=int, default=0, help="Limit articles (0=all)")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction step")
    parser.add_argument("--extracted-dir", default="", help="Pre-extracted directory")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract
    if args.extracted_dir:
        extracted_dir = args.extracted_dir
    elif args.skip_extract:
        extracted_dir = str(Path(args.input).parent / "wiki_extracted")
    else:
        extracted_dir = str(Path(args.input).parent / "wiki_extracted")
        extract_wiki_text(args.input, extracted_dir)

    # Step 2: Process
    assigner = ReadingAssigner()
    total_pairs = 0
    total_articles = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for article_text in iter_wiki_articles(extracted_dir):
            sentences = split_sentences(article_text)
            pairs = process_sentences(sentences, assigner)

            for pair in pairs:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += 1

            total_articles += 1
            if total_articles % 10000 == 0:
                print(f"  Processed {total_articles} articles, {total_pairs} pairs...")

            if args.max_articles and total_articles >= args.max_articles:
                break

    print(f"\nDone: {total_articles} articles → {total_pairs} sentence pairs")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
