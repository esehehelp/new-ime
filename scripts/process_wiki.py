"""Process Wikipedia dump into sentence pairs for kana-kanji conversion training.

Usage:
    uv run python scripts/process_wiki.py \
        --input datasets/src/jawiki-latest-pages-articles.xml.bz2 \
        --output datasets/wiki_sentences.jsonl

Pipeline:
    1. Parse XML dump via mwxml + mwparserfromhell (no wikiextractor)
    2. Strip wikitext markup → plain text
    3. Split into sentences
    4. Assign readings via MeCab
    5. Filter bad sentences
    6. Output (reading, surface, context) triples as JSONL
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import jaconv
import MeCab
import mwparserfromhell
import mwxml


# Sentence splitting regex
SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")

# Filter patterns
RE_URL = re.compile(r"https?://")
RE_MARKUP = re.compile(r"[{}\[\]|<>]")


def wikitext_to_plain(wikitext: str) -> str:
    """Convert wikitext to plain text using mwparserfromhell."""
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code(normalize=True, collapse=True)
    except Exception:
        return ""
    # Remove section headers (== title ==)
    text = re.sub(r"={2,}.*?={2,}", "", text)
    # Remove leftover markup artifacts
    text = re.sub(r"\[\[|\]\]|\{\{|\}\}", "", text)
    # Collapse multiple newlines/spaces
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = SENTENCE_SPLIT.split(text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    return result


def is_valid_sentence(text: str) -> bool:
    """Check if sentence is suitable for training."""
    if len(text) < 5 or len(text) > 100:
        return False
    if RE_URL.search(text):
        return False
    if RE_MARKUP.search(text):
        return False
    ascii_count = sum(1 for c in text if ord(c) < 0x80)
    if ascii_count / len(text) > 0.3:
        return False
    return True


class ReadingAssigner:
    """Assign hiragana readings to Japanese text using MeCab."""

    def __init__(self) -> None:
        self.tagger = MeCab.Tagger()
        self.tagger.parse("")

    def get_reading(self, text: str) -> str | None:
        """Get full hiragana reading. Returns None if unreliable."""
        node = self.tagger.parseToNode(text)
        readings = []

        while node:
            surface = node.surface
            if not surface:
                node = node.next
                continue

            features = node.feature.split(",")

            if features[0] == "未知語":
                return None

            reading = None
            if len(features) >= 8 and features[7] != "*":
                reading = features[7]
            elif len(features) >= 7 and features[6] != "*":
                reading = features[6]

            if reading:
                hira = jaconv.kata2hira(reading)
                # Reject if reading contains ASCII or other non-Japanese chars
                if any(c.isascii() and c.isalpha() for c in hira):
                    return None
                readings.append(hira)
            elif all("\u3040" <= c <= "\u309f" or not c.strip() for c in surface):
                readings.append(surface)
            elif all(c in "、。！？「」『』（）・…―─　\n\r\t " for c in surface):
                readings.append(surface)
            else:
                return None

            node = node.next

        return "".join(readings)


def iter_wiki_articles(dump_path: str):
    """Iterate over Wikipedia articles from XML bz2 dump."""
    import bz2

    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            if page.namespace != 0:
                continue
            for revision in page:
                if revision.text:
                    yield page.title, revision.text
                break


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia dump")
    parser.add_argument("--input", required=True, help="Path to jawiki XML bz2 dump")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-articles", type=int, default=0, help="Limit articles (0=all)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assigner = ReadingAssigner()
    total_pairs = 0
    total_articles = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for title, wikitext in iter_wiki_articles(args.input):
            plain_text = wikitext_to_plain(wikitext)
            if not plain_text:
                continue

            sentences = split_sentences(plain_text)
            prev_surface = ""

            for sent in sentences:
                if not is_valid_sentence(sent):
                    prev_surface = ""
                    continue

                reading = assigner.get_reading(sent)
                if reading is None:
                    prev_surface = ""
                    continue

                context = prev_surface[-40:] if prev_surface else ""
                pair = {
                    "reading": reading,
                    "surface": sent,
                    "context": context,
                }
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += 1
                prev_surface = sent

            total_articles += 1
            if total_articles % 10000 == 0:
                print(f"  {total_articles} articles, {total_pairs} pairs...")

            if args.max_articles and total_articles >= args.max_articles:
                break

    print(f"\nDone: {total_articles} articles -> {total_pairs} sentence pairs")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
