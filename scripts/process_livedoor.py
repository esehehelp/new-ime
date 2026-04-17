"""Process Livedoor News corpus into sentence pairs.

Usage:
    uv run python scripts/process_livedoor.py \
        --input datasets/src/text \
        --output datasets/livedoor_sentences.jsonl \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import jaconv
import MeCab


SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")
RE_URL = re.compile(r"https?://")


def _worker_init():
    import MeCab as _MeCab
    global _tagger
    _tagger = _MeCab.Tagger()
    _tagger.parse("")


def _get_reading(tagger, text: str) -> str | None:
    node = tagger.parseToNode(text)
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
        if len(features) >= 7 and features[6] != "*":
            reading = features[6]
        if reading:
            hira = jaconv.kata2hira(reading)
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


def _process_article(text: str) -> list[dict]:
    """Process a single article text into sentence pairs."""
    global _tagger
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    pairs = []
    for sent in sentences:
        if len(sent) < 8 or len(sent) > 100:
            continue
        if RE_URL.search(sent):
            continue
        ascii_count = sum(1 for c in sent if ord(c) < 0x80)
        if ascii_count / len(sent) > 0.3:
            continue
        reading = _get_reading(_tagger, sent)
        if reading is None:
            continue
        pairs.append({"reading": reading, "surface": sent})
    return pairs


def iter_livedoor_articles(text_dir: str):
    """Yield article texts from Livedoor corpus directory."""
    for category in sorted(os.listdir(text_dir)):
        cat_dir = os.path.join(text_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".txt") or fname.startswith("LICENSE"):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()
            # Livedoor format: line 1 = URL, line 2 = date, line 3 = title, rest = body
            if len(lines) > 3:
                body = "".join(lines[3:])
                yield body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Livedoor text/ directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    articles = list(iter_livedoor_articles(args.input))
    print(f"Articles: {len(articles)}")

    total = 0
    with open(args.output, "w", encoding="utf-8") as out, \
         Pool(processes=args.workers, initializer=_worker_init) as pool:
        for article_pairs in pool.imap_unordered(_process_article, articles, chunksize=50):
            prev = ""
            for pair in article_pairs:
                pair["context"] = prev[-40:] if prev else ""
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1
                prev = pair["surface"]

    print(f"Done: {len(articles)} articles -> {total} sentence pairs")


if __name__ == "__main__":
    main()
