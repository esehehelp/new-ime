"""Process Tatoeba Japanese sentences into sentence pairs.

Usage:
    uv run python scripts/process_tatoeba.py \
        --input datasets/src/jpn_sentences.tsv.bz2 \
        --output datasets/tatoeba_sentences.jsonl \
        --workers 8
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
from multiprocessing import Pool
from pathlib import Path

import jaconv
import MeCab


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


def _process_sentence(sent: str) -> dict | None:
    """Process a single sentence."""
    global _tagger
    sent = sent.strip()
    if len(sent) < 5 or len(sent) > 100:
        return None
    ascii_count = sum(1 for c in sent if ord(c) < 0x80)
    if len(sent) > 0 and ascii_count / len(sent) > 0.3:
        return None
    reading = _get_reading(_tagger, sent)
    if reading is None:
        return None
    return {"reading": reading, "surface": sent, "context": ""}


def iter_tatoeba(path: str):
    """Yield sentences from Tatoeba TSV.bz2."""
    open_fn = bz2.open if path.endswith(".bz2") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                yield parts[2]  # id, lang, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sentences = list(iter_tatoeba(args.input))
    print(f"Tatoeba sentences: {len(sentences)}")

    total = 0
    with open(args.output, "w", encoding="utf-8") as out, \
         Pool(processes=args.workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_process_sentence, sentences, chunksize=500):
            if result:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                total += 1

    print(f"Done: {len(sentences)} sentences -> {total} pairs")


if __name__ == "__main__":
    main()
