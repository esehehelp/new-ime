"""Pre-process clean JSONL into MeCab-analyzed TSV for Rust chunk generator.

Each input sentence becomes a block of morpheme lines, separated by blank lines.
Format per morpheme: surface\treading\tpos

Streams line by line — no memory accumulation.

Usage:
    uv run python scripts/mecab_to_tsv.py \
        --input datasets/wiki_clean_v3.jsonl \
        --output datasets/morphemes.tsv \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import jaconv
import MeCab


def _worker_init():
    global _tagger
    _tagger = MeCab.Tagger()
    _tagger.parse("")


def _analyze(surface_text: str) -> str | None:
    """Analyze a sentence and return TSV block (morphemes separated by \\n, ends with blank line)."""
    global _tagger

    node = _tagger.parseToNode(surface_text)
    lines = []
    while node:
        if node.surface:
            features = node.feature.split(",")
            pos = features[0]

            reading = None
            if len(features) >= 18 and features[17] != "*":
                reading = features[17]
            elif len(features) >= 7 and features[6] != "*":
                reading = features[6]

            if reading:
                reading_hira = jaconv.kata2hira(reading)
            elif all("\u3040" <= c <= "\u309f" or not c.strip() for c in node.surface):
                reading_hira = node.surface
            elif all(c in "、。！？「」『』（）・…―─　\n\r\t " for c in node.surface):
                reading_hira = node.surface
            else:
                reading_hira = ""

            lines.append(f"{node.surface}\t{reading_hira}\t{pos}")
        node = node.next

    if not lines:
        return None
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    num_workers = args.workers or max(1, os.cpu_count() - 2)

    # Stream sentences from input files
    def iter_sentences():
        for path in args.input:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)["surface"]

    total = 0
    with open(args.output, "w", encoding="utf-8") as out, \
         Pool(processes=num_workers, initializer=_worker_init) as pool:

        for block in pool.imap_unordered(_analyze, iter_sentences(), chunksize=1000):
            if block:
                out.write(block)
                out.write("\n")  # blank line separator
                total += 1
                if total % 500000 == 0:
                    print(f"  {total:,}...", flush=True)

    print(f"Done: {total:,} sentences → {args.output}")


if __name__ == "__main__":
    main()
