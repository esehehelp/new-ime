"""Process Wikipedia dump into sentence pairs for kana-kanji conversion training.

Usage:
    uv run python scripts/process_wiki.py \
        --input datasets/src/jawiki-latest-pages-articles.xml.bz2 \
        --output datasets/wiki_sentences.jsonl \
        --workers 8

Architecture:
    Producer (1 thread)  → XML bz2 read + mwparserfromhell strip
    Worker pool (N procs) → MeCab reading assignment + filtering
    Writer (1 thread)    → JSONL output

MeCab is CPU-bound and holds the GIL via C extension, so multiprocessing
(not threading) is required for actual parallelism.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
from multiprocessing import Pool, Queue, Process
from pathlib import Path
from queue import Empty

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
    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = re.sub(r"\[\[|\]\]|\{\{|\}\}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


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


def get_reading(tagger: MeCab.Tagger, text: str) -> str | None:
    """Get full hiragana reading. Returns None if unreliable."""
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
        if len(features) >= 8 and features[7] != "*":
            reading = features[7]
        elif len(features) >= 7 and features[6] != "*":
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


def process_article_batch(sentences_batch: list[list[str]]) -> list[list[dict]]:
    """Worker function: process a batch of articles (list of sentence lists).

    Each worker creates its own MeCab tagger (not shared across processes).
    Returns list of article results, each a list of {reading, surface} dicts.
    """
    tagger = MeCab.Tagger()
    tagger.parse("")

    results = []
    for sentences in sentences_batch:
        article_pairs = []
        for sent in sentences:
            if not is_valid_sentence(sent):
                continue
            reading = get_reading(tagger, sent)
            if reading is None:
                continue
            article_pairs.append({"reading": reading, "surface": sent})
        results.append(article_pairs)
    return results


def iter_wiki_articles(dump_path: str, max_articles: int = 0):
    """Iterate over Wikipedia articles, yielding (title, sentence_list) tuples."""
    count = 0
    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            if page.namespace != 0:
                continue
            for revision in page:
                if revision.text:
                    plain = wikitext_to_plain(revision.text)
                    if plain:
                        sentences = split_sentences(plain)
                        if sentences:
                            yield sentences
                            count += 1
                            if max_articles and count >= max_articles:
                                return
                break


def main():
    parser = argparse.ArgumentParser(description="Process Wikipedia dump")
    parser.add_argument("--input", required=True, help="Path to jawiki XML bz2 dump")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-articles", type=int, default=0, help="Limit articles (0=all)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")
    parser.add_argument("--batch-size", type=int, default=500, help="Articles per worker batch")
    args = parser.parse_args()

    num_workers = args.workers or max(1, os.cpu_count() - 2)
    print(f"Using {num_workers} worker processes, batch size {args.batch_size}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    total_articles = 0
    batch: list[list[str]] = []

    with open(output_path, "w", encoding="utf-8") as out, \
         Pool(processes=num_workers) as pool:

        pending = []

        for sentences in iter_wiki_articles(args.input, args.max_articles):
            batch.append(sentences)
            total_articles += 1

            if len(batch) >= args.batch_size:
                # Submit batch to pool
                pending.append(pool.apply_async(process_article_batch, (batch,)))
                batch = []

            # Drain completed results
            still_pending = []
            for future in pending:
                if future.ready():
                    for article_pairs in future.get():
                        prev_surface = ""
                        for pair in article_pairs:
                            context = prev_surface[-40:] if prev_surface else ""
                            pair["context"] = context
                            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                            total_pairs += 1
                            prev_surface = pair["surface"]
                else:
                    still_pending.append(future)
            pending = still_pending

            if total_articles % 10000 == 0:
                print(f"  {total_articles} articles, {total_pairs} pairs, "
                      f"{len(pending)} batches pending...")

        # Submit remaining batch
        if batch:
            pending.append(pool.apply_async(process_article_batch, (batch,)))

        # Drain all remaining
        for future in pending:
            for article_pairs in future.get():
                prev_surface = ""
                for pair in article_pairs:
                    context = prev_surface[-40:] if prev_surface else ""
                    pair["context"] = context
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_pairs += 1
                    prev_surface = pair["surface"]

    print(f"\nDone: {total_articles} articles -> {total_pairs} sentence pairs")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
