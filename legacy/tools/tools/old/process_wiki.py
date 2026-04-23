"""Process Wikipedia dump into sentence pairs for kana-kanji conversion training.

Usage:
    uv run python scripts/process_wiki.py \
        --input datasets/src/jawiki-latest-pages-articles.xml.bz2 \
        --output datasets/wiki_sentences.jsonl \
        --workers 8

Architecture:
    Producer (main process) → bz2 decompress + XML parse only (lightweight)
    Worker pool (N procs)   → mwparserfromhell + MeCab (heavy CPU work)
    Writer (main process)   → collect results + write JSONL

The producer yields raw wikitext strings. All heavy processing
(wikitext stripping, sentence splitting, MeCab) runs in workers.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import mwxml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.data.mecab_pipeline import text_to_pairs, worker_init as _worker_init


# ---- Wikitext-specific stripping runs in the worker, then hands plain text
# off to the shared mecab_pipeline. ----


def _process_article(wikitext: str) -> list[dict]:
    """Strip wikitext → plain text → shared sentence/reading pipeline."""
    import mwparserfromhell as mwp

    try:
        parsed = mwp.parse(wikitext)
        text = parsed.strip_code(normalize=True, collapse=True)
    except Exception:
        return []

    text = re.sub(r"={2,}.*?={2,}", "", text)
    text = re.sub(r"\[\[|\]\]|\{\{|\}\}", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()
    if not text:
        return []

    return text_to_pairs(text)


# ---- Producer: runs in main process ----

def iter_raw_wikitext(dump_path: str, max_articles: int = 0):
    """Yield raw wikitext strings from bz2 XML dump. Lightweight — no parsing."""
    count = 0
    with bz2.open(dump_path, "rt", encoding="utf-8") as f:
        dump = mwxml.Dump.from_file(f)
        for page in dump:
            if page.namespace != 0:
                continue
            for revision in page:
                if revision.text:
                    yield revision.text
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
    parser.add_argument("--chunksize", type=int, default=50, help="Articles per imap chunk")
    args = parser.parse_args()

    num_workers = args.workers or max(1, os.cpu_count() - 2)
    print(f"Using {num_workers} worker processes, chunksize {args.chunksize}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    total_articles = 0

    with open(output_path, "w", encoding="utf-8") as out, \
         Pool(processes=num_workers, initializer=_worker_init) as pool:

        # imap_unordered: stream results as they complete, don't wait for order
        results_iter = pool.imap_unordered(
            _process_article,
            iter_raw_wikitext(args.input, args.max_articles),
            chunksize=args.chunksize,
        )

        for article_pairs in results_iter:
            total_articles += 1

            # Add context (within-article sequential context)
            prev_surface = ""
            for pair in article_pairs:
                pair["context"] = prev_surface[-40:] if prev_surface else ""
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += 1
                prev_surface = pair["surface"]

            if total_articles % 1000 == 0:
                print(f"  {total_articles} articles, {total_pairs} pairs...", flush=True)

    print(f"\nDone: {total_articles} articles -> {total_pairs} sentence pairs")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
