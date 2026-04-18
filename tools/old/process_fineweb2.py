"""Process FineWeb-2 jpn_Jpan parquet shards into clean sentence pairs.

Each parquet row is a document with fields ``text``, ``language``,
``language_script``, ``language_score`` (Common Crawl derived). We keep
records where the detected language is confidently Japanese, hand the
document body to the shared MeCab pipeline, and attach per-document
context before writing to JSONL with ``source="fineweb2_ja"``.

Usage:
    uv run python scripts/process_fineweb2.py \
        --input datasets/src/fineweb2_ja/data/jpn_Jpan/train/000_00000.parquet [...] \
        --output datasets/fineweb2_ja_clean.jsonl \
        [--lang-score 0.9] [--max-docs 0] \
        [--contamination-ref datasets/eval_v3/test.jsonl] \
        [--workers 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.data.mecab_pipeline import attach_context, text_to_pairs, worker_init
from scripts.process_zenz_subset import (
    build_ngram_set,
    contains_contaminated_ngram,
)


def iter_parquet_docs(
    paths: list[Path],
    min_language_score: float,
    max_docs: int,
) -> Iterator[str]:
    count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=2048,
            columns=["text", "language", "language_script", "language_score"],
        ):
            texts = batch.column("text").to_pylist()
            langs = batch.column("language").to_pylist()
            scripts = batch.column("language_script").to_pylist()
            scores = batch.column("language_score").to_pylist()
            for text, lang, script, score in zip(texts, langs, scripts, scores, strict=True):
                if not text:
                    continue
                if lang != "jpn" or script != "Jpan":
                    continue
                try:
                    if float(score) < min_language_score:
                        continue
                except (TypeError, ValueError):
                    continue
                yield text
                count += 1
                if max_docs and count >= max_docs:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="FineWeb-2 parquet shard(s)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--lang-score", type=float, default=0.9, help="Min language_score")
    parser.add_argument("--max-docs", type=int, default=0, help="0 = all; cap input docs")
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (cpu_count - 2)")
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument(
        "--contamination-ref",
        default="",
        help="Optional eval JSONL to 6-gram-check against",
    )
    parser.add_argument("--contamination-n", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    paths = [Path(p) for p in args.input]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing shard: {p}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contamination: set[str] = set()
    if args.contamination_ref:
        print(f"Loading contamination reference: {args.contamination_ref}")
        contamination = build_ngram_set(
            Path(args.contamination_ref), n=args.contamination_n
        )
        print(f"  {len(contamination):,} n-grams")

    num_workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"workers={num_workers} chunksize={args.chunksize}")

    stats = {"docs": 0, "pairs_emitted": 0, "pairs_contam": 0}

    with output_path.open("w", encoding="utf-8") as out, Pool(
        processes=num_workers, initializer=worker_init
    ) as pool:
        results = pool.imap_unordered(
            text_to_pairs,
            iter_parquet_docs(paths, args.lang_score, args.max_docs),
            chunksize=args.chunksize,
        )
        for pairs in results:
            stats["docs"] += 1
            attach_context(pairs)
            for pair in pairs:
                if contamination and contains_contaminated_ngram(
                    pair["surface"], contamination, n=args.contamination_n
                ):
                    stats["pairs_contam"] += 1
                    continue
                pair["source"] = "fineweb2_ja"
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                stats["pairs_emitted"] += 1
            if stats["docs"] % args.progress_every == 0:
                print(
                    f"[{stats['docs']:>10,} docs] "
                    f"pairs={stats['pairs_emitted']:,} "
                    f"contam_dropped={stats['pairs_contam']:,}"
                )

    print(
        f"done: docs={stats['docs']:,} pairs={stats['pairs_emitted']:,} "
        f"contam_dropped={stats['pairs_contam']:,}"
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
