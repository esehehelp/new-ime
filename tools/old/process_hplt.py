"""Process HPLT jpn_Jpan shards into clean kana-kanji sentence pairs.

Input is one or more ``.jsonl.zst`` shards from ``download_hplt3_ja.py``.
Each record has fields including ``text`` (document-level plain text),
``lang`` (ordered language-ID list), and ``prob`` (detection probabilities).

We keep only records that are confidently Japanese, hand each document's
text to the shared MeCab worker pipeline, then attach per-document context
before writing to JSONL. All output rows carry ``source="hplt3_ja"`` for
downstream pool auditing.

Usage:
    uv run python scripts/process_hplt.py \
        --input datasets/src/hplt3_ja/10_1.jsonl.zst [...more shards...] \
        --output datasets/hplt3_ja_clean.jsonl \
        [--lang-prob 0.9] [--max-docs 0] \
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

import zstandard as zstd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.src.data.mecab_pipeline import attach_context, text_to_pairs, worker_init

# Contamination check shares logic with zenz processor.
from scripts.process_zenz_subset import (
    build_ngram_set,
    contains_contaminated_ngram,
)


def iter_jsonl_zst(path: Path) -> Iterator[dict]:
    """Stream-parse a zstd-compressed JSONL file."""
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as raw:
        with dctx.stream_reader(raw) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    if buf.strip():
                        try:
                            yield json.loads(buf.decode("utf-8"))
                        except json.JSONDecodeError:
                            pass
                    return
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[:nl], buf[nl + 1 :]
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue


def accept_record(record: dict, min_lang_prob: float) -> bool:
    langs = record.get("lang") or []
    probs = record.get("prob") or []
    if not langs or not probs:
        return False
    if langs[0] != "jpn_Jpan":
        return False
    try:
        if float(probs[0]) < min_lang_prob:
            return False
    except (TypeError, ValueError):
        return False
    return bool(record.get("text"))


def iter_doc_texts(
    paths: list[Path],
    min_lang_prob: float,
    max_docs: int,
) -> Iterator[str]:
    count = 0
    for path in paths:
        for record in iter_jsonl_zst(path):
            if not accept_record(record, min_lang_prob):
                continue
            yield record["text"]
            count += 1
            if max_docs and count >= max_docs:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Shard path(s) (.jsonl.zst)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--lang-prob",
        type=float,
        default=0.9,
        help="Minimum jpn_Jpan language-ID probability to accept (0..1)",
    )
    parser.add_argument("--max-docs", type=int, default=0, help="0 = all; cap input docs")
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (cpu_count - 2)")
    parser.add_argument("--chunksize", type=int, default=16, help="imap chunk size")
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
            iter_doc_texts(paths, args.lang_prob, args.max_docs),
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
                pair["source"] = "hplt3_ja"
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
