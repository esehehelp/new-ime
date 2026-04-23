"""Process the zenz-v2.5 llm-jp subset into our training schema.

The zenz file is already in (input=カタカナ, output=kana-kanji,
left_context=kana-kanji) form. We just:

1. Rename fields: ``input`` → ``reading``, ``output`` → ``surface``,
   ``left_context`` → ``context``.
2. Convert the reading from katakana to hiragana so it matches the
   alphabet used by ``eval_v3`` and our ``SharedCharTokenizer`` fallback.
3. Apply light length filtering consistent with the existing wiki pipeline
   (5-100 characters on surface).
4. Drop empty rows and tag each row with ``source="zenz_llmjp"`` so pool
   audits can identify provenance.
5. Optionally 6-gram-dedup against an evaluation set to prevent test
   contamination (off by default to keep the first pass fast; enable with
   ``--contamination-ref``).

Usage:
    uv run python scripts/process_zenz_subset.py \
        --input datasets/src/zenz_llmjp/train_llm-jp-corpus-v3.jsonl \
        --output datasets/zenz_llmjp_clean.jsonl \
        [--contamination-ref datasets/eval/general/test.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import jaconv


MIN_SURFACE_LEN = 5
MAX_SURFACE_LEN = 100
MIN_READING_LEN = 2
MAX_READING_LEN = 200


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_ngram_set(path: Path, n: int = 6) -> set[str]:
    ngrams: set[str] = set()
    for row in iter_jsonl(path):
        surface = row.get("surface") or row.get("output") or ""
        if len(surface) < n:
            ngrams.add(surface)
            continue
        for i in range(len(surface) - n + 1):
            ngrams.add(surface[i : i + n])
    return ngrams


def contains_contaminated_ngram(surface: str, contamination: set[str], n: int = 6) -> bool:
    if len(surface) < n:
        return surface in contamination
    for i in range(len(surface) - n + 1):
        if surface[i : i + n] in contamination:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to zenz JSONL (llm-jp subset)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--contamination-ref",
        default="",
        help="Optional eval JSONL to 6-gram-check against",
    )
    parser.add_argument(
        "--contamination-n", type=int, default=6, help="N for contamination 6-gram check"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Progress print interval (rows)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = all; cap number of output rows (for smoke)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    contamination: set[str] = set()
    if args.contamination_ref:
        print(f"Loading contamination reference from {args.contamination_ref}")
        contamination = build_ngram_set(Path(args.contamination_ref), n=args.contamination_n)
        print(f"Contamination n-grams: {len(contamination):,}")

    stats = {
        "read": 0,
        "written": 0,
        "skipped_empty": 0,
        "skipped_len": 0,
        "skipped_contam": 0,
    }

    with output_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(input_path):
            stats["read"] += 1

            raw_reading = row.get("input", "")
            surface = row.get("output", "")
            context = row.get("left_context", "") or ""

            if not raw_reading or not surface:
                stats["skipped_empty"] += 1
                continue

            # Katakana → hiragana to match eval_v3 / SharedCharTokenizer.
            reading = jaconv.kata2hira(raw_reading)

            if not (MIN_SURFACE_LEN <= len(surface) <= MAX_SURFACE_LEN):
                stats["skipped_len"] += 1
                continue
            if not (MIN_READING_LEN <= len(reading) <= MAX_READING_LEN):
                stats["skipped_len"] += 1
                continue

            if contamination and contains_contaminated_ngram(
                surface, contamination, n=args.contamination_n
            ):
                stats["skipped_contam"] += 1
                continue

            out.write(
                json.dumps(
                    {
                        "reading": reading,
                        "surface": surface,
                        "context": context,
                        "source": "zenz_llmjp",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stats["written"] += 1

            if args.max_rows and stats["written"] >= args.max_rows:
                break
            if stats["read"] % args.progress_every == 0:
                print(
                    f"[{stats['read']:>12,}] written={stats['written']:,} "
                    f"skip_empty={stats['skipped_empty']:,} "
                    f"skip_len={stats['skipped_len']:,} "
                    f"skip_contam={stats['skipped_contam']:,}"
                )

    print(
        f"done: read={stats['read']:,} written={stats['written']:,} "
        f"skip_empty={stats['skipped_empty']:,} "
        f"skip_len={stats['skipped_len']:,} "
        f"skip_contam={stats['skipped_contam']:,}"
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
