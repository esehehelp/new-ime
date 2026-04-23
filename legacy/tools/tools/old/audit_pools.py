"""Audit candidate data pools before Phase 3 training.

This script is intentionally lightweight and file-based. It computes:
- line counts
- average character lengths
- kanji-containing sample ratio
- exact lexical overlap against evaluation sets
- exact 6-gram overlap against evaluation sets

It is designed to run before the full pool-merging pipeline exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


KANJI_START = 0x4E00
KANJI_END = 0x9FFF

# Source tag → license mapping. Populated as new corpora land.
# "unknown" marks a legacy pool that predates the source-tag convention.
SOURCE_LICENSE: dict[str, str] = {
    "zenz_llmjp": "ODC-BY 1.0 (Common Crawl subset of llm-jp-corpus-v3)",
    "hplt3_ja": "CC0-1.0 (Common Crawl terms of use apply)",
    "fineweb2_ja": "ODC-BY 1.0 (Common Crawl terms of use apply)",
    "wiki": "CC-BY-SA 3.0 — derivative of Wikipedia",
    "aozora": "Public Domain",
    "livedoor": "CC BY-ND 2.1 JP — evaluation/exploration only",
    "tatoeba": "CC-BY 2.0 FR",
    "unknown": "unknown (no source tag in pool)",
}


@dataclass
class PoolStats:
    path: str
    lines: int = 0
    avg_reading_chars: float = 0.0
    avg_surface_chars: float = 0.0
    kanji_surface_ratio: float = 0.0
    lexical_overlap: int = 0
    sixgram_overlap: int = 0
    source_histogram: dict[str, int] = field(default_factory=dict)
    has_attribution_file: bool | None = None


def contains_kanji(text: str) -> bool:
    return any(KANJI_START <= ord(ch) <= KANJI_END for ch in text)


def read_jsonl(path: str, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def normalize_sample(sample: dict) -> tuple[str, str, str]:
    return (
        sample.get("reading", ""),
        sample.get("surface", ""),
        sample.get("context", ""),
    )


def surface_sixgrams(text: str) -> set[str]:
    if len(text) < 6:
        return {text} if text else set()
    return {text[i : i + 6] for i in range(len(text) - 5)}


def build_eval_indices(eval_paths: list[str], limit_per_file: int = 0) -> tuple[set[tuple[str, str, str]], set[str]]:
    lexical: set[tuple[str, str, str]] = set()
    sixgrams: set[str] = set()
    for path in eval_paths:
        for row in read_jsonl(path, limit=limit_per_file):
            norm = normalize_sample(row)
            lexical.add(norm)
            sixgrams.update(surface_sixgrams(norm[1]))
    return lexical, sixgrams


def audit_pool(path: str, eval_lexical: set[tuple[str, str, str]], eval_sixgrams: set[str], limit: int = 0) -> PoolStats:
    rows = read_jsonl(path, limit=limit)
    stats = PoolStats(path=path)
    if not rows:
        return stats

    reading_chars = 0
    surface_chars = 0
    kanji_count = 0
    lexical_overlap = 0
    sixgram_overlap = 0
    source_counter: Counter[str] = Counter()

    for row in rows:
        reading, surface, context = normalize_sample(row)
        stats.lines += 1
        reading_chars += len(reading)
        surface_chars += len(surface)
        kanji_count += int(contains_kanji(surface))
        lexical_overlap += int((reading, surface, context) in eval_lexical)
        if surface_sixgrams(surface) & eval_sixgrams:
            sixgram_overlap += 1
        source_counter[row.get("source") or "unknown"] += 1

    stats.avg_reading_chars = reading_chars / stats.lines
    stats.avg_surface_chars = surface_chars / stats.lines
    stats.kanji_surface_ratio = kanji_count / stats.lines
    stats.lexical_overlap = lexical_overlap
    stats.sixgram_overlap = sixgram_overlap
    stats.source_histogram = dict(source_counter)

    # ATTRIBUTION.md co-located with the source directory implies the
    # redistribute-with-attribution requirement is acknowledged upstream.
    # We look in both the pool's own dir and the corresponding src/<source>/ dir.
    attribution_candidates: list[Path] = [Path(path).parent / "ATTRIBUTION.md"]
    for source in source_counter:
        if source == "unknown":
            continue
        attribution_candidates.append(Path("datasets/src") / source / "ATTRIBUTION.md")
    stats.has_attribution_file = any(c.exists() for c in attribution_candidates)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 3 candidate data pools")
    parser.add_argument("--pools", nargs="+", required=True, help="JSONL pool files to audit")
    parser.add_argument(
        "--eval-sets",
        nargs="+",
        default=[
            "datasets/eval/general/dev.jsonl",
            "datasets/eval/general/test.jsonl",
            "datasets/gold_1k.jsonl",
        ],
        help="Evaluation JSONL files used for contamination checks",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional per-pool row cap for fast audits")
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=0,
        help="Optional per-eval-set row cap for fast audits",
    )
    parser.add_argument("--json", default="", help="Optional JSON output path")
    args = parser.parse_args()

    eval_lexical, eval_sixgrams = build_eval_indices(args.eval_sets, limit_per_file=args.eval_limit)

    report: dict = {"pools": [], "eval_sets": args.eval_sets}
    for pool in args.pools:
        stats = audit_pool(pool, eval_lexical, eval_sixgrams, limit=args.limit)
        source_licenses = sorted(
            {
                SOURCE_LICENSE.get(src, "(not in SOURCE_LICENSE table)")
                for src in stats.source_histogram
            }
        )
        report["pools"].append(
            {
                "path": stats.path,
                "lines": stats.lines,
                "avg_reading_chars": round(stats.avg_reading_chars, 2),
                "avg_surface_chars": round(stats.avg_surface_chars, 2),
                "kanji_surface_ratio": round(stats.kanji_surface_ratio, 4),
                "lexical_overlap": stats.lexical_overlap,
                "sixgram_overlap": stats.sixgram_overlap,
                "source_histogram": stats.source_histogram,
                "source_licenses": source_licenses,
                "has_attribution_file": stats.has_attribution_file,
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
