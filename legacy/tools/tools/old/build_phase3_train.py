"""Build the Phase 3 CTC-NAT train.jsonl from pool-mix proportions.

Target: a large (default 200M-row) JSONL biased toward short-sample pairs
so the 100M-class CTC-NAT student learns exact-match on short phrases
first.

Default mix (overridable via --ratio-* flags):
- 50%  chunks_main     — chunks_v3_100m.jsonl, surface length >= --super-cutoff
- 10%  super_short     — chunks_v3_100m.jsonl, surface length <  --super-cutoff
                         (<8 chars by default, captures most 四字熟語 pairs)
- 15%  zenz            — zenz_llmjp_clean.jsonl (ODC-BY)
- 10%  wiki+aozora     — wiki_clean_v3.jsonl + aozora_clean.jsonl
- 10%  fineweb2        — fineweb2_ja_clean.jsonl (ODC-By)
-  5%  hplt            — hplt3_ja_clean.jsonl   (CC0)

Contamination filtering: pools that were *not* filtered at processing time
(chunks / wiki / aozora) are 6-gram-checked against the contamination
reference here. Zenz / fineweb2 / hplt were already filtered upstream.

Oversampling: pools with fewer unique eligible rows than their target count
cycle through the source file repeatedly. Rows are emitted interleaved
across pools so the resulting file is already partially mixed — train-time
DataLoader shuffling completes the mix.

Note: This is the CTC-NAT student training set. The AR teacher used a
different vocabulary-building path and is not affected.

Usage (default 200M-row build):
    uv run python scripts/build_phase3_train.py \
        --output datasets/phase3/train.jsonl \
        --contamination-ref datasets/eval/general/test.jsonl datasets/eval/general/dev.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, IO, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.process_zenz_subset import (
    build_ngram_set,
    contains_contaminated_ngram,
)


@dataclass
class PoolSpec:
    name: str
    paths: list[Path]
    source_tag: str
    ratio: float
    length_predicate: Callable[[int], bool] | None = None
    filter_contamination: bool = False
    served: int = field(default=0, init=False)
    budget: int = field(default=0, init=False)


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


def open_compressed_writer(path: Path, mode: str, level: int) -> IO[str]:
    """Open path for text writing with optional compression.

    mode ∈ {"none", "zstd", "xz", "gzip"}. The caller picks the destination
    suffix via --output; this just picks the right writer.
    """
    if mode == "none":
        return path.open("w", encoding="utf-8")
    if mode == "xz":
        preset = max(0, min(level, 9))
        return lzma.open(path, "wt", encoding="utf-8", preset=preset)
    if mode == "gzip":
        compresslevel = max(1, min(level, 9))
        return gzip.open(path, "wt", encoding="utf-8", compresslevel=compresslevel)
    if mode == "zstd":
        import zstandard as zstd  # declared as project dep

        raw = path.open("wb")
        cctx = zstd.ZstdCompressor(level=max(1, min(level, 22)), threads=-1)
        binary_stream = cctx.stream_writer(raw)
        import io

        return io.TextIOWrapper(binary_stream, encoding="utf-8", write_through=True)
    raise ValueError(f"unknown compress mode: {mode}")


def pool_rows(
    pool: PoolSpec,
    contamination: set[str],
    contam_n: int,
    max_surface_len: int,
    min_surface_len: int,
) -> Iterator[dict]:
    """Yield rows from a pool, applying filters + source tag. Cycles files
    to allow oversampling; each cycle is a complete pass over all paths.
    """
    while True:
        emitted_this_cycle = 0
        for path in pool.paths:
            for row in iter_jsonl(path):
                surface = row.get("surface") or ""
                reading = row.get("reading") or ""
                if not surface or not reading:
                    continue
                if len(surface) < min_surface_len or len(surface) > max_surface_len:
                    continue
                if pool.length_predicate is not None and not pool.length_predicate(len(surface)):
                    continue
                if (
                    pool.filter_contamination
                    and contamination
                    and contains_contaminated_ngram(surface, contamination, n=contam_n)
                ):
                    continue
                context = row.get("context") or ""
                yield {
                    "reading": reading,
                    "surface": surface,
                    "context": context,
                    "source": pool.source_tag,
                }
                emitted_this_cycle += 1
        # A complete cycle emitted nothing — would loop forever. Give up.
        if emitted_this_cycle == 0:
            return


def build_pool_specs(args: argparse.Namespace) -> list[PoolSpec]:
    total = args.ratio_chunks + args.ratio_super + args.ratio_zenz + args.ratio_wiki + args.ratio_fineweb2 + args.ratio_hplt
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(
            f"Ratios must sum to 1.0, got {total:.4f}. "
            "Check --ratio-chunks / --ratio-super / --ratio-zenz / "
            "--ratio-wiki / --ratio-fineweb2 / --ratio-hplt."
        )

    super_cutoff = args.super_cutoff  # surface < super_cutoff → super-short
    specs = [
        PoolSpec(
            name="chunks_main",
            paths=[Path(args.chunks_path)],
            source_tag="chunks_main",
            ratio=args.ratio_chunks,
            length_predicate=lambda L, c=super_cutoff: L >= c,
            filter_contamination=True,
        ),
        PoolSpec(
            name="super_short",
            paths=[Path(args.chunks_path)],
            source_tag="chunks_super",
            ratio=args.ratio_super,
            length_predicate=lambda L, c=super_cutoff: L < c,
            filter_contamination=True,
        ),
        PoolSpec(
            name="zenz",
            paths=[Path(args.zenz_path)],
            source_tag="zenz_llmjp",
            ratio=args.ratio_zenz,
            filter_contamination=False,  # already filtered by processor
        ),
        PoolSpec(
            name="wiki_aozora",
            paths=[Path(args.wiki_path), Path(args.aozora_path)],
            source_tag="wiki_aozora",
            ratio=args.ratio_wiki,
            filter_contamination=True,  # legacy pools were not filtered
        ),
        PoolSpec(
            name="fineweb2",
            paths=[Path(args.fineweb2_path)],
            source_tag="fineweb2_ja",
            ratio=args.ratio_fineweb2,
            filter_contamination=False,
        ),
        PoolSpec(
            name="hplt",
            paths=[Path(args.hplt_path)],
            source_tag="hplt3_ja",
            ratio=args.ratio_hplt,
            filter_contamination=False,
        ),
    ]
    for spec in specs:
        spec.budget = int(round(args.total * spec.ratio))
    return specs


def weighted_least_served(specs: list[PoolSpec]) -> PoolSpec | None:
    """Pick the pool currently most under its budget. Returns None if all
    pools are at or past budget."""
    candidates = [s for s in specs if s.served < s.budget]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.served / s.budget if s.budget else 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--total",
        type=int,
        default=200_000_000,
        help="Total target rows (default 200M)",
    )
    parser.add_argument(
        "--super-cutoff",
        type=int,
        default=8,
        help="Surface length cutoff separating super-short (<) from chunks (>=)",
    )
    parser.add_argument(
        "--min-surface-len",
        type=int,
        default=1,
        help="Reject rows whose surface is shorter than this (absolute floor)",
    )
    parser.add_argument(
        "--max-surface-len",
        type=int,
        default=128,
        help="Reject rows whose surface exceeds this",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Pool paths (override to point to alternative files).
    parser.add_argument("--chunks-path", default="datasets/chunks_v3_100m.jsonl")
    parser.add_argument("--zenz-path", default="datasets/zenz_llmjp_clean.jsonl")
    parser.add_argument("--wiki-path", default="datasets/wiki_clean_v3.jsonl")
    parser.add_argument("--aozora-path", default="datasets/aozora_clean.jsonl")
    parser.add_argument("--fineweb2-path", default="datasets/fineweb2_ja_clean.jsonl")
    parser.add_argument("--hplt-path", default="datasets/hplt3_ja_clean.jsonl")

    # Ratios (must sum to 1.0).
    parser.add_argument("--ratio-chunks", type=float, default=0.50)
    parser.add_argument("--ratio-super", type=float, default=0.10)
    parser.add_argument("--ratio-zenz", type=float, default=0.15)
    parser.add_argument("--ratio-wiki", type=float, default=0.10)
    parser.add_argument("--ratio-fineweb2", type=float, default=0.10)
    parser.add_argument("--ratio-hplt", type=float, default=0.05)

    parser.add_argument(
        "--contamination-ref",
        nargs="+",
        default=["datasets/eval/general/test.jsonl"],
        help="Evaluation JSONL(s) to 6-gram-filter un-filtered pools against",
    )
    parser.add_argument("--contamination-n", type=int, default=6)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000_000,
        help="Emit a progress line every N rows written",
    )
    parser.add_argument(
        "--compress",
        choices=["none", "zstd", "xz", "gzip"],
        default="none",
        help=(
            "Compress the output file. Appends .zst/.xz/.gz to --output. "
            "For multi-GB transfers, zstd -19 is recommended (best size/speed "
            "tradeoff for JSONL). See memory feedback_dataset_distribution.md."
        ),
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=19,
        help="Compression level (zstd: 1-22, xz: 0-9, gzip: 1-9)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    output_path = Path(args.output)
    if args.compress != "none":
        suffix = {"zstd": ".zst", "xz": ".xz", "gzip": ".gz"}[args.compress]
        if not str(output_path).endswith(suffix):
            output_path = Path(str(output_path) + suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build contamination ngram set from all reference files.
    contamination: set[str] = set()
    for ref in args.contamination_ref:
        print(f"Loading contamination reference: {ref}")
        contamination |= build_ngram_set(Path(ref), n=args.contamination_n)
    print(f"  combined n-grams: {len(contamination):,}")

    specs = build_pool_specs(args)
    print(f"Target total: {args.total:,} rows → {output_path}")
    for spec in specs:
        print(
            f"  {spec.name:<12}  ratio={spec.ratio:.2f}  budget={spec.budget:>12,}  "
            f"src={spec.source_tag}  filter_contam={spec.filter_contamination}  "
            f"paths={[str(p) for p in spec.paths]}"
        )
    budget_total = sum(s.budget for s in specs)
    if abs(budget_total - args.total) > len(specs):
        print(
            f"WARNING: rounded budgets sum to {budget_total:,}, off from "
            f"--total {args.total:,} by {budget_total - args.total}"
        )

    # Per-pool iterators (created lazily so disk open happens once).
    iterators: dict[str, Iterator[dict]] = {
        spec.name: pool_rows(
            spec,
            contamination=contamination,
            contam_n=args.contamination_n,
            max_surface_len=args.max_surface_len,
            min_surface_len=args.min_surface_len,
        )
        for spec in specs
    }

    written = 0
    exhausted: set[str] = set()
    with open_compressed_writer(output_path, args.compress, args.compress_level) as out:
        while True:
            pool = weighted_least_served(
                [s for s in specs if s.name not in exhausted]
            )
            if pool is None:
                break
            try:
                row = next(iterators[pool.name])
            except StopIteration:
                # Pool source exhausted before reaching budget — mark and skip.
                print(
                    f"  [{written:>12,}] pool {pool.name} exhausted at "
                    f"{pool.served:,}/{pool.budget:,} — continuing without it"
                )
                exhausted.add(pool.name)
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            pool.served += 1
            written += 1
            if written % args.progress_every == 0:
                shares = " ".join(
                    f"{s.name}={s.served / max(s.budget,1):.2f}" for s in specs
                )
                print(f"  [{written:>12,}] served_fraction: {shares}")

    print(f"\ndone: {written:,} rows → {output_path}")
    for spec in specs:
        print(
            f"  {spec.name:<12}  served={spec.served:,}  "
            f"budget={spec.budget:,}  "
            f"{'EXHAUSTED' if spec.name in exhausted else 'OK'}"
        )
    total_bytes = output_path.stat().st_size
    print(f"  size: {total_bytes / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
