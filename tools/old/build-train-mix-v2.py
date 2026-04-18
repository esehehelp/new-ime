"""phase3_v2 training mix builder (Python, 300M target).

既存 Rust `build-train-mix` は固定 pool 構成 (chunks/zenz/wiki/aozora/fineweb/hplt)
で、v2_bunsetsu + synth_numeric を扱えないため Python で構築。

入力プール (存在するもののみ採用、欠けは比率から除外して正規化):
  - sentence (旧 phase3 既存 JSONL)
  - v2_bunsetsu span=1  (filter on span_bunsetsu==1)
  - v2_bunsetsu span=2  (filter on span_bunsetsu==2)
  - synth_numeric

出力: 単一 JSONL (zstd 圧縮オプション) の weighted-by-ratio 結合。
各プールはサイズが比率に届かない場合 **繰り返し oversampling** される。
出力順序は シャッフルされた deterministic stream。

Usage:
    uv run python -m tools.build-train-mix-v2.build \
        --output datasets/phase3_v2/train.jsonl \
        --total 300000000 \
        --sentence-src datasets/phase3/train.jsonl \
        --bunsetsu-src datasets/v2_bunsetsu \
        --synth-src datasets/v2_bunsetsu/synth_numeric.jsonl \
        --ratio-sentence 0.50 \
        --ratio-bunsetsu2 0.25 \
        --ratio-bunsetsu1 0.05 \
        --ratio-synth 0.10 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def count_lines(path: Path, cap: int = 0) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
            if cap and n >= cap:
                return n
    return n


def collect_bunsetsu_paths(bunsetsu_dir: Path, span: int) -> list[Path]:
    """Find all v2_bunsetsu pool files. Filter by span_bunsetsu at read time."""
    out = []
    for p in sorted(bunsetsu_dir.glob("*.jsonl")):
        if p.name == "synth_numeric.jsonl":
            continue
        out.append(p)
    return out


def stream_pool(
    paths: list[Path],
    span_filter: int | None,
    rng: random.Random,
) -> iter:
    """Yield JSONL lines from a pool, with optional span_bunsetsu filter.

    On exhaustion, loops (oversampling). Shuffles within each epoch via
    deterministic reservoir shuffling — but to keep memory bounded for
    multi-GB files we only shuffle the file order + line buffer of 10K.
    """
    # Indefinite loop: on each "epoch", shuffle the file-order and walk.
    while True:
        order = list(paths)
        rng.shuffle(order)
        for p in order:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if span_filter is not None:
                        # Quick substring test before full parse (faster).
                        needle = f'"span_bunsetsu": {span_filter}'
                        if needle not in line:
                            continue
                    yield line


def stream_sentence_pool(path: Path) -> iter:
    """Loop over a single JSONL file indefinitely."""
    while True:
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield line


def stream_multi_sentence_pool(paths: list[Path], rng: random.Random) -> iter:
    """Loop over multiple JSONL files with shuffled file-order each epoch.

    Unlike stream_sentence_pool, this mixes multiple sources (e.g.
    existing phase3 sentence + v2 .clean.jsonl) into one pool with
    balanced epoch boundaries — each file contributes its rows in
    proportion to its size within an epoch.
    """
    while True:
        order = list(paths)
        rng.shuffle(order)
        for p in order:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    yield line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--total", type=int, default=300_000_000,
                    help="Target row count (oversampling to reach)")
    ap.add_argument("--sentence-src", action="append", default=[],
                    help="Sentence-level JSONL (repeatable). Multiple files "
                         "are treated as one round-robin pool.")
    ap.add_argument("--bunsetsu-src", default="datasets/v2_bunsetsu",
                    help="Dir containing v2_bunsetsu JSONLs (one per source)")
    ap.add_argument("--synth-src", action="append", default=[],
                    help="Synth pool JSONL (repeatable). Multi-source "
                         "treated as one combined pool.")
    ap.add_argument("--ratio-sentence", type=float, default=0.50)
    ap.add_argument("--ratio-bunsetsu2", type=float, default=0.25)
    ap.add_argument("--ratio-bunsetsu1", type=float, default=0.05)
    ap.add_argument("--ratio-synth", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-buffer", type=int, default=100_000,
                    help="Rows held in memory for shuffled output")
    ap.add_argument("--report-every", type=int, default=1_000_000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bunsetsu_dir = Path(args.bunsetsu_src)
    synth_paths = [Path(p) for p in (args.synth_src or [])]
    existing_synth_paths = [p for p in synth_paths if p.exists()]
    missing_synth_paths = [p for p in synth_paths if not p.exists()]
    for p in missing_synth_paths:
        print(f"[warn] synth src not found: {p} — dropping", flush=True)

    # Check presence; drop missing pools and renormalize.
    pool_specs = []
    sentence_paths = [Path(p) for p in (args.sentence_src or [])]
    existing_sentence_paths = [p for p in sentence_paths if p.exists()]
    missing_sentence_paths = [p for p in sentence_paths if not p.exists()]
    for p in missing_sentence_paths:
        print(f"[warn] sentence src not found: {p} — dropping", flush=True)
    if existing_sentence_paths:
        print(f"  sentence pool = {len(existing_sentence_paths)} files:",
              flush=True)
        for p in existing_sentence_paths:
            print(f"    {p}", flush=True)
        pool_specs.append(("sentence", args.ratio_sentence,
                           stream_multi_sentence_pool(existing_sentence_paths,
                                                      rng=rng)))

    bun_paths = collect_bunsetsu_paths(bunsetsu_dir, span=2)
    if bun_paths:
        pool_specs.append(("bunsetsu2", args.ratio_bunsetsu2,
                           stream_pool(bun_paths, span_filter=2, rng=rng)))
    if bun_paths:
        pool_specs.append(("bunsetsu1", args.ratio_bunsetsu1,
                           stream_pool(bun_paths, span_filter=1, rng=rng)))
    if existing_synth_paths:
        print(f"  synth pool = {len(existing_synth_paths)} files:",
              flush=True)
        for p in existing_synth_paths:
            print(f"    {p}", flush=True)
        pool_specs.append(("synth", args.ratio_synth,
                           stream_multi_sentence_pool(existing_synth_paths,
                                                      rng=rng)))

    if not pool_specs:
        raise SystemExit("No pools available — check --sentence-src, "
                         "--bunsetsu-src, --synth-src paths")

    total_ratio = sum(r for _, r, _ in pool_specs)
    norm_specs = [(n, r / total_ratio, s) for n, r, s in pool_specs]
    print("=== Pool mix (normalized) ===", flush=True)
    for n, r, _ in norm_specs:
        print(f"  {n:<12} ratio={r:.3f}  target_rows={int(args.total * r):,}",
              flush=True)

    # Weighted sampling in a streaming manner: for each output row, pick a
    # pool weighted by (target_remaining / rows_remaining). This gives a
    # "least-served-wins" fill that hits exact target proportions.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = [int(args.total * r) for _, r, _ in norm_specs]
    # Adjust rounding to hit --total exactly.
    diff = args.total - sum(targets)
    targets[0] += diff
    served = [0] * len(norm_specs)

    def pick_pool() -> int:
        # Weight by remaining deficit.
        weights = []
        for i in range(len(norm_specs)):
            deficit = targets[i] - served[i]
            weights.append(max(deficit, 0))
        total_w = sum(weights)
        if total_w <= 0:
            return -1
        x = rng.random() * total_w
        acc = 0
        for i, w in enumerate(weights):
            acc += w
            if x < acc:
                return i
        return len(weights) - 1

    import time
    t0 = time.perf_counter()
    written = 0
    buffer: list[str] = []
    buffer_cap = max(args.shuffle_buffer, 1)

    # We write shuffled via a buffer: fill buffer, shuffle, flush, repeat.
    with out_path.open("w", encoding="utf-8") as f:
        while written < args.total:
            idx = pick_pool()
            if idx < 0:
                break
            _, _, stream = norm_specs[idx]
            try:
                line = next(stream)
            except StopIteration:
                break
            if not line.endswith("\n"):
                line = line + "\n"
            buffer.append(line)
            served[idx] += 1

            if len(buffer) >= buffer_cap:
                rng.shuffle(buffer)
                f.writelines(buffer)
                written += len(buffer)
                buffer.clear()
                if written % args.report_every < buffer_cap:
                    elapsed = time.perf_counter() - t0
                    rate = written / elapsed
                    eta = (args.total - written) / rate if rate > 0 else 0
                    print(
                        f"  written={written:,}/{args.total:,} "
                        f"rate={rate/1000:.1f}k rows/s  "
                        f"eta={eta/60:.1f}min",
                        flush=True,
                    )

        # Flush remaining.
        if buffer:
            rng.shuffle(buffer)
            f.writelines(buffer)
            written += len(buffer)

    elapsed = time.perf_counter() - t0
    print("\n=== Summary ===", flush=True)
    print(f"  total written: {written:,} rows -> {out_path}", flush=True)
    print(f"  elapsed: {elapsed:.0f}s  ({written/elapsed/1000:.1f}k rows/s)",
          flush=True)
    print(f"  file size: {out_path.stat().st_size / (1024**3):.1f} GiB",
          flush=True)
    print("  served per pool:", flush=True)
    for (n, r, _), got, tgt in zip(norm_specs, served, targets):
        print(f"    {n:<12} served={got:,}  target={tgt:,}  "
              f"({got/tgt*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
