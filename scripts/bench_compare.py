"""Cross-model bench diff.

Reads N verbose bench JSONL files (one per model) and reports:
  - Per-category EM1 side by side
  - Items where model A is correct but model B is wrong (and the reverse)
  - The pred / ref triple for each disagreement

Verbose JSONL is the per-item log written by `ime-bench -v`; one record
per item with keys: index, category, reading, context, references,
candidates, em1, em5, em1_nfkc, em5_nfkc, char_acc_top1, latency_ms.

Usage:
    uv run python tools/bench_compare.py \\
        results/bench/suiko-v1-small-greedy/probe_v3__greedy.full.jsonl \\
        results/bench/zenz-v3.1-small/probe_v3__beam.full.jsonl \\
        --names suiko-v1 zenz-v3.1 \\
        --bench probe_v3 \\
        --diff-limit 30

If --names is omitted, derives names from parent dir names.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_records(path: Path) -> dict[str, dict]:
    """Read verbose JSONL keyed by item index."""
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("index") or f"idx_{rec.get('i', len(out))}"
            out[key] = rec
    return out


def per_category(records: dict[str, dict], metric: str = "em1_nfkc") -> dict[str, tuple[int, int]]:
    """category -> (sum_metric, count). Falls back to em1 if em1_nfkc absent."""
    cats: dict[str, list[int]] = defaultdict(list)
    for rec in records.values():
        cat = rec.get("category", "?") or "?"
        v = rec.get(metric)
        if v is None:
            v = rec.get("em1", 0)
        cats[cat].append(int(bool(v)))
    return {c: (sum(vs), len(vs)) for c, vs in sorted(cats.items())}


def overall(records: dict[str, dict], metric: str = "em1_nfkc") -> tuple[int, int]:
    correct = 0
    total = 0
    for rec in records.values():
        v = rec.get(metric)
        if v is None:
            v = rec.get("em1", 0)
        correct += int(bool(v))
        total += 1
    return correct, total


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "  -  "
    return f"{num/den:.4f}"


def print_side_by_side(names: list[str], cat_tables: list[dict[str, tuple[int, int]]]) -> None:
    cats = sorted({c for t in cat_tables for c in t})
    width_n = max(max(len(n) for n in names), 8)
    print(f"\n=== per-category EM1 (em1_nfkc, fallback em1) ===")
    header = f"  {'category':<12}" + "".join(f"{n:>{width_n+2}s}" for n in names)
    print(header)
    for cat in cats:
        row = f"  {cat:<12}"
        for t in cat_tables:
            n_correct, n_total = t.get(cat, (0, 0))
            row += f"{fmt_pct(n_correct, n_total):>{width_n+2}s}"
        print(row)
    overall_row = f"  {'OVERALL':<12}"
    for t in cat_tables:
        n_correct = sum(c for c, _ in t.values())
        n_total = sum(n for _, n in t.values())
        overall_row += f"{fmt_pct(n_correct, n_total):>{width_n+2}s}"
    print(overall_row)


def diff_pairs(
    name_a: str,
    name_b: str,
    rec_a: dict[str, dict],
    rec_b: dict[str, dict],
    metric: str = "em1_nfkc",
) -> tuple[list[dict], list[dict]]:
    """(a_wins, b_wins) — items where one is correct and the other wrong."""
    a_wins: list[dict] = []
    b_wins: list[dict] = []
    for key, ra in rec_a.items():
        rb = rec_b.get(key)
        if rb is None:
            continue
        a_ok = bool(ra.get(metric, ra.get("em1", 0)))
        b_ok = bool(rb.get(metric, rb.get("em1", 0)))
        if a_ok and not b_ok:
            a_wins.append((key, ra, rb))
        elif b_ok and not a_ok:
            b_wins.append((key, ra, rb))
    return a_wins, b_wins


def print_diff(
    label: str,
    diff: list,
    name_a: str,
    name_b: str,
    limit: int,
) -> None:
    by_cat: dict[str, int] = defaultdict(int)
    for _, ra, _ in diff:
        by_cat[ra.get("category", "?") or "?"] += 1
    print(f"\n=== {label} (count: {len(diff)}) ===")
    print(f"  by category: " + " ".join(f"{c}={n}" for c, n in sorted(by_cat.items())))
    if limit <= 0:
        return
    print(f"  top {min(limit, len(diff))} examples:")
    for key, ra, rb in diff[:limit]:
        cat = ra.get("category", "?")
        reading = ra.get("reading", "")
        refs = ra.get("references", [])
        ref0 = refs[0] if refs else ""
        cands_a = ra.get("candidates", [])
        cands_b = rb.get("candidates", [])
        a_top = cands_a[0] if cands_a else "<none>"
        b_top = cands_b[0] if cands_b else "<none>"
        print(
            f"  [{cat:<10}] {key}: reading={reading!r}\n"
            f"             ref     ={ref0!r}\n"
            f"             {name_a}={a_top!r}\n"
            f"             {name_b}={b_top!r}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="verbose .full.jsonl paths")
    p.add_argument("--names", nargs="*", default=None, help="model name labels")
    p.add_argument("--bench", default="", help="optional label echoed in header")
    p.add_argument("--metric", default="em1_nfkc", choices=["em1", "em1_nfkc"])
    p.add_argument(
        "--diff-limit", type=int, default=20,
        help="how many disagreement examples to print per direction (0 = none)",
    )
    args = p.parse_args()

    paths = [Path(s) for s in args.inputs]
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    if args.names is not None and len(args.names) != len(paths):
        print(
            f"--names count ({len(args.names)}) must match inputs ({len(paths)})",
            file=sys.stderr,
        )
        return 2
    if args.names is None:
        names = [p.parent.name for p in paths]
    else:
        names = list(args.names)

    records = [load_records(p) for p in paths]
    cat_tables = [per_category(r, args.metric) for r in records]

    if args.bench:
        print(f"=== bench: {args.bench} ===")
    for name, path, recs in zip(names, paths, records):
        c, n = overall(recs, args.metric)
        print(f"  {name} ({path.name}): n={n} {args.metric}={fmt_pct(c, n)}")

    print_side_by_side(names, cat_tables)

    if len(names) == 2:
        a_wins, b_wins = diff_pairs(names[0], names[1], records[0], records[1], args.metric)
        print_diff(
            f"{names[0]} correct, {names[1]} wrong",
            a_wins, names[0], names[1], args.diff_limit,
        )
        print_diff(
            f"{names[1]} correct, {names[0]} wrong",
            b_wins, names[0], names[1], args.diff_limit,
        )
    elif len(names) > 2:
        print("\n(pairwise diff skipped: only computed for exactly 2 inputs)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
