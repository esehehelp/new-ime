"""Apply the current rule set to a round's samples.jsonl and split rows
into survivors + rejected (with per-row reason).

Rules are additive across iterations: each round reads `rules_vN.py`
where N = round_number - 1 accumulated so far. For round 2, apply
rules_v1; for round 3, apply rules_v2; etc.

Usage:
    python scripts/pool_filter.py --round 2 --rules-version 1 [--pool bunsetsu-wiktionary]

Outputs under <pool>/round_N/:
    survivors.jsonl
    rejected.jsonl          (each line: {reason, row})
    filter_report.tsv       (reason -> count)
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "datasets" / "audits" / "pool-qa"
RULES_DIR = REPO / "scripts" / "pool_rules"


def load_rules(version: int):
    path = RULES_DIR / f"rules_v{version}.py"
    if not path.exists():
        raise SystemExit(f"rules file not found: {path}")
    spec = importlib.util.spec_from_file_location(f"rules_v{version}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "RULES"):
        raise SystemExit(f"{path} does not export RULES")
    return module.RULES  # type: ignore[attr-defined]


def apply_rules(row: dict, rules) -> str | None:
    for name, fn in rules:
        reason = fn(row)
        if reason:
            return f"{name}:{reason}"
    return None


def filter_pool(pool_name: str, round_n: int, rules):
    round_dir = AUDIT / pool_name / f"round_{round_n}"
    samples = round_dir / "samples.jsonl"
    if not samples.exists():
        return None
    survivors = round_dir / "survivors.jsonl"
    rejected = round_dir / "rejected.jsonl"
    report = round_dir / "filter_report.tsv"

    n_in = n_out = 0
    reason_counts: dict[str, int] = {}

    with open(samples, encoding="utf-8") as fi, \
         open(survivors, "w", encoding="utf-8") as fs, \
         open(rejected, "w", encoding="utf-8") as fr:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            n_in += 1
            reason = apply_rules(row, rules)
            if reason is None:
                fs.write(line + "\n")
                n_out += 1
            else:
                fr.write(json.dumps({"reason": reason, "row": row}, ensure_ascii=False) + "\n")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    with open(report, "w", encoding="utf-8") as fo:
        fo.write("reason\tcount\n")
        for r, c in sorted(reason_counts.items(), key=lambda x: -x[1]):
            fo.write(f"{r}\t{c}\n")

    return {
        "pool": pool_name,
        "in": n_in,
        "out": n_out,
        "rejected": n_in - n_out,
        "reject_rate": (n_in - n_out) / max(n_in, 1),
        "top_reason": max(reason_counts.items(), key=lambda x: x[1]) if reason_counts else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--rules-version", type=int, required=True)
    ap.add_argument("--pool", default=None, help="single pool; default all")
    args = ap.parse_args()

    rules = load_rules(args.rules_version)
    pools = [args.pool] if args.pool else sorted(p.name for p in AUDIT.iterdir() if p.is_dir() and not p.name.startswith("_"))

    print(f"{'pool':<32} {'in':>5} {'out':>5} {'rej':>5} {'rate':>6}  top_reason")
    print("-" * 90)
    for pool in pools:
        rep = filter_pool(pool, args.round, rules)
        if rep is None:
            continue
        top = rep["top_reason"]
        top_str = f"{top[0]}({top[1]})" if top else "-"
        print(f"{rep['pool']:<32} {rep['in']:>5} {rep['out']:>5} {rep['rejected']:>5} {rep['reject_rate']*100:>5.1f}%  {top_str}")


if __name__ == "__main__":
    main()
