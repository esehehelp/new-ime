"""Compute exact-field deltas and latency errors against archive/pre-v2
anchors. Reports raw Δ values, no judgment.

Usage:
    .venv/Scripts/python.exe scripts/error_table.py [scenarios_dir]
    (default scenarios_dir = results/test_scenarios)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO / "results" / "bench_v1_vs_v1_2"

PAIRS = [
    ("Suiko-v1-small__greedy__probe_v3.json", "probe_v3__greedy.json", "probe_v3"),
    ("Suiko-v1-small__greedy__ajimee_jwtd.json", "ajimee_jwtd__greedy.json", "ajimee_jwtd"),
]
METRIC_FIELDS = [
    "n", "total", "em5",
    "exact_match_top1", "exact_match_top5", "exact_match_top10",
    "char_acc_top1", "char_acc_top5", "char_acc_top10",
]
LATENCY_FIELDS = ["p50", "p95", "mean"]


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    raw = path.read_bytes()
    for enc in ("utf-8", "cp932", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def fmt_delta(a, b):
    if a is None or b is None:
        return f"{a!r} / {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        if isinstance(a, float) or isinstance(b, float):
            return f"{a:.4f} -> {b:.4f}  Δ={d:+.4f}"
        return f"{a} -> {b}  Δ={d:+d}"
    return f"{a!r} -> {b!r}"


def report_pair(scenario_label: str, anchor: dict, new: dict, bench: str) -> None:
    print(f"  --- {scenario_label}/{bench} ---")
    print("  metrics (Δ should be 0):")
    any_metric_diff = False
    for k in METRIC_FIELDS:
        a, b = anchor.get(k), new.get(k)
        line = fmt_delta(a, b)
        nonzero = (
            a != b
            and not (a is None and b is None)
        )
        if nonzero:
            any_metric_diff = True
        marker = "  !! " if nonzero else "     "
        print(f"  {marker}{k:<22s} {line}")
    if not any_metric_diff:
        print("     (all metric fields match exactly)")

    a_lat = anchor.get("latency_ms", {}) or {}
    b_lat = new.get("latency_ms", {}) or {}
    print("  latency_ms:")
    for k in LATENCY_FIELDS:
        av, bv = a_lat.get(k), b_lat.get(k)
        if av is None or bv is None:
            print(f"     {k}: {av} / {bv}")
            continue
        d = bv - av
        pct = (d / av * 100.0) if av else 0.0
        print(f"     {k:<6s} {av:6.1f}ms -> {bv:6.1f}ms  Δ={d:+6.1f}ms  ({pct:+.1f}%)")


def main(argv: list[str]) -> int:
    sc_dir = Path(argv[1]) if len(argv) > 1 else (REPO / "results" / "test_scenarios")
    if not sc_dir.is_dir():
        print(f"[error_table] not a dir: {sc_dir}", file=sys.stderr)
        return 2

    anchors: dict[str, dict] = {}
    for anchor_name, _, bench_key in PAIRS:
        a = load_json(ANCHOR_DIR / anchor_name)
        if a is None:
            print(f"[error_table] anchor missing: {anchor_name}", file=sys.stderr)
            return 2
        anchors[bench_key] = a

    scenarios = sorted(p for p in sc_dir.iterdir() if p.is_dir())
    if not scenarios and (sc_dir / "suiko-v1-small-greedy").is_dir():
        # The scenarios_dir IS already a single-run output (e.g. canonical).
        scenarios = [sc_dir]

    for sc in scenarios:
        # Single-run outputs have run.name dirs directly under sc.
        # Multi-scenario outputs have sN_*/ -> run.name/ structure.
        runs: list[Path] = []
        for child in sorted(sc.iterdir()):
            if not child.is_dir():
                continue
            # Heuristic: if child contains *.json directly, it's a run dir.
            if any(p.suffix == ".json" for p in child.iterdir()):
                runs.append(child)

        # If we found run dirs directly, sc is the scenario boundary.
        # If not, sc itself might be a run dir (canonical case).
        if not runs and any(p.suffix == ".json" for p in sc.iterdir()):
            runs = [sc]
            sc_label = sc.parent.name + "/" + sc.name
        else:
            sc_label = sc.name

        if not runs:
            continue

        print(f"=== {sc_label} ===")
        for run_dir in runs:
            for anchor_name, v2_name, bench_key in PAIRS:
                v2_path = run_dir / v2_name
                if not v2_path.is_file():
                    continue
                v2 = load_json(v2_path)
                if v2 is None:
                    print(f"  [{run_dir.name}] failed to load {v2_path.name}")
                    continue
                report_pair(run_dir.name, anchors[bench_key], v2, bench_key)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
