"""Compare v2 bench outputs to archive/pre-v2 anchors.

For each scenario dir under results/test_scenarios/sN_*/, finds the
greedy probe_v3 / ajimee_jwtd JSON outputs and diffs them against
results/bench_v1_vs_v1_2/Suiko-v1-small__greedy__<bench>.json.

Reports differences in key metrics. Latency is compared with a tolerance
(CPU run-to-run variance is normal). Metrics (EM, CharAcc, n, total)
should match exactly since same model, same data, same decoder.

Usage (from repo root):
    .venv/Scripts/python.exe scripts/compare_bench_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid cp932 print errors
except Exception:
    pass

REPO = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO / "results" / "bench_v1_vs_v1_2"
SCENARIOS_DIR = REPO / "results" / "test_scenarios"

# (anchor_filename, v2_filename) for the greedy comparisons we can make.
ANCHOR_PAIRS = [
    ("Suiko-v1-small__greedy__probe_v3.json", "probe_v3__greedy.json"),
    ("Suiko-v1-small__greedy__ajimee_jwtd.json", "ajimee_jwtd__greedy.json"),
]

# Fields that should match EXACTLY (deterministic given same ckpt + data).
EXACT_FIELDS = [
    "total",
    "n",
    "em5",
    "char_acc_top1",
    "exact_match_top1",
    "char_acc_top5",
    "exact_match_top5",
    "char_acc_top10",
    "exact_match_top10",
]
# Latency tolerance in milliseconds (run-to-run variance on CPU).
LATENCY_TOLERANCE_MS = 25.0
# Mean tolerance is more permissive than p50.
LATENCY_FIELDS = ["p50", "p95", "mean"]


def load_json(path: Path) -> dict | None:
    """Load JSON, tolerating both utf-8 (v2 outputs) and cp932 (legacy
    Windows-locale-default Path.write_text outputs from pre-v2)."""
    if not path.is_file():
        return None
    raw = path.read_bytes()
    for enc in ("utf-8", "cp932", "utf-8-sig"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return json.loads(raw.decode("utf-8", errors="replace"))


def diff_one(anchor: dict, new: dict, *, label: str) -> list[str]:
    diffs: list[str] = []
    for k in EXACT_FIELDS:
        a = anchor.get(k)
        b = new.get(k)
        if a != b:
            diffs.append(f"  [{label}] {k}: anchor={a!r}  new={b!r}")
    a_lat = anchor.get("latency_ms", {}) or {}
    b_lat = new.get("latency_ms", {}) or {}
    for k in LATENCY_FIELDS:
        av = a_lat.get(k)
        bv = b_lat.get(k)
        if av is None or bv is None:
            if av != bv:
                diffs.append(f"  [{label}] latency.{k}: anchor={av!r}  new={bv!r}")
            continue
        if abs(av - bv) > LATENCY_TOLERANCE_MS:
            diffs.append(
                f"  [{label}] latency.{k}: anchor={av}ms  new={bv}ms  "
                f"(delta={bv - av:+.1f}ms, tolerance={LATENCY_TOLERANCE_MS}ms)"
            )
    # sample_failures: report if the count differs (the failing items
    # themselves vary slightly with truncation but cardinality should
    # match if metrics match).
    anc_fails = len(anchor.get("sample_failures") or [])
    new_fails = len(new.get("sample_failures") or [])
    if anc_fails != new_fails:
        diffs.append(
            f"  [{label}] sample_failures count: anchor={anc_fails} new={new_fails}"
        )
    return diffs


def find_v2_jsons(scenario_dir: Path, v2_filename: str) -> list[Path]:
    """A scenario dir holds one or more <run.name>/ subdirs; find the
    matching <bench>__<mode>.json under each.
    """
    out: list[Path] = []
    if not scenario_dir.is_dir():
        return out
    for run_dir in sorted(scenario_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        candidate = run_dir / v2_filename
        if candidate.is_file():
            out.append(candidate)
    return out


def main() -> int:
    if not ANCHOR_DIR.is_dir():
        print(f"[compare] anchor dir missing: {ANCHOR_DIR}", file=sys.stderr)
        return 2
    if not SCENARIOS_DIR.is_dir():
        print(f"[compare] scenarios dir missing: {SCENARIOS_DIR}", file=sys.stderr)
        return 2

    scenarios = sorted(p for p in SCENARIOS_DIR.iterdir() if p.is_dir())
    if not scenarios:
        print(f"[compare] no scenarios under {SCENARIOS_DIR}", file=sys.stderr)
        return 2

    print(f"[compare] anchors:   {ANCHOR_DIR}")
    print(f"[compare] scenarios: {SCENARIOS_DIR}")
    print(f"[compare] tolerance: latency +/- {LATENCY_TOLERANCE_MS}ms")
    print()

    total_diffs = 0
    for sc_dir in scenarios:
        print(f"=== {sc_dir.name} ===")
        any_seen = False
        for anchor_name, v2_name in ANCHOR_PAIRS:
            anchor_path = ANCHOR_DIR / anchor_name
            anchor = load_json(anchor_path)
            v2_paths = find_v2_jsons(sc_dir, v2_name)

            if not v2_paths:
                print(f"  [{v2_name}] (no output in this scenario)")
                continue

            for v2_path in v2_paths:
                any_seen = True
                rel = v2_path.relative_to(REPO)
                v2 = load_json(v2_path)
                if anchor is None:
                    print(f"  [{rel}] anchor missing: {anchor_name}")
                    continue
                if v2 is None:
                    print(f"  [{rel}] FAILED to load v2 output")
                    continue
                label = f"{sc_dir.name}/{v2_path.parent.name}/{v2_name}"
                diffs = diff_one(anchor, v2, label=label)
                if diffs:
                    total_diffs += len(diffs)
                    print(f"  [{rel}]  vs  {anchor_name}: {len(diffs)} differences")
                    for d in diffs:
                        print(d)
                else:
                    print(f"  [{rel}]  vs  {anchor_name}: OK")
        # also list non-greedy outputs (beam5 etc.) so we know what was
        # produced even without an anchor
        for run_dir in sorted(sc_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for j in sorted(run_dir.glob("*__*.json")):
                if j.name in {p[1] for p in ANCHOR_PAIRS}:
                    continue
                if j.name == "summary.json":
                    continue
                rel = j.relative_to(REPO)
                print(f"  [{rel}] (no anchor available — beam5)")
        if not any_seen:
            print("  (no greedy outputs found in this scenario)")
        print()

    print(f"[compare] total exact-field / over-tolerance diffs: {total_diffs}")
    return 0 if total_diffs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
