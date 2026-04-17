"""Print a unified comparison table from all results/eval_runs/*__*.json files."""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    files = sorted(glob.glob("results/eval_runs/*__*.json"))
    grouped: dict[str, dict[str, dict]] = {}
    for f in files:
        stem = Path(f).stem
        model, bench = stem.split("__", 1)
        grouped.setdefault(model, {})[bench] = json.loads(
            Path(f).read_text(encoding="utf-8")
        )

    order = [
        "zenz_v2_5_medium_greedy",
        "zenz_v2_5_small_greedy",
        "zenz_v2_5_xsmall_greedy",
        "ar_v3_local_greedy",
        "ar_v3_local_beam10",
        "ar_v3_vast_greedy",
        "ar_v3_vast_beam10",
        "ar_v3_chunks_greedy",
        "ar_v3_chunks_beam10",
    ]
    benches = ["manual_test", "ajimee_jwtd", "eval_v3_dev"]

    for b in benches:
        print(f"\n=== {b} ===")
        print(f'{"Model":<28} {"EM":>7} {"CharAcc":>8} {"p50ms":>7}')
        print("-" * 55)
        for m in order:
            r = grouped.get(m, {}).get(b, {})
            if not r:
                continue
            em = r.get("exact_match_top1", 0.0)
            ca = r.get("char_acc_top1", 0.0)
            lat = (r.get("latency") or {}).get("p50_ms", 0.0)
            print(f"{m:<28} {em:>7.3f} {ca:>8.3f} {lat:>7.0f}")


if __name__ == "__main__":
    main()
