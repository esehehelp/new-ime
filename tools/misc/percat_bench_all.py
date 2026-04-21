"""Per-category EM1 extractor for results/bench_all/summary.json."""
from __future__ import annotations
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

CATS = ["edge", "general", "homophone", "names", "numeric", "particle", "tech"]


def main():
    d = json.load(open("results/bench_all/summary.json"))
    print(f"{'model_cfg':<50} {'bench':<7} " + " ".join(f"{c[:5]:<6}" for c in CATS))
    print("-" * (50 + 8 + 7 * 7))
    for m in [
        "ctc-nat-30m-student-step160000__greedy",
        "ctc-nat-30m-student-step160000__kenlm",
        "ctc-nat-30m-student-step160000__kenlm-moe",
        "ctc-nat-30m-student-step160000__onnx-fp32-greedy",
        "ctc-nat-30m-student-step160000__onnx-int8-greedy",
        "zenz-v2.5-xsmall__beam5",
        "zenz-v2.5-small__beam5",
        "zenz-v2.5-medium__beam5",
        "zenz-v3.1-small__beam5",
        "teacher-150m-teacher-step200000__greedy",
    ]:
        for b in ["probe"]:  # ajimee has no per-cat (no category field)
            pc = d.get(m, {}).get(b, {}).get("per_category")
            if not pc:
                continue
            row = [m, b]
            for cat in CATS:
                v = pc.get(cat, {}).get("exact_match_top1", 0)
                row.append(f"{v:.3f}")
            print(f"{row[0]:<50} {row[1]:<7} " + " ".join(f"{c:<6}" for c in row[2:]))


if __name__ == "__main__":
    main()
