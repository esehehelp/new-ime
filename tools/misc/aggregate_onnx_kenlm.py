"""Aggregate ONNX × KenLM sweep results."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("results/onnx_kenlm_sweep")
MODELS = [
    "ctc-nat-30m-student-step160000.fp32",
    "ctc-nat-30m-student-step160000.int8",
]
CONFIGS = ["greedy", "beam5_nolm",
           "a0.2_b0.3", "a0.2_b0.6", "a0.4_b0.3", "a0.4_b0.6", "a0.6_b0.3", "a0.6_b0.6"]
CATS = ["edge", "general", "homophone", "names", "numeric", "particle", "tech"]


def load(model, cfg):
    p = ROOT / model / cfg / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d[next(iter(d))]


def main():
    header = f"{'model':<40} {'config':<14} {'EM1':<7} {'EM5':<7} {'CharAcc':<8} {'p50ms':<6}"
    print(header)
    print("-" * len(header))
    for m in MODELS:
        for cfg in CONFIGS:
            r = load(m, cfg)
            if r is None:
                continue
            em1 = r.get("exact_match_top1", 0)
            em5 = r.get("em5", 0)
            ca = r.get("char_acc_top1", 0)
            p50 = r.get("p50", "?")
            print(f"{m:<40} {cfg:<14} {em1:<7.3f} {em5:<7.3f} {ca:<8.3f} {p50}")

    print("\n=== BEST PER MODEL ===")
    for m in MODELS:
        best = None
        for cfg in CONFIGS:
            r = load(m, cfg)
            if r is None:
                continue
            em1 = r.get("exact_match_top1", 0)
            if best is None or em1 > best[0]:
                best = (em1, cfg, r)
        if best:
            print(f"{m:<40} {best[1]:<14} EM1={best[0]:.3f} EM5={best[2]['em5']:.3f} CharAcc={best[2].get('char_acc_top1', 0):.3f} p50={best[2].get('p50','?')}ms")

    # fp32 vs int8 per-cat at best config
    print("\n=== fp32 vs int8 (best α/β) per category ===")
    fp32_best = None
    int8_best = None
    for m in MODELS:
        for cfg in CONFIGS:
            r = load(m, cfg)
            if r is None:
                continue
            em1 = r.get("exact_match_top1", 0)
            target = fp32_best if "fp32" in m else int8_best
            if target is None or em1 > target[0]:
                if "fp32" in m:
                    fp32_best = (em1, cfg, r)
                else:
                    int8_best = (em1, cfg, r)
    if fp32_best and int8_best:
        print(f"{'cat':<10} {'fp32':<10} {'int8':<10} {'Δ':<8}")
        for c in CATS:
            f = fp32_best[2].get("per_category", {}).get(c, {}).get("em1", 0)
            i = int8_best[2].get("per_category", {}).get(c, {}).get("em1", 0)
            d = i - f
            print(f"{c:<10} {f:<10.3f} {i:<10.3f} {d:+.3f}")


if __name__ == "__main__":
    main()
