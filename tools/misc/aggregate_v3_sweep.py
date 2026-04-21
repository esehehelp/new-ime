"""Aggregate v3 bunsetsu probe_v3 × α/β sweep results."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("results/probe_v3_bunsetsu_kenlm_sweep")
CKPTS = [
    "ctc-nat-30m-bunsetsu-v3-best",
    "ctc-nat-30m-bunsetsu-v3-step60000",
    "ctc-nat-30m-bunsetsu-v3-step73000",
]
CONFIGS = ["greedy", "beam5_nolm",
           "a0.2_b0.3", "a0.2_b0.6", "a0.4_b0.3", "a0.4_b0.6", "a0.6_b0.3", "a0.6_b0.6"]
CATS = ["edge", "general", "homophone", "names", "numeric", "particle", "tech"]


def load(ckpt, cfg):
    p = ROOT / ckpt / cfg / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d:
        return None
    return d[next(iter(d))]


def main():
    header = f"{'ckpt':<40} {'config':<14} {'EM1':<7} {'EM5':<7} {'CharAcc':<8} {'p50ms':<6} | " + " ".join(f"{c:<5}" for c in CATS)
    print(header)
    print("-" * len(header))
    for ck in CKPTS:
        for cfg in CONFIGS:
            r = load(ck, cfg)
            if r is None:
                continue
            em1 = r.get("em1", 0)
            em5 = r.get("em5", 0)
            ca = r.get("char_acc", 0)
            p50 = r.get("latency_ms", {}).get("p50", "?")
            cats = r.get("per_category", {})
            cat_str = " ".join(
                f"{cats.get(c, {}).get('exact_match_top1', 0):.2f}"
                if cats.get(c) else " -   "
                for c in CATS
            )
            print(f"{ck:<40} {cfg:<14} {em1:<7.3f} {em5:<7.3f} {ca:<8.3f} {str(p50):<6} | {cat_str}")

    # Best per ckpt
    print("\n=== BEST CONFIG PER CKPT ===")
    for ck in CKPTS:
        best = None
        for cfg in CONFIGS:
            r = load(ck, cfg)
            if r is None:
                continue
            em1 = r.get("em1", 0)
            if best is None or em1 > best[0]:
                best = (em1, cfg, r)
        if best:
            print(f"{ck:<40} {best[1]:<14} EM1={best[0]:.3f} EM5={best[2]['em5']:.3f}")


if __name__ == "__main__":
    main()
