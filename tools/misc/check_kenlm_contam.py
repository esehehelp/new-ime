"""Check n-gram contamination of the KenLM training corpus by probe_v3 / AJIMEE.

Build a 6-gram set from probe/AJIMEE reference surfaces (expected_output +
original_text), then scan every line of general/train.jsonl counting how many
lines share at least one 6-gram. Report a few hit samples for inspection.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

N = 20
TRAIN = "datasets/eval/general/train.jsonl"
REFS = [
    "datasets/eval/probe/probe.json",
    "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json",
]


def ngrams(s: str, n: int = N) -> set[str]:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def build_contam_set() -> tuple[set[str], int]:
    contam: set[str] = set()
    item_count = 0
    for path in REFS:
        data = json.load(open(path, encoding="utf-8"))
        for it in data:
            item_count += 1
            for s in it.get("expected_output", []):
                if s:
                    contam |= ngrams(s)
            orig = it.get("original_text", "")
            if orig:
                contam |= ngrams(orig)
    return contam, item_count


def main():
    contam, n_items = build_contam_set()
    print(f"contam set: {len(contam)} distinct {N}-grams from {n_items} eval items", flush=True)

    total = 0
    hit = 0
    per_ref_hit = 0  # Redundant — same as hit here, but placeholder
    samples = []
    t0 = time.perf_counter()
    with open(TRAIN, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            surf = d.get("surface", "")
            if not surf:
                continue
            surf_ng = ngrams(surf)
            if surf_ng & contam:
                hit += 1
                if len(samples) < 10:
                    hit_ng = next(iter(surf_ng & contam))
                    samples.append({"surface": surf[:80], "hit_ngram": hit_ng})
            if total % 2_000_000 == 0:
                dt = time.perf_counter() - t0
                rate = total / dt
                print(f"  scanned {total:>10,} lines ({rate:,.0f}/s)  hit={hit:>8,} ({100*hit/total:.2f}%)",
                      flush=True)

    dt = time.perf_counter() - t0
    print(f"\n=== RESULT ({dt:.0f}s) ===")
    print(f"  total lines scanned: {total:,}")
    print(f"  contaminated lines : {hit:,} ({100*hit/max(total,1):.3f}%)")
    print("  sample hits:")
    for s in samples:
        print(f"    surface={s['surface']!r}  hit_ngram={s['hit_ngram']!r}")


if __name__ == "__main__":
    main()
