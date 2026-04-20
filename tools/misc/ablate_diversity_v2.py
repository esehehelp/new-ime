"""Extended ablation addressing analysis points #1b, #3, #4, #11.

Adds:
  - inner beam width sweep: beam=10 / 16 (output still top-5 surfaces)
  - wider temperature sweep: T ∈ {0.8, 1.5, 2.0, 2.5}
  - multi-alt mask refine: mask_refine_alt ∈ {3, 4}
  - EM3 / EM10 in addition to EM5

Fixed model: ctc-nat-30m-student step160000 (CPU).
Writes results/ablate_diversity_v2/{cfg}.json.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult

CKPT = "models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt"
LM = "models/kenlm/kenlm_general_train_4gram_probing.bin"
PROBE = "datasets/eval/probe/probe.json"
OUT = Path("results/ablate_diversity_v2")
KENLM_COMMON = dict(lm_path=LM, lm_alpha=0.2, lm_beta=0.6)


def eval_multi(backend, items, top_k=10):
    """Evaluate returning EM1/EM3/EM5/EM10, per-category, latency. Backend
    returns up to top_k surface candidates (passed via internal top_k arg)."""
    overall = EvalResult()
    per_cat_em1 = defaultdict(EvalResult)
    hits_at = {1: [], 3: [], 5: [], 10: []}
    per_cat_hits5 = defaultdict(list)
    lat = []
    n_cands = []

    for it in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(it["reading"], it["context"])
        except Exception as e:
            cands = [f"<err:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)
        n_cands.append(len(cands))

        refs = it["references"]
        # Truncate at top_k for EM@k
        cands_k = cands[:top_k] if cands else []

        # EM1 / CharAcc uses top-1
        overall.add_multi(refs, cands_k[:5])  # for CharAcc consistency
        per_cat_em1[it["category"]].add_multi(refs, cands_k[:5])

        for k in [1, 3, 5, 10]:
            hits_at[k].append(int(any(c in refs for c in cands_k[:k])))
        per_cat_hits5[it["category"]].append(
            int(any(c in refs for c in cands_k[:5]))
        )

    s = overall.summary()
    for k, hits in hits_at.items():
        s[f"em{k}"] = round(sum(hits) / len(hits), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["n_cands_avg"] = round(sum(n_cands) / len(n_cands), 2)
    s["per_cat_em5"] = {
        c: round(sum(v) / len(v), 3) for c, v in sorted(per_cat_hits5.items())
    }
    return s


class BackendWrapper:
    """Wraps CTCNATBackend so convert() asks for up to 10 surface candidates."""
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name

    def convert(self, reading, context):
        # Short inputs: call internal _decode_one with top_k=10 directly
        if self.inner.chunk_threshold <= 0 or len(reading) < self.inner.chunk_threshold:
            return self.inner._decode_one(reading, context, top_k=10)
        return self.inner.convert(reading, context)


def configs():
    return [
        ("A_kenlm_beam5", dict(beam_width=5, **KENLM_COMMON)),
        ("A_kenlm_beam10", dict(beam_width=10, **KENLM_COMMON)),
        ("A_kenlm_beam16", dict(beam_width=16, **KENLM_COMMON)),
        ("A_nolm_beam10", dict(beam_width=10)),
        ("AC_kenlm_T0.8", dict(beam_width=5, temperature=0.8, **KENLM_COMMON)),
        ("AC_kenlm_T1.5", dict(beam_width=5, temperature=1.5, **KENLM_COMMON)),
        ("AC_kenlm_T2.0", dict(beam_width=5, temperature=2.0, **KENLM_COMMON)),
        ("AC_kenlm_T2.5", dict(beam_width=5, temperature=2.5, **KENLM_COMMON)),
        ("D_greedy_mask15_alt3", dict(beam_width=1, mask_refine_k=15, mask_refine_alt=3)),
        ("D_greedy_mask20_alt4", dict(beam_width=1, mask_refine_k=20, mask_refine_alt=4)),
        ("D_greedy_mask30_alt4", dict(beam_width=1, mask_refine_k=30, mask_refine_alt=4)),
        ("AB_kenlm_div0.1", dict(beam_width=5, diversity_lambda=0.1, **KENLM_COMMON)),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_probe(PROBE)
    print(f"probe: {len(items)} items", flush=True)

    summary = {}
    for name, kwargs in configs():
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        inner = CTCNATBackend(CKPT, device="cpu", **kwargs)
        backend = BackendWrapper(inner)
        print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
        t0 = time.perf_counter()
        r = eval_multi(backend, items, top_k=10)
        dt = time.perf_counter() - t0
        em1, em3, em5, em10 = r["em1"], r["em3"], r["em5"], r["em10"]
        print(f"  EM1={em1:.3f} EM3={em3:.3f} EM5={em5:.3f} EM10={em10:.3f} "
              f"p50={r['p50']}ms cands/item={r['n_cands_avg']} ({dt:.0f}s)",
              flush=True)
        summary[name] = r
        (OUT / f"{name}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        del backend, inner

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n\n========= SUMMARY =========")
    header = f"{'config':<24} {'EM1':<7} {'EM3':<7} {'EM5':<7} {'EM10':<7} {'p50':<7} {'cands':<6}"
    print(header)
    print("-" * len(header))
    for name, r in summary.items():
        print(f"{name:<24} {r['em1']:<7.3f} {r['em3']:<7.3f} {r['em5']:<7.3f} "
              f"{r['em10']:<7.3f} {r['p50']:<7} {r['n_cands_avg']:<6}")


if __name__ == "__main__":
    main()
