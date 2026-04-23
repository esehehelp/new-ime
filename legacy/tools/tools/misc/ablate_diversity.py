"""Ablation: EM1 / EM5 on probe_v3 across A / B / C / D diversity approaches.

Fixed model: ctc-nat-30m-student step160000 (CPU).

Configurations:
    A_nolm_beam5              : plain beam=5, no LM
    A_kenlm_beam5             : beam=5 + KenLM α=0.2 β=0.6  (baseline after A fix)
    AB_kenlm_div0.5_beam5     : A + diversity_lambda=0.5
    AB_kenlm_div1.0_beam5     : A + diversity_lambda=1.0
    AC_kenlm_T1.3_beam5       : A + temperature=1.3
    AC_kenlm_top0.9_beam5     : A + top_p=0.9
    D_greedy_mask5            : greedy + mask_refine_k=5
    D_greedy_mask10           : greedy + mask_refine_k=10

Each prints EM1 / EM5 / CharAcc / p50 + per-category EM5 breakdown.
Writes results/ablate_diversity/{cfg}.json.
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
OUT = Path("results/ablate_diversity")


def evaluate(backend, items):
    overall = EvalResult()
    per_cat_em1 = defaultdict(EvalResult)
    em5 = []
    per_cat_em5 = defaultdict(list)
    lat = []
    for it in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(it["reading"], it["context"])
        except Exception as e:
            cands = [f"<err:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)
        refs = it["references"]
        cands_k = cands[:5] if cands else []
        overall.add_multi(refs, cands_k)
        per_cat_em1[it["category"]].add_multi(refs, cands_k)
        hit5 = int(any(c in refs for c in cands_k))
        em5.append(hit5)
        per_cat_em5[it["category"]].append(hit5)
    s = overall.summary()
    s["em5"] = round(sum(em5) / len(em5), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["per_cat_em1"] = {
        c: round(r.summary().get("exact_match_top1", 0), 3)
        for c, r in sorted(per_cat_em1.items())
    }
    s["per_cat_em5"] = {
        c: round(sum(v) / len(v), 3)
        for c, v in sorted(per_cat_em5.items())
    }
    s["n_cands_mean"] = round(
        sum(min(5, len(backend.convert(it["reading"], it["context"]) or [])) for it in items[:20]) / 20, 2
    )
    return s


def make_configs():
    return [
        ("A_nolm_beam5", dict(beam_width=5)),
        ("A_kenlm_beam5", dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6)),
        ("AB_kenlm_div0.5_beam5", dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6,
                                        diversity_lambda=0.5)),
        ("AB_kenlm_div1.0_beam5", dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6,
                                        diversity_lambda=1.0)),
        ("AC_kenlm_T1.3_beam5", dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6,
                                      temperature=1.3)),
        ("AC_kenlm_top0.9_beam5", dict(beam_width=5, lm_path=LM, lm_alpha=0.2, lm_beta=0.6,
                                        top_p=0.9)),
        ("D_greedy_mask5", dict(beam_width=1, mask_refine_k=5)),
        ("D_greedy_mask10", dict(beam_width=1, mask_refine_k=10)),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = load_probe(PROBE)
    print(f"probe: {len(items)} items", flush=True)

    summary = {}
    for name, kwargs in make_configs():
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        backend = CTCNATBackend(CKPT, device="cpu", **kwargs)
        print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
        t0 = time.perf_counter()
        r = evaluate(backend, items)
        dt = time.perf_counter() - t0
        em1 = r.get("exact_match_top1", 0)
        em5 = r["em5"]
        ca = r.get("char_acc_top1", 0)
        print(f"  EM1={em1:.3f} EM5={em5:.3f} CharAcc={ca:.3f} "
              f"p50={r['p50']}ms p95={r['p95']}ms ({dt:.0f}s)", flush=True)
        print(f"  EM5/cat: {r['per_cat_em5']}", flush=True)
        summary[name] = r
        (OUT / f"{name}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        del backend

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n\n========= ABLATION SUMMARY =========")
    print(f"{'config':<28} {'EM1':<7} {'EM5':<7} {'Δ':<7} {'CharAcc':<9} {'p50':<7}")
    for name, r in summary.items():
        em1 = r.get("exact_match_top1", 0)
        em5 = r["em5"]
        d = em5 - em1
        print(f"{name:<28} {em1:<7.3f} {em5:<7.3f} {d:<+7.3f} "
              f"{r.get('char_acc_top1', 0):<9.3f} {r['p50']}")


if __name__ == "__main__":
    main()
