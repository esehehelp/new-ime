"""probe_v3 + AJIMEE benchmark for KenLM-MoE vs single KenLM.

Compares on ctc-nat-30m-student step160000 (v2 best):
  - single KenLM (general-only, canonical: α=0.2 β=0.6, beam=5)
  - KenLM-MoE (general + tech + entity mixture via CategoryEstimator)

Runs both probe_v3 (348 items) and AJIMEE (200 items) on CPU via WSL.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import load_ajimee_jwtd, load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult

CKPT = "models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt"
PROBE = "datasets/eval/probe/probe.json"
AJIMEE = "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json"

LM_GENERAL = "models/kenlm/kenlm_general_train_4gram_probing.bin"
LM_TECH = "models/kenlm/kenlm_tech_4gram.bin"
LM_ENTITY = "models/kenlm/kenlm_entity_4gram.bin"

OUT_ROOT = Path("results/kenlm_moe_bench")


def evaluate(backend, items, top_k: int = 5) -> dict:
    overall = EvalResult()
    per_cat = defaultdict(EvalResult)
    em5: list[int] = []
    lat: list[float] = []
    for it in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(it["reading"], it["context"])
        except Exception as e:
            cands = [f"<err:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)
        refs = it["references"]
        cands_k = cands[:top_k] if cands else []
        overall.add_multi(refs, cands_k)
        cat = it.get("category")
        if cat:
            per_cat[cat].add_multi(refs, cands_k)
        em5.append(int(any(c in refs for c in cands_k)))
    s = overall.summary()
    s["em5"] = round(sum(em5) / max(len(em5), 1), 4)
    lat.sort()
    n = len(lat)
    s["p50"] = round(lat[n // 2], 1)
    s["p95"] = round(lat[int(n * 0.95)], 1)
    s["n"] = len(items)
    if per_cat:
        s["per_category"] = {
            c: {
                "em1": round(r.summary().get("exact_match_top1", 0), 3),
                "n": r.summary().get("n", 0),
            }
            for c, r in sorted(per_cat.items())
        }
    return s


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    probe_items = load_probe(PROBE)
    ajimee_items = load_ajimee_jwtd(AJIMEE)
    print(f"probe: {len(probe_items)}  ajimee: {len(ajimee_items)}", flush=True)

    lm_paths = {"general": LM_GENERAL}
    if Path(LM_TECH).exists():
        lm_paths["tech"] = LM_TECH
    if Path(LM_ENTITY).exists():
        lm_paths["entity"] = LM_ENTITY
    print(f"active LMs: {list(lm_paths.keys())}", flush=True)

    configs = [
        ("single_general_a0.2_b0.6", dict(
            beam_width=5, lm_path=LM_GENERAL, lm_alpha=0.2, lm_beta=0.6,
        )),
        ("moe_a0.2_b0.6", dict(
            beam_width=5, lm_paths_by_domain=lm_paths, lm_alpha=0.2, lm_beta=0.6,
        )),
        ("moe_a0.4_b0.6", dict(
            beam_width=5, lm_paths_by_domain=lm_paths, lm_alpha=0.4, lm_beta=0.6,
        )),
        ("moe_a0.2_b0.3", dict(
            beam_width=5, lm_paths_by_domain=lm_paths, lm_alpha=0.2, lm_beta=0.3,
        )),
    ]

    summary: dict[str, dict] = {}
    for cfg_name, kwargs in configs:
        print(f"\n=== {cfg_name} ===", flush=True)
        t0 = time.perf_counter()
        backend = CTCNATBackend(CKPT, device="cpu", **kwargs)
        print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
        cfg_res: dict[str, dict] = {}
        for bench_name, items in (("probe", probe_items), ("ajimee", ajimee_items)):
            t0 = time.perf_counter()
            r = evaluate(backend, items)
            dt = time.perf_counter() - t0
            em1 = r.get("exact_match_top1", 0)
            em5 = r["em5"]
            print(
                f"  {bench_name:<7} EM1={em1:.3f} EM5={em5:.3f} "
                f"CharAcc={r.get('char_acc_top1', 0):.3f} p50={r['p50']}ms ({dt:.0f}s)",
                flush=True,
            )
            cfg_res[bench_name] = r
        summary[cfg_name] = cfg_res
        del backend

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== COMPARISON ===")
    print(f"{'config':<30} {'probe EM1':<10} {'probe EM5':<10} {'ajimee EM1':<11} {'ajimee EM5':<11}")
    for name, r in summary.items():
        p = r.get("probe", {})
        a = r.get("ajimee", {})
        print(
            f"{name:<30} "
            f"{p.get('exact_match_top1', 0):<10.3f} "
            f"{p.get('em5', 0):<10.3f} "
            f"{a.get('exact_match_top1', 0):<11.3f} "
            f"{a.get('em5', 0):<11.3f}"
        )


if __name__ == "__main__":
    main()
