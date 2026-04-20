"""Run probe_v3 (348 items) + AJIMEE JWTD_v2 (200 items) across a fixed model set.

Uniform methodology (all runs):
    device           = CPU (forced — no CUDA)
    probe items      = all 348 (datasets/eval/probe/probe.json)
    ajimee items     = all 200 (references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json)
    top_k            = 5 (for EM5)
    latency          = wall-clock per item, sorted → p50/p95
    KenLM (CTC-NAT)  = models/kenlm/kenlm_general_train_4gram_probing.bin, α=0.2 β=0.6 beam=5
    beam (AR)        = beam=5, length_penalty=0.6 (existing default)
    zenz             = num_beams=5, num_return=5 (matches probe runner defaults)

Output per (model_cfg, bench): results/bench_all/{model_cfg}__{bench}.json
Aggregated table printed at end and saved to results/bench_all/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.ar_backend import ARCheckpointBackend
from models.src.eval.bench_loaders import load_ajimee_jwtd, load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult
from models.src.eval.teacher_backend import TeacherBackend
from models.src.eval.zenz_backend import ZenzV2Backend

DEVICE = "cpu"
LM_PATH = "models/kenlm/kenlm_general_train_4gram_probing.bin"
LM_ALPHA = 0.2
LM_BETA = 0.6
LM_BEAM = 5

OUT_ROOT = Path("results/bench_all")


def evaluate(backend, items: list[dict], top_k: int = 5) -> dict:
    overall = EvalResult()
    per_cat: dict[str, EvalResult] = defaultdict(EvalResult)
    per_cat_em5: dict[str, list[int]] = defaultdict(list)
    em5: list[int] = []
    lat: list[float] = []
    fails: list[dict] = []

    for item in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(item["reading"], item["context"])
        except Exception as e:
            cands = [f"<error:{e}>"]
        lat.append((time.perf_counter() - t0) * 1000)

        refs = item["references"]
        cands_k = cands[:top_k] if cands else []
        overall.add_multi(refs, cands_k)

        cat = item.get("category")
        if cat:
            per_cat[cat].add_multi(refs, cands_k)

        hit5 = int(any(c in refs for c in cands_k))
        em5.append(hit5)
        if cat:
            per_cat_em5[cat].append(hit5)

        if cands_k and cands_k[0] not in refs and len(fails) < 8:
            fails.append({
                "reading": item["reading"][:30],
                "ref": refs[0][:30],
                "pred": cands_k[0][:30],
                **({"cat": cat} if cat else {}),
            })

    s = overall.summary()
    s["backend"] = backend.name
    s["n"] = len(items)
    s["em5"] = round(sum(em5) / len(em5), 4) if em5 else 0
    lat.sort()
    n = len(lat)
    s["latency_ms"] = {
        "p50": round(lat[n // 2], 1),
        "p95": round(lat[int(n * 0.95)], 1),
        "mean": round(sum(lat) / n, 1),
    }
    if per_cat:
        cat_out = {}
        for c, r in sorted(per_cat.items()):
            cs = r.summary()
            hits = per_cat_em5[c]
            cs["em5"] = round(sum(hits) / len(hits), 4) if hits else 0
            cs["n"] = len(hits)
            cat_out[c] = cs
        s["per_category"] = cat_out
    s["sample_failures"] = fails
    return s


def ctc(ckpt: str, *, lm: bool) -> CTCNATBackend:
    return CTCNATBackend(
        ckpt, device=DEVICE,
        beam_width=LM_BEAM if lm else 1,
        lm_path=LM_PATH if lm else None,
        lm_alpha=LM_ALPHA if lm else 0.0,
        lm_beta=LM_BETA if lm else 0.0,
    )


REGISTRY: list[tuple[str, callable]] = [
    # --- CTC-NAT (own), latest step per checkpoint dir ---
    ("ctc-nat-30m-student-step160000__greedy",
        lambda: ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt", lm=False)),
    ("ctc-nat-30m-student-step160000__kenlm",
        lambda: ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt", lm=True)),

    ("ctc-nat-30m-scratch-step50000__greedy",
        lambda: ctc("models/checkpoints/ctc-nat-30m-scratch/checkpoint_step_50000.pt", lm=False)),
    ("ctc-nat-30m-scratch-step50000__kenlm",
        lambda: ctc("models/checkpoints/ctc-nat-30m-scratch/checkpoint_step_50000.pt", lm=True)),

    # 90m step30000 is broken (empty / single-char outputs across all inputs).
    # step27500 is the last working checkpoint; kept as the 90m entry.
    ("ctc-nat-90m-scratch-step27500__greedy",
        lambda: ctc("models/checkpoints/ctc-nat-90m-scratch/checkpoint_step_27500.pt", lm=False)),
    ("ctc-nat-90m-scratch-step27500__kenlm",
        lambda: ctc("models/checkpoints/ctc-nat-90m-scratch/checkpoint_step_27500.pt", lm=True)),

    # --- AR (own) ---
    ("ar-31m-scratch-step80000__greedy",
        lambda: ARCheckpointBackend(
            "models/checkpoints/ar-31m-scratch/checkpoint_step_80000.pt",
            device=DEVICE, beam_width=1)),
    ("ar-31m-scratch-step80000__beam5",
        lambda: ARCheckpointBackend(
            "models/checkpoints/ar-31m-scratch/checkpoint_step_80000.pt",
            device=DEVICE, beam_width=5)),

    # --- Teacher (own, AR encoder-decoder) ---
    ("teacher-150m-teacher-step200000__greedy",
        lambda: TeacherBackend(
            "models/checkpoints/teacher-150m-teacher/checkpoint_step_200000.pt",
            device=DEVICE)),

    # --- zenz (reference) ---
    ("zenz-v2.5-xsmall__beam5",
        lambda: ZenzV2Backend("references/zenz-v2.5-xsmall",
            device=DEVICE, num_beams=5, num_return=5)),
    ("zenz-v2.5-small__beam5",
        lambda: ZenzV2Backend("references/zenz-v2.5-small",
            device=DEVICE, num_beams=5, num_return=5)),
    ("zenz-v2.5-medium__beam5",
        lambda: ZenzV2Backend("references/zenz-v2.5-medium",
            device=DEVICE, num_beams=5, num_return=5)),
    ("zenz-v3.1-small__beam5",
        lambda: ZenzV2Backend("references/zenz-v3.1-small",
            device=DEVICE, num_beams=5, num_return=5)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma substring filter (empty = all)")
    ap.add_argument("--benches", default="probe,ajimee")
    ap.add_argument("--probe-path", default="datasets/eval/probe/probe.json")
    ap.add_argument("--ajimee-path", default="references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    benches: dict[str, list[dict]] = {}
    want = set(b.strip() for b in args.benches.split(",") if b.strip())
    if "probe" in want:
        benches["probe"] = load_probe(args.probe_path)
    if "ajimee" in want:
        benches["ajimee"] = load_ajimee_jwtd(args.ajimee_path)
    for b, items in benches.items():
        print(f"[bench] {b}: {len(items)} items", flush=True)

    pool = REGISTRY
    if args.models:
        wanted = [w.strip() for w in args.models.split(",") if w.strip()]
        pool = [(n, f) for n, f in pool if any(w in n for w in wanted)]

    print(f"[models] {len(pool)} to run", flush=True)

    agg: dict[str, dict[str, dict]] = {}
    t_overall = time.perf_counter()

    for name, factory in pool:
        print(f"\n### {name} ###", flush=True)
        try:
            t0 = time.perf_counter()
            backend = factory()
            print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  !! load failed: {e}", flush=True)
            agg[name] = {b: {"error": f"load: {e}"} for b in benches}
            continue

        agg[name] = {}
        for bench_name, items in benches.items():
            print(f"  -> {bench_name}", flush=True)
            try:
                t0 = time.perf_counter()
                res = evaluate(backend, items)
                dt = time.perf_counter() - t0
            except Exception as e:
                print(f"  !! eval error: {e}", flush=True)
                agg[name][bench_name] = {"error": str(e)}
                continue
            em1 = res.get("exact_match_top1", 0)
            em5 = res["em5"]
            ca = res.get("char_acc_top1", 0)
            p50 = res["latency_ms"]["p50"]
            print(f"     EM1={em1:.3f} EM5={em5:.3f} CharAcc={ca:.3f} "
                  f"p50={p50}ms ({dt:.0f}s total)", flush=True)
            agg[name][bench_name] = res
            (OUT_ROOT / f"{_safe(name)}__{bench_name}.json").write_text(
                json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

        del backend
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    (OUT_ROOT / "summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    # table
    print("\n\n========= COMPARISON =========")
    header = ["model_cfg", "probe EM1", "probe EM5", "probe CharAcc", "probe p50",
              "ajimee EM1", "ajimee EM5", "ajimee CharAcc", "ajimee p50"]
    rows = [header]
    for name, r in agg.items():
        p = r.get("probe", {})
        a = r.get("ajimee", {})
        row = [name]
        for src in (p, a):
            if "error" in src:
                row += ["err", "err", "err", "err"]
                continue
            row.append(f"{src.get('exact_match_top1', 0):.3f}")
            row.append(f"{src.get('em5', 0):.3f}")
            row.append(f"{src.get('char_acc_top1', 0):.3f}")
            row.append(str(src.get("latency_ms", {}).get("p50", "-")))
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for i, r in enumerate(rows):
        print(" | ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            print("-+-".join("-" * w for w in widths))

    total = time.perf_counter() - t_overall
    print(f"\nTotal: {total:.0f}s ({total/60:.1f} min)")
    print(f"Saved: {OUT_ROOT}/summary.json")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


if __name__ == "__main__":
    main()
