"""probe runner: category × backend EM/CharAcc/latency.

probe_v3 は AJIMEE 互換 JSON (+ category) で長文 + 文脈付き。probe_v2 の
tsv/phrase 短句版は `datasets/tools/probe/legacy/run_probe_v2.py` に退避済み。

Usage:
    uv run python -m datasets.tools.probe.run
    uv run python -m datasets.tools.probe.run --models ctc-nat-30m-student
    uv run python -m datasets.tools.probe.run --lm-path models/kenlm/... --alpha 0.4 --beta 0.6 --beam 5

metrics:
    EM1 / EM5 は multi-ref を許容 (references のいずれかにマッチで正解)。
    CharAcc はレーベンシュタインによる top1 文字単位一致率。
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
from models.src.eval.bench_loaders import load_probe
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult
from models.src.eval.teacher_backend import TeacherBackend
from models.src.eval.zenz_backend import ZenzV2Backend


def evaluate(backend, items: list[dict], top_k: int = 5) -> dict:
    overall = EvalResult()
    per_cat: dict[str, EvalResult] = defaultdict(EvalResult)
    latencies: list[float] = []
    fails: list[dict] = []
    per_cat_em5: dict[str, list[int]] = defaultdict(list)
    overall_em5: list[int] = []

    for item in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(item["reading"], item["context"])
        except Exception as e:
            cands = [f"<error:{e}>"]
        latencies.append((time.perf_counter() - t0) * 1000)
        refs = item["references"]
        cands_k = cands[:top_k] if cands else []
        overall.add_multi(refs, cands_k)
        per_cat[item["category"]].add_multi(refs, cands_k)
        hit5 = int(any(c in refs for c in cands_k))
        overall_em5.append(hit5)
        per_cat_em5[item["category"]].append(hit5)
        if cands_k and cands_k[0] not in refs and len(fails) < 12:
            fails.append({
                "cat": item["category"],
                "idx": item.get("_index", ""),
                "reading": item["reading"][:30],
                "ref": refs[0][:30],
                "pred": cands_k[0][:30],
            })

    summary = overall.summary()
    summary["backend"] = backend.name
    summary["n"] = len(items)
    summary["em5"] = round(sum(overall_em5) / len(overall_em5), 4) if overall_em5 else 0
    latencies.sort()
    n = len(latencies)
    summary["latency_ms"] = {
        "p50": round(latencies[n // 2], 1),
        "p95": round(latencies[int(n * 0.95)], 1),
        "mean": round(sum(latencies) / n, 1),
    }
    summary["per_category"] = {}
    for cat, r in sorted(per_cat.items()):
        s = r.summary()
        hits = per_cat_em5[cat]
        s["em5"] = round(sum(hits) / len(hits), 4) if hits else 0
        s["n"] = len(hits)
        summary["per_category"][cat] = s
    summary["sample_failures"] = fails
    return summary


def build_models(
    filter_substr: str = "",
    lm_path: str = "",
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
    beam_width: int = 1,
) -> list[tuple[str, callable]]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ctc(ckpt: str) -> CTCNATBackend:
        return CTCNATBackend(
            ckpt, device=device,
            beam_width=beam_width,
            lm_path=lm_path or None,
            lm_alpha=lm_alpha,
            lm_beta=lm_beta,
        )

    pool: list[tuple[str, callable]] = [
        ("zenz-v3.1-small",
            lambda: ZenzV2Backend("references/zenz-v3.1-small", device=device,
                                  num_beams=5, num_return=5, name_suffix="-b5")),
        ("zenz-v2.5-small",
            lambda: ZenzV2Backend("references/zenz-v2.5-small", device=device,
                                  num_beams=5, num_return=5, name_suffix="-b5")),
        ("ar_v3_vast-beam5",
            lambda: ARCheckpointBackend(
                "models/checkpoints/ar-31m-scratch/best.pt", device=device, beam_width=5)),
        ("ctc_nat_90m-step27500",
            lambda: _ctc("models/checkpoints/ctc-nat-90m-scratch/checkpoint_step_27500.pt")),
        ("ctc_nat_30m-best",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-scratch/best.pt")),
        ("ctc_nat_30m-step16000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-scratch/checkpoint_step_16000.pt")),
        ("ctc_nat_30m-final",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-scratch/final.pt")),
        ("ctc-nat-30m-student-best",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/best.pt")),
        ("ctc-nat-30m-student-final",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/final.pt")),
        ("ctc-nat-30m-student-step120000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_120000.pt")),
        ("ctc-nat-30m-student-step125000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_125000.pt")),
        ("ctc-nat-30m-student-step130000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_130000.pt")),
        ("ctc-nat-30m-student-step135000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_135000.pt")),
        ("ctc-nat-30m-student-step140000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_140000.pt")),
        ("ctc-nat-30m-student-step160000",
            lambda: _ctc("models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt")),
        ("teacher-150m-teacher-step200000",
            lambda: TeacherBackend(
                "models/checkpoints/teacher-150m-teacher/checkpoint_step_200000.pt",
                device=device)),
    ]
    if filter_substr:
        wanted = [w.strip() for w in filter_substr.split(",") if w.strip()]
        pool = [(n, f) for n, f in pool if any(w in n for w in wanted)]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="datasets/eval/probe/probe.json")
    ap.add_argument("--out-dir", default="results/probe")
    ap.add_argument("--models", default="", help="comma list filter")
    ap.add_argument("--lm-path", default="", help="KenLM .bin path")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    items = load_probe(args.probe)
    cats = sorted({i["category"] for i in items})
    with_ctx = sum(1 for i in items if i["context"])
    print(f"probe: {len(items)} items, {len(cats)} categories, {with_ctx} with context", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(args.models, lm_path=args.lm_path,
                          lm_alpha=args.alpha, lm_beta=args.beta,
                          beam_width=args.beam)
    summary: dict[str, dict] = {}

    for name, factory in models:
        print(f"\n=== {name} ===", flush=True)
        try:
            t0 = time.perf_counter()
            backend = factory()
            print(f"  load: {time.perf_counter()-t0:.1f}s", flush=True)
            res = evaluate(backend, items)
        except Exception as e:
            print(f"  !! error: {e}", flush=True)
            summary[name] = {"error": str(e)}
            continue

        em = res.get("exact_match_top1", 0)
        ca = res.get("char_acc_top1", 0)
        em5 = res["em5"]
        p50 = res["latency_ms"]["p50"]
        print(f"  EM1={em:.3f}  EM5={em5:.3f}  CharAcc={ca:.3f}  p50={p50}ms", flush=True)
        for cat, d in res["per_category"].items():
            cem = d.get("exact_match_top1", 0)
            cca = d.get("char_acc_top1", 0)
            cem5 = d.get("em5", 0)
            print(f"    {cat:<10} EM1={cem:.2f} EM5={cem5:.2f} CharAcc={cca:.2f} (n={d['n']})", flush=True)
        summary[name] = {
            "em1": em, "em5": em5, "char_acc": ca,
            "per_category": res["per_category"],
            "latency_ms": res["latency_ms"],
        }

        (out_dir / f"{_safe(name)}.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

        del backend
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Comparison table
    print("\n\n========= COMPARISON =========")
    header = ["model", "EM1", "EM5", "CharAcc"] + [f"{c}/EM1" for c in cats] + ["p50ms"]
    print(" | ".join(f"{h:<20}" for h in header))
    for name, r in summary.items():
        if "error" in r:
            print(f"{name:<20} | error: {r['error']}")
            continue
        row = [name, f"{r['em1']:.3f}", f"{r['em5']:.3f}", f"{r['char_acc']:.3f}"]
        for c in cats:
            v = r["per_category"].get(c, {}).get("exact_match_top1", "-")
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        row.append(str(r["latency_ms"]["p50"]))
        print(" | ".join(f"{c:<20}" for c in row))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_dir}/summary.json")


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


if __name__ == "__main__":
    main()
