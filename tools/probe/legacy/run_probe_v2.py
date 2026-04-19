"""4-way eval on probe_v2 (phrase-level): zenz-v3.1, zenz-v2.5, AR, CTC-NAT.

Reads datasets/eval/probe_v2.tsv (category, reading, expected_surface) and
runs each backend with top-5 candidates. Per-category EM / top-5 inclusion
are reported so we can see where each model is strong/weak at bunsetsu
level — this informs the corpus-v2 / phase3 integration design.

Usage:
    uv run python -m scripts.run_probe_v2
    uv run python -m scripts.run_probe_v2 --models zenz-v3.1,ctc_nat_90m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.ar_backend import ARCheckpointBackend
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.teacher_backend import TeacherBackend
from models.src.eval.zenz_backend import ZenzV2Backend


def load_probe(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        items.append({"category": parts[0], "reading": parts[1], "expected": parts[2]})
    return items


def evaluate(backend, items: list[dict], top_k: int = 5) -> dict:
    per_cat: dict[str, dict] = {}
    results = []
    latencies = []
    for item in items:
        t0 = time.perf_counter()
        try:
            cands = backend.convert(item["reading"], "")
        except Exception as e:
            cands = [f"<error:{e}>"]
        latencies.append((time.perf_counter() - t0) * 1000)

        cands = cands[:top_k] if cands else ["<empty>"]
        top1 = cands[0] if cands else ""
        em1 = 1 if top1 == item["expected"] else 0
        emk = 1 if item["expected"] in cands else 0

        cat = item["category"]
        if cat not in per_cat:
            per_cat[cat] = {"n": 0, "em1": 0, "emk": 0}
        per_cat[cat]["n"] += 1
        per_cat[cat]["em1"] += em1
        per_cat[cat]["emk"] += emk

        results.append({
            "category": cat,
            "reading": item["reading"],
            "expected": item["expected"],
            "top1": top1,
            "top5": cands,
            "em1": em1,
            "emk": emk,
        })

    overall_em1 = sum(r["em1"] for r in results) / len(results)
    overall_emk = sum(r["emk"] for r in results) / len(results)
    for c, d in per_cat.items():
        d["em1_rate"] = round(d["em1"] / d["n"], 3)
        d["emk_rate"] = round(d["emk"] / d["n"], 3)

    latencies.sort()
    n = len(latencies)
    return {
        "backend": backend.name,
        "n": len(results),
        "em1": round(overall_em1, 3),
        "em5": round(overall_emk, 3),
        "per_category": per_cat,
        "latency_ms": {
            "p50": round(latencies[n // 2], 1),
            "p95": round(latencies[int(n * 0.95)], 1),
            "mean": round(sum(latencies) / n, 1),
        },
        "results": results,
    }


def build_models(filter_substr: str = "",
                 lm_path: str = "",
                 lm_alpha: float = 0.0,
                 lm_beta: float = 0.0,
                 beam_width: int = 1) -> list[tuple[str, callable]]:
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

    pool = [
        ("zenz-v3.1-small",
            lambda: ZenzV2Backend("references/zenz-v3.1-small", device=device,
                                  num_beams=5, num_return=5, name_suffix="-b5")),
        ("zenz-v2.5-small",
            lambda: ZenzV2Backend("references/zenz-v2.5-small", device=device,
                                  num_beams=5, num_return=5, name_suffix="-b5")),
        ("ar_v3_vast-beam5",
            lambda: ARCheckpointBackend(
                "models/checkpoints/ar_v3_vast/best.pt", device=device, beam_width=5)),
        ("ctc_nat_30m-best",
            lambda: _ctc("models/checkpoints/ctc_nat_30m/best.pt")),
        ("ctc_nat_30m-step50000",
            lambda: _ctc("models/checkpoints/ctc_nat_30m/checkpoint_step_50000.pt")),
        ("ctc_nat_30m-final",
            lambda: _ctc("models/checkpoints/ctc_nat_30m/final.pt")),
        ("ctc_nat_90m-step27500",
            lambda: _ctc("models/checkpoints/ctc_nat_90m/checkpoint_step_27500.pt")),
        ("ctc_nat_30m_v2-step49000",
            lambda: _ctc("models/checkpoints/ctc_nat_30m_v2_dryrun/checkpoint_step_49000.pt")),
        ("teacher-v1-150m-step100000",
            lambda: TeacherBackend(
                "models/checkpoints/teacher-v1-150m/checkpoint_step_100000.pt",
                device=device)),
    ]
    if filter_substr:
        wanted = [w.strip() for w in filter_substr.split(",") if w.strip()]
        pool = [(n, f) for n, f in pool if any(w in n for w in wanted)]
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="datasets/eval/probe_v2.tsv")
    ap.add_argument("--out-dir", default="results/probe_v2")
    ap.add_argument("--models", default="", help="comma list filter")
    ap.add_argument("--lm-path", default="", help="KenLM .bin path")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    items = load_probe(Path(args.probe))
    print(f"Probe: {len(items)} items across "
          f"{len({i['category'] for i in items})} categories", flush=True)

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

        print(f"  EM1={res['em1']:.3f}  EM5={res['em5']:.3f}  "
              f"p50={res['latency_ms']['p50']}ms", flush=True)
        for cat, d in sorted(res["per_category"].items()):
            print(f"    {cat:<10} EM1={d['em1_rate']:.2f} "
                  f"EM5={d['emk_rate']:.2f} (n={d['n']})", flush=True)
        summary[name] = {
            "em1": res["em1"], "em5": res["em5"],
            "per_category": res["per_category"],
            "latency_ms": res["latency_ms"],
        }

        (out_dir / f"{name}.json").write_text(
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
    cats = sorted({c for r in summary.values() if "per_category" in r
                   for c in r["per_category"]})
    header = ["model", "EM1", "EM5"] + [f"{c}/EM1" for c in cats] + ["p50ms"]
    print(" | ".join(f"{h:<18}" for h in header))
    print("-" * (20 * len(header)))
    for name, r in summary.items():
        if "error" in r:
            print(f"{name:<18} | error: {r['error']}")
            continue
        row = [name, f"{r['em1']:.3f}", f"{r['em5']:.3f}"]
        for c in cats:
            v = r["per_category"].get(c, {}).get("em1_rate", "-")
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        row.append(str(r["latency_ms"]["p50"]))
        print(" | ".join(f"{c:<18}" for c in row))

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out_dir}/summary.json")


if __name__ == "__main__":
    main()
