"""Canonical benchmark per docs/benchmark_comparison.md.

Runs a checkpoint on the two canonical inputs (probe.json +
JWTD_v2/evaluation_items.json) with beams=5/return=5 on CPU and reports
EM1 / EM5 / CharAcc / p50 latency. Dumps per-bench JSON next to the
existing results/bench_all/ layout.

Usage (from repo root):
    uv run --project legacy/python python scripts/bench_checkpoint.py \\
        --checkpoint models/checkpoints/Suiko-v1.1-small/checkpoint_step_50000.pt \\
        --tag Suiko-v1.1-small-step50k

This is a minimal driver; the full rust-bench harness
(`cargo run -p rust-bench --features native-tch`) does the same thing
against any Rust-produced run-dir. For Python-pickled Suiko checkpoints
we go through `CTCNATBackend` directly.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "legacy" / "python"))

from models.src.eval.bench_loaders import load_ajimee_jwtd, load_probe  # noqa: E402
from models.src.eval.ctc_nat_backend import CTCNATBackend  # noqa: E402
from models.src.eval.metrics import EvalResult, character_accuracy  # noqa: E402


def run_bench(
    backend: CTCNATBackend,
    items: list[dict],
    num_return: int,
    label: str,
) -> tuple[dict, list[dict]]:
    result = EvalResult()
    latencies_ms: list[float] = []
    per_category: dict[str, EvalResult] = {}
    predictions: list[dict] = []
    for i, item in enumerate(items, 1):
        reading = item["reading"]
        context = item.get("context", "")
        references = item["references"]
        ref = references[0]
        cat = item.get("category", "general")

        t0 = time.perf_counter()
        candidates = backend.convert(reading, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        cand_list = list(candidates)[:num_return]
        top1 = cand_list[0] if cand_list else ""
        result.add(ref, cand_list)
        per_category.setdefault(cat, EvalResult()).add(ref, cand_list)

        predictions.append(
            {
                "index": item.get("_index", i),
                "reading": reading,
                "reference": ref,
                "top1": top1,
                "candidates": cand_list,
                "elapsed_ms": round(elapsed_ms, 2),
                "category": cat,
                "correct_top1": top1 == ref,
            }
        )
        if i % 50 == 0:
            print(f"  [{label}] {i}/{len(items)}  p50={statistics.median(latencies_ms):.1f}ms", flush=True)

    summary = result.summary()
    report = {
        "label": label,
        "total": len(items),
        "em1": summary.get("exact_match_top1", 0.0),
        "em5": summary.get(f"exact_match_top{num_return}", 0.0),
        "char_acc": summary.get("char_acc_top1", 0.0),
        "latency_ms": {
            "p50": round(statistics.median(latencies_ms), 2),
            "p95": round(sorted(latencies_ms)[int(0.95 * len(latencies_ms))], 2)
            if latencies_ms
            else 0.0,
            "mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        },
        "per_category": {
            cat: {
                "n": r.total,
                "em1": r.summary().get("exact_match_top1", 0.0),
                "em5": r.summary().get(f"exact_match_top{num_return}", 0.0),
                "char_acc": r.summary().get("char_acc_top1", 0.0),
            }
            for cat, r in sorted(per_category.items())
        },
    }
    return report, predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument(
        "--probe-path",
        default=str(REPO / "datasets" / "eval" / "probe" / "probe.json"),
    )
    ap.add_argument(
        "--ajimee-path",
        default=str(
            REPO
            / "references"
            / "AJIMEE-Bench"
            / "JWTD_v2"
            / "v1"
            / "evaluation_items.json"
        ),
    )
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--num-return", type=int, default=5)
    ap.add_argument(
        "--out-dir",
        default=str(REPO / "results" / "bench_all"),
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"loading checkpoint: {ckpt_path}  (CPU, beams={args.num_beams}, return={args.num_return})")
    t0 = time.time()
    backend = CTCNATBackend(
        checkpoint_path=str(ckpt_path),
        device="cpu",
        beam_width=args.num_beams,
        beam_top_k=max(16, args.num_beams),
    )
    print(f"  loaded in {time.time() - t0:.1f}s; step={backend.step}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for bench_name, loader, path in [
        ("probe", load_probe, args.probe_path),
        ("ajimee", load_ajimee_jwtd, args.ajimee_path),
    ]:
        print(f"\n== {bench_name}  ({path}) ==")
        items = loader(path)
        print(f"  {len(items)} items")
        report, predictions = run_bench(
            backend, items, num_return=args.num_return, label=bench_name
        )
        stem = f"{args.tag}__beam{args.num_beams}__{bench_name}"
        (out_dir / f"{stem}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"{stem}_predictions.jsonl").write_text(
            "\n".join(
                json.dumps(p, ensure_ascii=False) for p in predictions
            ),
            encoding="utf-8",
        )
        summary_rows.append((bench_name, report))

    print("\n=== Summary ===")
    print(
        f"{'bench':<10} {'n':>5} {'EM1':>7} {'EM5':>7} {'CharAcc':>8} {'p50_ms':>8}"
    )
    for bench, r in summary_rows:
        print(
            f"{bench:<10} {r['total']:>5} "
            f"{r['em1']:>7.4f} {r['em5']:>7.4f} "
            f"{r['char_acc']:>8.4f} "
            f"{r['latency_ms']['p50']:>8.1f}"
        )


if __name__ == "__main__":
    main()
