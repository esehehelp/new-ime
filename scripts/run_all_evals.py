"""Unified evaluation runner: zenz-v2.5-medium + own AR checkpoints on multiple benches.

Outputs per-(model, bench) JSON in results/eval_runs/, plus a final comparison table.

Usage:
    uv run python -m scripts.run_all_evals
    uv run python -m scripts.run_all_evals --manual 100 --ajimee 80 --evalv3 80
    uv run python -m scripts.run_all_evals --skip-beam
    uv run python -m scripts.run_all_evals --models zenz,ar_vast_greedy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from src.eval.ar_backend import ARCheckpointBackend
from src.eval.bench_loaders import (
    load_ajimee_jwtd,
    load_eval_v3,
    load_manual_test,
    sample_items,
)
from src.eval.ctc_nat_backend import CTCNATBackend
from src.eval.metrics import EvalResult
from src.eval.zenz_backend import ZenzV2Backend


def evaluate(backend, items: list[dict], verbose_every: int = 25) -> dict:
    """Run a backend over items; returns summary with per-source breakdown."""
    result = EvalResult()
    by_source: dict[str, EvalResult] = {}
    latencies: list[float] = []
    fails: list[dict] = []

    t_start = time.perf_counter()
    for i, item in enumerate(items):
        reading = item["reading"]
        ctx = item["context"]
        refs = item["references"]
        src = item["source"]
        t0 = time.perf_counter()
        cands = backend.convert(reading, ctx)
        latencies.append((time.perf_counter() - t0) * 1000)
        result.add_multi(refs, cands)
        by_source.setdefault(src, EvalResult()).add_multi(refs, cands)

        if cands and cands[0] not in refs and len(fails) < 12:
            fails.append({"reading": reading[:30], "ref": refs[0][:30], "pred": cands[0][:30]})

        if (i + 1) % verbose_every == 0 or i + 1 == len(items):
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(items) - i - 1) / rate if rate else 0
            print(
                f"    [{i+1}/{len(items)}] elapsed={elapsed:.0f}s "
                f"rate={rate:.2f}/s eta={eta:.0f}s",
                flush=True,
            )

    summary = result.summary()
    summary["backend"] = backend.name
    summary["total_time_s"] = round(time.perf_counter() - t_start, 1)

    latencies.sort()
    n = len(latencies)
    summary["latency"] = {
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "mean_ms": round(sum(latencies) / n, 1),
    }
    summary["per_source"] = {s: r.summary() for s, r in by_source.items()}
    summary["sample_failures"] = fails
    return summary


def run_one(backend, name: str, benches: dict[str, list[dict]], out_dir: Path) -> dict:
    print(f"\n=== {name} ===", flush=True)
    out: dict[str, dict] = {}
    for bench_name, items in benches.items():
        print(f"  -> {bench_name} ({len(items)} samples)", flush=True)
        try:
            summary = evaluate(backend, items)
        except Exception as e:
            print(f"  !! error in {bench_name}: {e}", flush=True)
            summary = {"error": str(e)}
        out[bench_name] = summary
        path = out_dir / f"{_safe(name)}__{bench_name}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def print_table(all_results: dict[str, dict[str, dict]]) -> None:
    print("\n\n========== COMPARISON TABLE ==========")
    bench_names = sorted({b for r in all_results.values() for b in r})
    headers = ["model"]
    for b in bench_names:
        headers += [f"{b}/EM", f"{b}/CharAcc", f"{b}/p50ms"]
    print(" | ".join(f"{h:<22}" for h in headers))
    print("-" * (24 * len(headers)))
    for name, runs in all_results.items():
        row = [name]
        for b in bench_names:
            r = runs.get(b, {})
            em = r.get("exact_match_top1", "-")
            ca = r.get("char_acc_top1", "-")
            lat = (r.get("latency") or {}).get("p50_ms", "-")
            row += [
                f"{em:.4f}" if isinstance(em, float) else str(em),
                f"{ca:.4f}" if isinstance(ca, float) else str(ca),
                f"{lat}",
            ]
        print(" | ".join(f"{c:<22}" for c in row))


def build_models(args) -> list[tuple[str, callable]]:
    """Returns list of (name, factory) so models load lazily."""
    pool = [
        (
            "zenz-v2.5-medium-greedy",
            lambda: ZenzV2Backend("references/zenz-v2.5-medium"),
        ),
        (
            "zenz-v2.5-small-greedy",
            lambda: ZenzV2Backend("references/zenz-v2.5-small"),
        ),
        (
            "zenz-v2.5-xsmall-greedy",
            lambda: ZenzV2Backend("references/zenz-v2.5-xsmall"),
        ),
        (
            "ar_v3_local-greedy",
            lambda: ARCheckpointBackend(
                "checkpoints/ar_v3_local/best.pt", beam_width=1
            ),
        ),
        (
            "ar_v3_vast-greedy",
            lambda: ARCheckpointBackend(
                "checkpoints/ar_v3_vast/checkpoint_step_70000.pt", beam_width=1
            ),
        ),
        (
            "ar_v3_chunks-greedy",
            lambda: ARCheckpointBackend(
                "checkpoints/ar_v3_chunks/best.pt", beam_width=1
            ),
        ),
        (
            "ctc_nat_20m_fast_kd-step8000-greedy",
            lambda: CTCNATBackend(
                "checkpoints/ctc_nat_local_20m_fast_kd/checkpoint_step_8000.pt"
            ),
        ),
        (
            "ctc_nat_20m_fast_kd-final-greedy",
            lambda: CTCNATBackend(
                "checkpoints/ctc_nat_local_20m_fast_kd/final.pt"
            ),
        ),
        (
            "ctc_nat_30m_fast_kd-best-greedy",
            lambda: CTCNATBackend(
                "checkpoints/ctc_nat_local_30m_fast_kd/best.pt"
            ),
        ),
        (
            "ctc_nat_30m_fast_kd-step12000-greedy",
            lambda: CTCNATBackend(
                "checkpoints/ctc_nat_local_30m_fast_kd/checkpoint_step_12000.pt"
            ),
        ),
        (
            "ctc_nat_20m_vocab4k-best-greedy",
            lambda: CTCNATBackend(
                "checkpoints/ctc_nat_local_20m_vocab4k/best.pt"
            ),
        ),
        (
            "ctc_nat_90m_step5000-greedy",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_5000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            ),
        ),
        (
            "ctc_nat_90m_step10000-greedy",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_10000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            ),
        ),
        (
            "ctc_nat_90m_step15000-greedy",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
            ),
        ),
        (
            "ctc_nat_90m_step15000-beam8",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                beam_width=8,
            ),
        ),
        (
            "ctc_nat_90m_step15000-chunk16x8",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                chunk_threshold=16,
                chunk_size=8,
            ),
        ),
        (
            "ctc_nat_90m_step15000-beam8+chunk",
            lambda: CTCNATBackend(
                "checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt",
                device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                beam_width=8,
                chunk_threshold=16,
                chunk_size=8,
            ),
        ),
    ]
    if not args.skip_beam:
        pool += [
            (
                "ar_v3_local-beam10",
                lambda: ARCheckpointBackend(
                    "checkpoints/ar_v3_local/best.pt", beam_width=10
                ),
            ),
            (
                "ar_v3_vast-beam10",
                lambda: ARCheckpointBackend(
                    "checkpoints/ar_v3_vast/checkpoint_step_70000.pt", beam_width=10
                ),
            ),
            (
                "ar_v3_chunks-beam10",
                lambda: ARCheckpointBackend(
                    "checkpoints/ar_v3_chunks/best.pt", beam_width=10
                ),
            ),
        ]
    if args.models:
        wanted = set(args.models.split(","))
        pool = [(n, f) for n, f in pool if any(w in n for w in wanted)]
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=int, default=100, help="manual test sample count (max 100)")
    parser.add_argument("--ajimee", type=int, default=80, help="AJIMEE-Bench sample count")
    parser.add_argument("--evalv3", type=int, default=80, help="eval_v3/dev sample count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ajimee-path",
        default="references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json",
    )
    parser.add_argument("--evalv3-path", default="datasets/eval_v3/dev.jsonl")
    parser.add_argument("--out-dir", default="results/eval_runs")
    parser.add_argument("--skip-beam", action="store_true", help="skip beam-search runs")
    parser.add_argument("--models", default="", help="comma list to filter model names")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benches...", flush=True)
    manual = load_manual_test()[: args.manual] if args.manual > 0 else []
    ajimee = sample_items(load_ajimee_jwtd(args.ajimee_path), args.ajimee, args.seed)
    evalv3 = sample_items(load_eval_v3(args.evalv3_path), args.evalv3, args.seed)
    benches: dict[str, list[dict]] = {}
    if manual:
        benches["manual_test"] = manual
    if ajimee:
        benches["ajimee_jwtd"] = ajimee
    if evalv3:
        benches["eval_v3_dev"] = evalv3
    print(
        f"  manual={len(manual)} ajimee={len(ajimee)} eval_v3={len(evalv3)} "
        f"(total {sum(len(v) for v in benches.values())} per model)",
        flush=True,
    )

    pool = build_models(args)
    print(f"Models to run ({len(pool)}): {', '.join(n for n, _ in pool)}", flush=True)

    all_results: dict[str, dict[str, dict]] = {}
    overall_start = time.perf_counter()
    for name, factory in pool:
        try:
            t = time.perf_counter()
            backend = factory()
            print(f"\n[load] {name}: {time.perf_counter()-t:.1f}s", flush=True)
            all_results[name] = run_one(backend, name, benches, out_dir)
            del backend
        except Exception as e:
            print(f"!! failed to load {name}: {e}", flush=True)
            all_results[name] = {}

    print_table(all_results)
    total = time.perf_counter() - overall_start
    print(f"\nTotal time: {total:.0f}s ({total/60:.1f} min)")

    # Save aggregated results
    agg_path = out_dir / "summary.json"
    agg_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Summary saved: {agg_path}")


if __name__ == "__main__":
    main()
