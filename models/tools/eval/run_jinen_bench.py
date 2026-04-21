from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from models.src.eval.bench_loaders import (
    load_ajimee_jwtd,
    load_probe,
)
from models.src.eval.jinen_backend import JinenV1Backend
from models.src.eval.metrics import EvalResult

sys.stdout.reconfigure(encoding="utf-8")


def evaluate(backend, items: list[dict], top_k: int = 5, verbose_every: int = 25) -> dict:
    result = EvalResult()
    per_cat: dict[str, EvalResult] = defaultdict(EvalResult)
    em5_hits: list[int] = []
    latencies: list[float] = []
    failures: list[dict] = []
    start = time.perf_counter()
    for i, item in enumerate(items):
        reading = item["reading"]
        context = item["context"]
        refs = item["references"]
        t0 = time.perf_counter()
        cands = backend.convert(reading, context)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        cands_k = cands[:top_k] if cands else []
        result.add_multi(refs, cands_k)
        cat = item.get("category")
        if cat:
            per_cat[cat].add_multi(refs, cands_k)
        em5_hits.append(int(any(c in refs for c in cands_k)))
        if cands_k and cands_k[0] not in refs and len(failures) < 12:
            failures.append(
                {
                    "reading": reading[:30],
                    "context": context[:30],
                    "ref": refs[0][:30],
                    "pred": cands_k[0][:30],
                    **({"cat": cat} if cat else {}),
                }
            )
        if (i + 1) % verbose_every == 0 or i + 1 == len(items):
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(items) - i - 1) / rate if rate else 0
            print(
                f"    [{i+1}/{len(items)}] elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                flush=True,
            )

    latencies.sort()
    summary = result.summary()
    n = len(latencies)
    summary["backend"] = backend.name
    summary["n"] = len(items)
    summary["em5"] = round(sum(em5_hits) / len(em5_hits), 4) if em5_hits else 0.0
    summary["latency"] = {
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "mean_ms": round(sum(latencies) / n, 1),
    }
    if per_cat:
        summary["per_category"] = {name: r.summary() for name, r in sorted(per_cat.items())}
    summary["sample_failures"] = failures
    summary["total_time_s"] = round(time.perf_counter() - start, 1)
    return summary


def load_benches(args: argparse.Namespace) -> dict[str, list[dict]]:
    benches: dict[str, list[dict]] = {}
    if args.probe:
        benches["probe_v3"] = load_probe(args.probe_path)
    if args.ajimee:
        benches["ajimee_jwtd"] = load_ajimee_jwtd(args.ajimee_path)
    return benches


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hugging Face jinen-v1 models.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe", action="store_true", default=True)
    parser.add_argument("--ajimee", action="store_true", default=True)
    parser.add_argument("--probe-path", default="datasets/eval/probe/probe.json")
    parser.add_argument(
        "--ajimee-path",
        default="references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json",
    )
    parser.add_argument("--out-dir", default="results/jinen_bench")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = args.device or "cpu"
    benches = load_benches(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict[str, dict]] = {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        backend = JinenV1Backend(
            model_name,
            device=device,
            num_beams=args.num_beams,
            num_return=args.num_return,
            max_new_tokens=args.max_new_tokens,
            torch_dtype=torch.float32,
        )
        model_results: dict[str, dict] = {}
        for bench_name, items in benches.items():
            print(f"  -> {bench_name} ({len(items)} samples)", flush=True)
            summary = evaluate(backend, items)
            model_results[bench_name] = summary
            out_path = out_dir / f"{model_name.split('/')[-1]}__{bench_name}.json"
            out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        table[model_name] = model_results

    print("\n=== Summary ===")
    for model_name, results in table.items():
        print(model_name)
        for bench_name, summary in results.items():
            print(
                f"  {bench_name}: EM1={summary['exact_match_top1']:.4f} "
                f"EM5={summary['em5']:.4f} "
                f"CharAcc={summary['char_acc_top1']:.4f} p50={summary['latency']['p50_ms']}ms"
            )


if __name__ == "__main__":
    main()
