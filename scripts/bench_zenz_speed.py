"""Latency benchmark for zenz-v2.5 (HuggingFace) over manual 100 cases.

Usage:
    uv run python -m scripts.bench_zenz_speed --model references/zenz-v2.5-small
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.manual_test import TEST_CASES
from src.eval.zenz_backend import ZenzV2Backend

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to zenz-v2.5-{xsmall,small,medium}")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backend = ZenzV2Backend(args.model, device=device, num_beams=1)
    n = sum(p.numel() for p in backend.model.parameters())
    print(f"model: {args.model}  device={device}  params={n/1e6:.2f}M")

    is_cuda = device == "cuda"

    for i in range(args.warmup):
        ctx, reading, _ = TEST_CASES[i % len(TEST_CASES)]
        _ = backend.convert(reading, ctx)
    if is_cuda:
        torch.cuda.synchronize()

    per_case_all: list[float] = []
    pass_totals: list[float] = []
    for r in range(args.runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0_pass = time.perf_counter()
        per_case: list[float] = []
        for ctx, reading, _ in TEST_CASES:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = backend.convert(reading, ctx)
            if is_cuda:
                torch.cuda.synchronize()
            per_case.append((time.perf_counter() - t0) * 1000.0)
        pass_totals.append(time.perf_counter() - t0_pass)
        per_case_all.extend(per_case)
        print(
            f"  run {r+1}: total={pass_totals[-1]*1000:.1f}ms  "
            f"mean={statistics.mean(per_case):.2f}ms  "
            f"median={statistics.median(per_case):.2f}ms  "
            f"p95={sorted(per_case)[int(0.95*len(per_case))-1]:.2f}ms  "
            f"max={max(per_case):.2f}ms"
        )

    n = len(per_case_all)
    sorted_lat = sorted(per_case_all)
    print()
    print(f"samples      : {n}  ({args.runs} runs x {len(TEST_CASES)} cases)")
    print(f"mean         : {statistics.mean(per_case_all):.2f} ms")
    print(f"median       : {statistics.median(per_case_all):.2f} ms")
    print(f"p90 / p95/p99: {sorted_lat[int(0.9*n)-1]:.2f} / {sorted_lat[int(0.95*n)-1]:.2f} / {sorted_lat[int(0.99*n)-1]:.2f} ms")
    print(f"min / max    : {min(per_case_all):.2f} / {max(per_case_all):.2f} ms")
    print(f"throughput   : {(args.runs * len(TEST_CASES)) / sum(pass_totals):.1f} req/s")


if __name__ == "__main__":
    main()
