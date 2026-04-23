"""Latency benchmark for CTC-NAT checkpoints over the manual 100 cases.

Usage:
    uv run python -m scripts.bench_ctc_nat_speed \
        --checkpoint checkpoints/ctc_nat_local_20m_fast_kd/checkpoint_step_8000.pt \
        [--device cuda|cpu] [--warmup 10] [--runs 3]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from scripts.manual.manual_test import TEST_CASES
from scripts.manual.manual_test_ctc_nat import build_model_from_checkpoint, predict
from models.src.data.tokenizer import SharedCharTokenizer
from models.src.model.ctc_nat import CTCNAT, PRESETS


def build_model_lenient(ckpt_path: Path, device: torch.device):
    """Like build_model_from_checkpoint, but fall back to tokenizer sidecar
    when ckpt['vocab_size'] is missing/None."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preset_name = ckpt.get("preset")
    if preset_name not in PRESETS:
        raise SystemExit(f"unknown preset in checkpoint: {preset_name!r}")
    tokenizer_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer sidecar: {tokenizer_path}")
    tokenizer = SharedCharTokenizer.load(str(tokenizer_path))
    vocab_size = ckpt.get("vocab_size")
    vocab_size = int(vocab_size) if vocab_size else tokenizer.vocab_size
    model = CTCNAT.from_preset(
        preset_name,
        vocab_size=vocab_size,
        use_cvae=bool(ckpt.get("use_cvae", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, tokenizer, ckpt

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3, help="full passes over 100 cases")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    model, tokenizer, ckpt = build_model_lenient(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {ckpt_path.name}")
    print(f"preset={ckpt.get('preset')}  vocab={tokenizer.vocab_size}  device={device}")
    print(f"params={n_params/1e6:.2f}M")

    cases = TEST_CASES
    is_cuda = device.type == "cuda"

    # warmup
    for i in range(args.warmup):
        ctx, reading, _ = cases[i % len(cases)]
        _ = predict(model, tokenizer, ctx, reading, device)
    if is_cuda:
        torch.cuda.synchronize()

    per_case_all: list[float] = []
    pass_totals: list[float] = []
    for r in range(args.runs):
        if is_cuda:
            torch.cuda.synchronize()
        t0_pass = time.perf_counter()
        per_case: list[float] = []
        for ctx, reading, _ in cases:
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = predict(model, tokenizer, ctx, reading, device)
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
    mean = statistics.mean(per_case_all)
    median = statistics.median(per_case_all)
    p90 = sorted_lat[int(0.90 * n) - 1]
    p95 = sorted_lat[int(0.95 * n) - 1]
    p99 = sorted_lat[int(0.99 * n) - 1]
    total = sum(pass_totals)
    throughput = (args.runs * len(cases)) / total

    print()
    print(f"samples      : {n}  ({args.runs} runs x {len(cases)} cases)")
    print(f"mean         : {mean:.2f} ms")
    print(f"median       : {median:.2f} ms")
    print(f"p90 / p95/p99: {p90:.2f} / {p95:.2f} / {p99:.2f} ms")
    print(f"min / max    : {min(per_case_all):.2f} / {max(per_case_all):.2f} ms")
    print(f"throughput   : {throughput:.1f} req/s")


if __name__ == "__main__":
    main()
