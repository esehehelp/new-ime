"""CTC-NAT + KenLM α/β sweep across manual/ajimee/general benches.

Mirrors `sweep_probe_v2_kenlm.sh` but targets the long-form benches instead
of the 467-item probe_v2. Used to check whether α/β optima on phrase-level
probe_v2 carry over to sentence-level benches (they often don't).

Usage (WSL, since KenLM isn't installed in the Windows venv):
    wsl -- bash -c "cd /mnt/d/Dev/new-ime && python3 -m models.tools.eval.sweep_ctc_kenlm_benches \\
        --ckpt models/checkpoints/ctc-nat-30m-student/checkpoint_step_49000.pt \\
        --lm models/kenlm/kenlm_general_train_4gram_probing.bin \\
        --out-dir results/eval_runs_30mv2_kenlm_sweep"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from models.src.eval.bench_loaders import (
    load_ajimee_jwtd,
    load_general,
    load_manual_test,
    sample_items,
)
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult


def evaluate(backend, items: list[dict]) -> dict:
    result = EvalResult()
    latencies: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        cands = backend.convert(item["reading"], item["context"])
        latencies.append((time.perf_counter() - t0) * 1000)
        result.add_multi(item["references"], cands)
    s = result.summary()
    latencies.sort()
    n = len(latencies)
    s["latency"] = {
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "mean_ms": round(sum(latencies) / n, 1),
    }
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--alphas", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--betas", default="0.0,0.3,0.6")
    ap.add_argument("--manual", type=int, default=100)
    ap.add_argument("--ajimee", type=int, default=80)
    ap.add_argument("--general", type=int, default=80)
    ap.add_argument(
        "--ajimee-path",
        default="references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json",
    )
    ap.add_argument("--general-path", default="datasets/eval/general/dev.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benches...", flush=True)
    benches: dict[str, list[dict]] = {}
    if args.manual > 0:
        benches["manual_test"] = load_manual_test()[: args.manual]
    if args.ajimee > 0:
        benches["ajimee_jwtd"] = sample_items(
            load_ajimee_jwtd(args.ajimee_path), args.ajimee, args.seed
        )
    if args.general > 0:
        benches["general_dev"] = sample_items(
            load_general(args.general_path), args.general, args.seed
        )
    for k, v in benches.items():
        print(f"  {k}: {len(v)}", flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Greedy baseline (α=β=0 explicit entry skipped; beam=1 no-LM).
    configs: list[tuple[str, dict]] = [
        ("greedy_nolm", {"beam_width": 1}),
        (f"beam{args.beam}_nolm", {"beam_width": args.beam}),
    ]
    for a in alphas:
        for b in betas:
            if a == 0.0 and b == 0.0:
                continue
            tag = f"a{a}_b{b}"
            configs.append((tag, {
                "beam_width": args.beam,
                "lm_path": args.lm,
                "lm_alpha": a,
                "lm_beta": b,
            }))

    all_results: dict[str, dict[str, dict]] = {}
    t0_all = time.perf_counter()
    for tag, kwargs in configs:
        print(f"\n=== {tag} ===", flush=True)
        tload = time.perf_counter()
        backend = CTCNATBackend(args.ckpt, device=device, **kwargs)
        print(f"  load: {time.perf_counter()-tload:.1f}s", flush=True)

        per_bench: dict[str, dict] = {}
        for bname, items in benches.items():
            tb = time.perf_counter()
            s = evaluate(backend, items)
            elapsed = time.perf_counter() - tb
            em = s.get("exact_match_top1", 0)
            ca = s.get("char_acc_top1", 0)
            p50 = s["latency"]["p50_ms"]
            print(
                f"  {bname}: EM={em:.4f} CharAcc={ca:.4f} p50={p50}ms  ({elapsed:.0f}s)",
                flush=True,
            )
            per_bench[bname] = s
        all_results[tag] = per_bench
        del backend

    (out_dir / "summary.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTotal: {time.perf_counter()-t0_all:.0f}s", flush=True)
    print(f"Saved: {out_dir}/summary.json", flush=True)

    # Compact table
    print("\n=== TABLE (EM / CharAcc) ===")
    benches_order = list(benches.keys())
    header = ["tag"] + [f"{b}/EM" for b in benches_order] + [f"{b}/CA" for b in benches_order] + ["p50(long)"]
    print(" | ".join(f"{h:<20}" for h in header))
    for tag, runs in all_results.items():
        row = [tag]
        for b in benches_order:
            em = runs[b].get("exact_match_top1", 0)
            row.append(f"{em:.4f}")
        for b in benches_order:
            ca = runs[b].get("char_acc_top1", 0)
            row.append(f"{ca:.4f}")
        longest = max(benches_order, key=lambda b: runs[b]["latency"]["p50_ms"])
        row.append(f"{runs[longest]['latency']['p50_ms']}")
        print(" | ".join(f"{c:<20}" for c in row))


if __name__ == "__main__":
    main()
