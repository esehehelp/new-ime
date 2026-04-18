"""CPU fusion bench for CTC-NAT + KenLM shallow fusion.

Run in an env where `kenlm` and `torch` import cleanly (on this project that
is WSL — the Windows `kenlm` sdist fails to build its Cython wrapper). On
WSL the repo lives at /mnt/d/Dev/new-ime; this script is cwd-agnostic.

    python3 scripts/bench_ctc_nat_fusion.py \
        --checkpoint checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt \
        --lm-path /home/$USER/kenlm_work/model.bin \
        --alpha 0.3 --beta 0.5 --beam 8 \
        --bench manual,ajimee,evalv3

Emits a per-(lm config, bench) summary. No result JSON is written; the
caller is expected to capture stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.src.eval.bench_loaders import (
    load_ajimee_jwtd,
    load_eval_v3,
    load_manual_test,
    sample_items,
)
from models.src.eval.ctc_nat_backend import CTCNATBackend
from models.src.eval.metrics import EvalResult


def evaluate(backend, items):
    result = EvalResult()
    latencies = []
    t_start = time.perf_counter()
    for item in items:
        t0 = time.perf_counter()
        cands = backend.convert(item["reading"], item["context"])
        latencies.append((time.perf_counter() - t0) * 1000)
        result.add_multi(item["references"], cands)
    summary = result.summary()
    summary["total_s"] = round(time.perf_counter() - t_start, 1)
    latencies.sort()
    n = len(latencies)
    summary["p50_ms"] = round(latencies[n // 2], 1)
    summary["p95_ms"] = round(latencies[int(n * 0.95)], 1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lm-path", required=True)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument(
        "--gate-confs",
        default="",
        help=(
            "comma list of low-conf gate thresholds to sweep. Negative values "
            "enable the gate: LM fires only when greedy mean top-1 logp is "
            "below the threshold. Empty string disables the gate entirely "
            "(LM fires on every call). Typical values: -1.0, -2.0, -3.0."
        ),
    )
    parser.add_argument(
        "--alphas",
        default="0.0,0.15,0.3,0.5",
        help="comma list of LM weights to sweep",
    )
    parser.add_argument(
        "--betas",
        default="0.0,0.5,1.0",
        help="comma list of length penalties to sweep",
    )
    parser.add_argument("--bench", default="manual,ajimee,evalv3")
    parser.add_argument("--manual", type=int, default=100)
    parser.add_argument("--ajimee", type=int, default=80)
    parser.add_argument("--evalv3", type=int, default=200)
    parser.add_argument(
        "--ajimee-path",
        default="references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json",
    )
    parser.add_argument("--evalv3-path", default="datasets/eval_v3/dev.jsonl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    gate_confs = [float(x) for x in args.gate_confs.split(",") if x.strip()]
    if not gate_confs:
        gate_confs = [0.0]  # 0.0 = gate disabled
    wanted = set(args.bench.split(","))

    benches = {}
    if "manual" in wanted:
        benches["manual"] = load_manual_test()[: args.manual]
    if "ajimee" in wanted:
        benches["ajimee"] = sample_items(load_ajimee_jwtd(args.ajimee_path), args.ajimee, 42)
    if "evalv3" in wanted:
        benches["evalv3"] = sample_items(load_eval_v3(args.evalv3_path), args.evalv3, 42)

    print(f"checkpoint: {args.checkpoint}")
    print(f"lm_path:    {args.lm_path}")
    print(f"beam:       {args.beam}")
    print(f"alphas:     {alphas}")
    print(f"betas:      {betas}")
    print(f"benches:    {list(benches.keys())}  sizes={[len(v) for v in benches.values()]}")
    print()

    rows = []
    for alpha in alphas:
        for beta in betas:
            for gate in gate_confs:
                use_lm = alpha != 0.0 or beta != 0.0
                backend = CTCNATBackend(
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    beam_width=args.beam,
                    lm_path=args.lm_path if use_lm else None,
                    lm_alpha=alpha,
                    lm_beta=beta,
                    lm_gate_min_conf=gate,
                )
                for bench_name, items in benches.items():
                    s = evaluate(backend, items)
                    row = {
                        "alpha": alpha,
                        "beta": beta,
                        "gate": gate,
                        "bench": bench_name,
                        "em": s.get("exact_match_top1"),
                        "char_acc": s.get("char_acc_top1"),
                        "p50_ms": s.get("p50_ms"),
                        "p95_ms": s.get("p95_ms"),
                        "n": len(items),
                    }
                    rows.append(row)
                    print(
                        f"  a={alpha:.2f} b={beta:.2f} g={gate:+.2f} {bench_name:<7} "
                        f"EM={row['em']:.4f} CharAcc={row['char_acc']:.4f} "
                        f"p50={row['p50_ms']}ms p95={row['p95_ms']}ms",
                        flush=True,
                    )
                del backend

    print()
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
