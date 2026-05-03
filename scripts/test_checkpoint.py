"""Run the 100-case manual benchmark on a CTC-NAT checkpoint (CPU-only).

Self-contained replacement for `models/tools/manual/manual_test_ctc_nat.py`
whose `scripts.manual.*` import was broken by the legacy reshuffle. Forces
CPU so a running GPU training job isn't disturbed.

Usage (from legacy/python/):
    uv run python ../../scripts/test_checkpoint.py \\
        --checkpoint ../../models/checkpoints/Suiko-v1.1-small/checkpoint_step_50000.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "legacy" / "python"))

import torch  # noqa: E402

from models.src.data.tokenizer import SharedCharTokenizer  # noqa: E402
from models.src.eval.metrics import EvalResult  # noqa: E402
from models.src.model.ctc_nat import CTCNAT, PRESETS  # noqa: E402
from models.tools.manual.manual_test import TEST_CASES  # noqa: E402


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preset_name = ckpt.get("preset")
    if preset_name not in PRESETS:
        raise SystemExit(f"unknown preset in checkpoint: {preset_name!r}")
    vocab_size = int(ckpt["vocab_size"])
    tokenizer_path = Path(str(checkpoint_path).replace(".pt", "_tokenizer.json"))
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer sidecar: {tokenizer_path}")
    tokenizer = SharedCharTokenizer.load(str(tokenizer_path))
    if tokenizer.vocab_size != vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != checkpoint vocab {vocab_size}"
        )
    model = CTCNAT.from_preset(
        preset_name,
        vocab_size=vocab_size,
        use_cvae=bool(ckpt.get("use_cvae", False)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def predict(model, tokenizer, context, reading, device, max_context=40, max_seq_len=128):
    ids = tokenizer.encode_with_special(context[-max_context:], reading)
    ids = ids[:max_seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    decoded = model.greedy_decode(input_ids, attention_mask)
    return tokenizer.decode(decoded[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="default cpu to avoid interfering with concurrent training",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    t0 = time.time()
    model, tokenizer, ckpt = build_model_from_checkpoint(ckpt_path, device)
    load_s = time.time() - t0

    print(
        f"checkpoint: {ckpt_path.name}  step={ckpt.get('step', '?')}"
        f"  epoch={ckpt.get('epoch', '?')}"
        f"  best_metric={ckpt.get('best_metric', '?')}"
    )
    print(
        f"preset={ckpt.get('preset')}  vocab={tokenizer.vocab_size}"
        f"  device={device}  load={load_s:.1f}s"
    )

    cases = TEST_CASES if not args.max_cases else TEST_CASES[: args.max_cases]
    result = EvalResult()
    t_inf = time.time()
    for idx, (context, reading, expected) in enumerate(cases, 1):
        prediction = predict(model, tokenizer, context, reading, device)
        result.add(expected, [prediction])
        if not args.quiet:
            mark = "OK" if prediction == expected else "xx"
            ctx_disp = f"  [{context[-20:]}]" if context else ""
            print(
                f"  {idx:>3} {mark}  ref={expected}{ctx_disp}\n"
                f"       pred={prediction}"
            )
    inf_s = time.time() - t_inf

    summary = result.summary()
    em = summary.get("exact_match_top1", 0.0)
    char_acc = summary.get("char_acc_top1", 0.0)
    print()
    print(f"EM       : {em:.4f}  ({int(round(em * len(cases)))}/{len(cases)})")
    print(f"CharAcc  : {char_acc:.4f}")
    print(f"cases    : {len(cases)}  inference_total={inf_s:.1f}s  per_case={inf_s / max(len(cases), 1):.2f}s")


if __name__ == "__main__":
    main()
