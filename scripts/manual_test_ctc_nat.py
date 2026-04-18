"""Manual test for CTC-NAT checkpoints with the same 100 cases as the AR
baseline. Use this to compare phase3 student checkpoints against the AR
teacher on a curated benchmark.

Usage:
    uv run python -m scripts.manual_test_ctc_nat \
        --checkpoint checkpoints/ctc_nat_90m_phase3mix_vast/checkpoint_step_10000.pt

Prints per-case ref/pred, then aggregate EM / CharAcc.

Expects a companion ``<checkpoint>_tokenizer.json`` alongside the .pt (that
is how ``src.training.train_ctc_nat.save_checkpoint`` writes it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.data.tokenizer import SharedCharTokenizer
from src.eval.metrics import EvalResult
from src.model.ctc_nat import CTCNAT, PRESETS
from scripts.manual_test import TEST_CASES

sys.stdout.reconfigure(encoding="utf-8")


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[CTCNAT, SharedCharTokenizer, dict]:
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
    if "blank_logit_bias" in ckpt:
        model.blank_logit_bias = float(ckpt["blank_logit_bias"])
    model = model.to(device).eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def predict(
    model: CTCNAT,
    tokenizer: SharedCharTokenizer,
    context: str,
    reading: str,
    device: torch.device,
    max_context: int = 40,
    max_seq_len: int = 128,
) -> str:
    ids = tokenizer.encode_with_special(context[-max_context:], reading)
    ids = ids[:max_seq_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    decoded = model.greedy_decode(input_ids, attention_mask)
    return tokenizer.decode(decoded[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="CTC-NAT .pt path")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all")
    parser.add_argument("--quiet", action="store_true", help="skip per-case output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    model, tokenizer, ckpt = build_model_from_checkpoint(ckpt_path, device)
    print(
        f"checkpoint: {ckpt_path.name}  "
        f"step={ckpt.get('step','?')}  epoch={ckpt.get('epoch','?')}  "
        f"best_metric={ckpt.get('best_metric','?')}"
    )
    print(f"preset={ckpt.get('preset')}  vocab={tokenizer.vocab_size}  device={device}")

    cases = TEST_CASES if not args.max_cases else TEST_CASES[: args.max_cases]
    result = EvalResult()
    for idx, (context, reading, expected) in enumerate(cases, 1):
        prediction = predict(model, tokenizer, context, reading, device)
        result.add(expected, [prediction])
        if not args.quiet:
            mark = "✓" if prediction == expected else "✗"
            ctx_disp = f"  [{context[-20:]}]" if context else ""
            print(
                f"  {idx:>3} {mark}  ref={expected}{ctx_disp}\n"
                f"       pred={prediction}"
            )

    summary = result.summary()
    em = summary.get("exact_match_top1", 0.0)
    char_acc = summary.get("char_acc_top1", 0.0)
    print()
    print(f"EM       : {em:.4f}  ({int(round(em * len(cases)))}/{len(cases)})")
    print(f"CharAcc  : {char_acc:.4f}")


if __name__ == "__main__":
    main()
