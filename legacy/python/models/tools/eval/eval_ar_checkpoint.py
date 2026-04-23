"""Evaluate an AR baseline checkpoint with autoregressive generation.

Usage:
    uv run python scripts/eval_ar_checkpoint.py \
        --checkpoint checkpoints/ar_baseline/best.pt \
        --dev datasets/eval/dev.jsonl \
        --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from models.src.data.dataset import ARCollator, KanaKanjiDataset
from models.src.eval.metrics import EvalResult
from models.src.training.train_ar import SimpleGPT2, autoregressive_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--show-examples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab
    vocab_path = args.checkpoint.replace(".pt", "_vocab.json")
    collator = ARCollator()
    collator.load_vocab(vocab_path)
    print(f"Vocab: {collator.vocab_size} tokens")

    # Load model — infer hidden_size and max_positions from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    hidden_size = state["embed_tokens.weight"].shape[1]
    max_positions = state["embed_positions.weight"].shape[0]
    # Count transformer layers
    layer_keys = [k for k in state if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight")]
    num_layers = len(layer_keys)
    num_heads = 8  # default, not stored in checkpoint
    print(f"Model config: hidden={hidden_size}, layers={num_layers}, max_pos={max_positions}")

    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_positions=max_positions,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint (step {ckpt['step']})")

    # Load dev data
    dev = KanaKanjiDataset(args.dev)
    print(f"Dev samples: {len(dev)}")

    # Evaluate with autoregressive generation
    result = EvalResult()
    examples = []
    t_start = time.time()

    for i in range(min(args.max_samples, len(dev))):
        sample = dev.data[i]
        context = sample.get("context", "")[-40:]
        reading = sample["reading"]
        reference = sample["surface"]

        ctx_ids = collator.encode_text(context)
        read_ids = collator.encode_text(reading)
        prefix = ctx_ids + [collator.SEP] + read_ids + [collator.OUT]

        gen_ids = autoregressive_generate(
            model, prefix, device,
            max_new_tokens=len(reference) + 20,
            max_seq_len=max_positions,
            eos_id=collator.EOS,
            pad_id=collator.PAD,
        )
        pred_text = collator.decode_ids(gen_ids)
        result.add(reference, [pred_text])

        if i < args.show_examples:
            examples.append({
                "reading": reading[:40],
                "reference": reference[:40],
                "predicted": pred_text[:40],
            })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  {i + 1}/{args.max_samples} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== Autoregressive Evaluation ===")
    print(f"Samples: {result.total}")
    print(f"Time: {elapsed:.1f}s ({elapsed / result.total:.2f}s/sample)")
    print(result.report())

    print(f"\n--- Examples ---")
    for ex in examples:
        print(f"  reading:  {ex['reading']}")
        print(f"  ref:      {ex['reference']}")
        print(f"  pred:     {ex['predicted']}")
        print()


if __name__ == "__main__":
    main()
