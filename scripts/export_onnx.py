"""Export AR baseline model to ONNX format.

Usage:
    uv run python scripts/export_onnx.py \
        --checkpoint checkpoints/ar_v3_vast/checkpoint_step_70000.pt \
        --output models/ar_v3_vast.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.data.dataset import ARCollator
from src.training.train_ar import SimpleGPT2

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    # Load model
    collator = ARCollator()
    vocab_path = args.checkpoint.replace(".pt", "_vocab.json")
    collator.load_vocab(vocab_path)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    hidden = state["embed_tokens.weight"].shape[1]
    max_pos = state["embed_positions.weight"].shape[0]
    n_layers = len([
        k for k in state
        if k.startswith("transformer.layers.") and k.endswith(".self_attn.in_proj_weight")
    ])

    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=hidden,
        num_layers=n_layers,
        num_heads=8,
        max_positions=max_pos,
    )
    model.load_state_dict(state)
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"  hidden={hidden}, layers={n_layers}, vocab={collator.vocab_size}, max_pos={max_pos}")

    # Export ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input
    seq_len = 32
    dummy_input_ids = torch.randint(0, collator.vocab_size, (1, seq_len))
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    print(f"Exporting to {args.output} (opset {args.opset})...")

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        opset_version=args.opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )

    print(f"ONNX exported: {output_path} ({output_path.stat().st_size / 1024**2:.1f} MB)")

    # Also save vocab JSON alongside
    vocab_out = output_path.with_suffix(".vocab.json")
    collator.save_vocab(str(vocab_out))
    print(f"Vocab saved: {vocab_out}")

    # Save model config
    config = {
        "hidden_size": hidden,
        "num_layers": n_layers,
        "num_heads": 8,
        "vocab_size": collator.vocab_size,
        "max_positions": max_pos,
        "step": ckpt["step"],
        "special_tokens": {
            "PAD": collator.PAD,
            "SEP": collator.SEP,
            "OUT": collator.OUT,
            "EOS": collator.EOS,
            "UNK": collator.UNK,
        },
    }
    config_out = output_path.with_suffix(".config.json")
    config_out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Config saved: {config_out}")

    # Verify ONNX
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path))
    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)
    print(f"ONNX verification: output shape {ort_outputs[0].shape}")

    # Compare PyTorch vs ONNX
    with torch.no_grad():
        pt_output = model(dummy_input_ids, dummy_attention_mask)
    import numpy as np

    diff = np.abs(pt_output.numpy() - ort_outputs[0]).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")
    if diff < 1e-4:
        print("OK: outputs match")
    else:
        print(f"WARNING: outputs differ by {diff}")


if __name__ == "__main__":
    main()
