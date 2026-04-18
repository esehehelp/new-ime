"""Export a CTC-NAT checkpoint to ONNX for the C++ / Win32 interactive path.

The PyTorch ``CTCNAT.forward`` returns a dict and accepts training-only
arguments (target_ids, target_lengths, cvae IDs). For ONNX we want a clean
``(input_ids, attention_mask) -> logits`` graph, so we wrap the model in
``CTCNATInferenceWrapper`` that matches that shape and has ``use_cvae=False``
at export time.

Usage:
    uv run python -m scripts.export_onnx_ctc_nat \
        --checkpoint checkpoints/vast_mirror/ctc_nat_90m_phase3mix/checkpoint_step_15000.pt \
        --out models/ctc_nat_90m_step15000.onnx \
        --seq-len 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.data.tokenizer import SharedCharTokenizer
from src.model.ctc_nat import CTCNAT, PRESETS

sys.stdout.reconfigure(encoding="utf-8")


class CTCNATInferenceWrapper(torch.nn.Module):
    """Inference-only wrapper: (input_ids, attention_mask) -> logits."""

    def __init__(self, model: CTCNAT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_out = self.model.encoder(input_ids, attention_mask)
        encoder_padding_mask = ~attention_mask.bool()
        decoder_out = self.model.decoder(
            encoder_out,
            encoder_padding_mask,
            film_conditioning=None,
        )
        return self.model.ctc_head(decoder_out)  # (B, T, V)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=128, help="sample seq len for tracing")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preset = ckpt.get("preset")
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset: {preset!r}")

    tokenizer_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer sidecar: {tokenizer_path}")
    tokenizer = SharedCharTokenizer.load(str(tokenizer_path))
    vocab_size = int(ckpt.get("vocab_size") or tokenizer.vocab_size)

    model = CTCNAT.from_preset(
        preset, vocab_size=vocab_size, use_cvae=False
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    device = torch.device(args.device)
    wrapper = CTCNATInferenceWrapper(model).to(device).eval()

    # Sample input for tracing. Actual input sizes are dynamic.
    dummy_ids = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=device)

    print(f"exporting {preset} ({vocab_size} vocab, {args.seq_len} seq) -> {out_path}")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # Sanity: reload and diff against PyTorch.
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        print("onnxruntime not installed; skipping parity check")
        return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    # Keep the test seq_len identical to export-time seq_len — MultiheadAttention
    # bakes attention shape constants into the graph under dynamic_axes on
    # older opsets, so runtime inputs must be padded to args.seq_len anyway.
    test_ids = torch.randint(0, vocab_size, (1, args.seq_len), dtype=torch.long)
    test_mask = torch.zeros_like(test_ids)
    actual_len = min(40, args.seq_len)
    test_mask[:, :actual_len] = 1
    with torch.no_grad():
        pt_logits = wrapper(test_ids, test_mask).cpu().numpy()
    ort_logits = sess.run(
        ["logits"],
        {"input_ids": test_ids.numpy(), "attention_mask": test_mask.numpy()},
    )[0]
    max_abs_full = abs(pt_logits - ort_logits).max()
    max_abs_valid = abs(
        pt_logits[:, :actual_len, :] - ort_logits[:, :actual_len, :]
    ).max()
    print(f"max abs diff, full tensor:   {max_abs_full:.6f}")
    print(f"max abs diff, valid region:  {max_abs_valid:.6f}")
    # Only the valid region (non-padded) feeds the CTC decoder, so judge
    # parity on that.
    if max_abs_valid > 1e-3:
        print("WARNING: ONNX output differs non-trivially in valid region")
    else:
        print("OK: ONNX matches PyTorch in valid region within 1e-3")

    # Also save tokenizer sidecar next to the ONNX for the C++ consumer.
    sidecar = out_path.with_suffix(".tokenizer.json")
    tokenizer_bytes = tokenizer_path.read_bytes()
    sidecar.write_bytes(tokenizer_bytes)
    print(f"copied tokenizer to {sidecar}")

    # And a small meta file for C++ consumers.
    meta_path = out_path.with_suffix(".meta.txt")
    meta_path.write_text(
        f"preset={preset}\nvocab_size={vocab_size}\nblank_id={model.blank_id}\n"
        f"step={ckpt.get('step', '?')}\n"
        f"max_seq_len={ckpt.get('max_seq_len', 128)}\n",
        encoding="utf-8",
    )
    print(f"wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
