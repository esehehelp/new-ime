"""CTC-NAT checkpoint → ONNX (fp32 + int8 dynamic quant) + vocab.hex.tsv sidecar.

Output layout (matches `crates/new-ime-engine-core/src/session.rs::derive_*`):

    <out_dir>/<name>.fp32.onnx
    <out_dir>/<name>.int8.onnx
    <out_dir>/<name>.fp32.tokenizer.json.vocab.hex.tsv

The exported graph wraps `CTCNAT.proposal_logits` so its single output `logits`
(shape [1, T, V]) is what the Rust engine reads. Inputs are i64
`input_ids` / `attention_mask` of shape (1, T) where T is dynamic.

Why dynamic T: with a fixed seq_len=128 export, ORT pays full 128-token
compute even for short prompts (typical IME readings are 5-30 chars).
Probe_v3 latency was ~3x higher than legacy PyTorch greedy under fixed
shape. The legacy TorchScript export path constant-folds seq_len=128
inside nn.MultiheadAttention reshape; switching to `dynamo=True` forces
torch.export which preserves seq_len as a symbolic dimension.

Usage:
    uv run --with onnx --with onnxruntime python scripts/export_onnx.py \
        --ckpt checkpoints/suiko-v1-small/checkpoint_step_100000.pt \
        --out-dir models/onnx \
        --name suiko-v1-small-step100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from new_ime.data.tokenizer import SharedCharTokenizer
from new_ime.model.ctc_nat import CTCNAT, PRESETS


class ProposalWrapper(nn.Module):
    """Thin export-only wrapper: returns only the proposal CTC logits."""

    def __init__(self, model: CTCNAT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.model.proposal_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_posterior=False,
        )
        return out["logits"]


def write_vocab_hex_tsv(tokenizer: SharedCharTokenizer, dst: Path) -> None:
    """Emit `id\\thex(utf8(token))` lines, one per id, sorted by id ascending."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="ascii", newline="\n") as f:
        for tid in sorted(tokenizer.id_to_token):
            tok = tokenizer.id_to_token[tid]
            f.write(f"{tid}\t{tok.encode('utf-8').hex()}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--name",
        required=True,
        help="basename without extension, e.g. suiko-v1-small-step100000",
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path: Path = args.ckpt
    out_dir: Path = args.out_dir
    name: str = args.name

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preset = ckpt.get("preset")
    if preset not in PRESETS:
        raise SystemExit(f"unknown preset in ckpt: {preset!r}")

    tokenizer_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
    if not tokenizer_path.exists():
        raise SystemExit(f"missing tokenizer sidecar: {tokenizer_path}")
    tokenizer = SharedCharTokenizer.load(tokenizer_path)
    vocab_size = int(ckpt.get("vocab_size") or tokenizer.vocab_size)

    model = CTCNAT.from_preset(
        preset,
        vocab_size=vocab_size,
        use_cvae=bool(ckpt.get("use_cvae", False)),
    )
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if load_result.unexpected_keys:
        print(f"[warn] unexpected state keys: {load_result.unexpected_keys}")
    if load_result.missing_keys:
        # MaskCTC heads (refine_*, remask_, stop_) are unused by proposal export.
        print(f"[info] missing state keys (proposal-only export): {len(load_result.missing_keys)}")
    model.eval()
    wrapper = ProposalWrapper(model).eval()

    seq_len = int(ckpt.get("max_seq_len", args.seq_len))
    dummy_ids = torch.zeros((1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, seq_len), dtype=torch.long)

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / f"{name}.fp32.onnx"
    int8_path = out_dir / f"{name}.int8.onnx"
    tsv_path = out_dir / f"{name}.fp32.tokenizer.json.vocab.hex.tsv"

    print(f"[1/3] export fp32 ONNX → {fp32_path}")
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        fp32_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {1: "seq"},
            "attention_mask": {1: "seq"},
            "logits": {1: "seq"},
        },
        dynamo=True,
    )

    print(f"[2/3] write vocab hex TSV → {tsv_path}")
    write_vocab_hex_tsv(tokenizer, tsv_path)

    print(f"[3/3] dynamic int8 quantize → {int8_path}")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(
        model_input=fp32_path.as_posix(),
        model_output=int8_path.as_posix(),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )

    fp32_mb = fp32_path.stat().st_size / 1024 / 1024
    int8_mb = int8_path.stat().st_size / 1024 / 1024
    print(
        f"done. fp32={fp32_mb:.1f} MB  int8={int8_mb:.1f} MB  vocab_size={vocab_size}  "
        f"seq_len={seq_len}  preset={preset}  step={ckpt.get('step')}"
    )


if __name__ == "__main__":
    main()
