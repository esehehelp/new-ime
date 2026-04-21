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
import hashlib
import sys
from pathlib import Path

import torch

from models.src.data.tokenizer import SharedCharTokenizer
from models.src.model.ctc_nat import CTCNAT, PRESETS

# Bumped whenever the exported graph's input/output contract or the
# tokenizer sidecar format changes. The Rust consumer verifies this
# against its expected value before wiring up a session.
ARTIFACT_VERSION = "2"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

sys.stdout.reconfigure(encoding="utf-8")


class CTCNATInferenceWrapper(torch.nn.Module):
    """Inference-only wrapper: (input_ids, attention_mask) -> logits."""

    def __init__(self, model: CTCNAT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.proposal_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_posterior=False,
        )["logits"]


class MaskCTCRefinerInferenceWrapper(torch.nn.Module):
    """Inference-only wrapper.

    Inputs: (input_ids, attention_mask, hypothesis_ids, hypothesis_attention_mask)
    Outputs: (logits, remask_logits, stop_logit) — the two extra outputs feed
    the iterative refinement loop on the Rust side (learned re-mask policy +
    learned stop condition). Consumers that only need `logits` can ignore the
    additional outputs.
    """

    def __init__(self, model: CTCNAT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hypothesis_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.model.refine_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
            sample_posterior=False,
        )
        return result["logits"], result["remask_logits"], result["stop_logit"]


def export_graph(
    wrapper: torch.nn.Module,
    args_tuple: tuple[torch.Tensor, ...],
    out_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    torch.onnx.export(
        wrapper,
        args_tuple,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--export-refiner",
        action="store_true",
        help="Also export a dedicated Mask-CTC refiner graph next to the proposal graph.",
    )
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
    proposal_wrapper = CTCNATInferenceWrapper(model).to(device).eval()
    refiner_wrapper = MaskCTCRefinerInferenceWrapper(model).to(device).eval()

    # Sample input for tracing. Actual input sizes are dynamic.
    dummy_ids = torch.zeros((1, args.seq_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=device)
    dummy_hypothesis = torch.full((1, args.seq_len), tokenizer.mask_id, dtype=torch.long, device=device)
    dummy_hypothesis_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=device)

    print(f"exporting proposal {preset} ({vocab_size} vocab, {args.seq_len} seq) -> {out_path}")
    export_graph(
        proposal_wrapper,
        (dummy_ids, dummy_mask),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset=args.opset,
    )
    refiner_out_path = out_path.with_name(f"{out_path.stem}.refiner{out_path.suffix}")
    if args.export_refiner:
        print(f"exporting refiner {preset} ({vocab_size} vocab, {args.seq_len} seq) -> {refiner_out_path}")
        export_graph(
            refiner_wrapper,
            (dummy_ids, dummy_mask, dummy_hypothesis, dummy_hypothesis_mask),
            refiner_out_path,
            input_names=[
                "input_ids",
                "attention_mask",
                "hypothesis_ids",
                "hypothesis_attention_mask",
            ],
            output_names=["logits", "remask_logits", "stop_logit"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "hypothesis_ids": {0: "batch", 1: "hyp_seq"},
                "hypothesis_attention_mask": {0: "batch", 1: "hyp_seq"},
                "logits": {0: "batch", 1: "hyp_seq"},
                "remask_logits": {0: "batch", 1: "hyp_seq"},
                "stop_logit": {0: "batch"},
            },
            opset=args.opset,
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
    test_hyp = torch.full((1, args.seq_len), tokenizer.mask_id, dtype=torch.long)
    test_hyp[:, :actual_len] = test_ids[:, :actual_len]
    if actual_len > 3:
        # Mask one early and one late position so the refiner parity check
        # exercises non-trivial updates across the valid span.
        test_hyp[:, 1] = tokenizer.mask_id
        test_hyp[:, actual_len - 2] = tokenizer.mask_id
    test_hyp_mask = torch.zeros_like(test_hyp)
    test_hyp_mask[:, :actual_len] = 1
    with torch.no_grad():
        pt_logits = proposal_wrapper(test_ids, test_mask).cpu().numpy()
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
    if args.export_refiner:
        refiner_sess = ort.InferenceSession(str(refiner_out_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt_logits, pt_remask, pt_stop = refiner_wrapper(
                test_ids, test_mask, test_hyp, test_hyp_mask
            )
            pt_logits = pt_logits.cpu().numpy()
            pt_remask = pt_remask.cpu().numpy()
            pt_stop = pt_stop.cpu().numpy()
        ort_logits, ort_remask, ort_stop = refiner_sess.run(
            ["logits", "remask_logits", "stop_logit"],
            {
                "input_ids": test_ids.numpy(),
                "attention_mask": test_mask.numpy(),
                "hypothesis_ids": test_hyp.numpy(),
                "hypothesis_attention_mask": test_hyp_mask.numpy(),
            },
        )
        max_refine = abs(pt_logits[:, :actual_len, :] - ort_logits[:, :actual_len, :]).max()
        max_remask = abs(pt_remask[:, :actual_len] - ort_remask[:, :actual_len]).max()
        max_stop = abs(pt_stop - ort_stop).max()
        print(f"max abs diff, refiner logits valid: {max_refine:.6f}")
        print(f"max abs diff, remask_logits valid: {max_remask:.6f}")
        print(f"max abs diff, stop_logit:          {max_stop:.6f}")
        if max(max_refine, max_remask, max_stop) > 1e-3:
            print("WARNING: refiner ONNX output differs non-trivially")
        else:
            print("OK: refiner ONNX matches PyTorch within 1e-3")

    # Also save tokenizer sidecar next to the ONNX for the C++ consumer.
    sidecar = out_path.with_suffix(".tokenizer.json")
    tokenizer_bytes = tokenizer_path.read_bytes()
    sidecar.write_bytes(tokenizer_bytes)
    print(f"copied tokenizer to {sidecar}")

    # And a small meta file for C++ / Rust consumers. Includes sha256
    # digests of both graphs so the loader can detect mismatched
    # proposal/refiner pairs (e.g. re-export of only one side).
    proposal_sha = _sha256_file(out_path)
    refiner_sha = (
        _sha256_file(refiner_out_path)
        if args.export_refiner and refiner_out_path.exists()
        else ""
    )
    meta_path = out_path.with_suffix(".meta.txt")
    meta_path.write_text(
        f"artifact_version={ARTIFACT_VERSION}\n"
        f"preset={preset}\nvocab_size={vocab_size}\nblank_id={model.blank_id}\n"
        f"mask_id={tokenizer.mask_id}\n"
        f"proposal_inputs=2\n"
        f"proposal_path={out_path.name}\n"
        f"proposal_sha256={proposal_sha}\n"
        f"refiner_enabled={int(args.export_refiner)}\n"
        f"refiner_inputs=4\n"
        f"refiner_path={refiner_out_path.name if args.export_refiner else ''}\n"
        f"refiner_sha256={refiner_sha}\n"
        f"step={ckpt.get('step', '?')}\n"
        f"max_seq_len={ckpt.get('max_seq_len', 128)}\n",
        encoding="utf-8",
    )
    print(f"wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
