"""Teacher Protocol + factory for KD."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from new_ime.config.train import KdSection


@runtime_checkable
class TeacherProtocol(Protocol):
    """Minimum surface a KD teacher must expose.

    `arch_tag` is a free-text identifier embedded in the checkpoint blob's
    KD metadata so a resume can detect a teacher swap.

    Logits-based teachers (CTC, Seq2Seq same-vocab) implement `forward(batch)
    -> {"logits": (B, T, V), "confidence": (B,)}`. Text-roundtrip teachers
    (AR, Seq2Seq cross-vocab) implement `generate(batch) -> {"texts":
    list[str], "confidence": list[float]}` and pair with compute_kd_ctc_loss.
    """

    arch_tag: str


def build_teacher(
    cfg: KdSection,
    *,
    device: torch.device,
    expected_vocab_size: int,
):
    if cfg.teacher_type == "ctc":
        from new_ime.training.teacher.ctc import CTCTeacher

        return CTCTeacher.from_checkpoint(
            cfg.teacher_path, device=device, expected_vocab_size=expected_vocab_size
        )
    if cfg.teacher_type == "ar":
        from new_ime.training.teacher.ar import ARTeacher

        return ARTeacher.from_checkpoint(
            cfg.teacher_path,
            teacher_vocab_path=cfg.teacher_vocab,
            device=device,
        )
    if cfg.teacher_type == "seq2seq":
        from new_ime.training.teacher.seq2seq import Seq2SeqTeacher

        return Seq2SeqTeacher.from_checkpoint(
            cfg.teacher_path,
            teacher_vocab_path=cfg.teacher_vocab,
            device=device,
        )
    raise ValueError(f"unknown teacher_type: {cfg.teacher_type}")
