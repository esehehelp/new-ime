"""KD teacher loaders for v2.5 (AR-only)."""

from __future__ import annotations

import torch

from new_ime.config.train import KdSection


def build_teacher(cfg: KdSection, *, device: torch.device):
    """Build the AR teacher described by `cfg`. Raises if checkpoint missing.

    v2.5 ships only the AR text-roundtrip teacher. Future archs (CTC
    logits-based KL, Seq2Seq cross-vocab) reintroduce a dispatch tag in
    KdSection when they land.
    """
    from new_ime.training.teacher.ar import ARTeacher

    return ARTeacher.from_checkpoint(
        cfg.teacher_path,
        teacher_vocab_path=cfg.teacher_vocab,
        device=device,
        hidden_size=cfg.teacher_hidden,
        num_layers=cfg.teacher_layers,
        num_heads=cfg.teacher_heads,
        max_seq_len=cfg.teacher_max_seq_len,
        max_new_tokens=cfg.max_new_tokens,
    )
