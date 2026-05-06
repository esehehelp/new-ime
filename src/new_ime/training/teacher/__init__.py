"""KD teacher loaders.

Dispatched by `KdSection.teacher_type`:
    "ar"   → SimpleGPT2 + char-level vocab (legacy ar_v3 series)
    "zenz" → HuggingFace GPT-2 directory (zenz-v2.5-* / zenz-v3.1-*)
"""

from __future__ import annotations

import torch

from new_ime.config.train import KdSection


def build_teacher(cfg: KdSection, *, device: torch.device):
    if cfg.teacher_type == "zenz":
        from new_ime.training.teacher.zenz import ZenzTeacher

        return ZenzTeacher.from_checkpoint(
            cfg.teacher_path,
            device=device,
            max_new_tokens=cfg.max_new_tokens,
            max_context_chars=getattr(cfg, "max_context_chars", 40),
        )
    if cfg.teacher_type == "ar":
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
    raise ValueError(
        f"unknown teacher_type: {cfg.teacher_type!r} (supported: ar, zenz)"
    )
