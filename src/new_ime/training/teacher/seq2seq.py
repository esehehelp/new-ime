"""Seq2Seq encoder-decoder teacher (text round-trip). v1.1 — placeholder."""

from __future__ import annotations

from pathlib import Path

import torch


class Seq2SeqTeacher:
    arch_tag = "seq2seq-teacher"

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        teacher_vocab_path: Path | None,
        device: torch.device,
    ) -> "Seq2SeqTeacher":
        raise NotImplementedError(
            "Seq2SeqTeacher is not implemented in v1.0. See the AR teacher "
            "scaffolding for the intended interface."
        )

    @torch.no_grad()
    def generate(self, batch: dict) -> dict:
        raise NotImplementedError("Seq2SeqTeacher.generate not implemented in v1.0")
