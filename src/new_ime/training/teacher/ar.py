"""AR teacher (text round-trip, cross-vocab). v1.1 — not yet implemented.

In pre-v2 this loaded a SimpleGPT2 + its own vocab JSON, ran greedy
generation per row, and the generated text was re-encoded with the student
tokenizer to act as a CTC target via compute_kd_ctc_loss. The model class
itself has not been ported to v2 yet, so this file ships as a clear
placeholder for the future v1.1 work.
"""

from __future__ import annotations

from pathlib import Path

import torch


class ARTeacher:
    arch_tag = "ar-teacher"

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        teacher_vocab_path: Path | None,
        device: torch.device,
    ) -> "ARTeacher":
        raise NotImplementedError(
            "ARTeacher is not implemented in v1.0. The AR model class itself "
            "has not been ported from pre-v2 yet — see "
            "train/research/ar-arch sidebranch in plans/train-linear-blanket.md."
        )

    @torch.no_grad()
    def generate(self, batch: dict) -> dict:
        raise NotImplementedError("ARTeacher.generate not implemented in v1.0")
