"""CTC teacher: same-tokenizer CTCNAT checkpoint, logits → KL loss path.

Pre-v2 anchored CTC teacher KD on a same-vocab teacher so the KL is
well-defined position-by-position. Cross-vocab (AR / Seq2Seq) teachers
go through text round-trip via training/teacher/{ar,seq2seq}.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from new_ime.model.ctc_nat import CTCNAT


class CTCTeacher:
    arch_tag = "ctc-teacher"

    def __init__(self, model: CTCNAT, device: torch.device):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        device: torch.device,
        expected_vocab_size: int,
    ) -> "CTCTeacher":
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        preset = blob["preset"]
        vocab_size = int(blob.get("vocab_size", expected_vocab_size))
        if vocab_size != expected_vocab_size:
            raise ValueError(
                f"CTC teacher vocab mismatch: ckpt={vocab_size} "
                f"student={expected_vocab_size}"
            )
        model = CTCNAT.from_preset(
            preset,
            vocab_size=vocab_size,
            use_cvae=bool(blob.get("use_cvae", False)),
            max_positions=int(blob.get("max_seq_len", 128)),
        )
        model.load_state_dict(blob["model_state_dict"], strict=False)
        return cls(model, device)

    @torch.no_grad()
    def forward(self, batch: dict) -> dict:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        # Mean top-1 probability per row → confidence in [0, 1].
        confidence = log_probs.max(dim=-1).values.exp().mean(dim=1)
        return {"logits": logits, "confidence": confidence}
