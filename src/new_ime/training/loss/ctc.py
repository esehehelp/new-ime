"""CTC loss helper.

CTCNAT.forward already computes `F.ctc_loss` internally and returns it
under outputs["loss"]; this module provides a stand-alone version for
ad-hoc use (e.g. computing a separate CTC loss against a teacher hypothesis
without going through the model.forward path) and a blank-fraction
diagnostic shared with evaluate.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_ctc_loss(
    log_probs: torch.Tensor,        # (T, B, V)
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_id: int = 4,
) -> torch.Tensor:
    return F.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def blank_fraction(
    logits: torch.Tensor,           # (B, T, V) or (T, B, V)
    attention_mask: torch.Tensor,   # (B, T)
    blank_id: int,
    *,
    time_first: bool = False,
) -> float:
    if time_first:
        logits = logits.permute(1, 0, 2)
    argmax = logits.argmax(dim=-1)
    valid = attention_mask.bool()
    blanks = ((argmax == blank_id) & valid).sum().item()
    total = valid.sum().item()
    return float(blanks) / max(int(total), 1)
