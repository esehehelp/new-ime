"""KD loss helpers and aux-fn factory.

`alpha_at(step, cfg)` reproduces the pre-v2 schedule: linear warmup of α
from 0 to `cfg.alpha` over `warmup_steps`, then optional linear decay to
`alpha_final` between `alpha_decay_start` and `alpha_decay_start +
alpha_decay_steps`.

`compute_kd_kl_loss` is logits-based and arch-agnostic: any teacher whose
output shape matches the student plugs in. `compute_kd_ctc_loss` covers
the text-roundtrip path (AR / Seq2Seq teachers) for v1.1.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from new_ime.config.train import KdSection


def alpha_at(step: int, cfg: KdSection) -> float:
    if step < cfg.start_step:
        return 0.0
    if cfg.warmup_steps <= 0:
        alpha = float(cfg.alpha)
    else:
        progress = min((step - cfg.start_step) / cfg.warmup_steps, 1.0)
        alpha = float(cfg.alpha) * progress
    if cfg.alpha_decay_steps <= 0:
        return alpha
    if step < cfg.alpha_decay_start:
        return alpha
    decay_progress = min(
        (step - cfg.alpha_decay_start) / cfg.alpha_decay_steps, 1.0
    )
    return alpha + (float(cfg.alpha_final) - alpha) * decay_progress


def hard_example_mask(
    confidences: torch.Tensor,
    threshold: float,
    mode: str,
) -> torch.Tensor:
    if mode == "all":
        return torch.ones_like(confidences, dtype=torch.bool)
    if mode == "high_conf":
        return confidences >= threshold
    if mode == "low_conf":
        return confidences < threshold
    raise ValueError(f"unknown gate_mode: {mode!r}")


def compute_kd_kl_loss(
    student_logits: torch.Tensor,    # (B, T, V)
    teacher_logits: torch.Tensor,    # (B, T, V)
    attention_mask: torch.Tensor,    # (B, T)
    hard_mask: torch.Tensor,         # (B,) bool
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    device = student_logits.device
    valid = hard_mask.to(device)
    num_hard = int(valid.sum().item())
    if num_hard == 0:
        return (
            torch.zeros((), device=device, dtype=student_logits.dtype),
            0,
        )
    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    s_log = F.log_softmax(
        student_logits.index_select(0, idx) / temperature, dim=-1
    )
    t_log = F.log_softmax(
        teacher_logits.index_select(0, idx) / temperature, dim=-1
    )
    t_prob = t_log.exp()
    kl = (t_prob * (t_log - s_log)).sum(dim=-1)
    mask = attention_mask.index_select(0, idx).float()
    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    loss = loss * (temperature ** 2)
    return loss, num_hard


def compute_kd_ctc_loss(
    student_log_probs: torch.Tensor,   # (T, B, V) time-first
    input_lengths: torch.Tensor,
    teacher_ids: torch.Tensor,         # (B, target_len)
    teacher_lengths: torch.Tensor,
    hard_mask: torch.Tensor,           # (B,) bool
    blank_id: int = 4,
) -> tuple[torch.Tensor, int]:
    """For text-roundtrip teachers: train student to reproduce teacher tokens
    via CTC. Used by AR / Seq2Seq teacher path (v1.1)."""
    device = student_log_probs.device
    valid = hard_mask.to(device) & (teacher_lengths.to(device) > 0)
    num_hard = int(valid.sum().item())
    if num_hard == 0:
        return (
            torch.zeros((), device=device, dtype=student_log_probs.dtype),
            0,
        )
    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    sel_log_probs = student_log_probs.index_select(1, idx)
    sel_input_lengths = input_lengths.to(device).index_select(0, idx)
    sel_teacher_ids = teacher_ids.to(device).index_select(0, idx)
    sel_teacher_lengths = teacher_lengths.to(device).index_select(0, idx)

    keepable = sel_input_lengths >= sel_teacher_lengths
    if not bool(keepable.all().item()):
        keep = keepable.nonzero(as_tuple=False).squeeze(-1)
        if keep.numel() == 0:
            return (
                torch.zeros((), device=device, dtype=student_log_probs.dtype),
                0,
            )
        sel_log_probs = sel_log_probs.index_select(1, keep)
        sel_input_lengths = sel_input_lengths.index_select(0, keep)
        sel_teacher_ids = sel_teacher_ids.index_select(0, keep)
        sel_teacher_lengths = sel_teacher_lengths.index_select(0, keep)
        num_hard = int(keep.numel())

    loss = F.ctc_loss(
        log_probs=sel_log_probs,
        targets=sel_teacher_ids,
        input_lengths=sel_input_lengths,
        target_lengths=sel_teacher_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return loss, num_hard


def build_kd_loss_fn(
    cfg: KdSection,
    teacher,
) -> Callable[[torch.nn.Module, dict, dict, int], dict[str, torch.Tensor]]:
    """Closure for training/loop.run_loop.aux_loss_fns. Logits-based KL only.

    Text-roundtrip teachers (AR / Seq2Seq) have their own factory in v1.1.
    """

    def fn(model, batch, outputs, step):
        if step < cfg.start_step:
            return {}
        if cfg.every > 1 and (step - cfg.start_step) % cfg.every != 0:
            return {}
        alpha = alpha_at(step, cfg)
        if alpha <= 0.0:
            return {}
        teacher_out = teacher.forward(batch)
        teacher_logits = teacher_out["logits"]
        confidence = teacher_out["confidence"]
        hard_mask = hard_example_mask(
            confidence, cfg.hard_threshold, cfg.gate_mode
        )
        student_logits = outputs["logits"]
        loss, n = compute_kd_kl_loss(
            student_logits,
            teacher_logits,
            batch["attention_mask"],
            hard_mask,
            cfg.temperature,
        )
        if n == 0:
            return {}
        return {"kd_kl": alpha * loss}

    return fn
