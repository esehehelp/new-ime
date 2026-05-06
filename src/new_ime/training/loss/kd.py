"""KD loss helpers and aux-fn factory for v2.5 (AR text-roundtrip only).

`alpha_at(step, cfg)` reproduces the pre-v2 schedule: linear warmup of α
from 0 to `cfg.alpha` over `warmup_steps` (gated by `cfg.start_step`), then
optional linear decay to `alpha_final` between `alpha_decay_start` and
`alpha_decay_start + alpha_decay_steps`.

The student is trained to reproduce the AR teacher's text via CTC against
the teacher tokens. This is the same path used in the 8cdf0df trace.
Logits-based KL (CTC teacher) is out of v2.5 scope.
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


def compute_kd_ctc_loss(
    student_log_probs: torch.Tensor,   # (T, B, V) time-first
    input_lengths: torch.Tensor,
    teacher_ids: torch.Tensor,         # (B, target_len)
    teacher_lengths: torch.Tensor,
    hard_mask: torch.Tensor,           # (B,) bool
    blank_id: int = 4,
) -> tuple[torch.Tensor, int]:
    """Train student to reproduce teacher tokens via CTC."""
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
    tokenizer,
) -> Callable[[torch.nn.Module, dict, dict, int], dict[str, torch.Tensor]]:
    """Closure for run.py's `kd_loss_fn` argument.

    Each call (subject to `every` / `start_step` / `alpha`) runs the AR
    teacher on `(context_texts, reading_texts)` to obtain teacher
    surface text, encodes that with the *student* tokenizer, then adds a
    CTC loss against the teacher surface (gated by teacher confidence).
    """

    def fn(model, batch, outputs, step):
        if step < cfg.start_step:
            return {}
        if cfg.every > 1 and (step - cfg.start_step) % cfg.every != 0:
            return {}
        alpha = alpha_at(step, cfg)
        if alpha <= 0.0:
            return {}

        contexts = batch["context_texts"]
        readings = batch["reading_texts"]
        with torch.no_grad():
            texts, confidences = teacher.generate(
                contexts, readings, max_new_tokens=cfg.max_new_tokens
            )
        confidences = confidences.to(outputs["logits"].device)
        hard_mask = hard_example_mask(
            confidences, cfg.hard_threshold, cfg.gate_mode
        )
        if not bool(hard_mask.any().item()):
            return {}

        # Encode teacher text with student tokenizer.
        teacher_id_lists = [tokenizer.encode(t) for t in texts]
        teacher_lengths = torch.tensor(
            [len(ids) for ids in teacher_id_lists],
            dtype=torch.long,
            device=outputs["logits"].device,
        )
        max_len = max(int(teacher_lengths.max().item()), 1)
        pad_id = getattr(tokenizer, "pad_id", 0)
        teacher_ids = torch.full(
            (len(teacher_id_lists), max_len),
            pad_id,
            dtype=torch.long,
            device=outputs["logits"].device,
        )
        for i, ids in enumerate(teacher_id_lists):
            if ids:
                teacher_ids[i, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=outputs["logits"].device
                )

        # CTC needs (T, B, V).
        student_logits = outputs["logits"]
        log_probs = F.log_softmax(student_logits, dim=-1).transpose(0, 1)
        input_lengths = batch["attention_mask"].sum(dim=1).long()

        loss, n_hard = compute_kd_ctc_loss(
            log_probs,
            input_lengths,
            teacher_ids,
            teacher_lengths,
            hard_mask,
            blank_id=getattr(model, "blank_id", 4),
        )
        if n_hard == 0:
            return {}

        diag = {
            "kd_loss": loss.detach(),
            "kd_alpha": torch.tensor(alpha, device=loss.device),
            "kd_hard": torch.tensor(
                n_hard / max(len(texts), 1), device=loss.device
            ),
            "kd_conf": confidences.mean(),
        }
        # The training loop adds the weighted loss back; diag entries are
        # passed through for logging.
        return {"kd_ctc": alpha * loss, **{f"_diag_{k}": v for k, v in diag.items()}}

    return fn
