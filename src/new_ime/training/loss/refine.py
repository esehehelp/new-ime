"""Mask-CTC refinement 3-part loss (CE masked + BCE remask + BCE stop).

The loss is computed against a fresh hypothesis built by random-masking
the gold target (refine_source='target') or the CTC argmax proposal
(refine_source='proposal'). The model's `refine_from_proposal()` is called
on the cached encoder output stored inside the proposal dict; the refine
decoder predicts: (a) original token at masked positions, (b) per-token
remask probability, (c) sequence-level stop probability.

`build_refine_loss_fn(cfg, mask_id)` returns a closure with signature
`(model, batch, outputs, step) -> dict[str, Tensor]` suitable to plug into
`training/run.py`'s explicit `refine_loss_fn` argument.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F

from new_ime.config.train import RefineSection


def resolve_refine_loss_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    if max_weight <= 0:
        return 0.0
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_weight
    return max_weight * (step / warmup_steps)


def resolve_refine_mask_ratio(mask_ratio_min: float, mask_ratio_max: float) -> float:
    lo = max(0.0, min(1.0, mask_ratio_min))
    hi = max(lo, min(1.0, mask_ratio_max))
    if hi <= lo:
        return lo
    return random.uniform(lo, hi)


def build_refinement_batch(
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    mask_id: int,
    mask_ratio: float,
    *,
    source_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Random-mask `mask_ratio` of valid positions in source_ids (default: target_ids).

    Returns hypothesis_ids (with mask_id at chosen positions), mask_positions
    bool, valid_positions bool, hypothesis_attention_mask long.
    """
    base = source_ids if source_ids is not None else target_ids
    B, T = base.shape
    device = base.device
    positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid = positions < target_lengths.to(device).unsqueeze(1)

    rand = torch.rand((B, T), device=device)
    mask_positions = (rand < mask_ratio) & valid
    hypothesis_ids = torch.where(
        mask_positions,
        torch.full_like(base, mask_id),
        base,
    )
    return {
        "hypothesis_ids": hypothesis_ids,
        "mask_positions": mask_positions,
        "valid_positions": valid,
        "hypothesis_attention_mask": valid.long(),
    }


def build_refine_loss_fn(
    cfg: RefineSection,
    mask_id: int,
) -> Callable[[torch.nn.Module, dict, dict, int], dict[str, torch.Tensor]]:
    """Closure suitable as `refine_loss_fn` for run_loop.

    The closure inspects `step` to honor the loss_weight warmup and skips
    computation entirely when the resolved weight is 0. Single-iteration
    refine only — repeated iterations are out of v2.5 scope.
    """

    def fn(
        model: torch.nn.Module,
        batch: dict,
        outputs: dict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        weight = resolve_refine_loss_weight(step, cfg.warmup_steps, cfg.loss_weight)
        if weight <= 0.0:
            return {}
        if not hasattr(model, "refine_from_proposal"):
            return {}

        target_ids = batch["target_ids"]
        target_lengths = batch["target_lengths"]

        if cfg.refine_source == "proposal":
            input_lengths = batch["attention_mask"].sum(dim=1).long()
            from new_ime.model.ctc_nat import CTCNAT

            collapsed = CTCNAT.collapse_alignment_tensors(
                outputs["logits"], input_lengths, model.blank_id
            )
            source_ids = collapsed["token_ids"]
            source_lengths = collapsed["lengths"]
            B, Ttgt = target_ids.shape
            Tsrc = source_ids.shape[1]
            if Tsrc < Ttgt:
                pad = torch.zeros(
                    (B, Ttgt - Tsrc),
                    dtype=source_ids.dtype,
                    device=source_ids.device,
                )
                source_ids = torch.cat([source_ids, pad], dim=1)
            elif Tsrc > Ttgt:
                source_ids = source_ids[:, :Ttgt]
            refine_input_ids = source_ids
            refine_input_lengths = source_lengths.clamp(max=Ttgt)
        else:
            refine_input_ids = target_ids
            refine_input_lengths = target_lengths

        mask_ratio = resolve_refine_mask_ratio(cfg.mask_ratio_min, cfg.mask_ratio_max)
        refinement = build_refinement_batch(
            target_ids,
            refine_input_lengths,
            mask_id=mask_id,
            mask_ratio=mask_ratio,
            source_ids=refine_input_ids,
        )
        hypothesis_ids = refinement["hypothesis_ids"]
        mask_positions = refinement["mask_positions"]
        valid_positions = refinement["valid_positions"]
        hypothesis_attention_mask = refinement["hypothesis_attention_mask"]

        refine_result = model.refine_from_proposal(
            proposal=outputs,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        refine_logits = refine_result["logits"]
        ce = F.cross_entropy(
            refine_logits.reshape(-1, refine_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        denom = mask_positions.float().sum().clamp(min=1.0)
        ce_loss = (ce * mask_positions.float()).sum() / denom

        accum: dict[str, torch.Tensor] = {"refine_loss": weight * ce_loss}

        if cfg.remask_loss_weight > 0.0:
            argmax = refine_logits.argmax(dim=-1)
            filled = torch.where(mask_positions, argmax, hypothesis_ids)
            remask_target = ((filled != target_ids) & valid_positions).float()
            bce = F.binary_cross_entropy_with_logits(
                refine_result["remask_logits"],
                remask_target,
                reduction="none",
            )
            denom = valid_positions.float().sum().clamp(min=1.0)
            remask_loss = (bce * valid_positions.float()).sum() / denom
            accum["remask_loss"] = cfg.remask_loss_weight * remask_loss

        if cfg.stop_loss_weight > 0.0:
            argmax = refine_logits.argmax(dim=-1)
            filled = torch.where(mask_positions, argmax, hypothesis_ids)
            row_correct = (
                (filled == target_ids) | ~valid_positions
            ).all(dim=1).float()
            stop_bce = F.binary_cross_entropy_with_logits(
                refine_result["stop_logit"],
                row_correct,
                reduction="mean",
            )
            accum["stop_loss"] = cfg.stop_loss_weight * stop_bce

        return accum

    return fn
