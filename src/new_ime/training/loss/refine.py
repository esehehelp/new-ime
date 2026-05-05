"""Mask-CTC refinement 3-part loss (CE masked + BCE remask + BCE stop).

The loss is computed against a fresh hypothesis built by random-masking
the gold target (refine_source='target') or the CTC argmax proposal
(refine_source='proposal'). The model's `refine_from_proposal()` is called
on the cached encoder output stored inside the proposal dict; the refine
decoder predicts: (a) original token at masked positions, (b) per-token
remask probability, (c) sequence-level stop probability.

`build_refine_loss_fn(cfg, mask_id)` returns a closure with signature
`(model, batch, outputs, step) -> dict[str, Tensor]` suitable to plug into
`training/loop.py`'s aux loss list.
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


def _apply_glat_leak(
    *,
    model: torch.nn.Module,
    proposal: dict,
    hypothesis_ids: torch.Tensor,
    hypothesis_attention_mask: torch.Tensor,
    mask_positions: torch.Tensor,
    target_ids: torch.Tensor,
    glance_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GLAT-style leak: run a no-grad refine pass, find masked positions
    whose prediction misses the target, and replace mask_token with oracle
    at `glance_ratio` × per-row miss_count of those positions.

    Returns updated `(hypothesis_ids, mask_positions)`. Loss is still
    computed only on positions where mask_positions is True after the
    leak — leaked positions are treated as "given" and excluded from the
    CE so the model focuses on the remaining masks.

    Reference: DAT `_glance_hint_ids` (model/dat.py:370-430) — same
    number-random strategy adapted to refine's masked-hypothesis setup.
    """
    if glance_ratio <= 0.0:
        return hypothesis_ids, mask_positions

    with torch.no_grad():
        first_pass = model.refine_from_proposal(
            proposal=proposal,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        first_pred = first_pass["logits"].argmax(dim=-1)
        missed = (first_pred != target_ids) & mask_positions

    miss_count = missed.sum(dim=1)                                # (B,)
    leak_count = ((miss_count.float() * glance_ratio) + 0.5).long()
    if int(leak_count.max().item()) == 0:
        return hypothesis_ids, mask_positions

    rand = torch.rand_like(missed, dtype=torch.float)
    rand = rand.masked_fill(~missed, -float("inf"))
    sorted_rand, _ = rand.sort(dim=-1, descending=True)
    idx = (leak_count - 1).clamp(min=0).unsqueeze(-1)             # (B, 1)
    thresh = sorted_rand.gather(-1, idx).squeeze(-1)              # (B,)
    # Rows with leak_count == 0 must accept no entries → push threshold to +inf.
    thresh = torch.where(
        leak_count == 0,
        torch.full_like(thresh, float("inf")),
        thresh,
    )
    leak_mask = (rand >= thresh.unsqueeze(-1)) & missed

    hypothesis_ids = torch.where(leak_mask, target_ids, hypothesis_ids)
    mask_positions = mask_positions & ~leak_mask
    return hypothesis_ids, mask_positions


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
    """Closure suitable as an aux_loss_fn for training/loop.run_loop.

    The closure inspects `step` to honor the loss_weight warmup and
    skips computation entirely when the resolved weight is 0.
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
            with torch.no_grad():
                argmax_ids = outputs["logits"].argmax(dim=-1)
            from new_ime.model.ctc_nat import CTCNAT

            collapsed = CTCNAT.collapse_alignment_tensors(
                outputs["logits"], input_lengths, model.blank_id
            )
            source_ids = collapsed["token_ids"]
            source_lengths = collapsed["lengths"]
            # Pad source_ids/lengths to target shape so refine batch shapes match.
            B, Ttgt = target_ids.shape
            Tsrc = source_ids.shape[1]
            if Tsrc < Ttgt:
                pad = torch.zeros((B, Ttgt - Tsrc), dtype=source_ids.dtype, device=source_ids.device)
                source_ids = torch.cat([source_ids, pad], dim=1)
            elif Tsrc > Ttgt:
                source_ids = source_ids[:, :Ttgt]
            refine_input_ids = source_ids
            refine_input_lengths = source_lengths.clamp(max=Ttgt)
        else:
            refine_input_ids = target_ids
            refine_input_lengths = target_lengths

        accum: dict[str, torch.Tensor] = {}
        ce_total = None
        last_refine_result = None
        first_refine_result = None

        # GLAT schedule (Phase 1 γ). parse_anneal lives in dat.py since DAT
        # also uses it; reuse to keep one source of truth.
        glat_spec = getattr(cfg, "glat_p", "0.0")
        glance_strategy = getattr(cfg, "glance_strategy", "none")
        if glance_strategy == "number-random" and model.training:
            from new_ime.training.loss.dat import parse_anneal

            glance_ratio = parse_anneal(glat_spec, step)
        else:
            glance_ratio = 0.0

        for it in range(max(cfg.refine_iterations, 1)):
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

            # GLAT leak — only on the first iteration (rest of the loop
            # uses argmax-fill from prior iteration anyway, so leak there
            # would be redundant).
            if it == 0 and glance_ratio > 0.0:
                hypothesis_ids, mask_positions = _apply_glat_leak(
                    model=model,
                    proposal=outputs,
                    hypothesis_ids=hypothesis_ids,
                    hypothesis_attention_mask=hypothesis_attention_mask,
                    mask_positions=mask_positions,
                    target_ids=target_ids,
                    glance_ratio=glance_ratio,
                )

            refine_result = model.refine_from_proposal(
                proposal=outputs,
                hypothesis_ids=hypothesis_ids,
                hypothesis_attention_mask=hypothesis_attention_mask,
            )
            if first_refine_result is None:
                first_refine_result = refine_result
                first_mask_positions = mask_positions
                first_valid_positions = valid_positions
                first_hypothesis_ids = hypothesis_ids
            last_refine_result = refine_result

            refine_logits = refine_result["logits"]
            ce = F.cross_entropy(
                refine_logits.reshape(-1, refine_logits.shape[-1]),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape_as(target_ids)
            denom = mask_positions.float().sum().clamp(min=1.0)
            iter_ce = (ce * mask_positions.float()).sum() / denom
            ce_total = iter_ce if ce_total is None else ce_total + iter_ce

            if cfg.refine_iterations > 1 and it < cfg.refine_iterations - 1:
                with torch.no_grad():
                    pred = refine_logits.argmax(dim=-1)
                    refine_input_ids = torch.where(
                        mask_positions, pred, refine_input_ids
                    )

        accum["refine_loss"] = weight * ce_total

        if cfg.remask_loss_weight > 0.0 and first_refine_result is not None:
            first_argmax = first_refine_result["logits"].argmax(dim=-1)
            first_filled = torch.where(
                first_mask_positions, first_argmax, first_hypothesis_ids
            )
            remask_target = (
                (first_filled != target_ids) & first_valid_positions
            ).float()
            bce = F.binary_cross_entropy_with_logits(
                first_refine_result["remask_logits"],
                remask_target,
                reduction="none",
            )
            denom = first_valid_positions.float().sum().clamp(min=1.0)
            bce = (bce * first_valid_positions.float()).sum() / denom
            accum["remask_loss"] = cfg.remask_loss_weight * bce

        if cfg.stop_loss_weight > 0.0 and last_refine_result is not None:
            last_argmax = last_refine_result["logits"].argmax(dim=-1)
            last_filled = torch.where(
                first_mask_positions, last_argmax, first_hypothesis_ids
            )
            # Phase 1 ζ: per-position stop BCE. Each position learns whether
            # it is converged on the gold target. Use stop_logits (per-pos)
            # if surfaced by the model; otherwise fall back to the scalar
            # path for older model versions.
            position_correct = (last_filled == target_ids).float()
            if "stop_logits" in last_refine_result:
                stop_logits = last_refine_result["stop_logits"]
                bce = F.binary_cross_entropy_with_logits(
                    stop_logits, position_correct, reduction="none"
                )
                denom = first_valid_positions.float().sum().clamp(min=1.0)
                stop_bce = (bce * first_valid_positions.float()).sum() / denom
            else:
                row_correct = (
                    (last_filled == target_ids) | ~first_valid_positions
                ).all(dim=1).float()
                stop_bce = F.binary_cross_entropy_with_logits(
                    last_refine_result["stop_logit"],
                    row_correct,
                    reduction="mean",
                )
            accum["stop_loss"] = cfg.stop_loss_weight * stop_bce

        return accum

    return fn
