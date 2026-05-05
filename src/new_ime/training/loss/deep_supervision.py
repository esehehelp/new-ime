"""Deep-supervision CTC loss on intermediate decoder layers.

Training-only auxiliary signal that forces middle decoder layers to be
useful CTC predictors on their own. Adds zero learnable params (the
shared `ctc_head` — already tied to the encoder embedding — is reused
for each captured mid-layer hidden), and inference cost is unchanged
(model only runs the deep-supervision path when `_deep_sup_layers` is
non-empty, which run.py only sets in training).

Wired into the training loop via the `aux_loss_fns` list pattern (same
shape as `build_refine_loss_fn` / `build_kd_loss_fn`):

    aux_loss_fns.append(build_deep_supervision_loss_fn(cfg.deep_supervision))

The model surfaces the captured logits as
    outputs["intermediate_logits"] = [(layer_idx, logits[B, T, V]), ...]

and this loss closure consumes them directly.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from new_ime.config.train import DeepSupervisionSection


def build_deep_supervision_loss_fn(
    cfg: DeepSupervisionSection,
) -> Callable[[torch.nn.Module, dict, dict, int], dict[str, torch.Tensor]]:
    """Return an aux-loss closure that adds CTC loss on captured mid-layer
    logits. No-op (returns empty dict) when `cfg.layers` is empty.
    """
    layers = list(cfg.layers)
    weights = list(cfg.weights) if cfg.weights else [1.0] * len(layers)
    if len(weights) != len(layers):
        raise ValueError(
            f"deep_supervision.weights ({weights}) length mismatch with layers ({layers})"
        )
    weight_by_layer = {int(layer): float(w) for layer, w in zip(layers, weights, strict=True)}
    warmup_steps = max(0, int(cfg.warmup_steps))

    def fn(
        model: torch.nn.Module,
        batch: dict,
        outputs: dict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        if not weight_by_layer:
            return {}
        captures = outputs.get("intermediate_logits")
        if not captures:
            return {}
        target_ids = batch.get("target_ids")
        target_lengths = batch.get("target_lengths")
        attention_mask = batch.get("attention_mask")
        if target_ids is None or target_lengths is None or attention_mask is None:
            return {}

        input_lengths = attention_mask.sum(dim=1).long()
        # Linear warmup on the aggregate aux weight (each layer keeps its own
        # static ratio inside `weight_by_layer`).
        ramp = 1.0
        if warmup_steps > 0 and step < warmup_steps:
            ramp = max(0.0, step / float(warmup_steps))
        if ramp <= 0.0:
            return {}

        blank_id = int(getattr(model, "blank_id", 4))
        aux: dict[str, torch.Tensor] = {}
        for layer_idx, logits in captures:
            w = weight_by_layer.get(int(layer_idx))
            if w is None or w == 0.0:
                continue
            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = F.ctc_loss(
                log_probs=log_probs,
                targets=target_ids,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=blank_id,
                reduction="mean",
                zero_infinity=True,
            )
            aux[f"deep_sup_layer{layer_idx}"] = ramp * w * loss
        return aux

    return fn
