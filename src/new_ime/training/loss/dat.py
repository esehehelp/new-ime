"""DAT loss orchestration.

The DP loss itself lives on `DAT.forward()` (it depends on the model's
own logits/links), so this module is intentionally thin: it exposes a
factory that returns a no-op aux closure for parity with the CTC-NAT
`build_refine_loss_fn` signature, and a `parse_anneal()` helper used by
the GLAT schedule (Stage 4). The training loop already adds
`outputs["loss"]` from the model — no aux contribution is needed.
"""

from __future__ import annotations

from typing import Callable

import torch

from new_ime.config.train import DatSection


def parse_anneal(spec: str, step: int) -> float:
    """Linear anneal spec parser, e.g. ``"0.5:0.1@200000"`` or ``"0.3:0.0@100k"``.

    ``a:b@N`` linearly interpolates from ``a`` at step 0 to ``b`` at step ``N``.
    A bare ``"0.3"`` is treated as the constant 0.3. ``k`` / ``m`` suffixes
    on the step count expand to thousands / millions for readability.
    """
    if "@" not in spec:
        return float(spec)
    rates, end = spec.split("@", 1)
    if ":" not in rates:
        raise ValueError(f"Invalid anneal spec {spec!r}: expected 'start:end@steps'")
    start_s, end_s = rates.split(":", 1)
    start = float(start_s)
    target = float(end_s)
    end_lower = end.lower().strip()
    if end_lower.endswith("k"):
        end_steps = int(float(end_lower[:-1]) * 1_000)
    elif end_lower.endswith("m"):
        end_steps = int(float(end_lower[:-1]) * 1_000_000)
    else:
        end_steps = int(end_lower)
    if end_steps <= 0:
        return target
    if step >= end_steps:
        return target
    if step <= 0:
        return start
    frac = step / end_steps
    return start + (target - start) * frac


def build_dat_loss_fn(
    cfg: DatSection,
) -> Callable[[torch.nn.Module, dict, dict, int], dict[str, torch.Tensor]]:
    """Returns a no-op aux closure. The actual DAG DP loss is computed
    inside `DAT.forward()` and surfaced as `outputs["loss"]`, which the
    training loop already accounts for.
    """

    def fn(
        model: torch.nn.Module,
        batch: dict,
        outputs: dict,
        step: int,
    ) -> dict[str, torch.Tensor]:
        # Reserved for future DAT-only side losses (e.g. label smoothing
        # contribution beyond the DP). Currently empty.
        return {}

    return fn
