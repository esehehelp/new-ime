"""Optimizer and LR scheduler factories.

`build_scheduler` returns a `LambdaLR` whose multiplier is `warmup → decay`,
with three modes selected via `OptimSection.schedule`:

    cosine   — warmup then cosine decay from 1.0 to lr_min_ratio
    linear   — warmup then linear decay 1.0 → lr_min_ratio
    constant — warmup then hold at 1.0 (no decay)

Step is the optimizer-step count, not the microbatch index.
"""

from __future__ import annotations

import math

import torch

from new_ime.config.train import OptimSection


def build_optimizer(
    model: torch.nn.Module,
    cfg: OptimSection,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: OptimSection,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    schedule = cfg.schedule
    warmup = max(int(cfg.warmup_steps), 1)
    floor = max(0.0, min(1.0, float(cfg.lr_min_ratio)))
    total_steps = max(int(max_steps), warmup + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "linear":
            return floor + (1.0 - floor) * (1.0 - progress)
        if schedule == "cosine":
            cos_unit = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cos_unit
        raise ValueError(f"unknown schedule: {schedule!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
