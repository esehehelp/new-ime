#!/usr/bin/env python
"""Emit parity fixtures for the Rust tch trainer.

The Rust trainer ports a piece of the Python CTC-NAT trainer at a time.
For each piece we save the exact inputs + outputs of the Python reference
so the corresponding Rust test can load them and check numerical
equivalence without having to keep Python and Rust running side-by-side
on every cargo test.

Scope of this fixture set (matches the plan Step 5):
  1. CTC loss  — random logits/targets, F.ctc_loss with reduction='mean',
     zero_infinity=True, blank=BLANK_ID. Rust ctc_proposal_loss must
     match within 1e-5.
  2. Warmup-cosine LR schedule — sampled at a handful of steps using the
     same formula as Python's LambdaLR warmup_cosine. Rust TchOptimizer
     must match within 1e-9.

Full architectural parity (end-to-end model forward + 1 optim step) is
deferred to a follow-up; it requires a custom Python model mirroring
the Rust layout exactly, which is a larger undertaking.

Output layout:
  parity-fixtures/
    ctc_random.safetensors   — logits, targets, input_lengths, target_lengths
    ctc_random.json          — scalars: blank_id, expected_loss
    lr_schedule.json         — config + sampled (step, lr) points
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "parity-fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def emit_ctc_loss_fixture(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch = 3
    time_steps = 12
    vocab = 16
    blank_id = 0
    max_target = 6

    logits = torch.randn(batch, time_steps, vocab, dtype=torch.float32)
    input_lengths = torch.tensor([time_steps, time_steps - 1, time_steps - 3], dtype=torch.int64)
    target_lengths = torch.tensor([4, 3, 5], dtype=torch.int64)
    targets = torch.zeros(batch, max_target, dtype=torch.int64)
    for b in range(batch):
        # Draw non-blank ids for the first `target_lengths[b]` positions.
        for t in range(int(target_lengths[b].item())):
            tok = int(torch.randint(1, vocab, (1,)).item())
            targets[b, t] = tok

    # CTC loss expects [T, B, V] log_probs.
    log_probs = logits.log_softmax(-1).permute(1, 0, 2)
    loss = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )

    save_file(
        {
            "logits": logits.numpy(),
            "targets": targets.numpy(),
            "input_lengths": input_lengths.numpy(),
            "target_lengths": target_lengths.numpy(),
        },
        OUT_DIR / "ctc_random.safetensors",
    )
    (OUT_DIR / "ctc_random.json").write_text(
        json.dumps(
            {
                "blank_id": blank_id,
                "batch": batch,
                "time_steps": time_steps,
                "vocab": vocab,
                "max_target": max_target,
                "expected_loss": float(loss.item()),
                "seed": seed,
            },
            indent=2,
        )
    )
    print(f"[ctc] expected_loss={loss.item():.9f}")


def warmup_cosine_lr(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float,
) -> float:
    """Python reference for the warmup-cosine schedule the Rust trainer
    ships. Matches the CPU `OptimizerState::current_lr` impl used by
    the existing Phase 3 trainer, so keeping parity lets us reuse the
    existing Python logs for sanity checks.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(
        max((step - warmup_steps) / (total_steps - warmup_steps), 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_scale + (1.0 - min_lr_scale) * cosine)


def emit_schedule_fixture() -> None:
    cfg = dict(base_lr=1e-3, warmup_steps=100, total_steps=1000, min_lr_scale=0.1)
    samples = []
    for step in [0, 1, 50, 100, 101, 250, 500, 750, 999, 1000, 2000]:
        lr = warmup_cosine_lr(step, **cfg)
        samples.append({"step": step, "lr": lr})
    payload = {"config": cfg, "samples": samples}
    (OUT_DIR / "lr_schedule.json").write_text(json.dumps(payload, indent=2))
    print(f"[schedule] wrote {len(samples)} samples")


def main() -> None:
    emit_ctc_loss_fixture()
    emit_schedule_fixture()
    print(f"wrote parity fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
