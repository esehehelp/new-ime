"""Smoke test: tiny model + tiny shard + 10 optimizer steps.

Asserts the loss trajectory is decreasing (last-3 mean < first-3 mean × 0.95).
This is the minimum signal that all wiring (shard reader, collator,
dataloader, optimizer, scheduler, loop) is correct and the loss is
backpropagating.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable
from new_ime.training.loop import run_loop
from new_ime.training.optim import build_optimizer, build_scheduler


class _OptimCfg:
    lr = 1e-3
    warmup_steps = 2
    schedule = "constant"
    lr_min_ratio = 1.0
    weight_decay = 0.0
    grad_clip = 1.0
    lr_restart_period = 80000
    lr_restart_decay = 0.9


def test_smoke_loss_decreases(
    mock_shard: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    device = torch.device("cpu")
    model = tiny_model_factory(seed=0).to(device)

    optim_cfg = _OptimCfg()
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, max_steps=10)

    ds = KanaKanjiShardIterable(
        mock_shard,
        block_size=8,
        shuffle=True,
        seed=0,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds,
        batch_size=8,
        num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )

    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=loader,
        device=device,
        max_steps=10,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
    )

    assert result.final_step == 10
    losses = [r.loss for r in result.history]
    assert len(losses) == 10
    head = sum(losses[:3]) / 3
    tail = sum(losses[-3:]) / 3
    assert tail < head * 0.95, f"loss not decreasing: head={head:.4f} tail={tail:.4f}"
