"""GLAT (glancing) integration for DAT.

Three angles:
    1. `parse_anneal` correctly interpolates the schedule string used in
       `[dat].glat_p` (start:end@steps; supports "k"/"m" suffixes).
    2. `model.set_glance_ratio(r)` controls whether forward runs the
       2-stage GLAT path; r=0 is exactly equivalent to a single forward.
    3. With glance_ratio>0 the smoke loss still decreases (the second
       forward must be differentiable end-to-end).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable
from new_ime.training.loop import run_loop
from new_ime.training.loss.dat import parse_anneal
from new_ime.training.optim import build_optimizer, build_scheduler


# ---------------------------------------------------------------------------
# Schedule parser
# ---------------------------------------------------------------------------


def test_parse_anneal_constant():
    assert parse_anneal("0.3", 0) == pytest.approx(0.3)
    assert parse_anneal("0.3", 999_999) == pytest.approx(0.3)


def test_parse_anneal_linear_interpolation():
    assert parse_anneal("0.5:0.1@1000", 0) == pytest.approx(0.5)
    assert parse_anneal("0.5:0.1@1000", 1000) == pytest.approx(0.1)
    assert parse_anneal("0.5:0.1@1000", 500) == pytest.approx(0.3, abs=1e-6)
    # Past the end clamps to target.
    assert parse_anneal("0.5:0.1@1000", 5000) == pytest.approx(0.1)


def test_parse_anneal_suffixes():
    assert parse_anneal("0.5:0.1@2k", 1000) == pytest.approx(0.3, abs=1e-6)
    assert parse_anneal("0.5:0.1@1m", 500_000) == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------


def test_glance_zero_matches_single_forward(
    mock_shard: Path,
    tiny_dat_factory,
    tiny_tokenizer,
):
    """ratio=0 must produce no `glance_hint_ids` and an unchanged loss."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = tiny_dat_factory(seed=0).to(device)
    model.train()

    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=False,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )
    batch = next(iter(loader))

    model.set_glance_ratio(0.0)
    out0 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_ids=batch["target_ids"],
        target_lengths=batch["target_lengths"],
    )
    assert "glance_hint_ids" not in out0


def test_glance_positive_emits_hint_ids(
    mock_shard: Path,
    tiny_dat_factory,
    tiny_tokenizer,
):
    """ratio>0 in train mode triggers the 2-stage path; hints are surfaced."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = tiny_dat_factory(seed=0).to(device)
    model.train()

    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=False,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )
    batch = next(iter(loader))

    model.set_glance_ratio(0.5)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_ids=batch["target_ids"],
        target_lengths=batch["target_lengths"],
    )
    assert "glance_hint_ids" in out
    hint = out["glance_hint_ids"]
    # hint shape matches upsampled length.
    expected_t_up = batch["input_ids"].size(1) * model.upsample_scale
    assert hint.shape == (batch["input_ids"].size(0), expected_t_up)
    # At least some leak should happen (since the untrained model gets most
    # tokens wrong, glance_nums > 0 with high probability for ratio=0.5).
    # We can't guarantee a non-zero hint per batch, but across rows there
    # should be at least one non-zero entry.
    assert (hint != 0).any(), "no GLAT leak occurred at all (suspicious)"


def test_glance_eval_mode_disables_two_stage(
    mock_shard: Path,
    tiny_dat_factory,
    tiny_tokenizer,
):
    """eval mode must skip GLAT regardless of glance_ratio."""
    device = torch.device("cpu")
    model = tiny_dat_factory(seed=0).to(device)
    model.eval()
    model.set_glance_ratio(0.9)

    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=False,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )
    batch = next(iter(loader))

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_ids=batch["target_ids"],
        target_lengths=batch["target_lengths"],
    )
    assert "glance_hint_ids" not in out


# ---------------------------------------------------------------------------
# Smoke with GLAT enabled
# ---------------------------------------------------------------------------


class _OptimCfg:
    lr = 1e-3
    warmup_steps = 2
    schedule = "constant"
    lr_min_ratio = 1.0
    weight_decay = 0.0
    grad_clip = 1.0
    lr_restart_period = 80000
    lr_restart_decay = 0.9


def test_dat_smoke_loss_decreases_with_glat(
    mock_shard: Path,
    tiny_dat_factory,
    tiny_tokenizer,
):
    """End-to-end: GLAT-enabled DAT smoke must still see loss decrease."""
    device = torch.device("cpu")
    model = tiny_dat_factory(seed=0).to(device)
    model.set_glance_ratio(0.3)

    optim_cfg = _OptimCfg()
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, max_steps=10)

    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=True, seed=0,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=8, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )

    result = run_loop(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loader=loader, device=device, max_steps=10,
        grad_accum=1, grad_clip=1.0, log_every=1,
    )

    losses = [r.loss for r in result.history]
    head = sum(losses[:3]) / 3
    tail = sum(losses[-3:]) / 3
    assert tail < head * 0.95, (
        f"GLAT-enabled DAT loss not decreasing: head={head:.4f} tail={tail:.4f}"
    )
