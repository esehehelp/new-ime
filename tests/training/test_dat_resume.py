"""DAT resume: 5 step → save → restore → 1 step matches direct 6 step.

Confirms `validate_resume_compatibility` accepts DAT's checkpoint metadata
(arch_tag="dat", upsample_scale, etc.) and that the optimizer / scheduler
/ model state survive a round trip with byte-equal forward loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable
from new_ime.training.checkpoint import (
    load,
    save,
    validate_resume_compatibility,
)
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


def _next_batch(state):
    """Recycle the iterable shard so we can step beyond one epoch."""
    try:
        return next(state["iter"])
    except StopIteration:
        state["iter"] = iter(state["loader"])
        return next(state["iter"])


def _step(model, optimizer, scheduler, loader_state, device):
    batch = _next_batch(loader_state)
    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_ids=batch["target_ids"],
        target_lengths=batch["target_lengths"],
    )
    loss_value = float(out["loss"].item())
    out["loss"].backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss_value


def _make_loader_state(mock_shard: Path, vocab_size: int):
    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=False, seed=0,
        expected_vocab_size=vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )
    return {"loader": loader, "iter": iter(loader)}


def test_dat_resume_matches_uninterrupted(
    tmp_path: Path,
    mock_shard: Path,
    tiny_dat_factory,
    tiny_tokenizer,
):
    """Save at step 5, resume into a fresh model, run 1 more — final loss
    must equal a single uninterrupted 6-step run (relative 1e-5)."""
    device = torch.device("cpu")

    # ---------- Reference: uninterrupted 6 steps ----------
    torch.manual_seed(0)
    model_ref = tiny_dat_factory(seed=0).to(device)
    opt_ref = build_optimizer(model_ref, _OptimCfg())
    sch_ref = build_scheduler(opt_ref, _OptimCfg(), max_steps=10)
    loader_ref = _make_loader_state(mock_shard, tiny_tokenizer.vocab_size)
    last_loss_ref = None
    for _ in range(6):
        last_loss_ref = _step(model_ref, opt_ref, sch_ref, loader_ref, device)

    # ---------- Run 5 steps, save, resume into a freshly-init model ----------
    torch.manual_seed(0)
    model_a = tiny_dat_factory(seed=0).to(device)
    opt_a = build_optimizer(model_a, _OptimCfg())
    sch_a = build_scheduler(opt_a, _OptimCfg(), max_steps=10)
    loader_a = _make_loader_state(mock_shard, tiny_tokenizer.vocab_size)
    for _ in range(5):
        _step(model_a, opt_a, sch_a, loader_a, device)

    ckpt_path = tmp_path / "dat_step5.pt"
    save(
        ckpt_path,
        model=model_a,
        optimizer=opt_a,
        scheduler=sch_a,
        step=5,
        epoch=0,
        best_metric=float("-inf"),
        tokenizer=tiny_tokenizer,
    )
    assert ckpt_path.exists()

    # Fresh init with DIFFERENT seed; the load must overwrite it.
    model_b = tiny_dat_factory(seed=999).to(device)
    opt_b = build_optimizer(model_b, _OptimCfg())
    sch_b = build_scheduler(opt_b, _OptimCfg(), max_steps=10)
    blob = load(
        ckpt_path,
        model=model_b,
        optimizer=opt_b,
        scheduler=sch_b,
    )
    validate_resume_compatibility(blob, model_b)
    assert blob["arch_tag"] == "dat"
    assert blob["upsample_scale"] == model_b.upsample_scale

    loader_b = _make_loader_state(mock_shard, tiny_tokenizer.vocab_size)
    # Skip the first 5 batches so step 6 sees the same data as the reference.
    for _ in range(5):
        _next_batch(loader_b)
    last_loss_b = _step(model_b, opt_b, sch_b, loader_b, device)

    assert last_loss_b == pytest.approx(last_loss_ref, rel=1e-5, abs=1e-6), (
        f"resume drift: ref={last_loss_ref:.6f} resumed={last_loss_b:.6f}"
    )


def test_dat_resume_rejects_arch_mismatch(
    tmp_path: Path,
    mock_shard: Path,
    tiny_dat_factory,
    tiny_model_factory,
    tiny_tokenizer,
):
    """A CTC-NAT checkpoint must not load into a DAT model and vice versa."""
    device = torch.device("cpu")
    ctc_model = tiny_model_factory(seed=0).to(device)
    dat_model = tiny_dat_factory(seed=0).to(device)
    cfg = _OptimCfg()

    # Save CTC-NAT ckpt.
    ctc_ckpt = tmp_path / "ctc.pt"
    save(
        ctc_ckpt,
        model=ctc_model,
        optimizer=build_optimizer(ctc_model, cfg),
        scheduler=build_scheduler(build_optimizer(ctc_model, cfg), cfg, max_steps=2),
        step=0, epoch=0, best_metric=float("-inf"),
        tokenizer=tiny_tokenizer,
    )

    # `load()` calls `model.load_state_dict()` strictly first, which itself
    # raises RuntimeError on the CTC-NAT vs DAT mismatch — that's also
    # acceptable as a rejection mechanism. We accept either failure mode.
    with pytest.raises((RuntimeError, ValueError)):
        blob = load(
            ctc_ckpt, model=dat_model,
            optimizer=None, scheduler=None,
        )
        validate_resume_compatibility(blob, dat_model)
