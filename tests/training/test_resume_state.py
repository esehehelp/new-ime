"""Resume test: save → fresh-init → load → state_dict identity.

Verifies the checkpoint module preserves model/optimizer/scheduler state
faithfully and that `validate_resume_compatibility` accepts a matching
load and rejects an arch-tag mismatch.
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
    schedule = "cosine"
    lr_min_ratio = 0.1
    weight_decay = 0.01
    grad_clip = 1.0
    lr_restart_period = 80000
    lr_restart_decay = 0.9


def _train_steps(model, optimizer, scheduler, loader, device, n: int):
    iterator = iter(loader)
    for _ in range(n):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_ids=batch["target_ids"],
            target_lengths=batch["target_lengths"],
        )
        outputs["loss"].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)


def test_save_load_state_identity(
    tmp_path: Path,
    mock_shard: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    device = torch.device("cpu")
    model_a = tiny_model_factory(seed=0).to(device)
    opt_a = build_optimizer(model_a, _OptimCfg())
    sch_a = build_scheduler(opt_a, _OptimCfg(), max_steps=10)
    ds = KanaKanjiShardIterable(mock_shard, block_size=8, shuffle=False, seed=0)
    loader = DataLoader(
        ds, batch_size=8, num_workers=0, collate_fn=CTCShardCollator(max_seq_len=64)
    )
    _train_steps(model_a, opt_a, sch_a, loader, device, n=5)

    ckpt_path = tmp_path / "step5.pt"
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
    sidecar = ckpt_path.with_name(ckpt_path.stem + "_tokenizer.json")
    assert sidecar.exists()

    # Build a fresh model (different init seed) and load the checkpoint.
    model_b = tiny_model_factory(seed=99).to(device)
    opt_b = build_optimizer(model_b, _OptimCfg())
    sch_b = build_scheduler(opt_b, _OptimCfg(), max_steps=10)
    blob = load(
        ckpt_path,
        model=model_b,
        optimizer=opt_b,
        scheduler=sch_b,
    )
    assert blob["step"] == 5
    assert blob["arch_tag"] == "ctc-nat"

    sa = model_a.state_dict()
    sb = model_b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"state_dict[{k}] differs after load"

    # validate_resume passes with matching arch metadata.
    validate_resume_compatibility(blob, model_b)


def test_validate_resume_rejects_arch_mismatch(
    tmp_path: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    model = tiny_model_factory(seed=0)
    opt = build_optimizer(model, _OptimCfg())
    sch = build_scheduler(opt, _OptimCfg(), max_steps=10)
    ckpt_path = tmp_path / "step0.pt"
    save(
        ckpt_path,
        model=model,
        optimizer=opt,
        scheduler=sch,
        step=0,
        epoch=0,
        best_metric=float("-inf"),
        tokenizer=tiny_tokenizer,
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    blob["arch_tag"] = "ar"  # tamper

    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_compatibility(blob, model)
