"""Refine 3-part loss: returns refine_loss + remask_loss + stop_loss > 0
for an untrained model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from new_ime.config.train import RefineSection
from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable
from new_ime.training.loss.refine import build_refine_loss_fn


def test_refine_loss_components(
    mock_shard: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    device = torch.device("cpu")
    model = tiny_model_factory(seed=0).to(device)
    cfg = RefineSection(
        loss_weight=1.0,
        warmup_steps=0,
        mask_ratio_min=0.3,
        mask_ratio_max=0.3,
        refine_source="target",
        remask_loss_weight=0.5,
        stop_loss_weight=0.5,
    )
    fn = build_refine_loss_fn(cfg, mask_id=tiny_tokenizer.mask_id)

    ds = KanaKanjiShardIterable(mock_shard, block_size=8, shuffle=False)
    loader = DataLoader(
        ds, batch_size=8, num_workers=0, collate_fn=CTCShardCollator(max_seq_len=64)
    )
    batch = next(iter(loader))
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
    aux = fn(model, batch, outputs, step=100)

    assert "refine_loss" in aux
    assert "remask_loss" in aux
    assert "stop_loss" in aux
    for k, v in aux.items():
        assert v.requires_grad, f"{k} should require grad"
        assert v.item() > 0, f"{k} should be > 0 for untrained model, got {v.item()}"


def test_refine_loss_skipped_during_warmup(
    mock_shard: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    """loss_weight warmup: when step < warmup_steps the resolved weight is < max,
    and at step=0 with warmup>0 the weight is 0 → fn returns {}.
    """
    device = torch.device("cpu")
    model = tiny_model_factory(seed=0).to(device)
    cfg = RefineSection(
        loss_weight=1.0,
        warmup_steps=100,
    )
    fn = build_refine_loss_fn(cfg, mask_id=tiny_tokenizer.mask_id)

    ds = KanaKanjiShardIterable(mock_shard, block_size=8, shuffle=False)
    loader = DataLoader(
        ds, batch_size=8, num_workers=0, collate_fn=CTCShardCollator(max_seq_len=64)
    )
    batch = next(iter(loader))
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
    aux = fn(model, batch, outputs, step=0)
    assert aux == {}, "step=0 with warmup_steps=100 should return empty dict"
