"""Smoke test for evaluate_model: returns the expected schema, no crash."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable
from new_ime.training.evaluate import evaluate_model


def test_evaluate_returns_expected_keys(
    mock_shard: Path,
    tiny_model_factory,
    tiny_tokenizer,
):
    device = torch.device("cpu")
    model = tiny_model_factory(seed=0).to(device)
    ds = KanaKanjiShardIterable(mock_shard, block_size=8, shuffle=False)
    loader = DataLoader(
        ds, batch_size=8, num_workers=0, collate_fn=CTCShardCollator(max_seq_len=64)
    )

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        tokenizer=tiny_tokenizer,
        max_batches=1,
    )

    assert set(metrics.keys()) == {
        "loss",
        "exact_match_top1",
        "char_acc_top1",
        "blank_fraction",
        "num_samples",
    }
    assert metrics["num_samples"] == 8
    assert 0.0 <= metrics["exact_match_top1"] <= 1.0
    assert 0.0 <= metrics["blank_fraction"] <= 1.0
    assert metrics["loss"] > 0  # untrained tiny model: loss should be > 0
