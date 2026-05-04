"""Stage 4 helpers: probe EM1, curriculum collator update, memory estimate."""

from __future__ import annotations

import torch

from new_ime.data.shards import CTCShardCollator
from new_ime.eval.loaders import BenchItem
from new_ime.training.curriculum import apply_short_sample_warmup
from new_ime.training.evaluate import evaluate_probe_em1
from new_ime.training.memory import (
    estimate_training_memory,
    format_memory_table,
)


def test_evaluate_probe_em1_returns_schema(tiny_model_factory, tiny_tokenizer):
    device = torch.device("cpu")
    model = tiny_model_factory(seed=0).to(device)
    items = [
        BenchItem(reading="あい", context="", references=["愛"], source="probe", category="t"),
        BenchItem(reading="うえ", context="", references=["上"], source="probe", category="t"),
        BenchItem(reading="おか", context="", references=["岡"], source="probe", category="t"),
    ]
    result = evaluate_probe_em1(
        model=model,
        probe_items=items,
        tokenizer=tiny_tokenizer,
        device=device,
        max_seq_len=64,
        max_context=32,
        limit=0,
    )
    assert set(result.keys()) == {"em1", "n"}
    assert result["n"] == 3
    assert 0.0 <= result["em1"] <= 1.0


def test_curriculum_warmup_toggle():
    collator = CTCShardCollator(max_seq_len=128, short_sample_max_chars=0)

    apply_short_sample_warmup(
        collator, step=0, warmup_steps=100, short_max_chars=8
    )
    assert collator.short_sample_max_chars == 8

    apply_short_sample_warmup(
        collator, step=99, warmup_steps=100, short_max_chars=8
    )
    assert collator.short_sample_max_chars == 8

    apply_short_sample_warmup(
        collator, step=100, warmup_steps=100, short_max_chars=8
    )
    assert collator.short_sample_max_chars == 0

    apply_short_sample_warmup(
        collator, step=0, warmup_steps=0, short_max_chars=8
    )
    assert collator.short_sample_max_chars == 0


def test_memory_estimate_table(tiny_model_factory):
    model = tiny_model_factory(seed=0)
    est = estimate_training_memory(
        model, batch_size=8, seq_len=64, bytes_per_param=4
    )
    assert est.params_gb > 0
    assert est.optimizer_gb > 0
    assert est.activations_gb >= 0
    assert est.total_gb >= est.params_gb + est.optimizer_gb
    table = format_memory_table(est, peak_vram_gb=None)
    assert "params" in table
    assert "total" in table
