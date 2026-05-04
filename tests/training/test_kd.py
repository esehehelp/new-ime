"""KD module: alpha schedule, hard-example mask, KL loss, factory gating."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from new_ime.config.train import KdSection
from new_ime.training.loss.kd import (
    alpha_at,
    build_kd_loss_fn,
    compute_kd_kl_loss,
    hard_example_mask,
)


def _kd_cfg(**overrides) -> KdSection:
    base = dict(
        teacher_type="ctc",
        teacher_path=Path("/dev/null"),
        alpha=0.5,
        alpha_final=0.0,
        start_step=0,
        warmup_steps=0,
        alpha_decay_start=0,
        alpha_decay_steps=0,
        every=1,
        gate_mode="all",
        hard_threshold=0.5,
        temperature=1.0,
    )
    base.update(overrides)
    return KdSection(**base)


def test_alpha_at_warmup_then_decay():
    cfg = _kd_cfg(
        alpha=1.0, alpha_final=0.1,
        start_step=10, warmup_steps=10,
        alpha_decay_start=50, alpha_decay_steps=50,
    )
    assert alpha_at(0, cfg) == 0.0
    assert alpha_at(10, cfg) == 0.0
    assert alpha_at(15, cfg) == pytest.approx(0.5)
    assert alpha_at(20, cfg) == pytest.approx(1.0)
    assert alpha_at(40, cfg) == pytest.approx(1.0)
    assert alpha_at(75, cfg) == pytest.approx(1.0 + (0.1 - 1.0) * 0.5)
    assert alpha_at(100, cfg) == pytest.approx(0.1)


def test_hard_example_mask_modes():
    conf = torch.tensor([0.2, 0.6, 0.9])
    assert hard_example_mask(conf, 0.5, "all").all().item()
    assert hard_example_mask(conf, 0.5, "low_conf").tolist() == [True, False, False]
    assert hard_example_mask(conf, 0.5, "high_conf").tolist() == [False, True, True]
    with pytest.raises(ValueError):
        hard_example_mask(conf, 0.5, "bogus")


def test_compute_kd_kl_zero_when_logits_match():
    B, T, V = 2, 4, 8
    logits = torch.randn(B, T, V)
    attn = torch.ones(B, T, dtype=torch.long)
    hard = torch.ones(B, dtype=torch.bool)
    loss, n = compute_kd_kl_loss(logits, logits.clone(), attn, hard, temperature=1.0)
    assert n == 2
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)


def test_compute_kd_kl_skipped_when_no_hard_examples():
    B, T, V = 2, 4, 8
    student = torch.randn(B, T, V)
    teacher = torch.randn(B, T, V)
    attn = torch.ones(B, T, dtype=torch.long)
    hard = torch.zeros(B, dtype=torch.bool)
    loss, n = compute_kd_kl_loss(student, teacher, attn, hard)
    assert n == 0
    assert float(loss.item()) == 0.0


class _StubTeacher:
    arch_tag = "stub-teacher"

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def forward(self, batch: dict) -> dict:
        B, T = batch["input_ids"].shape
        return {
            "logits": torch.randn(B, T, self.vocab_size),
            "confidence": torch.full((B,), 0.8),
        }


def test_build_kd_loss_fn_skip_alpha_zero():
    cfg = _kd_cfg(alpha=0.0)
    teacher = _StubTeacher(vocab_size=16)
    fn = build_kd_loss_fn(cfg, teacher)
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    outputs = {"logits": torch.randn(2, 4, 16)}
    assert fn(model=None, batch=batch, outputs=outputs, step=10) == {}


def test_build_kd_loss_fn_emits_kd_kl_when_active():
    cfg = _kd_cfg(alpha=0.3)
    teacher = _StubTeacher(vocab_size=16)
    fn = build_kd_loss_fn(cfg, teacher)
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    outputs = {"logits": torch.randn(2, 4, 16, requires_grad=True)}
    aux = fn(model=None, batch=batch, outputs=outputs, step=10)
    assert set(aux.keys()) == {"kd_kl"}
    assert aux["kd_kl"].item() > 0


def test_build_kd_loss_fn_skip_before_start_step():
    cfg = _kd_cfg(alpha=0.5, start_step=100)
    teacher = _StubTeacher(vocab_size=16)
    fn = build_kd_loss_fn(cfg, teacher)
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    outputs = {"logits": torch.randn(2, 4, 16)}
    assert fn(model=None, batch=batch, outputs=outputs, step=50) == {}
