"""Kill switch + final checkpoint guarantees.

Two paths trip the same shutdown logic:
    1. STOP file appears in out_dir.
    2. SIGINT / SIGTERM is delivered to the process.

In both cases the loop must exit at the next step boundary AND the
caller's `on_checkpoint` must be invoked once more so the run is
resumable from the very last optimizer step (not just from the last
periodic checkpoint).
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from new_ime.training.loop import run_loop
from new_ime.training.optim import build_optimizer, build_scheduler


class _StubModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_lengths: torch.Tensor,
        **_kwargs,
    ) -> dict:
        h = self.emb(input_ids).mean(dim=1)
        logits = self.head(h)
        loss = nn.functional.cross_entropy(logits, target_ids[:, 0])
        return {"logits": logits, "loss": loss}


class _OptimCfg:
    lr = 1e-2
    warmup_steps = 1
    schedule = "constant"
    lr_min_ratio = 1.0
    weight_decay = 0.0
    grad_clip = 1.0
    lr_restart_period = 1
    lr_restart_decay = 1.0


def _toy_loader(num_batches: int = 5, batch_size: int = 4, vocab: int = 32):
    torch.manual_seed(0)
    inp = torch.randint(0, vocab, (num_batches, batch_size, 8))
    tgt = torch.randint(0, vocab, (num_batches, batch_size, 4))
    while True:
        for i in range(num_batches):
            yield {
                "input_ids": inp[i],
                "attention_mask": torch.ones_like(inp[i]),
                "target_ids": tgt[i],
                "target_lengths": torch.full((batch_size,), 4, dtype=torch.long),
            }


def _build(max_steps: int = 100):
    model = _StubModel()
    cfg = _OptimCfg()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max_steps=max_steps)
    return model, optimizer, scheduler


def test_stop_file_triggers_final_checkpoint(tmp_path: Path) -> None:
    """STOP file appears at step 3 → loop exits, on_checkpoint called once with step=3."""
    stop_file = tmp_path / "STOP"
    ckpt_calls: list[int] = []

    def on_checkpoint(step: int, _metrics: dict | None) -> None:
        ckpt_calls.append(step)

    def on_log(rec) -> None:  # touch STOP after step 3 logs
        if rec.step == 3:
            stop_file.touch()

    model, optimizer, scheduler = _build()
    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=_toy_loader(),
        device=torch.device("cpu"),
        max_steps=100,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
        checkpoint_every=0,  # disable periodic; only the kill-switch save should fire
        on_checkpoint=on_checkpoint,
        stop_file=stop_file,
        on_log=on_log,
    )

    assert result.stopped_via_file is True
    assert result.interrupted is False
    # on_log(3) touches STOP after step 3 logs (post-completion). The next
    # loop iteration's top-of-loop check sees STOP and breaks; final_step==3.
    assert result.final_step == 3
    assert ckpt_calls == [3], f"expected single final-save at step 3, got {ckpt_calls}"


def test_stop_file_does_not_double_checkpoint(tmp_path: Path) -> None:
    """If the periodic save just fired at the same step, don't save twice."""
    stop_file = tmp_path / "STOP"
    ckpt_calls: list[int] = []

    def on_checkpoint(step: int, _metrics: dict | None) -> None:
        ckpt_calls.append(step)
        # Touch STOP immediately after the periodic save at step 4 returns.
        # The next loop iteration's top check will see STOP and exit at step=4
        # — and the shutdown save must skip because last_checkpointed_step==4.
        if step == 4:
            stop_file.touch()

    model, optimizer, scheduler = _build()
    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=_toy_loader(),
        device=torch.device("cpu"),
        max_steps=100,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
        checkpoint_every=2,
        on_checkpoint=on_checkpoint,
        stop_file=stop_file,
    )

    assert result.stopped_via_file is True
    assert result.final_step == 4
    assert ckpt_calls == [2, 4], f"unexpected ckpt sequence: {ckpt_calls}"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="signal.raise_signal of SIGINT on Windows console is racy under pytest",
)
def test_sigint_triggers_final_checkpoint() -> None:
    """First SIGINT sets the flag; loop exits and saves at the next boundary."""
    ckpt_calls: list[int] = []

    def on_checkpoint(step: int, _metrics: dict | None) -> None:
        ckpt_calls.append(step)

    def on_log(rec) -> None:
        if rec.step == 3:
            os.kill(os.getpid(), signal.SIGINT)

    model, optimizer, scheduler = _build()
    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=_toy_loader(),
        device=torch.device("cpu"),
        max_steps=100,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
        checkpoint_every=0,
        on_checkpoint=on_checkpoint,
        on_log=on_log,
    )

    assert result.interrupted is True
    assert result.stopped_via_file is False
    # SIGINT delivered during on_step_start of step 3; signal handler runs
    # before step 4's top-of-loop check, so we exit at step 4 boundary at the
    # latest. Allow a small slack (3..5) for delivery timing.
    assert 3 <= result.final_step <= 5
    assert ckpt_calls == [result.final_step]
