"""The loop must run on any model whose forward returns {"loss": Tensor, ...}.

This guards the design constraint that loop.py never imports CTCNAT or any
other arch-specific class. A future AR / DAT model that respects the same
output contract should be plug-compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from new_ime.training.loop import run_loop
from new_ime.training.optim import build_optimizer, build_scheduler


class _StubModel(nn.Module):
    """Minimal arch-agnostic stand-in. Linear → CE on first target token."""

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


def test_loop_runs_on_stub_model():
    """No reference to CTCNAT / refine / KD: loop should still complete."""
    model = _StubModel()
    cfg = _OptimCfg()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max_steps=8)

    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=_toy_loader(),
        device=torch.device("cpu"),
        max_steps=8,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
    )
    assert result.final_step == 8
    assert len(result.history) == 8


def test_loop_runs_on_dat_model(mock_shard, tiny_dat_factory, tiny_tokenizer):
    """The same arch-agnostic loop accepts DAT (different output dict shape,
    different aux structure) without any DAT-specific branches. Rough proxy
    for the contract `model.forward(...)["loss"]` only."""
    from torch.utils.data import DataLoader

    from new_ime.data.shards import CTCShardCollator, KanaKanjiShardIterable

    device = torch.device("cpu")
    model = tiny_dat_factory(seed=0).to(device)
    cfg = _OptimCfg()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max_steps=4)
    ds = KanaKanjiShardIterable(
        mock_shard, block_size=8, shuffle=True, seed=0,
        expected_vocab_size=tiny_tokenizer.vocab_size,
    )
    loader = DataLoader(
        ds, batch_size=4, num_workers=0,
        collate_fn=CTCShardCollator(max_seq_len=64),
    )

    result = run_loop(
        model=model, optimizer=optimizer, scheduler=scheduler,
        loader=loader, device=device,
        max_steps=4, grad_accum=1, grad_clip=1.0, log_every=1,
    )
    assert result.final_step == 4
    assert len(result.history) == 4


def test_loop_resumes_from_start_step():
    """Resume path treats max_steps as the absolute target optimizer step."""
    model = _StubModel()
    cfg = _OptimCfg()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, max_steps=8)

    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=_toy_loader(),
        device=torch.device("cpu"),
        max_steps=8,
        start_step=5,
        grad_accum=1,
        grad_clip=1.0,
        log_every=1,
    )
    assert result.final_step == 8
    assert [r.step for r in result.history] == [6, 7, 8]
