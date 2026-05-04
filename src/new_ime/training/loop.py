"""Model-agnostic training loop.

Pulls a `loss` tensor out of `model(**batch)`'s output dict, optionally
adds aux losses via `model.compute_aux_losses(batch, outputs)`, and steps
the optimizer with grad accumulation + clip. Does not import a specific
architecture: any model whose forward returns `{"loss": Tensor, ...}` plugs
in.

Stage 2 surface area:
    - AMP: pass `amp_dtype=torch.float16` (FP16 path with GradScaler) or
      `torch.bfloat16` (bf16 path, no scaler). `None` runs in fp32.
    - eval rhythm: `eval_every` + `on_eval(step) -> dict | None`. The
      returned dict (best metric candidate) is propagated to `on_checkpoint`
      so the caller can use it for `best.pt` selection.
    - checkpoint rhythm: `checkpoint_every` + `on_checkpoint(step, eval_metrics)`.
    - STOP file: when `out_dir/STOP` appears, the loop exits cleanly at
      the next microbatch boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch


@dataclass
class StepRecord:
    step: int
    loss: float
    lr: float


@dataclass
class LoopResult:
    final_step: int
    history: list[StepRecord]
    last_eval: dict | None


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _compute_aux(
    model: torch.nn.Module,
    batch: dict,
    outputs: dict,
    aux_loss_fns: list | None,
    step: int,
) -> dict:
    aux: dict = {}
    fn = getattr(model, "compute_aux_losses", None)
    if fn is not None:
        result = fn(batch, outputs)
        if result:
            aux.update(result)
    if aux_loss_fns:
        for f in aux_loss_fns:
            result = f(model, batch, outputs, step)
            if result:
                aux.update(result)
    return aux


def _forward_tensor_kwargs(batch: dict) -> dict:
    return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}


def run_loop(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: Iterable[dict],
    device: torch.device,
    max_steps: int,
    grad_accum: int = 1,
    grad_clip: float = 1.0,
    log_every: int = 100,
    amp_dtype: torch.dtype | None = None,
    aux_loss_fns: list[Callable[[torch.nn.Module, dict, dict, int], dict]] | None = None,
    eval_every: int = 0,
    on_eval: Callable[[int], dict | None] | None = None,
    checkpoint_every: int = 0,
    on_checkpoint: Callable[[int, dict | None], None] | None = None,
    stop_file: Path | None = None,
    on_log: Callable[[StepRecord], None] | None = None,
    on_step_start: Callable[[int], None] | None = None,
) -> LoopResult:
    """Run up to `max_steps` optimizer steps over `loader` (which may be infinite)."""
    use_amp = amp_dtype is not None and device.type == "cuda"
    needs_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    model.train()
    history: list[StepRecord] = []
    last_eval: dict | None = None
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    iterator = iter(loader)
    micro_idx = 0
    while step < max_steps:
        if stop_file is not None and stop_file.exists():
            break
        if on_step_start is not None and micro_idx % grad_accum == 0:
            on_step_start(step)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = _move_to_device(batch, device)

        autocast_ctx = (
            torch.amp.autocast(device.type, dtype=amp_dtype)
            if use_amp
            else _NullCtx()
        )
        with autocast_ctx:
            outputs = model(**_forward_tensor_kwargs(batch))
            loss = outputs["loss"]
            for v in _compute_aux(model, batch, outputs, aux_loss_fns, step).values():
                loss = loss + v

        scaled = loss / grad_accum
        if needs_scaler:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        accum_loss += float(loss.detach().item())
        micro_idx += 1

        if micro_idx % grad_accum != 0:
            continue

        if needs_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if needs_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0 or step == max_steps:
            avg = accum_loss / grad_accum
            lr = scheduler.get_last_lr()[0]
            rec = StepRecord(step=step, loss=avg, lr=lr)
            history.append(rec)
            if on_log is not None:
                on_log(rec)
        accum_loss = 0.0

        eval_due = eval_every > 0 and on_eval is not None and step % eval_every == 0
        if eval_due:
            last_eval = on_eval(step)

        if (
            checkpoint_every > 0
            and on_checkpoint is not None
            and step % checkpoint_every == 0
        ):
            on_checkpoint(step, last_eval)

    return LoopResult(final_step=step, history=history, last_eval=last_eval)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False
