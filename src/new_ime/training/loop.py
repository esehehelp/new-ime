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
      the next microbatch boundary and the final step is checkpointed.
    - SIGINT / SIGTERM: first signal sets a flag and exits at the next
      step boundary, calling `on_checkpoint` so the run is resumable.
      A second signal restores the default handler and re-raises so the
      user can force-quit if the graceful path stalls.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from tqdm import tqdm


@dataclass
class StepRecord:
    step: int
    loss: float
    lr: float
    steps_per_sec: float = 0.0


@dataclass
class LoopResult:
    final_step: int
    history: list[StepRecord]
    last_eval: dict | None
    interrupted: bool = False
    stopped_via_file: bool = False


class _SignalCatcher:
    """Catch SIGINT / SIGTERM so the loop can save before exiting.

    First signal flips `flag`; the loop notices at the next step boundary
    and breaks. A second signal restores the previous handler and re-raises
    so the user can force-quit when graceful shutdown stalls (e.g. a
    teacher forward hangs on a long batch).

    Best-effort: signals can only be installed on the main thread, and not
    every signal exists on every platform. Failures are swallowed so the
    training still runs (just without the kill switch).
    """

    def __init__(self) -> None:
        self.flag = False
        self._prev: dict[int, object] = {}

    def __enter__(self) -> "_SignalCatcher":
        def handler(signum, frame):  # noqa: ANN001
            if self.flag:
                signal.signal(signum, self._prev.get(signum, signal.SIG_DFL))  # type: ignore[arg-type]
                raise KeyboardInterrupt
            self.flag = True
            print(
                f"[train] signal {signum} received; saving checkpoint at next step "
                "boundary (send signal again to force-quit)",
                file=sys.stderr,
                flush=True,
            )

        for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                self._prev[int(sig)] = signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *_exc) -> None:
        for sig, prev in self._prev.items():
            try:
                signal.signal(sig, prev)  # type: ignore[arg-type]
            except (ValueError, OSError):
                pass


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
    start_step: int = 0,
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
    """Run until optimizer step `max_steps` over `loader` (which may be infinite)."""
    use_amp = amp_dtype is not None and device.type == "cuda"
    needs_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)

    model.train()
    history: list[StepRecord] = []
    last_eval: dict | None = None
    step = int(start_step)
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    last_checkpointed_step = step
    interrupted = False
    stopped_via_file = False

    # tqdm progress bar over optimizer steps (not microbatches). Output to
    # stderr so stdout-redirected log files capture only the [train]/[eval]
    # lines while the terminal sees a live bar; use `tqdm.write()` for any
    # in-loop logs to avoid tearing the bar.
    pbar_total = max(0, int(max_steps) - int(start_step))
    pbar = tqdm(
        total=pbar_total,
        initial=0,
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.1,
        desc=f"train step {step}",
        leave=True,
    )

    with _SignalCatcher() as catcher:
        iterator = iter(loader)
        micro_idx = 0
        last_log_time = time.monotonic()
        last_log_step = step
        while step < max_steps:
            if catcher.flag:
                interrupted = True
                break
            if stop_file is not None and stop_file.exists():
                stopped_via_file = True
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
            pbar.update(1)
            pbar.set_description(f"train step {step}")

            if step % log_every == 0 or step == max_steps:
                avg = accum_loss / grad_accum
                lr = scheduler.get_last_lr()[0]
                now = time.monotonic()
                elapsed = now - last_log_time
                steps_done = step - last_log_step
                sps = (steps_done / elapsed) if elapsed > 0 and steps_done > 0 else 0.0
                last_log_time = now
                last_log_step = step
                rec = StepRecord(step=step, loss=avg, lr=lr, steps_per_sec=sps)
                history.append(rec)
                if on_log is not None:
                    on_log(rec)
                # Surface latest loss/lr/rate on the tqdm bar without redrawing
                # noisy text inside the loop. Postfix re-renders on next refresh.
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}", rate=f"{sps:.2f}/s")
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
                last_checkpointed_step = step

    pbar.close()

    if (
        (interrupted or stopped_via_file)
        and on_checkpoint is not None
        and step > int(start_step)
        and step != last_checkpointed_step
    ):
        try:
            on_checkpoint(step, last_eval)
        except Exception as e:
            print(
                f"[train] WARNING: final checkpoint save failed: {e}",
                file=sys.stderr,
            )

    return LoopResult(
        final_step=step,
        history=history,
        last_eval=last_eval,
        interrupted=interrupted,
        stopped_via_file=stopped_via_file,
    )


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False
