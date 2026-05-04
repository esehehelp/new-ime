"""Training entry point. Builds device / tokenizer / model / loaders / loop.

Dataset dispatch:
    cfg.data.train ending in `.kkc`        → KanaKanjiShardIterable
    anything else                          → KanaKanjiJsonlDataset
    cfg.data.dev: same dispatch (map-style for shard, eager for JSONL).

AMP:
    cfg.loop.bf16    → bfloat16 (CUDA only; falls back to fp16 if GPU
                       lacks bf16 support).
    cfg.loop.fp16    → float16 (with GradScaler).
    neither / no CUDA → fp32.

eval / checkpoint rhythm + STOP file are wired here as callbacks; the
loop itself stays model-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from new_ime.config.train import TrainConfig
from new_ime.data.jsonl import KanaKanjiJsonlDataset
from new_ime.data.shards import (
    CTCShardCollator,
    KanaKanjiShardDataset,
    KanaKanjiShardIterable,
)
from new_ime.data.tokenizer import SharedCharTokenizer
from new_ime.model.ctc_nat import CTCNAT
from new_ime.training.checkpoint import rolling_keep, save
from new_ime.training.curriculum import apply_short_sample_warmup
from new_ime.training.evaluate import evaluate_model, evaluate_probe_em1
from new_ime.training.loop import StepRecord, run_loop
from new_ime.training.loss.refine import build_refine_loss_fn
from new_ime.training.memory import (
    estimate_training_memory,
    format_memory_table,
    measure_peak_vram,
)
from new_ime.training.optim import build_optimizer, build_scheduler


def _resolve_device(cfg: TrainConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.loop.fp16 or cfg.loop.bf16:
        print(
            "[train] WARNING: AMP requested but CUDA unavailable; "
            "running on CPU in fp32",
            file=sys.stderr,
        )
    return torch.device("cpu")


def _resolve_amp_dtype(cfg: TrainConfig, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if cfg.loop.bf16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print(
            "[train] WARNING: bf16 not supported by this GPU; "
            "falling back to fp16 (memory: feedback_local_precision_fp16)",
            file=sys.stderr,
        )
        return torch.float16
    if cfg.loop.fp16:
        return torch.float16
    return None


def _build_model(cfg: TrainConfig, vocab_size: int) -> torch.nn.Module:
    if cfg.model.arch != "ctc-nat":
        raise NotImplementedError(
            f"arch={cfg.model.arch!r} not implemented in v1.0 "
            "(only ctc-nat ships; ar / dat scaffolding only)"
        )
    return CTCNAT.from_preset(
        cfg.model.preset,
        vocab_size=vocab_size,
        use_cvae=cfg.model.use_cvae,
        max_positions=cfg.model.max_seq_len,
    )


def _build_train_loader(
    cfg: TrainConfig, tokenizer: SharedCharTokenizer
) -> DataLoader:
    train_path = Path(cfg.data.train)
    collator = CTCShardCollator(max_seq_len=cfg.model.max_seq_len)
    pin = torch.cuda.is_available()
    if train_path.suffix == ".kkc":
        ds = KanaKanjiShardIterable(
            train_path,
            block_size=1024,
            shuffle=True,
            seed=cfg.run.seed,
            expected_vocab_size=tokenizer.vocab_size,
        )
        return DataLoader(
            ds,
            batch_size=cfg.loop.batch_size,
            num_workers=cfg.loop.num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )
    ds = KanaKanjiJsonlDataset(
        train_path,
        tokenizer=tokenizer,
        max_context_chars=cfg.model.max_context,
    )
    return DataLoader(
        ds,
        batch_size=cfg.loop.batch_size,
        shuffle=True,
        num_workers=cfg.loop.num_workers,
        collate_fn=collator,
        pin_memory=pin,
    )


def _build_dev_loader(
    cfg: TrainConfig, tokenizer: SharedCharTokenizer
) -> DataLoader:
    dev_path = Path(cfg.data.dev)
    collator = CTCShardCollator(max_seq_len=cfg.model.max_seq_len)
    pin = torch.cuda.is_available()
    if dev_path.suffix == ".kkc":
        ds = KanaKanjiShardDataset(
            dev_path, expected_vocab_size=tokenizer.vocab_size
        )
    else:
        ds = KanaKanjiJsonlDataset(
            dev_path,
            tokenizer=tokenizer,
            max_context_chars=cfg.model.max_context,
        )
    return DataLoader(
        ds,
        batch_size=cfg.loop.eval_batch_size,
        shuffle=False,
        num_workers=cfg.loop.num_workers,
        collate_fn=collator,
        pin_memory=pin,
    )


def run(cfg: TrainConfig, config_path: Path) -> int:
    torch.manual_seed(cfg.run.seed)
    out_dir = Path(cfg.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg)
    amp_dtype = _resolve_amp_dtype(cfg, device)
    tokenizer = SharedCharTokenizer.load(cfg.data.tokenizer)
    model = _build_model(cfg, vocab_size=tokenizer.vocab_size).to(device)
    if cfg.model.use_cvae and hasattr(model, "set_cvae_kl_weight"):
        model.set_cvae_kl_weight(cfg.model.cvae_kl_weight)

    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.loop.max_steps)

    train_loader = _build_train_loader(cfg, tokenizer)
    dev_loader = _build_dev_loader(cfg, tokenizer)
    train_collator = train_loader.collate_fn

    aux_loss_fns: list = []
    if cfg.refine is not None:
        aux_loss_fns.append(
            build_refine_loss_fn(cfg.refine, mask_id=tokenizer.mask_id)
        )
    if cfg.kd is not None:
        from new_ime.training.loss.kd import build_kd_loss_fn
        from new_ime.training.teacher import build_teacher

        teacher = build_teacher(
            cfg.kd, device=device, expected_vocab_size=tokenizer.vocab_size
        )
        aux_loss_fns.append(build_kd_loss_fn(cfg.kd, teacher))
        print(
            f"[train] KD: teacher={cfg.kd.teacher_type} "
            f"alpha={cfg.kd.alpha} warmup={cfg.kd.warmup_steps} "
            f"gate={cfg.kd.gate_mode} T={cfg.kd.temperature}",
            file=sys.stderr,
        )

    # Memory estimate (and actual peak VRAM on CUDA).
    estimate = estimate_training_memory(
        model,
        batch_size=cfg.loop.batch_size,
        seq_len=cfg.model.max_seq_len,
        bytes_per_param=2 if amp_dtype is not None else 4,
    )
    peak = measure_peak_vram(
        model,
        batch_size=cfg.loop.batch_size,
        seq_len=cfg.model.max_seq_len,
        vocab_size=tokenizer.vocab_size,
        device=device,
    )
    print(
        "[train] memory:\n" + format_memory_table(estimate, peak),
        file=sys.stderr,
    )

    probe_items = None
    if cfg.probe is not None:
        from new_ime.eval.loaders import load_bench

        probe_items = load_bench("probe_v3", cfg.probe.path)
        print(
            f"[train] probe loaded: n={len(probe_items)} "
            f"every={cfg.probe.every} metric={cfg.probe.metric_priority}",
            file=sys.stderr,
        )

    print(
        f"[train] {cfg.run.name}: arch={cfg.model.arch} preset={cfg.model.preset} "
        f"vocab={tokenizer.vocab_size} device={device} amp={amp_dtype}",
        file=sys.stderr,
    )

    best_metric = float("-inf")
    eval_max_batches = (
        cfg.data.max_dev_samples // max(cfg.loop.eval_batch_size, 1)
        if cfg.data.max_dev_samples
        else 0
    )

    def _log(rec: StepRecord) -> None:
        print(
            f"[train] step={rec.step} loss={rec.loss:.4f} lr={rec.lr:.2e}",
            file=sys.stderr,
        )

    def _select_metric(metrics: dict) -> float:
        priority = (
            cfg.probe.metric_priority if cfg.probe is not None else "exact_match_top1"
        )
        if priority == "loss_neg":
            return -float(metrics.get("loss", float("inf")))
        return float(metrics.get(priority, float("-inf")))

    def _on_eval(step: int) -> dict:
        m = evaluate_model(
            model=model,
            loader=dev_loader,
            device=device,
            tokenizer=tokenizer,
            max_batches=eval_max_batches,
        )
        if (
            probe_items is not None
            and cfg.probe is not None
            and step % cfg.probe.every == 0
        ):
            probe_m = evaluate_probe_em1(
                model=model,
                probe_items=probe_items,
                tokenizer=tokenizer,
                device=device,
                max_seq_len=cfg.model.max_seq_len,
                max_context=cfg.model.max_context,
                limit=cfg.probe.limit,
            )
            m["probe_em1"] = probe_m["em1"]
            print(
                f"[probe] step={step} EM1={probe_m['em1']:.4f} n={probe_m['n']}",
                file=sys.stderr,
            )
        print(
            f"[eval] step={step} loss={m['loss']:.4f} "
            f"EM1={m['exact_match_top1']:.4f} "
            f"charAcc={m['char_acc_top1']:.4f} "
            f"blank={m['blank_fraction']:.3f} n={m['num_samples']}",
            file=sys.stderr,
        )
        return m

    def _on_checkpoint(step: int, metrics: dict | None) -> None:
        nonlocal best_metric
        ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
        save(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            epoch=0,
            best_metric=best_metric,
            tokenizer=tokenizer,
        )
        rolling_keep(out_dir, cfg.logging.keep_last_k)
        if metrics is not None:
            cur = _select_metric(metrics)
            if cur > best_metric:
                best_metric = cur
                save(
                    out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    epoch=0,
                    best_metric=best_metric,
                    tokenizer=tokenizer,
                )

    def _on_step_start(step: int) -> None:
        if cfg.loop.warmup_short_sample_steps > 0:
            apply_short_sample_warmup(
                train_collator,
                step=step,
                warmup_steps=cfg.loop.warmup_short_sample_steps,
                short_max_chars=cfg.loop.short_sample_max_chars,
            )

    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=train_loader,
        device=device,
        max_steps=cfg.loop.max_steps,
        grad_accum=cfg.loop.grad_accum,
        grad_clip=cfg.optim.grad_clip,
        log_every=cfg.logging.log_every,
        amp_dtype=amp_dtype,
        aux_loss_fns=aux_loss_fns,
        eval_every=cfg.logging.eval_every,
        on_eval=_on_eval,
        checkpoint_every=cfg.logging.checkpoint_every,
        on_checkpoint=_on_checkpoint,
        stop_file=out_dir / "STOP",
        on_log=_log,
        on_step_start=_on_step_start,
    )

    print(
        f"[train] done: final_step={result.final_step} "
        f"best_metric={best_metric:.4f}",
        file=sys.stderr,
    )
    return 0
