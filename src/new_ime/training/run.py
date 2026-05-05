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
from torch.utils.data import DataLoader, Subset

from new_ime.config.train import TrainConfig
from new_ime.data.jsonl import KanaKanjiJsonlDataset
from new_ime.data.shards import (
    CTCShardCollator,
    KanaKanjiShardDataset,
    KanaKanjiShardIterable,
)
from new_ime.data.tokenizer import SharedCharTokenizer
from new_ime.model.ctc_nat import CTCNAT
from new_ime.training.checkpoint import (
    load,
    rolling_keep,
    save,
    validate_resume_compatibility,
)
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
    if cfg.model.arch == "ctc-nat":
        return CTCNAT.from_preset(
            cfg.model.preset,
            vocab_size=vocab_size,
            use_cvae=cfg.model.use_cvae,
            max_positions=cfg.model.max_seq_len,
        )
    if cfg.model.arch == "dat":
        from new_ime.model.dat import DAT

        if cfg.refine is not None:
            raise ValueError(
                "[refine] is CTC-NAT-specific and incompatible with arch=dat; "
                "remove the [refine] section or switch arch to ctc-nat"
            )
        if cfg.dat is None:
            raise ValueError("arch=dat requires a [dat] section in the TOML")
        return DAT.from_preset(
            cfg.model.preset,
            vocab_size=vocab_size,
            upsample_scale=cfg.dat.upsample_scale,
            num_link_heads=cfg.dat.num_link_heads,
            max_positions=cfg.model.max_seq_len,
        )
    raise NotImplementedError(
        f"arch={cfg.model.arch!r} not implemented in v1.0 "
        "(supported: ctc-nat, dat)"
    )


def _build_train_loader(
    cfg: TrainConfig, tokenizer: SharedCharTokenizer
) -> DataLoader:
    train_path = Path(cfg.data.train)
    collator = CTCShardCollator(max_seq_len=cfg.model.max_seq_len)
    pin = torch.cuda.is_available()
    if cfg.loop.tiny_overfit_samples > 0:
        if train_path.suffix == ".kkc":
            ds = KanaKanjiShardDataset(
                train_path, expected_vocab_size=tokenizer.vocab_size
            )
        else:
            ds = KanaKanjiJsonlDataset(
                train_path,
                tokenizer=tokenizer,
                max_context_chars=cfg.model.max_context,
            )
        n = min(int(cfg.loop.tiny_overfit_samples), len(ds))
        ds = Subset(ds, range(n))
        return DataLoader(
            ds,
            batch_size=cfg.loop.batch_size,
            shuffle=True,
            num_workers=cfg.loop.num_workers,
            collate_fn=collator,
            pin_memory=pin,
        )
    if train_path.suffix == ".kkc":
        if cfg.data.max_train_samples > 0:
            ds = KanaKanjiShardDataset(
                train_path, expected_vocab_size=tokenizer.vocab_size
            )
            n = min(int(cfg.data.max_train_samples), len(ds))
            ds = Subset(ds, range(n))
            return DataLoader(
                ds,
                batch_size=cfg.loop.batch_size,
                shuffle=True,
                num_workers=cfg.loop.num_workers,
                collate_fn=collator,
                pin_memory=pin,
            )
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
    if cfg.data.max_train_samples > 0:
        n = min(int(cfg.data.max_train_samples), len(ds))
        ds = Subset(ds, range(n))
    return DataLoader(
        ds,
        batch_size=cfg.loop.batch_size,
        shuffle=True,
        num_workers=cfg.loop.num_workers,
        collate_fn=collator,
        pin_memory=pin,
    )


def _compile_model_if_requested(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> torch.nn.Module:
    if not cfg.loop.compile:
        return model
    if not hasattr(torch, "compile"):
        print(
            "[train] WARNING: torch.compile unavailable; running eager",
            file=sys.stderr,
        )
        return model
    try:
        compiled = torch.compile(model)
    except Exception as e:
        print(
            f"[train] WARNING: torch.compile failed at setup: {e}; running eager",
            file=sys.stderr,
        )
        return model
    print("[train] torch.compile enabled", file=sys.stderr)
    return compiled


def _print_eval_samples(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    tokenizer: SharedCharTokenizer,
    limit: int,
) -> None:
    if limit <= 0 or not hasattr(model, "greedy_decode"):
        return
    was_training = model.training
    model.eval()
    printed = 0
    try:
        batch = next(iter(loader))
    except StopIteration:
        return
    batch = {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    with torch.no_grad():
        decoded = model.greedy_decode(batch["input_ids"], batch["attention_mask"])
    for b, ids in enumerate(decoded):
        if printed >= limit:
            break
        tlen = int(batch["target_lengths"][b].item())
        ref_ids = batch["target_ids"][b, :tlen].tolist()
        pred = tokenizer.decode(ids)
        ref = tokenizer.decode(ref_ids)
        print(
            f"[sample] pred={pred!r} ref={ref!r}",
            file=sys.stderr,
        )
        printed += 1
    if was_training:
        model.train()


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
    raw_model = _build_model(cfg, vocab_size=tokenizer.vocab_size).to(device)
    if cfg.model.use_cvae and hasattr(raw_model, "set_cvae_kl_weight"):
        raw_model.set_cvae_kl_weight(cfg.model.cvae_kl_weight)

    optimizer = build_optimizer(raw_model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.loop.max_steps)
    start_step = 0
    best_metric = float("-inf")

    if cfg.resume is not None:
        resume_path = Path(cfg.resume.checkpoint)
        blob = load(
            resume_path,
            model=raw_model,
            optimizer=None if cfg.resume.reset_optimizer else optimizer,
            scheduler=None if cfg.resume.reset_scheduler else scheduler,
            reset_scheduler=cfg.resume.reset_scheduler,
            map_location=device,
        )
        validate_resume_compatibility(blob, raw_model)
        start_step = int(blob.get("step", 0))
        if not cfg.resume.reset_best_metric:
            best_metric = float(blob.get("best_metric", best_metric))
        print(
            f"[train] resumed: checkpoint={resume_path} "
            f"step={start_step} best_metric={best_metric:.4f} "
            f"reset_optimizer={cfg.resume.reset_optimizer} "
            f"reset_scheduler={cfg.resume.reset_scheduler}",
            file=sys.stderr,
        )

    model = _compile_model_if_requested(raw_model, cfg)

    train_loader = _build_train_loader(cfg, tokenizer)
    dev_loader = _build_dev_loader(cfg, tokenizer)
    train_collator = train_loader.collate_fn

    aux_loss_fns: list = []
    if cfg.refine is not None:
        aux_loss_fns.append(
            build_refine_loss_fn(cfg.refine, mask_id=tokenizer.mask_id)
        )
    if cfg.dat is not None:
        from new_ime.training.loss.dat import build_dat_loss_fn

        aux_loss_fns.append(build_dat_loss_fn(cfg.dat))
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
        raw_model,
        batch_size=cfg.loop.batch_size,
        seq_len=cfg.model.max_seq_len,
        bytes_per_param=2 if amp_dtype is not None else 4,
    )
    peak = measure_peak_vram(
        raw_model,
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
    print(
        f"[train] kill switch: touch {out_dir / 'STOP'} or send SIGINT/SIGTERM "
        "(Ctrl-C); final checkpoint is written before exit",
        file=sys.stderr,
    )

    eval_max_batches = (
        max(
            1,
            (cfg.data.max_dev_samples + max(cfg.loop.eval_batch_size, 1) - 1)
            // max(cfg.loop.eval_batch_size, 1),
        )
        if cfg.data.max_dev_samples
        else 0
    )

    def _log(rec: StepRecord) -> None:
        rate_str = (
            f"rate={rec.steps_per_sec:.2f}step/s"
            if rec.steps_per_sec > 0
            else "rate=-"
        )
        print(
            f"[train] step={rec.step} loss={rec.loss:.4f} lr={rec.lr:.2e} {rate_str}",
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
        _print_eval_samples(
            model=model,
            loader=dev_loader,
            device=device,
            tokenizer=tokenizer,
            limit=cfg.logging.print_samples,
        )
        return m

    def _on_checkpoint(step: int, metrics: dict | None) -> None:
        nonlocal best_metric
        ckpt_path = out_dir / f"checkpoint_step_{step}.pt"
        save(
            ckpt_path,
            model=raw_model,
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
                    model=raw_model,
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
        # DAT GLAT schedule: feed the annealed glance ratio into the model
        # so its forward(...) decides whether to run the 2-stage GLAT path.
        if cfg.dat is not None and hasattr(raw_model, "set_glance_ratio"):
            from new_ime.training.loss.dat import parse_anneal

            ratio = parse_anneal(cfg.dat.glat_p, step)
            raw_model.set_glance_ratio(ratio)

    result = run_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader=train_loader,
        device=device,
        max_steps=cfg.loop.max_steps,
        start_step=start_step,
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

    if result.interrupted:
        print(
            f"[train] interrupted: final_step={result.final_step} "
            f"best_metric={best_metric:.4f} (final checkpoint written)",
            file=sys.stderr,
        )
    elif result.stopped_via_file:
        print(
            f"[train] STOP file detected: final_step={result.final_step} "
            f"best_metric={best_metric:.4f} (final checkpoint written)",
            file=sys.stderr,
        )
        try:
            (out_dir / "STOP").unlink()
        except OSError:
            pass
    else:
        print(
            f"[train] done: final_step={result.final_step} "
            f"best_metric={best_metric:.4f}",
            file=sys.stderr,
        )
    return 0
