"""Phase 3: local CTC-NAT training scaffold.

This trainer is intentionally conservative:
- supports a lightweight 20M-class preset for local experiments
- checkpoints every N steps (default 2000)
- resume-first workflow
- keeps the VRAM estimation / smoke-test path available

It is not the final Phase 3 curriculum trainer. The goal here is to make
interrupted local training practical before the full data pipeline lands.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from dataclasses import dataclass
import time

import torch
from torch.utils.data import DataLoader

from src.data.dataset import KanaKanjiDataset
from src.data.tokenizer import BLANK_ID, PAD_ID, SharedCharTokenizer
from src.eval.metrics import EvalResult
from src.model.ctc_nat import CTCNAT, PRESETS
from src.training.kd import (
    ARTeacher,
    KDConfig,
    TeacherConfig,
    compute_kd_ctc_loss,
    encode_texts_for_student,
    hard_example_mask,
)


@dataclass
class MemoryEstimate:
    params_m: float
    param_gb: float
    optimizer_gb: float
    activation_gb: float
    total_gb: float


class CTCCollator:
    """Builds encoder inputs and CTC targets for the shared-char tokenizer."""

    def __init__(
        self,
        tokenizer: SharedCharTokenizer,
        max_seq_len: int = 128,
        max_context: int = 40,
        short_sample_max_chars: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_context = max_context
        self.short_sample_max_chars = short_sample_max_chars

    def _encode_input(self, context: str, reading: str) -> list[int]:
        context = context[-self.max_context :] if context else ""
        ids = self.tokenizer.encode_with_special(context, reading)
        return ids[: self.max_seq_len]

    def _encode_target(self, surface: str) -> list[int]:
        ids = self.tokenizer.encode(surface)
        ids = [tid for tid in ids if tid != BLANK_ID]
        return ids[: self.max_seq_len]

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if self.short_sample_max_chars > 0:
            filtered = []
            for sample in batch:
                reading = sample["reading"]
                surface = sample["surface"]
                if len(reading) <= self.short_sample_max_chars and len(surface) <= self.short_sample_max_chars:
                    filtered.append(sample)
            if filtered:
                batch = filtered

        encoded_inputs: list[list[int]] = []
        encoded_targets: list[list[int]] = []
        target_lengths: list[int] = []
        writer_ids: list[int] = []
        domain_ids: list[int] = []
        source_ids: list[int] = []
        contexts: list[str] = []
        readings: list[str] = []
        surfaces: list[str] = []

        for sample in batch:
            context = sample.get("context", "") or ""
            reading = sample["reading"]
            surface = sample["surface"]
            inp = self._encode_input(context, reading)
            tgt = self._encode_target(surface)
            encoded_inputs.append(inp)
            encoded_targets.append(tgt)
            target_lengths.append(len(tgt))
            writer_ids.append(int(sample.get("writer_id", 0)))
            domain_ids.append(int(sample.get("domain_id", 0)))
            source_ids.append(int(sample.get("source_id", 0)))
            contexts.append(context)
            readings.append(reading)
            surfaces.append(surface)

        max_input_len = max(len(x) for x in encoded_inputs)
        max_target_len = max(max(target_lengths), 1)

        input_ids = []
        attention_mask = []
        target_ids = []
        for inp, tgt in zip(encoded_inputs, encoded_targets, strict=True):
            input_pad = max_input_len - len(inp)
            target_pad = max_target_len - len(tgt)
            input_ids.append(inp + [PAD_ID] * input_pad)
            attention_mask.append([1] * len(inp) + [0] * input_pad)
            target_ids.append(tgt + [PAD_ID] * target_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "writer_ids": torch.tensor(writer_ids, dtype=torch.long),
            "domain_ids": torch.tensor(domain_ids, dtype=torch.long),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "_contexts": contexts,
            "_readings": readings,
            "_surfaces": surfaces,
        }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    return 2 if device.type == "cuda" else 0


def should_run_kd_microbatch(
    step: int,
    batch_idx: int,
    grad_accum: int,
    teacher: ARTeacher | None,
    kd_config: KDConfig,
) -> bool:
    if teacher is None:
        return False
    if grad_accum <= 1:
        return kd_config.active(step)
    is_optimizer_boundary = ((batch_idx + 1) % grad_accum) == 0
    return is_optimizer_boundary and kd_config.active(step)


def build_model(preset: str, vocab_size: int, use_cvae: bool) -> CTCNAT:
    return CTCNAT.from_preset(
        preset,
        vocab_size=vocab_size,
        use_cvae=use_cvae,
        blank_id=BLANK_ID,
    )


def build_tokenizer(args: argparse.Namespace) -> SharedCharTokenizer:
    if getattr(args, "tokenizer_path", ""):
        return SharedCharTokenizer.load(args.tokenizer_path)
    return SharedCharTokenizer(max_kanji=args.max_kanji)


def estimate_training_memory(
    model: CTCNAT,
    preset_name: str,
    batch_size: int,
    seq_len: int,
    fp16: bool = True,
    use_adamw: bool = True,
) -> MemoryEstimate:
    preset = PRESETS[preset_name]
    num_params = sum(p.numel() for p in model.parameters())
    param_bytes = 2 if fp16 else 4

    param_mem = num_params * param_bytes
    grad_mem = num_params * param_bytes
    master_mem = num_params * 4 if fp16 else 0
    optimizer_mem = num_params * 8 if use_adamw else 0

    layers_total = preset.encoder_layers + preset.decoder_layers
    activation_elements = batch_size * seq_len * preset.hidden_size * layers_total * 18
    activation_mem = activation_elements * param_bytes

    total = param_mem + grad_mem + master_mem + optimizer_mem + activation_mem
    return MemoryEstimate(
        params_m=round(num_params / 1_000_000, 2),
        param_gb=param_mem / 1024**3,
        optimizer_gb=(grad_mem + master_mem + optimizer_mem) / 1024**3,
        activation_gb=activation_mem / 1024**3,
        total_gb=total / 1024**3,
    )


def measure_peak_vram(
    model: CTCNAT,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    use_cvae: bool,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None

    model = model.to(device)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    target_len = max(4, seq_len // 2)
    target_ids = torch.randint(6, vocab_size, (batch_size, target_len), device=device)
    target_lengths = torch.full((batch_size,), target_len, dtype=torch.long, device=device)
    kwargs = {}
    if use_cvae:
        kwargs.update(
            writer_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            domain_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
            source_ids=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    result = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_ids,
        target_lengths=target_lengths,
        **kwargs,
    )
    loss = result["loss"] + result.get("kl", torch.zeros((), device=device)) * 0.0
    loss.backward()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    model.zero_grad(set_to_none=True)
    return peak


def format_memory_table(estimate: MemoryEstimate, peak_gb: float | None) -> str:
    lines = [
        f"params:        {estimate.params_m:.2f}M",
        f"param mem:     {estimate.param_gb:.2f} GB",
        f"opt+grad mem:  {estimate.optimizer_gb:.2f} GB",
        f"activation:    {estimate.activation_gb:.2f} GB",
        f"total est:     {estimate.total_gb:.2f} GB",
    ]
    if peak_gb is not None:
        lines.append(f"cuda peak:     {peak_gb:.2f} GB")
    return "\n".join(lines)


def smoke_dataloader(path: str, tokenizer: SharedCharTokenizer, batch_size: int, max_seq_len: int):
    dataset = KanaKanjiDataset(path, max_samples=max(batch_size * 4, batch_size))
    collator = CTCCollator(tokenizer, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def save_checkpoint(
    path: str,
    model: CTCNAT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    tokenizer: SharedCharTokenizer,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_metric": best_metric,
        "preset": args.preset,
        "use_cvae": args.use_cvae,
        "max_seq_len": args.max_seq_len,
        "max_kanji": args.max_kanji,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_path": getattr(args, "tokenizer_path", ""),
        "blank_logit_bias": float(getattr(args, "blank_logit_bias", 0.0)),
        "kd": {
            "teacher_path": getattr(args, "kd_teacher_path", "") or "",
            "teacher_vocab": getattr(args, "kd_teacher_vocab", "") or "",
            "alpha": float(getattr(args, "kd_alpha", 0.0)),
            "hard_threshold": float(getattr(args, "kd_hard_threshold", 0.0)),
            "gate_mode": getattr(args, "kd_gate_mode", "low_conf"),
            "start_step": int(getattr(args, "kd_start_step", 0)),
            "warmup_steps": int(getattr(args, "kd_warmup_steps", 0)),
            "every": int(getattr(args, "kd_every", 1)),
            "max_new_tokens": int(getattr(args, "kd_max_new_tokens", 0)),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    tokenizer.save(path.replace(".pt", "_tokenizer.json"))


def load_checkpoint(
    path: str,
    model: CTCNAT,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def validate_resume_compatibility(
    checkpoint: dict,
    args: argparse.Namespace,
    tokenizer: SharedCharTokenizer | None = None,
) -> None:
    """Fail fast if a resume checkpoint does not match the current run."""

    mismatches: list[str] = []
    if checkpoint.get("preset") != args.preset:
        mismatches.append(f"preset: ckpt={checkpoint.get('preset')} current={args.preset}")
    if bool(checkpoint.get("use_cvae", False)) != bool(args.use_cvae):
        mismatches.append(
            f"use_cvae: ckpt={checkpoint.get('use_cvae', False)} current={args.use_cvae}"
        )
    if int(checkpoint.get("max_seq_len", args.max_seq_len)) != int(args.max_seq_len):
        mismatches.append(
            f"max_seq_len: ckpt={checkpoint.get('max_seq_len')} current={args.max_seq_len}"
        )
    if int(checkpoint.get("max_kanji", args.max_kanji)) != int(args.max_kanji):
        mismatches.append(
            f"max_kanji: ckpt={checkpoint.get('max_kanji')} current={args.max_kanji}"
        )
    if abs(float(checkpoint.get("blank_logit_bias", 0.0)) - float(args.blank_logit_bias)) > 1e-9:
        mismatches.append(
            "blank_logit_bias: "
            f"ckpt={checkpoint.get('blank_logit_bias', 0.0)} current={args.blank_logit_bias}"
        )
    if tokenizer is not None and int(checkpoint.get("vocab_size", tokenizer.vocab_size)) != int(
        tokenizer.vocab_size
    ):
        mismatches.append(
            f"vocab_size: ckpt={checkpoint.get('vocab_size')} current={tokenizer.vocab_size}"
        )

    # KD metadata: resume must use matching teacher + hyperparameters, otherwise
    # the optimizer/scheduler state belongs to a different objective.
    kd_prev = checkpoint.get("kd") or {}
    kd_fields = [
        ("teacher_path", "kd_teacher_path", ""),
        ("teacher_vocab", "kd_teacher_vocab", ""),
        ("alpha", "kd_alpha", 0.0),
        ("hard_threshold", "kd_hard_threshold", 0.0),
        ("gate_mode", "kd_gate_mode", "low_conf"),
        ("start_step", "kd_start_step", 0),
        ("warmup_steps", "kd_warmup_steps", 0),
        ("every", "kd_every", 1),
        ("max_new_tokens", "kd_max_new_tokens", 0),
    ]
    for ckpt_key, arg_key, default in kd_fields:
        ckpt_val = kd_prev.get(ckpt_key, default) if kd_prev else default
        cur_val = getattr(args, arg_key, default)
        if type(default) is float:
            if abs(float(ckpt_val) - float(cur_val)) > 1e-9:
                mismatches.append(f"kd.{ckpt_key}: ckpt={ckpt_val} current={cur_val}")
        else:
            if ckpt_val != cur_val:
                mismatches.append(f"kd.{ckpt_key}: ckpt={ckpt_val} current={cur_val}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Resume checkpoint is incompatible with current run: {details}")


@torch.no_grad()
def evaluate_model(
    model: CTCNAT,
    dataloader: DataLoader,
    tokenizer: SharedCharTokenizer,
    device: torch.device,
    use_cvae: bool,
    max_batches: int = 20,
    print_samples: int = 0,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    eval_result = EvalResult()
    total_loss = 0.0
    num_batches = 0
    samples: list[dict[str, str]] = []
    blank_fraction_sum = 0.0
    decoded_len_sum = 0.0
    target_len_sum = 0.0
    sample_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        kwargs = {}
        if use_cvae:
            kwargs.update(
                writer_ids=batch["writer_ids"],
                domain_ids=batch["domain_ids"],
                source_ids=batch["source_ids"],
            )
        result = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_ids=batch["target_ids"],
            target_lengths=batch["target_lengths"],
            sample_posterior=False,
            **kwargs,
        )
        total_loss += result["loss"].item()
        num_batches += 1
        predictions = result["logits"].argmax(dim=-1)
        valid_positions = batch["attention_mask"].bool()
        blank_fraction_sum += (
            ((predictions == BLANK_ID) & valid_positions).sum().item()
            / max(valid_positions.sum().item(), 1)
        )

        decoded = model.greedy_decode(
            batch["input_ids"],
            batch["attention_mask"],
            writer_ids=batch["writer_ids"] if use_cvae else None,
            domain_ids=batch["domain_ids"] if use_cvae else None,
            source_ids=batch["source_ids"] if use_cvae else None,
        )

        for target_ids, target_len, pred_ids in zip(
            batch["target_ids"], batch["target_lengths"], decoded, strict=True
        ):
            reference = tokenizer.decode(target_ids[: target_len.item()].tolist())
            hypothesis = tokenizer.decode(pred_ids)
            eval_result.add(reference, [hypothesis])
            decoded_len_sum += len(hypothesis)
            target_len_sum += len(reference)
            sample_count += 1
            if len(samples) < print_samples:
                samples.append({"reference": reference, "prediction": hypothesis})

    summary = eval_result.summary()
    summary["loss"] = total_loss / max(num_batches, 1)
    summary["blank_fraction"] = blank_fraction_sum / max(num_batches, 1)
    summary["mean_decoded_chars"] = decoded_len_sum / max(sample_count, 1)
    summary["mean_target_chars"] = target_len_sum / max(sample_count, 1)
    return summary, samples


def make_dataloader(
    path: str,
    tokenizer: SharedCharTokenizer,
    batch_size: int,
    max_seq_len: int,
    max_samples: int,
    shuffle: bool,
    num_workers: int,
    seed: int = 42,
    short_sample_max_chars: int = 0,
    max_context: int = 40,
    pin_memory: bool = False,
):
    dataset = KanaKanjiDataset(path, max_samples=max_samples, seed=seed)
    if short_sample_max_chars > 0:
        dataset.data = [
            sample
            for sample in dataset.data
            if len(sample["reading"]) <= short_sample_max_chars
            and len(sample["surface"]) <= short_sample_max_chars
        ]
        if not dataset.data:
            raise ValueError(
                f"No samples left after short-sample filter (<= {short_sample_max_chars} chars)."
            )
    collator = CTCCollator(
        tokenizer,
        max_seq_len=max_seq_len,
        max_context=max_context,
        short_sample_max_chars=short_sample_max_chars,
    )
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        **loader_kwargs,
    )


def build_kd(args: argparse.Namespace, device: torch.device) -> tuple[ARTeacher | None, KDConfig]:
    """Construct the KD teacher + config from CLI args (teacher is optional)."""
    kd_config = KDConfig(
        alpha=args.kd_alpha,
        hard_threshold=args.kd_hard_threshold,
        gate_mode=args.kd_gate_mode,
        start_step=args.kd_start_step,
        warmup_steps=args.kd_warmup_steps,
        every=max(args.kd_every, 1),
        max_new_tokens=args.kd_max_new_tokens,
    )
    if not args.kd_teacher_path or args.kd_alpha <= 0.0:
        return None, kd_config
    teacher_config = TeacherConfig(
        checkpoint_path=args.kd_teacher_path,
        vocab_path=args.kd_teacher_vocab or "",
        hidden_size=args.kd_teacher_hidden,
        num_layers=args.kd_teacher_layers,
        num_heads=args.kd_teacher_heads,
        max_seq_len=args.kd_teacher_max_seq_len,
        max_new_tokens=args.kd_max_new_tokens,
        max_context_chars=args.max_context,
        fp16=args.fp16,
    )
    teacher = ARTeacher.from_checkpoint(teacher_config, device=device)
    print(
        f"KD teacher loaded: {args.kd_teacher_path} "
        f"(vocab={teacher.collator.vocab_size}, fp16={teacher_config.fp16}) "
        f"α={kd_config.alpha}, threshold={kd_config.hard_threshold}, mode={kd_config.gate_mode}, "
        f"start={kd_config.start_step}, warmup={kd_config.warmup_steps}, "
        f"every={kd_config.every}"
    )
    return teacher, kd_config


def train_local(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    tokenizer = build_tokenizer(args)
    model = build_model(args.preset, vocab_size=tokenizer.vocab_size, use_cvae=args.use_cvae).to(device)
    model.blank_logit_bias = float(args.blank_logit_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / max(args.warmup_steps, 1), 1.0),
    )

    teacher, kd_config = build_kd(args, device)

    start_step = 0
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        validate_resume_compatibility(checkpoint, args, tokenizer=tokenizer)
        start_step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", 0)
        best_metric = checkpoint.get("best_metric", float("-inf"))
        print(f"Resumed from {args.resume} @ step {start_step}, epoch {start_epoch}")

    train_loader = make_dataloader(
        args.train,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_samples=args.tiny_overfit_samples or args.max_train_samples,
        shuffle=True,
        num_workers=num_workers,
        seed=args.seed,
        short_sample_max_chars=args.short_sample_max_chars,
        max_context=args.max_context,
        pin_memory=pin_memory,
    )
    eval_path = args.dev
    eval_max_samples = args.max_dev_samples
    eval_seed = args.seed
    if args.tiny_overfit_samples and args.tiny_overfit_eval_train:
        eval_path = args.train
        eval_max_samples = args.tiny_overfit_samples
        print(
            f"tiny-overfit mode: evaluating on the same {args.tiny_overfit_samples} "
            "training samples instead of dev"
        )
    dev_loader = make_dataloader(
        eval_path,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_seq_len=args.max_seq_len,
        max_samples=eval_max_samples,
        shuffle=False,
        num_workers=0,
        seed=eval_seed,
        short_sample_max_chars=args.short_sample_max_chars if args.tiny_overfit_samples else 0,
        max_context=args.max_context,
        pin_memory=pin_memory,
    )

    warmup_loader = None
    if (
        args.warmup_short_sample_steps > 0
        and args.warmup_short_sample_max_chars > 0
        and args.tiny_overfit_samples == 0
        and start_step < args.warmup_short_sample_steps
    ):
        warmup_loader = make_dataloader(
            args.train,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_samples=args.max_train_samples,
            shuffle=True,
            num_workers=num_workers,
            seed=args.seed,
            short_sample_max_chars=args.warmup_short_sample_max_chars,
            max_context=args.max_context,
            pin_memory=pin_memory,
        )

    estimate = estimate_training_memory(
        model,
        preset_name=args.preset,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        fp16=args.fp16 and device.type == "cuda",
        use_adamw=True,
    )
    print(format_memory_table(estimate, peak_gb=None))
    print(f"blank bias:     {args.blank_logit_bias:.2f}")
    print(
        f"dataloader: workers={num_workers} pin_memory={pin_memory} "
        f"persistent_workers={num_workers > 0}"
    )
    if warmup_loader is not None:
        print(
            "short-sample warmup: "
            f"steps={args.warmup_short_sample_steps} "
            f"max_chars={args.warmup_short_sample_max_chars}"
        )
    if args.tiny_overfit_samples:
        effective_batches = math.ceil(args.tiny_overfit_samples / args.batch_size)
        effective_steps_per_epoch = math.ceil(effective_batches / args.grad_accum)
        print(
            "tiny-overfit summary: "
            f"samples={args.tiny_overfit_samples} "
            f"mini_batches/epoch={effective_batches} "
            f"optimizer_steps/epoch={effective_steps_per_epoch}"
        )
        if args.log_every > effective_steps_per_epoch * max(args.epochs, 1):
            print("warning: log_every exceeds total expected optimizer steps.")
        if args.eval_every > effective_steps_per_epoch * max(args.epochs, 1):
            print("warning: eval_every exceeds total expected optimizer steps.")

    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    step = start_step
    last_log = time.perf_counter()
    running_losses: deque[float] = deque(maxlen=args.loss_window)
    kd_stats = {"loss_sum": 0.0, "hard_sum": 0, "total_sum": 0, "conf_sum": 0.0, "batches": 0}

    def fetch_next_batch(loader_iter, loader):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        return batch, loader_iter

    def run_microbatch(batch, batch_idx: int, epoch_idx: int) -> None:
        nonlocal step, last_log, kd_stats, best_metric

        contexts = batch.get("_contexts", [])
        readings = batch.get("_readings", [])
        batch = move_batch_to_device(batch, device)
        kwargs = {}
        if args.use_cvae:
            kwargs.update(
                writer_ids=batch["writer_ids"],
                domain_ids=batch["domain_ids"],
                source_ids=batch["source_ids"],
            )

        run_kd = should_run_kd_microbatch(step, batch_idx, args.grad_accum, teacher, kd_config)
        with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_ids=batch["target_ids"],
                target_lengths=batch["target_lengths"],
                **kwargs,
            )
            loss = result["loss"]
            if args.use_cvae and "kl" in result:
                loss = loss + args.kl_weight * result["kl"]

            if run_kd:
                alpha_now = kd_config.alpha_at(step)
                if alpha_now <= 0.0:
                    run_kd = False

            if run_kd:
                with torch.no_grad():
                    teacher_texts, teacher_conf = teacher.generate(
                        contexts=contexts,
                        readings=readings,
                        max_new_tokens=kd_config.max_new_tokens,
                    )
                conf_tensor = torch.tensor(teacher_conf, device=device)
                hard_mask = hard_example_mask(
                    conf_tensor,
                    kd_config.hard_threshold,
                    mode=kd_config.gate_mode,
                )
                teacher_ids, teacher_lengths = encode_texts_for_student(
                    teacher_texts,
                    tokenizer=tokenizer,
                    max_len=args.max_seq_len,
                )
                kd_loss_value, num_hard = compute_kd_ctc_loss(
                    student_log_probs=result["log_probs"],
                    input_lengths=batch["attention_mask"].sum(dim=1).long(),
                    teacher_ids=teacher_ids,
                    teacher_lengths=teacher_lengths,
                    hard_mask=hard_mask,
                    blank_id=BLANK_ID,
                )
                if num_hard > 0 and alpha_now > 0.0:
                    # KD fires on only 1 of grad_accum microbatches per active
                    # optimizer step. The subsequent `loss / grad_accum` would
                    # otherwise shrink the KD contribution by that same factor.
                    # Pre-multiply so --kd-alpha acts as a grad_accum-invariant
                    # weight on the KD term.
                    loss = loss + alpha_now * args.grad_accum * kd_loss_value
                kd_stats["loss_sum"] += float(kd_loss_value.detach().item())
                kd_stats["hard_sum"] += num_hard
                kd_stats["total_sum"] += len(teacher_texts)
                kd_stats["conf_sum"] += (
                    float(conf_tensor.mean().item()) if conf_tensor.numel() else 0.0
                )
                kd_stats["batches"] += 1

            loss = loss / args.grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            current_loss = loss.item() * args.grad_accum
            running_losses.append(current_loss)

            if step % args.log_every == 0:
                now = time.perf_counter()
                rate = args.log_every / max(now - last_log, 1e-6)
                last_log = now
                avg_loss = sum(running_losses) / max(len(running_losses), 1)
                lr = optimizer.param_groups[0]["lr"]
                line = (
                    f"[step {step}] loss={current_loss:.4f} "
                    f"avg{args.loss_window}={avg_loss:.4f} "
                    f"lr={lr:.6f} rate={rate:.2f} steps/s"
                )
                if teacher is not None and kd_stats["batches"] > 0:
                    kd_avg = kd_stats["loss_sum"] / kd_stats["batches"]
                    hard_ratio = kd_stats["hard_sum"] / max(kd_stats["total_sum"], 1)
                    conf_avg = kd_stats["conf_sum"] / kd_stats["batches"]
                    alpha_now = kd_config.alpha_at(step)
                    line += (
                        f" kd_loss={kd_avg:.4f} "
                        f"kd_hard={hard_ratio:.2f} "
                        f"kd_conf={conf_avg:.2f} "
                        f"kd_alpha={alpha_now:.3f}"
                    )
                print(line)
                kd_stats = {
                    "loss_sum": 0.0,
                    "hard_sum": 0,
                    "total_sum": 0,
                    "conf_sum": 0.0,
                    "batches": 0,
                }

            if step % args.eval_every == 0:
                metrics, samples = evaluate_model(
                    model,
                    dev_loader,
                    tokenizer=tokenizer,
                    device=device,
                    use_cvae=args.use_cvae,
                    max_batches=args.eval_batches,
                    print_samples=args.print_samples,
                )
                print(
                    f"[eval {step}] loss={metrics['loss']:.4f} "
                    f"EM={metrics.get('exact_match_top1', 0):.4f} "
                    f"CharAcc={metrics.get('char_acc_top1', 0):.4f} "
                    f"blank={metrics.get('blank_fraction', 0):.3f} "
                    f"pred_len={metrics.get('mean_decoded_chars', 0):.1f}/"
                    f"{metrics.get('mean_target_chars', 0):.1f}"
                )
                for idx, sample in enumerate(samples, start=1):
                    print(
                        f"  sample{idx}: ref={sample['reference'][:40]} "
                        f"pred={sample['prediction'][:40]}"
                    )
                metric_key = metrics.get("exact_match_top1", 0.0)
                if metric_key > best_metric:
                    best_metric = metric_key
                    save_checkpoint(
                        os.path.join(args.output, "best.pt"),
                        model,
                        optimizer,
                        scheduler,
                        step=step,
                        epoch=epoch_idx,
                        tokenizer=tokenizer,
                        best_metric=best_metric,
                        args=args,
                    )

            if step % args.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(args.output, f"checkpoint_step_{step}.pt"),
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    epoch=epoch_idx,
                    tokenizer=tokenizer,
                    best_metric=best_metric,
                    args=args,
                )

    try:
        if warmup_loader is not None:
            warmup_iter = iter(warmup_loader)
            while step < args.warmup_short_sample_steps:
                for warmup_batch_idx in range(args.grad_accum):
                    if args.max_steps and step >= args.max_steps:
                        raise StopIteration
                    batch, warmup_iter = fetch_next_batch(warmup_iter, warmup_loader)
                    run_microbatch(batch, batch_idx=warmup_batch_idx, epoch_idx=start_epoch)
                    if step >= args.warmup_short_sample_steps:
                        break

        for epoch in range(start_epoch, args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_loader):
                if args.max_steps and step >= args.max_steps:
                    raise StopIteration
                run_microbatch(batch, batch_idx=batch_idx, epoch_idx=epoch)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint before exit...")
        save_checkpoint(
            os.path.join(args.output, f"interrupted_step_{step}.pt"),
            model,
            optimizer,
            scheduler,
            step=step,
            epoch=epoch if "epoch" in locals() else 0,
            tokenizer=tokenizer,
            best_metric=best_metric,
            args=args,
        )
        return
    except StopIteration:
        pass

    save_checkpoint(
        os.path.join(args.output, "final.pt"),
        model,
        optimizer,
        scheduler,
        step=step,
        epoch=epoch if "epoch" in locals() else 0,
        tokenizer=tokenizer,
        best_metric=best_metric,
        args=args,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CTC-NAT local training scaffold")
    parser.add_argument("--train", default="", help="Training JSONL path")
    parser.add_argument("--dev", default="", help="Dev JSONL path")
    parser.add_argument("--output", default="checkpoints/ctc_nat_local", help="Output directory")
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="phase3_20m")
    parser.add_argument("--use-cvae", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = no explicit cap")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-kanji", type=int, default=6000)
    parser.add_argument("--small-vocab-max-kanji", type=int, default=0)
    parser.add_argument("--tokenizer-path", default="", help="Path to a saved SharedCharTokenizer JSON")
    parser.add_argument("--max-train-samples", type=int, default=200_000)
    parser.add_argument("--max-dev-samples", type=int, default=2_000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="-1 auto-selects 2 workers on CUDA and 0 on CPU",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--tiny-overfit-samples", type=int, default=0)
    parser.add_argument("--short-sample-max-chars", type=int, default=0)
    parser.add_argument("--warmup-short-sample-steps", type=int, default=0)
    parser.add_argument("--warmup-short-sample-max-chars", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--blank-logit-bias",
        type=float,
        default=0.0,
        help="Subtract this value from the CTC blank logit before softmax/decode",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--loss-window", type=int, default=100)
    parser.add_argument("--print-samples", type=int, default=3)
    parser.add_argument("--tiny-overfit-eval-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--max-context",
        type=int,
        default=40,
        help="Max context chars fed to both the student encoder and the AR teacher",
    )

    kd_group = parser.add_argument_group("online KD (AR teacher)")
    kd_group.add_argument("--kd-teacher-path", default="", help="AR teacher checkpoint (.pt)")
    kd_group.add_argument("--kd-teacher-vocab", default="", help="AR teacher vocab JSON (default: auto)")
    kd_group.add_argument("--kd-teacher-hidden", type=int, default=512)
    kd_group.add_argument("--kd-teacher-layers", type=int, default=8)
    kd_group.add_argument("--kd-teacher-heads", type=int, default=8)
    kd_group.add_argument("--kd-teacher-max-seq-len", type=int, default=256)
    kd_group.add_argument("--kd-alpha", type=float, default=0.0, help="KD loss weight (0 disables)")
    kd_group.add_argument("--kd-hard-threshold", type=float, default=0.6)
    kd_group.add_argument(
        "--kd-gate-mode",
        choices=["low_conf", "high_conf", "all"],
        default="low_conf",
        help=(
            "Which teacher outputs contribute to KD: "
            "low_conf=uncertain only, high_conf=confident only, all=no gating"
        ),
    )
    kd_group.add_argument("--kd-start-step", type=int, default=0)
    kd_group.add_argument("--kd-warmup-steps", type=int, default=0)
    kd_group.add_argument(
        "--kd-every",
        type=int,
        default=4,
        help=(
            "Apply KD every N optimizer steps. KD only runs on the microbatch "
            "that triggers optimizer.step(), not on every accumulation shard."
        ),
    )
    kd_group.add_argument("--kd-max-new-tokens", type=int, default=96)

    args = parser.parse_args()
    if args.small_vocab_max_kanji > 0:
        args.max_kanji = args.small_vocab_max_kanji

    tokenizer = build_tokenizer(args)
    model = build_model(args.preset, vocab_size=tokenizer.vocab_size, use_cvae=args.use_cvae)
    model.blank_logit_bias = float(args.blank_logit_bias)
    estimate = estimate_training_memory(
        model,
        preset_name=args.preset,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        fp16=args.fp16 and torch.cuda.is_available(),
        use_adamw=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peak = measure_peak_vram(
        model,
        batch_size=args.batch_size,
        seq_len=args.max_seq_len,
        vocab_size=tokenizer.vocab_size,
        use_cvae=args.use_cvae,
        device=device,
    )
    if device.type == "cuda":
        model = model.cpu()
    print(format_memory_table(estimate, peak))

    if args.estimate_only:
        return

    if not args.train or not args.dev:
        raise SystemExit("--train and --dev are required unless --estimate-only is set")

    train_local(args)


if __name__ == "__main__":
    main()
