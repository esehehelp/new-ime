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
import re
from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.src.data.dataset import KanaKanjiDataset
from models.src.data.tokenizer import BLANK_ID, MASK_ID, PAD_ID, SharedCharTokenizer
from models.src.eval.metrics import EvalResult
from models.src.model.ctc_nat import CTCAlignmentToken, CTCNAT, PRESETS
from models.src.training.kd import (
    ARTeacher,
    CTCTeacher,
    CTCTeacherConfig,
    KDConfig,
    TeacherConfig,
    compute_kd_ctc_loss,
    compute_kd_kl_loss,
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


def build_refinement_batch(
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    mask_ratio: float,
    mask_id: int = MASK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask a subset of valid target positions for the refinement objective."""
    seq_len = target_ids.shape[1]
    positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
    hypothesis_attention_mask = positions < target_lengths.unsqueeze(1)
    random_mask = torch.rand_like(target_ids.float()) < mask_ratio
    mask_positions = random_mask & hypothesis_attention_mask

    no_mask_rows = (mask_positions.sum(dim=1) == 0) & (target_lengths > 0)
    if no_mask_rows.any():
        row_ids = no_mask_rows.nonzero(as_tuple=False).squeeze(1)
        forced_cols = []
        for row_idx in row_ids.tolist():
            forced_cols.append(int(torch.randint(0, int(target_lengths[row_idx].item()), (1,), device=target_ids.device).item()))
        mask_positions[row_ids, forced_cols] = True

    hypothesis_ids = target_ids.clone()
    hypothesis_ids[mask_positions] = mask_id
    return hypothesis_ids, hypothesis_attention_mask.long(), mask_positions


def resolve_refine_mask_ratio(args: argparse.Namespace) -> float:
    if args.refine_mask_ratio_min is not None and args.refine_mask_ratio_max is not None:
        lo = float(min(args.refine_mask_ratio_min, args.refine_mask_ratio_max))
        hi = float(max(args.refine_mask_ratio_min, args.refine_mask_ratio_max))
        if hi <= lo:
            return lo
        return float(torch.empty((), device="cpu").uniform_(lo, hi).item())
    return float(args.refine_mask_ratio)


def resolve_refine_loss_weight(args: argparse.Namespace, step: int) -> float:
    base = float(args.refine_loss_weight)
    warmup_steps = int(getattr(args, "refine_warmup_steps", 0))
    if base <= 0.0 or warmup_steps <= 0:
        return base
    scale = min(max(step, 0) / warmup_steps, 1.0)
    return base * scale


def build_refinement_batch_from_proposal_tensors(
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    proposal_token_ids: torch.Tensor,
    proposal_min_log_prob: torch.Tensor,
    proposal_min_margin: torch.Tensor,
    proposal_lengths: torch.Tensor,
    mask_ratio: float,
    mask_id: int = MASK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor-input version of `build_refinement_batch_from_proposal`.

    Takes the output of `CTCNAT.collapse_alignment_tensors` directly (no
    per-frame Python loop or dataclass construction). Rows whose collapsed
    proposal length matches the target length use the proposal hypothesis
    with the lowest-confidence positions masked; the rest fall back to the
    plain target-masked path.
    """
    device = target_ids.device
    hypothesis_ids, hypothesis_attention_mask, mask_positions = build_refinement_batch(
        target_ids,
        target_lengths,
        mask_ratio=mask_ratio,
        mask_id=mask_id,
    )
    used_proposal_rows = torch.zeros(target_ids.shape[0], dtype=torch.bool, device=device)

    proposal_lengths = proposal_lengths.to(device)
    target_lengths = target_lengths.to(device)
    proposal_token_ids = proposal_token_ids.to(device)
    proposal_min_log_prob = proposal_min_log_prob.to(device)
    proposal_min_margin = proposal_min_margin.to(device)

    match_rows = (proposal_lengths == target_lengths) & (target_lengths > 0)
    if not bool(match_rows.any().item()):
        return (
            hypothesis_ids,
            hypothesis_attention_mask.long(),
            mask_positions,
            used_proposal_rows,
        )

    seq_len = hypothesis_ids.shape[1]
    match_idx = match_rows.nonzero(as_tuple=False).squeeze(1).tolist()
    # Convert the small tensors to numpy once to avoid device sync inside loop.
    tgt_lens_list = target_lengths.detach().cpu().tolist()
    prop_ids_cpu = proposal_token_ids.detach().cpu()
    prop_conf_cpu = proposal_min_log_prob.detach().cpu()
    prop_margin_cpu = proposal_min_margin.detach().cpu()

    for row_idx in match_idx:
        tgt_len = int(tgt_lens_list[row_idx])
        if tgt_len <= 0:
            continue

        proposal_slice = prop_ids_cpu[row_idx, :tgt_len]
        hypothesis_ids[row_idx, :tgt_len] = proposal_slice.to(device)
        hypothesis_ids[row_idx, tgt_len:seq_len] = PAD_ID
        hypothesis_attention_mask[row_idx, :tgt_len] = 1
        hypothesis_attention_mask[row_idx, tgt_len:seq_len] = 0
        mask_positions[row_idx] = False

        num_masks = max(1, int(round(tgt_len * mask_ratio)))
        num_masks = min(num_masks, tgt_len)
        # Rank by (min_log_prob asc, min_margin asc, idx asc) — i.e. least
        # confident + smallest margin first, stable on ties.
        conf_slice = prop_conf_cpu[row_idx, :tgt_len]
        margin_slice = prop_margin_cpu[row_idx, :tgt_len]
        # Primary sort by conf_slice, secondary by margin, stable.
        order = torch.argsort(margin_slice, stable=True)
        order = order[torch.argsort(conf_slice[order], stable=True)]
        ranked = order[:num_masks].to(device)

        mask_positions[row_idx, ranked] = True
        hypothesis_ids[row_idx, ranked] = mask_id
        used_proposal_rows[row_idx] = True

    return (
        hypothesis_ids,
        hypothesis_attention_mask.long(),
        mask_positions,
        used_proposal_rows,
    )


def build_refinement_batch_from_proposal(
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    proposal_alignments: list[list[CTCAlignmentToken]],
    mask_ratio: float,
    mask_id: int = MASK_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build refinement hypotheses from the model proposal when lengths match.

    Rows whose collapsed proposal length does not match the target length fall
    back to the plain target-masked path to keep the objective well-defined.
    Returns `(hypothesis_ids, hypothesis_attention_mask, mask_positions,
    used_proposal_rows)`.
    """
    hypothesis_ids, hypothesis_attention_mask, mask_positions = build_refinement_batch(
        target_ids,
        target_lengths,
        mask_ratio=mask_ratio,
        mask_id=mask_id,
    )
    used_proposal_rows = torch.zeros(target_ids.shape[0], dtype=torch.bool, device=target_ids.device)

    for row_idx, aligned_tokens in enumerate(proposal_alignments):
        tgt_len = int(target_lengths[row_idx].item())
        if tgt_len <= 0 or len(aligned_tokens) != tgt_len:
            continue

        proposal_ids = torch.tensor(
            [tok.token_id for tok in aligned_tokens],
            dtype=target_ids.dtype,
            device=target_ids.device,
        )
        hypothesis_ids[row_idx, :tgt_len] = proposal_ids
        hypothesis_ids[row_idx, tgt_len:] = PAD_ID
        hypothesis_attention_mask[row_idx, :tgt_len] = 1
        hypothesis_attention_mask[row_idx, tgt_len:] = 0
        mask_positions[row_idx] = False

        num_masks = max(1, int(round(tgt_len * mask_ratio)))
        ranked_positions = sorted(
            range(tgt_len),
            key=lambda idx: (
                aligned_tokens[idx].confidence,
                aligned_tokens[idx].min_margin,
                idx,
            ),
        )[:num_masks]
        mask_positions[row_idx, ranked_positions] = True
        hypothesis_ids[row_idx, ranked_positions] = mask_id
        used_proposal_rows[row_idx] = True

    return (
        hypothesis_ids,
        hypothesis_attention_mask.long(),
        mask_positions,
        used_proposal_rows,
    )


def resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    return 2 if device.type == "cuda" else 0


def should_run_kd_microbatch(
    step: int,
    batch_idx: int,
    grad_accum: int,
    teacher: "ARTeacher | CTCTeacher | None",
    kd_config: KDConfig,
) -> bool:
    if teacher is None:
        return False
    if grad_accum <= 1:
        return kd_config.active(step)
    is_optimizer_boundary = ((batch_idx + 1) % grad_accum) == 0
    return is_optimizer_boundary and kd_config.active(step)


def build_model(preset: str, vocab_size: int, use_cvae: bool,
                max_positions: int | None = None) -> CTCNAT:
    return CTCNAT.from_preset(
        preset,
        vocab_size=vocab_size,
        use_cvae=use_cvae,
        blank_id=BLANK_ID,
        max_positions=max_positions,
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


_CKPT_STEP_RE = re.compile(r"^checkpoint_step_(\d+)\.pt$")


@torch.no_grad()
def evaluate_probe_em1(
    model: "CTCNAT",
    probe_items: list[dict],
    tokenizer: "SharedCharTokenizer",
    device: torch.device,
    use_cvae: bool,
    max_items: int = 0,
    max_seq_len: int = 128,
    max_context: int = 40,
) -> dict[str, float]:
    """Run greedy CTC decode over probe_v3 items and return EM1 + per-cat EM1.

    Uses the same `model.greedy_decode` path as the production backend, so
    numbers are directly comparable to probe runner output.
    """
    from collections import defaultdict
    items = probe_items[:max_items] if max_items > 0 else probe_items
    model.eval()
    overall_hits = 0
    per_cat_hits: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    for it in items:
        ids = tokenizer.encode_with_special(
            it["context"][-max_context:] if it["context"] else "",
            it["reading"],
        )[:max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        cvae_kwargs = {}
        if use_cvae:
            cvae_kwargs.update(
                writer_ids=torch.zeros(1, dtype=torch.long, device=device),
                domain_ids=torch.zeros(1, dtype=torch.long, device=device),
                source_ids=torch.zeros(1, dtype=torch.long, device=device),
            )
        decoded = model.greedy_decode(input_ids, attention_mask, **cvae_kwargs)
        pred = tokenizer.decode(decoded[0])
        hit = int(pred in it["references"])
        overall_hits += hit
        cat = it.get("category", "_unk")
        per_cat_hits[cat] += hit
        per_cat_total[cat] += 1
    out = {"em1": overall_hits / max(len(items), 1), "n": len(items)}
    for c in sorted(per_cat_total.keys()):
        out[f"em1_{c}"] = per_cat_hits[c] / max(per_cat_total[c], 1)
    return out


def _rolling_keep_checkpoints(output_dir: str, keep_last_k: int) -> None:
    """Delete old numbered checkpoints, keeping only the last K.

    best.pt / final.pt / *_tokenizer.json are never touched. keep_last_k=0
    disables rolling-keep (preserves historical all-steps behaviour).
    """
    if keep_last_k <= 0 or not os.path.isdir(output_dir):
        return
    entries: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        m = _CKPT_STEP_RE.match(name)
        if m:
            entries.append((int(m.group(1)), name))
    if len(entries) <= keep_last_k:
        return
    entries.sort(key=lambda x: x[0])
    to_delete = entries[: len(entries) - keep_last_k]
    for _, name in to_delete:
        pt_path = os.path.join(output_dir, name)
        tok_path = pt_path.replace(".pt", "_tokenizer.json")
        for p in (pt_path, tok_path):
            try:
                os.remove(p)
            except OSError:
                pass


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
        "kd": {
            "teacher_type": getattr(args, "kd_teacher_type", "ar"),
            "teacher_path": getattr(args, "kd_teacher_path", "") or "",
            "teacher_vocab": getattr(args, "kd_teacher_vocab", "") or "",
            "alpha": float(getattr(args, "kd_alpha", 0.0)),
            "alpha_final": getattr(args, "kd_alpha_final", None),
            "alpha_decay_start": int(getattr(args, "kd_alpha_decay_start", 0)),
            "alpha_decay_steps": int(getattr(args, "kd_alpha_decay_steps", 0)),
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
    reset_scheduler: bool = False,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if (
        scheduler is not None
        and checkpoint.get("scheduler_state_dict") is not None
        and not bool(reset_scheduler)
    ):
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
    if tokenizer is not None and int(checkpoint.get("vocab_size", tokenizer.vocab_size)) != int(
        tokenizer.vocab_size
    ):
        mismatches.append(
            f"vocab_size: ckpt={checkpoint.get('vocab_size')} current={tokenizer.vocab_size}"
        )

    # KD metadata: resume must use matching teacher + hyperparameters, otherwise
    # the optimizer/scheduler state belongs to a different objective.
    kd_prev = checkpoint.get("kd") or {}
    # Strict: changing these across resume changes the loss definition or
    # picks a different teacher, so the previous gradients no longer apply.
    kd_strict = [
        ("teacher_type", "kd_teacher_type", "ar"),
        ("teacher_path", "kd_teacher_path", ""),
        ("teacher_vocab", "kd_teacher_vocab", ""),
        ("gate_mode", "kd_gate_mode", "low_conf"),
    ]
    # Tunable: schedule/budget knobs that can legitimately change mid-training
    # (e.g. dialing KD down for OOM safety). Warn but do not block.
    kd_tunable = [
        ("alpha", "kd_alpha", 0.0),
        ("alpha_final", "kd_alpha_final", None),
        ("alpha_decay_start", "kd_alpha_decay_start", 0),
        ("alpha_decay_steps", "kd_alpha_decay_steps", 0),
        ("hard_threshold", "kd_hard_threshold", 0.0),
        ("start_step", "kd_start_step", 0),
        ("warmup_steps", "kd_warmup_steps", 0),
        ("every", "kd_every", 1),
        ("max_new_tokens", "kd_max_new_tokens", 0),
    ]

    def _diff(ckpt_val, cur_val, default) -> bool:
        if type(default) is float:
            return abs(float(ckpt_val) - float(cur_val)) > 1e-9
        return ckpt_val != cur_val

    for ckpt_key, arg_key, default in kd_strict:
        ckpt_val = kd_prev.get(ckpt_key, default) if kd_prev else default
        cur_val = getattr(args, arg_key, default)
        if _diff(ckpt_val, cur_val, default):
            if getattr(args, "allow_resume_kd_swap", False):
                print(
                    f"[resume] WARNING: kd.{ckpt_key} mismatch (ckpt={ckpt_val!r} "
                    f"current={cur_val!r}) — allowed via --allow-resume-kd-swap. "
                    "Optimizer/scheduler state inherited but the KD objective is "
                    "now defined by the new teacher; expect transient instability "
                    "for the first few hundred steps.",
                    flush=True,
                )
            else:
                mismatches.append(f"kd.{ckpt_key}: ckpt={ckpt_val} current={cur_val}")

    for ckpt_key, arg_key, default in kd_tunable:
        ckpt_val = kd_prev.get(ckpt_key, default) if kd_prev else default
        cur_val = getattr(args, arg_key, default)
        if _diff(ckpt_val, cur_val, default):
            print(
                f"[resume] kd.{ckpt_key} changed: ckpt={ckpt_val} -> current={cur_val}",
                flush=True,
            )

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
    preload: bool = False,
):
    dataset = KanaKanjiDataset(
        path, max_samples=max_samples, seed=seed, preload=preload
    )
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
        loader_kwargs["prefetch_factor"] = int(
            __import__("os").environ.get("NEWIME_PREFETCH_FACTOR", "4")
        )
    return DataLoader(
        **loader_kwargs,
    )


def build_kd(
    args: argparse.Namespace,
    device: torch.device,
    student_tokenizer: SharedCharTokenizer,
) -> tuple[ARTeacher | CTCTeacher | None, KDConfig]:
    """Construct the KD teacher + config from CLI args (teacher is optional).

    Teacher type is selected by `--kd-teacher-type`:
      * "ar"      (default): decoder-only AR (SimpleGPT2) teacher via text
                  round-trip + CTC loss. Teacher has its own ARCollator vocab.
      * "seq2seq": TeacherSeq2Seq (encoder-decoder) teacher via text
                  round-trip + CTC loss. Shares SharedCharTokenizer with the
                  student → no vocab remap.
      * "ctc":    CTC-NAT teacher via direct logit KL. Requires the teacher's
                  tokenizer sidecar (`*_tokenizer.json`) to match the student.
    """
    kd_config = KDConfig(
        alpha=args.kd_alpha,
        alpha_final=args.kd_alpha_final,
        alpha_decay_start=args.kd_alpha_decay_start,
        alpha_decay_steps=args.kd_alpha_decay_steps,
        hard_threshold=args.kd_hard_threshold,
        gate_mode=args.kd_gate_mode,
        start_step=args.kd_start_step,
        warmup_steps=args.kd_warmup_steps,
        every=max(args.kd_every, 1),
        max_new_tokens=args.kd_max_new_tokens,
    )
    if not args.kd_teacher_path or args.kd_alpha <= 0.0:
        return None, kd_config

    teacher_type = getattr(args, "kd_teacher_type", "ar")
    if teacher_type == "seq2seq":
        from models.src.training.kd import Seq2SeqTeacher, Seq2SeqTeacherConfig
        teacher_config = Seq2SeqTeacherConfig(
            checkpoint_path=args.kd_teacher_path,
            max_context_chars=args.max_context,
            max_seq_len=args.kd_teacher_max_seq_len,
            max_new_tokens=args.kd_max_new_tokens,
            fp16=args.fp16,
        )
        teacher = Seq2SeqTeacher.from_checkpoint(
            teacher_config, device=device,
            student_tokenizer=student_tokenizer,
        )
        print(
            f"KD teacher (Seq2Seq) loaded: {args.kd_teacher_path} "
            f"(vocab={teacher.tokenizer.vocab_size}, fp16={teacher_config.fp16}) "
            f"α={kd_config.alpha}, threshold={kd_config.hard_threshold}, "
            f"mode={kd_config.gate_mode}, start={kd_config.start_step}, "
            f"warmup={kd_config.warmup_steps}, alpha_final={kd_config.alpha_final}, "
            f"decay_start={kd_config.alpha_decay_start}, "
            f"decay_steps={kd_config.alpha_decay_steps}, every={kd_config.every}"
        )
        return teacher, kd_config

    if teacher_type == "ctc":
        teacher_config = CTCTeacherConfig(
            checkpoint_path=args.kd_teacher_path,
            fp16=args.fp16,
        )
        teacher = CTCTeacher.from_checkpoint(
            teacher_config, device=device,
            student_tokenizer=student_tokenizer,
        )
        print(
            f"KD teacher (CTC-NAT) loaded: {args.kd_teacher_path} "
            f"(vocab={teacher.tokenizer.vocab_size}, fp16={teacher_config.fp16}) "
            f"α={kd_config.alpha}, threshold={kd_config.hard_threshold}, "
            f"mode={kd_config.gate_mode}, start={kd_config.start_step}, "
            f"warmup={kd_config.warmup_steps}, alpha_final={kd_config.alpha_final}, "
            f"decay_start={kd_config.alpha_decay_start}, "
            f"decay_steps={kd_config.alpha_decay_steps}, every={kd_config.every}, "
            f"T={args.kd_temperature}"
        )
        return teacher, kd_config

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
        f"KD teacher (AR) loaded: {args.kd_teacher_path} "
        f"(vocab={teacher.collator.vocab_size}, fp16={teacher_config.fp16}) "
        f"α={kd_config.alpha}, threshold={kd_config.hard_threshold}, mode={kd_config.gate_mode}, "
        f"start={kd_config.start_step}, warmup={kd_config.warmup_steps}, "
        f"alpha_final={kd_config.alpha_final}, decay_start={kd_config.alpha_decay_start}, "
        f"decay_steps={kd_config.alpha_decay_steps}, "
        f"every={kd_config.every}"
    )
    return teacher, kd_config


def train_local(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    tokenizer = build_tokenizer(args)

    # Probe auto-eval items (loaded once; evaluated via greedy decode at
    # eval-every intervals when --probe-eval-every > 0).
    probe_items: list[dict] = []
    if getattr(args, "probe_eval_every", 0) > 0 and getattr(args, "probe_eval_path", ""):
        from models.src.eval.bench_loaders import load_probe
        probe_items = load_probe(args.probe_eval_path)
        print(
            f"[probe] loaded {len(probe_items)} items from {args.probe_eval_path} "
            f"(eval every {args.probe_eval_every} steps; "
            f"best.pt selected by probe EM1)"
        )

    model = build_model(args.preset, vocab_size=tokenizer.vocab_size, use_cvae=args.use_cvae, max_positions=args.max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LR schedule: warmup then optional decay. `flat` (default, legacy) holds
    # at peak after warmup (the source of the "LR plateau" seen in earlier 30M
    # runs). `cosine` decays from peak to 0 across remaining steps.
    import math

    def lr_lambda(step: int) -> float:
        warmup = max(args.warmup_steps, 1)
        if step < warmup:
            return (step + 1) / warmup
        schedule = getattr(args, "lr_schedule", "flat")
        floor = max(0.0, min(1.0, getattr(args, "lr_min_ratio", 0.0)))
        if schedule == "cosine":
            total = max(args.max_steps, warmup + 1)
            progress = (step - warmup) / max(total - warmup, 1)
            progress = min(max(progress, 0.0), 1.0)
            # Cosine from 1.0 at warmup end to `floor` at max_steps (floor=0.0
            # reproduces the legacy decay-to-zero behaviour).
            cos_unit = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cos_unit
        if schedule == "cosine_warm_restarts":
            period = max(int(getattr(args, "lr_restart_period", 80000)), 1)
            decay = float(getattr(args, "lr_restart_decay", 0.9))
            rel = step - warmup
            cycle = rel // period
            rel_in_cycle = rel % period
            progress = rel_in_cycle / period
            cos_unit = 0.5 * (1.0 + math.cos(math.pi * progress))
            amplitude = decay ** cycle
            return floor + (amplitude - floor) * cos_unit
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    teacher, kd_config = build_kd(args, device, tokenizer)

    start_step = 0
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            reset_scheduler=bool(getattr(args, "reset_scheduler", False)),
        )
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
        preload=bool(getattr(args, "preload_dataset", False)),
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
    refine_stats = {
        "loss_sum": 0.0,
        "tokens_sum": 0,
        "batches": 0,
        "proposal_rows": 0,
        "rows": 0,
        "weight_sum": 0.0,
        "iter_loss_sum": [0.0] * max(int(getattr(args, "refine_iterations", 1)), 1),
        "iter_batches": [0] * max(int(getattr(args, "refine_iterations", 1)), 1),
        "masked_acc_sum": 0.0,
        "em_before_sum": 0.0,
        "em_after_sum": 0.0,
        "top1_improved_sum": 0.0,
        "top1_degraded_sum": 0.0,
        "metric_batches": 0,
        "remask_loss_sum": 0.0,
        "remask_batches": 0,
        "stop_loss_sum": 0.0,
        "stop_batches": 0,
    }

    def fetch_next_batch(loader_iter, loader):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        return batch, loader_iter

    def run_microbatch(batch, batch_idx: int, epoch_idx: int) -> None:
        nonlocal step, last_log, kd_stats, refine_stats, best_metric

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

            refine_weight_now = resolve_refine_loss_weight(args, step)
            if refine_weight_now > 0.0:
                refine_mask_ratio = resolve_refine_mask_ratio(args)
                used_proposal_rows = torch.zeros(
                    batch["target_ids"].shape[0],
                    dtype=torch.bool,
                    device=batch["target_ids"].device,
                )
                if args.refine_source == "target":
                    hyp_ids, hyp_attn, mask_positions = build_refinement_batch(
                        batch["target_ids"],
                        batch["target_lengths"],
                        mask_ratio=refine_mask_ratio,
                    )
                else:
                    alignment = CTCNAT.collapse_alignment_tensors(
                        result["logits"].detach(),
                        batch["attention_mask"].sum(dim=1).long(),
                        BLANK_ID,
                    )
                    hyp_ids, hyp_attn, mask_positions, used_proposal_rows = (
                        build_refinement_batch_from_proposal_tensors(
                            batch["target_ids"],
                            batch["target_lengths"],
                            alignment["token_ids"],
                            alignment["min_log_prob"],
                            alignment["min_margin"],
                            alignment["lengths"],
                            mask_ratio=refine_mask_ratio,
                        )
                    )
                    if args.refine_source == "proposal" and not bool(used_proposal_rows.any().item()):
                        hyp_ids, hyp_attn, mask_positions = build_refinement_batch(
                            batch["target_ids"],
                            batch["target_lengths"],
                            mask_ratio=refine_mask_ratio,
                        )

                num_iterations = max(int(getattr(args, "refine_iterations", 1)), 1)
                target_ids_t = batch["target_ids"]
                valid_positions = hyp_attn.bool()
                current_ids = hyp_ids
                current_mask = mask_positions
                initial_hyp_ids = hyp_ids.clone()
                batch_total_refine_loss = 0.0
                first_refine_logits: torch.Tensor | None = None
                first_refine_decoder_out: torch.Tensor | None = None
                last_refine_result: dict | None = None
                for it in range(num_iterations):
                    refine_result = model.refine_from_proposal(
                        proposal=result,
                        hypothesis_ids=current_ids,
                        hypothesis_attention_mask=hyp_attn,
                    )
                    last_refine_result = refine_result
                    if it == 0:
                        first_refine_logits = refine_result["logits"]
                        first_refine_decoder_out = refine_result["decoder_out"]
                    ce = F.cross_entropy(
                        refine_result["logits"].reshape(-1, refine_result["logits"].shape[-1]),
                        target_ids_t.reshape(-1),
                        reduction="none",
                    ).reshape_as(target_ids_t)
                    iter_masked = int(current_mask.sum().item())
                    if iter_masked > 0:
                        iter_loss = ce[current_mask].mean()
                        batch_total_refine_loss = batch_total_refine_loss + iter_loss
                        refine_stats["iter_loss_sum"][it] += float(iter_loss.detach().item())
                        refine_stats["iter_batches"][it] += 1
                        if it == 0:
                            refine_stats["loss_sum"] += float(iter_loss.detach().item())
                            refine_stats["tokens_sum"] += iter_masked
                            refine_stats["batches"] += 1
                            refine_stats["proposal_rows"] += int(used_proposal_rows.sum().item())
                            refine_stats["rows"] += int(target_ids_t.shape[0])
                            refine_stats["weight_sum"] += refine_weight_now
                    if it == num_iterations - 1:
                        break
                    # Decide next-iteration masks: fill argmax into current
                    # masked positions, then re-mask the positions that are
                    # still likely to be wrong. Use the learned remask head
                    # when available (falls back to a confidence threshold).
                    with torch.no_grad():
                        argmax = refine_result["logits"].argmax(dim=-1)
                        filled = torch.where(current_mask, argmax, current_ids)
                        wrong = (filled != target_ids_t) & valid_positions
                        remask_prob = torch.sigmoid(refine_result["remask_logits"])
                        next_mask = (remask_prob >= float(args.remask_threshold)) & valid_positions
                        # Guarantee at least one mask if the sequence is still
                        # wrong anywhere — otherwise the next iteration sees
                        # no signal.
                        still_wrong_rows = wrong.any(dim=1)
                        for b_idx in still_wrong_rows.nonzero(as_tuple=False).squeeze(1).tolist():
                            if not bool(next_mask[b_idx].any().item()):
                                wrong_pos = wrong[b_idx].nonzero(as_tuple=False).squeeze(1)
                                if wrong_pos.numel() > 0:
                                    pick = int(wrong_pos[0].item())
                                    next_mask[b_idx, pick] = True
                        # If no wrong positions remain, stop iterating early.
                        if not bool(still_wrong_rows.any().item()):
                            break
                    current_ids = torch.where(
                        next_mask,
                        torch.full_like(current_ids, MASK_ID),
                        filled,
                    )
                    current_mask = next_mask
                if isinstance(batch_total_refine_loss, torch.Tensor):
                    loss = loss + refine_weight_now * batch_total_refine_loss

                # Remask head: BCE on "is this position wrong after this
                # iteration's argmax fill". Use the first iteration's
                # decoder output (most signal early in training).
                remask_weight_now = float(getattr(args, "remask_loss_weight", 0.0))
                if (
                    remask_weight_now > 0.0
                    and first_refine_logits is not None
                    and first_refine_decoder_out is not None
                ):
                    with torch.no_grad():
                        first_argmax = first_refine_logits.argmax(dim=-1)
                        first_filled = torch.where(mask_positions, first_argmax, initial_hyp_ids)
                        remask_target = ((first_filled != target_ids_t) & valid_positions).float()
                    remask_bce = F.binary_cross_entropy_with_logits(
                        last_refine_result["remask_logits"],
                        remask_target,
                        reduction="none",
                    )
                    remask_bce = (remask_bce * valid_positions.float()).sum() / valid_positions.float().sum().clamp(min=1)
                    loss = loss + remask_weight_now * remask_bce
                    refine_stats["remask_loss_sum"] += float(remask_bce.detach().item())
                    refine_stats["remask_batches"] += 1

                # Stop head: BCE on "refined sequence equals target in full".
                stop_weight_now = float(getattr(args, "stop_loss_weight", 0.0))
                if (
                    stop_weight_now > 0.0
                    and last_refine_result is not None
                ):
                    with torch.no_grad():
                        last_argmax = last_refine_result["logits"].argmax(dim=-1)
                        last_filled = torch.where(current_mask, last_argmax, current_ids)
                        row_correct = ((last_filled == target_ids_t) | ~valid_positions).all(dim=1).float()
                    stop_bce = F.binary_cross_entropy_with_logits(
                        last_refine_result["stop_logit"],
                        row_correct,
                        reduction="mean",
                    )
                    loss = loss + stop_weight_now * stop_bce
                    refine_stats["stop_loss_sum"] += float(stop_bce.detach().item())
                    refine_stats["stop_batches"] += 1

                # Refiner metrics (B): masked accuracy + EM before/after +
                # top1 improve/degrade. Cheap — computed on the final
                # iteration's output against targets.
                if last_refine_result is not None and first_refine_logits is not None:
                    with torch.no_grad():
                        first_argmax = first_refine_logits.argmax(dim=-1)
                        first_filled = torch.where(mask_positions, first_argmax, initial_hyp_ids)
                        last_argmax = last_refine_result["logits"].argmax(dim=-1)
                        last_filled = torch.where(current_mask, last_argmax, current_ids)
                        masked_acc = (
                            (last_argmax == target_ids_t)[mask_positions].float().mean()
                            if bool(mask_positions.any().item())
                            else torch.zeros((), device=target_ids_t.device)
                        )
                        valid_and_target = valid_positions
                        em_before = ((initial_hyp_ids == target_ids_t) | ~valid_and_target).all(dim=1).float().mean()
                        em_after = ((last_filled == target_ids_t) | ~valid_and_target).all(dim=1).float().mean()
                        was_wrong = (initial_hyp_ids != target_ids_t) & valid_and_target
                        was_right = (initial_hyp_ids == target_ids_t) & valid_and_target
                        improved = (was_wrong & (last_filled == target_ids_t)).float().sum() / was_wrong.float().sum().clamp(min=1)
                        degraded = (was_right & (last_filled != target_ids_t)).float().sum() / was_right.float().sum().clamp(min=1)
                    refine_stats["masked_acc_sum"] += float(masked_acc.item())
                    refine_stats["em_before_sum"] += float(em_before.item())
                    refine_stats["em_after_sum"] += float(em_after.item())
                    refine_stats["top1_improved_sum"] += float(improved.item())
                    refine_stats["top1_degraded_sum"] += float(degraded.item())
                    refine_stats["metric_batches"] += 1

            if run_kd:
                alpha_now = kd_config.alpha_at(step)
                if alpha_now <= 0.0:
                    run_kd = False

            if run_kd:
                if isinstance(teacher, CTCTeacher):
                    # Direct-logit KL path. Same input as student, single
                    # forward, soft-KL on output distributions.
                    with torch.no_grad():
                        teacher_logits, teacher_conf_tensor = teacher(
                            batch["input_ids"], batch["attention_mask"]
                        )
                    hard_mask = hard_example_mask(
                        teacher_conf_tensor,
                        kd_config.hard_threshold,
                        mode=kd_config.gate_mode,
                    )
                    kd_loss_value, num_hard = compute_kd_kl_loss(
                        student_logits=result["logits"],
                        teacher_logits=teacher_logits,
                        attention_mask=batch["attention_mask"],
                        hard_mask=hard_mask,
                        temperature=float(getattr(args, "kd_temperature", 2.0)),
                    )
                    if num_hard > 0 and alpha_now > 0.0:
                        loss = loss + alpha_now * args.grad_accum * kd_loss_value
                    kd_stats["loss_sum"] += float(kd_loss_value.detach().item())
                    kd_stats["hard_sum"] += num_hard
                    kd_stats["total_sum"] += int(batch["input_ids"].shape[0])
                    kd_stats["conf_sum"] += (
                        float(teacher_conf_tensor.mean().item())
                        if teacher_conf_tensor.numel() else 0.0
                    )
                    kd_stats["batches"] += 1
                else:
                    with torch.no_grad():
                        kd_chunk = int(getattr(args, "max_kd_batch_size", 0))
                        n_ctx = len(contexts)
                        if kd_chunk <= 0 or kd_chunk >= n_ctx:
                            teacher_texts, teacher_conf = teacher.generate(
                                contexts=contexts,
                                readings=readings,
                                max_new_tokens=kd_config.max_new_tokens,
                            )
                        else:
                            teacher_texts = []
                            teacher_conf = []
                            for s in range(0, n_ctx, kd_chunk):
                                t_texts, t_conf = teacher.generate(
                                    contexts=contexts[s : s + kd_chunk],
                                    readings=readings[s : s + kd_chunk],
                                    max_new_tokens=kd_config.max_new_tokens,
                                )
                                teacher_texts.extend(t_texts)
                                teacher_conf.extend(t_conf)
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
                if refine_stats["batches"] > 0:
                    refine_avg = refine_stats["loss_sum"] / refine_stats["batches"]
                    masked_avg = refine_stats["tokens_sum"] / refine_stats["batches"]
                    proposal_ratio = refine_stats["proposal_rows"] / max(refine_stats["rows"], 1)
                    refine_weight_avg = refine_stats["weight_sum"] / refine_stats["batches"]
                    line += (
                        f" refine_loss={refine_avg:.4f}n/tok "
                        f"refine_masked={masked_avg:.1f} "
                        f"refine_prop={proposal_ratio:.2f} "
                        f"refine_w={refine_weight_avg:.3f}"
                    )
                    per_iter = []
                    for i, (s, b) in enumerate(zip(
                        refine_stats["iter_loss_sum"], refine_stats["iter_batches"]
                    )):
                        if b > 0:
                            per_iter.append(f"it{i}={s / b:.3f}")
                    if per_iter:
                        line += " refine_iter_loss=" + ",".join(per_iter)
                    if refine_stats["metric_batches"] > 0:
                        mb = refine_stats["metric_batches"]
                        line += (
                            f" masked_acc={refine_stats['masked_acc_sum'] / mb:.3f}"
                            f" em_before={refine_stats['em_before_sum'] / mb:.3f}"
                            f" em_after={refine_stats['em_after_sum'] / mb:.3f}"
                            f" top1_imp={refine_stats['top1_improved_sum'] / mb:.3f}"
                            f" top1_deg={refine_stats['top1_degraded_sum'] / mb:.3f}"
                        )
                    if refine_stats["remask_batches"] > 0:
                        line += (
                            f" remask_loss="
                            f"{refine_stats['remask_loss_sum'] / refine_stats['remask_batches']:.4f}"
                        )
                    if refine_stats["stop_batches"] > 0:
                        line += (
                            f" stop_loss="
                            f"{refine_stats['stop_loss_sum'] / refine_stats['stop_batches']:.4f}"
                        )
                print(line)
                kd_stats = {
                    "loss_sum": 0.0,
                    "hard_sum": 0,
                    "total_sum": 0,
                    "conf_sum": 0.0,
                    "batches": 0,
                }
                refine_stats = {
                    "loss_sum": 0.0,
                    "tokens_sum": 0,
                    "batches": 0,
                    "proposal_rows": 0,
                    "rows": 0,
                    "weight_sum": 0.0,
                    "iter_loss_sum": [0.0] * max(int(getattr(args, "refine_iterations", 1)), 1),
                    "iter_batches": [0] * max(int(getattr(args, "refine_iterations", 1)), 1),
                    "masked_acc_sum": 0.0,
                    "em_before_sum": 0.0,
                    "em_after_sum": 0.0,
                    "top1_improved_sum": 0.0,
                    "top1_degraded_sum": 0.0,
                    "metric_batches": 0,
                    "remask_loss_sum": 0.0,
                    "remask_batches": 0,
                    "stop_loss_sum": 0.0,
                    "stop_batches": 0,
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

                # Optional probe_v3 auto-eval. When --probe-eval-every > 0 and
                # we're on an eval-every boundary that is also a probe-eval
                # multiple, run greedy decode over probe items and use the
                # resulting EM1 as the best.pt selection metric instead of dev.
                probe_em1 = None
                if (
                    getattr(args, "probe_eval_every", 0) > 0
                    and step % args.probe_eval_every == 0
                    and probe_items
                ):
                    probe_summary = evaluate_probe_em1(
                        model, probe_items, tokenizer, device,
                        use_cvae=args.use_cvae,
                        max_items=getattr(args, "probe_eval_limit", 0),
                        max_seq_len=args.max_seq_len,
                        max_context=args.max_context,
                    )
                    probe_em1 = probe_summary["em1"]
                    cat_bits = " ".join(
                        f"{k[4:]}={v:.2f}" for k, v in probe_summary.items()
                        if k.startswith("em1_")
                    )
                    print(
                        f"[probe {step}] EM1={probe_em1:.4f} n={probe_summary['n']} {cat_bits}"
                    )

                metric_key = probe_em1 if probe_em1 is not None else metrics.get("exact_match_top1", 0.0)
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
                _rolling_keep_checkpoints(args.output, getattr(args, "keep_last_k", 0))

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
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="On resume, discard the saved scheduler state and rebuild a fresh "
             "schedule from the current CLI args (--lr / --warmup-steps / "
             "--lr-schedule / --max-steps). Use when extending a finished run "
             "past its original max-steps — otherwise the cosine schedule "
             "stays pinned at its minimum LR for the entire resumed phase.",
    )
    parser.add_argument(
        "--allow-resume-kd-swap",
        action="store_true",
        help="Demote KD strict-match (teacher_type / teacher_path / teacher_vocab / "
             "gate_mode) to warnings when resuming. Use when intentionally changing "
             "the KD teacher between runs (v2 → v2.1 style). Optimizer/scheduler "
             "state is still inherited; only the KD objective changes.",
    )
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
    parser.add_argument(
        "--refine-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the dedicated Mask-CTC refinement CE loss. 0 disables it.",
    )
    parser.add_argument(
        "--refine-warmup-steps",
        type=int,
        default=0,
        help="Linearly ramp refinement loss weight from 0 to --refine-loss-weight over this many optimizer steps.",
    )
    parser.add_argument(
        "--refine-mask-ratio",
        type=float,
        default=0.3,
        help="Fraction of valid target tokens masked when training the refinement branch.",
    )
    parser.add_argument(
        "--refine-mask-ratio-min",
        type=float,
        default=None,
        help="If set with --refine-mask-ratio-max, sample the refine mask ratio uniformly per batch.",
    )
    parser.add_argument(
        "--refine-mask-ratio-max",
        type=float,
        default=None,
        help="If set with --refine-mask-ratio-min, sample the refine mask ratio uniformly per batch.",
    )
    parser.add_argument(
        "--refine-source",
        choices=["target", "proposal", "mixed"],
        default="target",
        help=(
            "Source of refinement hypotheses: target=mask the gold target only; "
            "proposal=use current proposal when collapsed length matches, else fallback; "
            "mixed=same as proposal but report mixed usage explicitly."
        ),
    )
    parser.add_argument(
        "--refine-iterations",
        type=int,
        default=1,
        help="Number of refinement iterations to supervise. Iter 2+ re-masks "
             "positions still predicted incorrectly after iter 1 (using the "
             "learned remask head when enabled, confidence fallback otherwise).",
    )
    parser.add_argument(
        "--remask-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the learned re-mask head BCE loss (target = refined "
             "argmax mismatches gold target). 0 disables, inference falls back "
             "to confidence-based remasking.",
    )
    parser.add_argument(
        "--remask-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold on the remask head used to pick next-iteration "
             "mask positions during training and inference.",
    )
    parser.add_argument(
        "--stop-loss-weight",
        type=float,
        default=0.0,
        help="Weight for the learned stop-head BCE loss (target = refined "
             "sequence equals gold target in full). 0 disables.",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold on the stop head used to halt iteration early "
             "at inference time.",
    )
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument(
        "--lr-schedule",
        choices=["flat", "cosine", "cosine_warm_restarts"],
        default="flat",
        help=(
            "LR schedule after warmup: 'flat' holds peak (legacy, caused "
            "LR plateau in 30M training). 'cosine' decays to --lr-min-ratio "
            "across --max-steps. 'cosine_warm_restarts' restarts every "
            "--lr-restart-period steps with amplitude * --lr-restart-decay "
            "per cycle."
        ),
    )
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.0,
        help="Cosine LR floor as fraction of peak. 0.0 legacy (decay-to-zero). 0.1 "
             "recommended for long runs to avoid end-of-schedule lr=0 plateau.",
    )
    parser.add_argument(
        "--lr-restart-period",
        type=int,
        default=80000,
        help="Steps per cosine cycle when --lr-schedule=cosine_warm_restarts.",
    )
    parser.add_argument(
        "--lr-restart-decay",
        type=float,
        default=0.9,
        help="Peak amplitude multiplier per cycle (SGDR-style) for warm restarts.",
    )
    parser.add_argument("--tiny-overfit-samples", type=int, default=0)
    parser.add_argument("--short-sample-max-chars", type=int, default=0)
    parser.add_argument("--warmup-short-sample-steps", type=int, default=0)
    parser.add_argument("--warmup-short-sample-max-chars", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--preload-dataset",
        action="store_true",
        help="Read + json.loads every sampled training row into a Python list "
             "at startup instead of streaming from the mmap'd jsonl. Eliminates "
             "per-step disk seek and JSON parse from the dataloader hot path. "
             "Requires enough cgroup RAM: ~1 KiB per sample (30M rows ≈ 30 GiB).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=0,
        help="Rolling-keep only the last K numbered checkpoints. 0 disables "
             "(keeps every checkpoint, legacy). best.pt / final.pt are never deleted.",
    )
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--probe-eval-path",
        default="",
        help="Path to probe_v3 JSON (AJIMEE-compatible items). When set with "
             "--probe-eval-every > 0, the train loop runs greedy decode over "
             "these items at eval-every intervals and uses the resulting EM1 "
             "as the best.pt selection metric (replacing dev EM).",
    )
    parser.add_argument(
        "--probe-eval-every",
        type=int,
        default=0,
        help="Interval (steps) between probe auto-evals. 0 disables.",
    )
    parser.add_argument(
        "--probe-eval-limit",
        type=int,
        default=0,
        help="Cap probe items per auto-eval (0 = all).",
    )
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

    kd_group = parser.add_argument_group("online KD (AR / Seq2Seq / CTC teacher)")
    kd_group.add_argument(
        "--kd-teacher-type",
        choices=["ar", "seq2seq", "ctc"],
        default="ar",
        help=(
            "Teacher kind. 'ar' = autoregressive SimpleGPT2 (vocab-bridged via "
            "text round-trip + CTC loss against teacher text). "
            "'seq2seq' = TeacherSeq2Seq encoder-decoder (shares SharedCharTokenizer "
            "with student; text round-trip + CTC loss). "
            "'ctc' = CTC-NAT teacher (same tokenizer as student required; KD "
            "via direct logit KL, much faster forward pass)."
        ),
    )
    kd_group.add_argument(
        "--kd-temperature",
        type=float,
        default=2.0,
        help="Soft-KL temperature for CTC teacher (ignored for AR teacher)",
    )
    kd_group.add_argument("--kd-teacher-path", default="", help="Teacher checkpoint (.pt)")
    kd_group.add_argument("--kd-teacher-vocab", default="", help="AR teacher vocab JSON (default: auto; AR only)")
    kd_group.add_argument("--kd-teacher-hidden", type=int, default=512)
    kd_group.add_argument("--kd-teacher-layers", type=int, default=8)
    kd_group.add_argument("--kd-teacher-heads", type=int, default=8)
    kd_group.add_argument("--kd-teacher-max-seq-len", type=int, default=256)
    kd_group.add_argument("--kd-alpha", type=float, default=0.0, help="KD loss weight (0 disables)")
    kd_group.add_argument(
        "--kd-alpha-final",
        type=float,
        default=None,
        help="Final KD alpha after optional post-warmup decay (default: keep constant)",
    )
    kd_group.add_argument(
        "--kd-alpha-decay-start",
        type=int,
        default=0,
        help="Optimizer step at which KD alpha starts decaying toward --kd-alpha-final",
    )
    kd_group.add_argument(
        "--kd-alpha-decay-steps",
        type=int,
        default=0,
        help="Number of optimizer steps used to linearly decay KD alpha",
    )
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
    kd_group.add_argument(
        "--max-kd-batch-size",
        type=int,
        default=0,
        help=(
            "If >0, split teacher.generate() into chunks of this many samples "
            "and concatenate outputs. Peak VRAM of the teacher pass scales with "
            "this chunk size instead of the full student batch; useful when KD "
            "fires on a large student batch and AR teacher generate spikes VRAM."
        ),
    )

    args = parser.parse_args()
    if args.small_vocab_max_kanji > 0:
        args.max_kanji = args.small_vocab_max_kanji

    tokenizer = build_tokenizer(args)
    model = build_model(args.preset, vocab_size=tokenizer.vocab_size, use_cvae=args.use_cvae, max_positions=args.max_seq_len)
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
