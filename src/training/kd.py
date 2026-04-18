"""Online knowledge distillation for Phase 3 CTC-NAT training.

Design (see `docs/phase3_plan.md` Step D / memory `feedback_kd_strategy.md`):

- Teacher is the Phase 2 AR baseline (`SimpleGPT2`, 32M, ~6997 vocab) loaded
  from a `.pt` checkpoint + its companion `_vocab.json`.
- Teacher runs on the same device as the student (fp16, eval, no_grad).
- Online batched greedy generation — **no pre-expanded JSONL**.
- Teacher has its own tokenizer (`ARCollator`), vocab-incompatible with the
  student `SharedCharTokenizer`. We bridge via text round-trip: teacher text
  → student CTC target IDs.
- Hard-example gating: only samples where the teacher's mean top-1 confidence
  over the generated sequence is *below* `hard_threshold` contribute to the
  KD loss. Easy examples (teacher confident) are skipped so the teacher does
  not cap the student.
- KD loss: CTC loss against teacher text, averaged only over the hard subset.
  Combined with the real CTC loss via weight α (optionally linearly ramped).

Not handled here (future work):
- Separate-GPU teacher placement
- Cross-tokenizer soft-KL / probability matching
- Persistent LRU cache of teacher outputs across epochs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import ARCollator
from src.data.tokenizer import BLANK_ID, PAD_ID, SharedCharTokenizer
from src.training.train_ar import SimpleGPT2


@dataclass
class TeacherConfig:
    """Hyperparameters needed to reconstruct the AR teacher."""

    checkpoint_path: str
    vocab_path: str = ""
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    max_seq_len: int = 256
    max_new_tokens: int = 128
    max_context_chars: int = 40
    fp16: bool = True


@dataclass
class KDConfig:
    """Runtime controls for online KD."""

    alpha: float = 0.3
    alpha_final: float | None = None
    alpha_decay_start: int = 0
    alpha_decay_steps: int = 0
    hard_threshold: float = 0.6
    gate_mode: Literal["low_conf", "high_conf", "all"] = "low_conf"
    start_step: int = 0
    warmup_steps: int = 0
    every: int = 1
    max_new_tokens: int = 128

    def alpha_at(self, step: int) -> float:
        """α with optional linear warmup from start_step and optional decay."""
        if step < self.start_step:
            return 0.0

        if self.warmup_steps <= 0:
            alpha_now = self.alpha
        else:
            progress = min((step - self.start_step) / self.warmup_steps, 1.0)
            alpha_now = self.alpha * progress

        if self.alpha_final is None or self.alpha_decay_steps <= 0:
            return alpha_now
        if step < self.alpha_decay_start:
            return alpha_now

        decay_progress = min((step - self.alpha_decay_start) / self.alpha_decay_steps, 1.0)
        return alpha_now + (self.alpha_final - alpha_now) * decay_progress

    def active(self, step: int) -> bool:
        """Whether KD should run at the given optimizer step.

        Evaluated on the *optimizer* step (not per microbatch). When True for a
        given step, every microbatch inside that step's grad-accum window will
        run the teacher; the caller owns that semantic.
        """
        if self.alpha <= 0.0:
            return False
        if step < self.start_step:
            return False
        if self.every <= 1:
            return True
        return (step % self.every) == 0


class ARTeacher(nn.Module):
    """Frozen AR teacher for online KD.

    Implements batched greedy generation. All parameters are frozen and the
    module is kept in eval mode.
    """

    def __init__(
        self,
        model: SimpleGPT2,
        collator: ARCollator,
        config: TeacherConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.collator = collator
        self.config = config
        self.device = device
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.dtype = torch.float16 if config.fp16 and device.type == "cuda" else torch.float32
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model = self.model.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        config: TeacherConfig,
        device: torch.device,
    ) -> "ARTeacher":
        ckpt_path = Path(config.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
        vocab_path = config.vocab_path or str(ckpt_path).replace(".pt", "_vocab.json")
        if not Path(vocab_path).exists():
            raise FileNotFoundError(f"Teacher vocab not found: {vocab_path}")

        collator = ARCollator(max_seq_len=config.max_seq_len)
        collator.load_vocab(vocab_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vocab_size = int(checkpoint.get("vocab_size", collator.vocab_size))
        if vocab_size != collator.vocab_size:
            raise ValueError(
                f"Teacher vocab size mismatch: checkpoint={vocab_size} "
                f"vocab_json={collator.vocab_size}"
            )

        model = SimpleGPT2(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_positions=config.max_seq_len,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, collator=collator, config=config, device=device)

    def train(self, mode: bool = True) -> "ARTeacher":
        """Force eval mode regardless of parent-module train() calls."""
        super().train(False)
        self.model.eval()
        return self

    def _encode_frozen(self, text: str) -> list[int]:
        """Map characters to teacher IDs without mutating the vocab.

        Unknown characters fall back to UNK so the teacher embedding stays
        within its fixed `vocab_size`. Using `ARCollator.encode_text` directly
        would extend the vocabulary at inference time and produce out-of-range
        IDs (teacher embedding is frozen at checkpoint load).
        """
        table = self.collator._char_to_id
        unk = self.collator.UNK
        return [table.get(ch, unk) for ch in text]

    def _encode_prompt(self, context: str, reading: str) -> list[int]:
        """Build `[context][SEP][reading][OUT]` prompt using the teacher vocab."""
        context = context[-self.config.max_context_chars :] if context else ""
        ctx_ids = self._encode_frozen(context)
        read_ids = self._encode_frozen(reading)
        return ctx_ids + [self.collator.SEP] + read_ids + [self.collator.OUT]

    @torch.no_grad()
    def generate(
        self,
        contexts: Sequence[str],
        readings: Sequence[str],
        max_new_tokens: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Batched greedy generation.

        Returns:
            texts: teacher greedy output per sample (decoded with teacher vocab)
            confidences: mean top-1 softmax probability over generated tokens
        """
        if len(contexts) != len(readings):
            raise ValueError("contexts and readings must have the same length")
        batch_size = len(contexts)
        if batch_size == 0:
            return [], []

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        prompts = [self._encode_prompt(c, r) for c, r in zip(contexts, readings)]
        prompt_lengths = [len(p) for p in prompts]
        max_prompt = max(prompt_lengths)

        cap = self.config.max_seq_len
        if max_prompt >= cap:
            # Prompt alone fills the budget — nothing to generate.
            return ["" for _ in range(batch_size)], [0.0 for _ in range(batch_size)]

        total_len = min(max_prompt + max_new_tokens, cap)
        input_ids = torch.full(
            (batch_size, total_len),
            self.collator.PAD,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, total_len), dtype=torch.long, device=self.device
        )
        for i, prompt in enumerate(prompts):
            prompt = prompt[:cap]
            length = len(prompt)
            input_ids[i, :length] = torch.tensor(prompt, dtype=torch.long, device=self.device)
            attention_mask[i, :length] = 1

        lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        confidence_sums = torch.zeros(batch_size, device=self.device)
        confidence_counts = torch.zeros(batch_size, device=self.device)

        for step in range(max_new_tokens):
            current_max = int(lengths.max().item())
            if current_max >= total_len:
                break

            active = ~finished
            if not active.any():
                break

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.dtype == torch.float16,
                dtype=self.dtype if self.dtype == torch.float16 else torch.float32,
            ):
                logits = self.model(
                    input_ids[:, :current_max],
                    attention_mask[:, :current_max],
                )  # (batch, current_max, vocab)

            last_idx = lengths - 1  # position of the last real token
            last_idx = last_idx.clamp(max=current_max - 1)
            batch_arange = torch.arange(batch_size, device=self.device)
            step_logits = logits[batch_arange, last_idx].float()
            probs = F.softmax(step_logits, dim=-1)
            top_probs, top_ids = probs.max(dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                tok = int(top_ids[i].item())
                conf = float(top_probs[i].item())
                confidence_sums[i] += conf
                confidence_counts[i] += 1.0
                if tok == self.collator.EOS or tok == self.collator.PAD:
                    finished[i] = True
                    continue
                pos = int(lengths[i].item())
                if pos >= total_len:
                    finished[i] = True
                    continue
                input_ids[i, pos] = tok
                attention_mask[i, pos] = 1
                lengths[i] += 1
                generated[i].append(tok)

        texts = [self.collator.decode_ids(ids) for ids in generated]
        confidences = [
            float(confidence_sums[i].item() / max(confidence_counts[i].item(), 1.0))
            for i in range(batch_size)
        ]
        return texts, confidences


def encode_texts_for_student(
    texts: Sequence[str],
    tokenizer: SharedCharTokenizer,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-encode teacher texts into the student tokenizer.

    Blanks are filtered (CTC targets must not contain the blank symbol).
    Empty sequences are represented by length 0; callers must mask these out
    (CTC loss is undefined for zero-length targets).
    """
    encoded: list[list[int]] = []
    lengths: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        ids = [tid for tid in ids if tid != BLANK_ID]
        ids = ids[:max_len]
        encoded.append(ids)
        lengths.append(len(ids))

    max_observed = max(lengths) if lengths else 0
    padded_len = max(max_observed, 1)
    padded = []
    for ids in encoded:
        pad_count = padded_len - len(ids)
        padded.append(ids + [PAD_ID] * pad_count)
    ids_tensor = torch.tensor(padded, dtype=torch.long) if padded else torch.zeros(
        (0, padded_len), dtype=torch.long
    )
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    return ids_tensor, length_tensor


def hard_example_mask(
    confidences: torch.Tensor,
    threshold: float,
    mode: Literal["low_conf", "high_conf", "all"] = "low_conf",
) -> torch.Tensor:
    """Select which samples contribute to KD.

    Modes:
    - low_conf: teacher is uncertain, useful for "hard-example" KD.
    - high_conf: teacher is confident, useful for stabilizing the student.
    - all: apply KD to every sample and ignore the threshold.
    """
    if mode == "all":
        return torch.ones_like(confidences, dtype=torch.bool)
    if mode == "high_conf":
        return confidences >= threshold
    return confidences < threshold


def compute_kd_ctc_loss(
    student_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_lengths: torch.Tensor,
    hard_mask: torch.Tensor,
    blank_id: int = BLANK_ID,
) -> tuple[torch.Tensor, int]:
    """CTC loss against teacher greedy outputs, averaged over hard examples.

    Args:
        student_log_probs: (time, batch, vocab) — CTC log-probs in time-first form.
        input_lengths: (batch,) student input lengths (encoder-side).
        teacher_ids: (batch, target_len) teacher outputs re-encoded for the student.
        teacher_lengths: (batch,) non-padded teacher lengths.
        hard_mask: (batch,) True where the example is "hard" (KD applies).
        blank_id: CTC blank symbol id (matches the student tokenizer).

    Returns:
        (loss, num_hard): loss is 0 when no hard examples with non-empty teacher
        output remain. `num_hard` reflects the effective count used.
    """
    device = student_log_probs.device
    valid = hard_mask.to(device) & (teacher_lengths.to(device) > 0)
    num_hard = int(valid.sum().item())
    if num_hard == 0:
        return torch.zeros((), device=device, dtype=student_log_probs.dtype), 0

    time_len, batch_size, _ = student_log_probs.shape
    if teacher_ids.numel() == 0:
        return torch.zeros((), device=device, dtype=student_log_probs.dtype), 0

    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    selected_log_probs = student_log_probs.index_select(1, idx)
    selected_input_lengths = input_lengths.to(device).index_select(0, idx)
    selected_teacher_ids = teacher_ids.to(device).index_select(0, idx)
    selected_teacher_lengths = teacher_lengths.to(device).index_select(0, idx)

    # CTC requires input_length >= target_length (monotonic alignment). Skip any
    # surviving rows that violate this; zero-contribute instead of raising.
    keepable = selected_input_lengths >= selected_teacher_lengths
    if not keepable.all():
        keep_idx = keepable.nonzero(as_tuple=False).squeeze(-1)
        if keep_idx.numel() == 0:
            return torch.zeros((), device=device, dtype=student_log_probs.dtype), 0
        selected_log_probs = selected_log_probs.index_select(1, keep_idx)
        selected_input_lengths = selected_input_lengths.index_select(0, keep_idx)
        selected_teacher_ids = selected_teacher_ids.index_select(0, keep_idx)
        selected_teacher_lengths = selected_teacher_lengths.index_select(0, keep_idx)
        num_hard = int(keep_idx.numel())

    loss = F.ctc_loss(
        log_probs=selected_log_probs,
        targets=selected_teacher_ids,
        input_lengths=selected_input_lengths,
        target_lengths=selected_teacher_lengths,
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return loss, num_hard
