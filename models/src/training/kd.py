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

from models.src.data.dataset import ARCollator
from models.src.data.tokenizer import BLANK_ID, CLS_ID, PAD_ID, SEP_ID, SharedCharTokenizer
from models.src.training.train_ar import SimpleGPT2


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


@dataclass
class CTCTeacherConfig:
    """Hyperparameters needed to reconstruct a CTC-NAT teacher.

    Unlike AR teacher:
      - No vocab_path; teacher tokenizer is the sidecar JSON of the ckpt.
      - No max_new_tokens; CTC teacher is non-autoregressive (one forward pass).
      - Same tokenizer as student is **required** (enforced at load time).
    """
    checkpoint_path: str
    fp16: bool = True


class CTCTeacher(nn.Module):
    """Frozen CTC-NAT teacher for direct-logit KD.

    - Same tokenizer as student (enforced).
    - Single forward pass (no autoregressive loop) → fast.
    - Returns per-position logits + per-sample mean confidence.
    - KD loss is soft-KL on output softmax, applied per time step.

    The student and teacher consume the same input_ids / attention_mask so
    the caller passes them in directly; no re-tokenization needed.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: SharedCharTokenizer,
        config: CTCTeacherConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
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
        config: CTCTeacherConfig,
        device: torch.device,
        student_tokenizer: SharedCharTokenizer,
    ) -> "CTCTeacher":
        from models.src.model.ctc_nat import CTCNAT, PRESETS
        ckpt_path = Path(config.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CTC teacher checkpoint not found: {ckpt_path}")

        tok_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
        if not tok_path.exists():
            raise FileNotFoundError(
                f"CTC teacher tokenizer sidecar not found: {tok_path}"
            )
        teacher_tokenizer = SharedCharTokenizer.load(str(tok_path))
        if teacher_tokenizer.vocab_size != student_tokenizer.vocab_size:
            raise ValueError(
                f"CTC teacher vocab size {teacher_tokenizer.vocab_size} does not "
                f"match student {student_tokenizer.vocab_size}. Direct-logit KD "
                f"requires identical tokenizers."
            )
        # Spot-check: the first, last, and a middle id should map back to the
        # same character in both tokenizers. Deeper verification is costly but
        # vocab-size match + id 0/N check catches most mismatches.
        for probe_id in (0, teacher_tokenizer.vocab_size // 2,
                         teacher_tokenizer.vocab_size - 1):
            s_char = student_tokenizer.id_to_token.get(probe_id, "")
            t_char = teacher_tokenizer.id_to_token.get(probe_id, "")
            if s_char != t_char:
                raise ValueError(
                    f"Tokenizer mismatch at id {probe_id}: student={s_char!r} "
                    f"teacher={t_char!r}. Teacher must share student's tokenizer."
                )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        preset_name = checkpoint.get("preset")
        if preset_name not in PRESETS:
            raise ValueError(f"unknown preset in teacher: {preset_name!r}")
        vocab_size = int(checkpoint.get("vocab_size") or teacher_tokenizer.vocab_size)
        model = CTCNAT.from_preset(
            preset_name,
            vocab_size=vocab_size,
            use_cvae=bool(checkpoint.get("use_cvae", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, tokenizer=teacher_tokenizer,
                   config=config, device=device)

    def train(self, mode: bool = True) -> "CTCTeacher":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One forward pass. Returns (logits, mean_confidence_per_sample).

        logits shape:       (B, T, V)
        confidence shape:   (B,) — mean top-1 softmax prob over non-blank
                                    positions of greedy path
        """
        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.dtype == torch.float16,
            dtype=self.dtype if self.dtype == torch.float16 else torch.float32,
        ):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"].float()   # (B, T, V)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.max(dim=-1)  # (B, T)
        blank_id = self.model.blank_id
        nonblank = (top_ids != blank_id) & (attention_mask.bool())
        # Avoid div-by-zero: if a sample has only blanks, fall back to
        # attention-mask mean (still valid, just includes blank positions).
        has_nonblank = nonblank.any(dim=-1)
        mean_conf = torch.where(
            has_nonblank,
            (top_probs * nonblank.float()).sum(dim=-1)
                / nonblank.float().sum(dim=-1).clamp_min(1.0),
            (top_probs * attention_mask.float()).sum(dim=-1)
                / attention_mask.float().sum(dim=-1).clamp_min(1.0),
        )
        return logits, mean_conf


@dataclass
class Seq2SeqTeacherConfig:
    """Hyperparameters for a TeacherSeq2Seq (encoder-decoder) teacher.

    Uses SharedCharTokenizer (same as student) → teacher output tokens can be
    fed directly as CTC targets after stripping BOS/EOS, skipping the AR
    teacher's text round-trip.
    """
    checkpoint_path: str
    max_context_chars: int = 40
    max_seq_len: int = 192
    max_new_tokens: int = 128
    fp16: bool = True


class Seq2SeqTeacher(nn.Module):
    """Frozen TeacherSeq2Seq teacher for online KD via text round-trip.

    Produces per-sample greedy surface text + mean top-1 confidence, matching
    the :class:`ARTeacher.generate` contract so :func:`encode_texts_for_student`
    + :func:`compute_kd_ctc_loss` plug in unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: SharedCharTokenizer,
        config: Seq2SeqTeacherConfig,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
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
        config: Seq2SeqTeacherConfig,
        device: torch.device,
        student_tokenizer: SharedCharTokenizer,
    ) -> "Seq2SeqTeacher":
        from models.src.model.teacher_seq2seq import TEACHER_PRESETS, TeacherSeq2Seq

        ckpt_path = Path(config.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Seq2Seq teacher checkpoint not found: {ckpt_path}")

        tok_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
        if not tok_path.exists():
            raise FileNotFoundError(
                f"Seq2Seq teacher tokenizer sidecar not found: {tok_path}"
            )
        teacher_tokenizer = SharedCharTokenizer.load(str(tok_path))
        if teacher_tokenizer.vocab_size != student_tokenizer.vocab_size:
            raise ValueError(
                f"Seq2Seq teacher vocab size {teacher_tokenizer.vocab_size} does not "
                f"match student {student_tokenizer.vocab_size}. Seq2Seq KD requires "
                f"identical tokenizers."
            )
        for probe_id in (0, teacher_tokenizer.vocab_size // 2,
                         teacher_tokenizer.vocab_size - 1):
            s_char = student_tokenizer.id_to_token.get(probe_id, "")
            t_char = teacher_tokenizer.id_to_token.get(probe_id, "")
            if s_char != t_char:
                raise ValueError(
                    f"Tokenizer mismatch at id {probe_id}: student={s_char!r} "
                    f"teacher={t_char!r}."
                )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        preset_name = checkpoint.get("preset")
        # train_teacher.py saves `preset` as the class name ("TeacherSeq2Seq")
        # rather than the TEACHER_PRESETS key. Fall back to dim-based matching
        # against the known presets using token_embedding hidden size.
        vocab_size = int(checkpoint.get("vocab_size") or teacher_tokenizer.vocab_size)
        if preset_name not in TEACHER_PRESETS:
            state = checkpoint["model_state_dict"]
            try:
                hidden = int(state["token_embedding.weight"].shape[1])
            except KeyError as exc:
                raise ValueError(
                    f"cannot infer preset from checkpoint (missing token_embedding.weight): "
                    f"{ckpt_path}"
                ) from exc
            matched = [k for k, p in TEACHER_PRESETS.items() if p.hidden_size == hidden]
            if not matched:
                raise ValueError(
                    f"no TEACHER_PRESETS match hidden_size={hidden} "
                    f"(checkpoint preset={preset_name!r})"
                )
            preset_name = matched[0]
        model = TeacherSeq2Seq.from_preset(preset_name, vocab_size=vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, tokenizer=teacher_tokenizer,
                   config=config, device=device)

    def train(self, mode: bool = True) -> "Seq2SeqTeacher":
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def generate(
        self,
        contexts: Sequence[str],
        readings: Sequence[str],
        max_new_tokens: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """Batched greedy generation; returns (texts, confidences)."""
        if len(contexts) != len(readings):
            raise ValueError("contexts and readings must have the same length")
        batch_size = len(contexts)
        if batch_size == 0:
            return [], []

        max_new = max_new_tokens or self.config.max_new_tokens
        cap = self.config.max_seq_len

        # Encode (context, reading) via SharedCharTokenizer, pad to max_enc.
        enc_ids: list[list[int]] = []
        for ctx, reading in zip(contexts, readings):
            ctx_trim = ctx[-self.config.max_context_chars :] if ctx else ""
            ids = self.tokenizer.encode_with_special(ctx_trim, reading)[:cap]
            enc_ids.append(ids)
        max_enc = max((len(x) for x in enc_ids), default=1)
        if max_enc == 0:
            return ["" for _ in range(batch_size)], [0.0 for _ in range(batch_size)]

        input_ids = torch.full(
            (batch_size, max_enc), PAD_ID, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (batch_size, max_enc), dtype=torch.long, device=self.device
        )
        for i, ids in enumerate(enc_ids):
            if not ids:
                continue
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[i, : len(ids)] = 1

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=self.dtype == torch.float16,
            dtype=self.dtype if self.dtype == torch.float16 else torch.float32,
        ):
            tokens, mean_conf = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
            )
        # tokens: (B, 1+gen_len) with BOS prefix; strip BOS and everything from
        # first EOS onward. Decode via SharedCharTokenizer.
        token_lists = tokens.tolist()
        bos_id = getattr(self.model, "bos_id", CLS_ID)
        eos_id = getattr(self.model, "eos_id", SEP_ID)
        texts: list[str] = []
        for row in token_lists:
            if row and row[0] == bos_id:
                row = row[1:]
            out_ids: list[int] = []
            for tid in row:
                if tid == eos_id or tid == PAD_ID:
                    break
                out_ids.append(tid)
            texts.append(self.tokenizer.decode(out_ids))
        confidences = [float(x) for x in mean_conf.tolist()]
        return texts, confidences


def compute_kd_kl_loss(
    student_logits: torch.Tensor,   # (B, T, V)
    teacher_logits: torch.Tensor,   # (B, T, V)
    attention_mask: torch.Tensor,   # (B, T)
    hard_mask: torch.Tensor,        # (B,)
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Soft-KL KD loss on CTC output distributions.

    KL(teacher || student) averaged over non-padded positions of hard
    examples. Temperature-scaled per standard KD.

    Returns (loss, num_hard). num_hard is the number of samples that
    contributed. Loss is 0 when no hard samples remain.
    """
    device = student_logits.device
    valid = hard_mask.to(device)
    num_hard = int(valid.sum().item())
    if num_hard == 0:
        return torch.zeros((), device=device, dtype=student_logits.dtype), 0

    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    s_log = F.log_softmax(student_logits.index_select(0, idx) / temperature, dim=-1)
    t_log = F.log_softmax(teacher_logits.index_select(0, idx) / temperature, dim=-1)
    t_prob = t_log.exp()

    # KL(t || s) = sum t * (log t - log s), positionwise
    kl = (t_prob * (t_log - s_log)).sum(dim=-1)  # (B_hard, T)
    mask = attention_mask.index_select(0, idx).float()  # (B_hard, T)
    loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)
    # Temperature scaling correction (standard KD).
    loss = loss * (temperature ** 2)
    return loss, num_hard


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
