"""Zenz (HuggingFace GPT-2) teacher for KD via text-roundtrip.

zenz models (referenced under references/zenz-*) are AR GPT-2 variants
with strong context modelling — probe_v3 EM1 ≥ 0.695 (xsmall, 30M)
through 0.747 (medium, 310M). They solve the homophone disambiguation
that pure CTC-NAT struggles with, so distilling them into a student CTC
model is the natural next attack on Suiko's homophone gap.

Prompt format (matches eval/zenz_backend.py):
    INPUT(\\u{EE00}) + reading_kata + [CTX(\\u{EE02}) + context] + OUTPUT(\\u{EE01})

zenz expects KATAKANA reading; hiragana inputs are converted via jaconv.

Generate path: batched greedy generate, returns (texts, confidences)
where confidence is the per-sequence mean softmax probability over the
generated tokens (>0 means the model committed; closer to 1 means high
confidence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import jaconv
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


_INPUT_TAG = ""
_OUTPUT_TAG = ""
_CTX_TAG = ""
_TERMINATORS = ("</s>", _INPUT_TAG, _OUTPUT_TAG, _CTX_TAG, "[PAD]")


class ZenzTeacher(torch.nn.Module):
    """Frozen zenz GPT-2 used as KD teacher."""

    arch_tag = "zenz-gpt2"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: torch.device,
        max_new_tokens: int = 48,
        max_context_chars: int = 40,
        fp16: bool = True,
    ):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        dtype = (
            torch.float16
            if (fp16 and device.type == "cuda")
            else torch.float32
        )
        self.model = GPT2LMHeadModel.from_pretrained(
            str(model_path), torch_dtype=dtype
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.max_context_chars = int(max_context_chars)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        teacher_vocab_path: str | Path | None = None,
        device: torch.device,
        max_new_tokens: int = 48,
        max_context_chars: int = 40,
        fp16: bool = True,
        **_unused,
    ) -> "ZenzTeacher":
        # zenz checkpoint = HF directory; vocab/tokenizer come from the
        # same dir, teacher_vocab_path is unused and accepted only for
        # API parity with ARTeacher.
        path = Path(ckpt_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Zenz teacher dir not found: {ckpt_path} (expected HF model dir)"
            )
        return cls(
            path,
            device=device,
            max_new_tokens=max_new_tokens,
            max_context_chars=max_context_chars,
            fp16=fp16,
        )

    def train(self, mode: bool = True) -> "ZenzTeacher":  # type: ignore[override]
        super().train(False)
        self.model.eval()
        return self

    def _prompt(self, context: str, reading: str) -> str:
        kata = jaconv.hira2kata(reading or "")
        ctx = (context or "")[-self.max_context_chars :]
        if ctx:
            return _INPUT_TAG + kata + _CTX_TAG + ctx + _OUTPUT_TAG
        return _INPUT_TAG + kata + _OUTPUT_TAG

    @torch.no_grad()
    def generate(
        self,
        contexts: Sequence[str],
        readings: Sequence[str],
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[list[str], torch.Tensor]:
        """Batched greedy generate. Returns (texts, confidences[B] in [0,1])."""
        if len(contexts) != len(readings):
            raise ValueError("contexts and readings length mismatch")
        B = len(contexts)
        if B == 0:
            return [], torch.zeros((0,), device=self.device)

        prompts = [self._prompt(c, r) for c, r in zip(contexts, readings)]
        # Pad token: GPT-2 has no native PAD; reuse eos for left-padding.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Encode without padding first to know lengths.
        encs = [self.tokenizer.encode(p) for p in prompts]
        max_in = max(len(e) for e in encs)
        # Left-pad so all prompts end at the same column.
        input_ids = torch.full(
            (B, max_in), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (B, max_in), dtype=torch.long, device=self.device
        )
        for i, ids in enumerate(encs):
            input_ids[i, max_in - len(ids) :] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
            attention_mask[i, max_in - len(ids) :] = 1

        max_new = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = out.sequences  # (B, max_in + new_T)
        scores = out.scores  # tuple of (B, V) per generated step
        gen_len = len(scores)

        # Confidence per row: mean over valid generated steps of softmax
        # prob of the chosen token. Stop at first eos / terminator.
        confidences = torch.zeros(B, device=self.device)
        valid_counts = torch.zeros(B, device=self.device)
        for t, score_t in enumerate(scores):
            probs = F.softmax(score_t.float(), dim=-1)
            chosen = sequences[:, max_in + t]
            top_p = probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
            confidences = confidences + top_p
            valid_counts = valid_counts + 1.0
        confidences = confidences / valid_counts.clamp_min(1.0)

        texts: list[str] = []
        eos = self.tokenizer.eos_token_id
        for i in range(B):
            gen_ids = sequences[i, max_in : max_in + gen_len].tolist()
            # truncate at eos
            if eos in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos)]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            for marker in _TERMINATORS:
                if marker in text:
                    text = text.split(marker)[0]
            texts.append(text)
        return texts, confidences
