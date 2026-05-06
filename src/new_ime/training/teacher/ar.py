"""AR teacher (SimpleGPT2 + char-level vocab) for v2.5 KD via text-roundtrip.

Self-contained — does not depend on the legacy `models.src.training.train_ar`
module hierarchy. The model and tokenizer are reimplemented here from the
8cdf0df pre-v2 reference to keep the rebuild scope inside src/new_ime/.

Pre-v2 trained an AR baseline (`SimpleGPT2`, 32M, ~6997 vocab) and shipped
its checkpoint as `checkpoints/ar_v3_vast/best.pt` + a sidecar `_vocab.json`
(or a separately-specified `teacher_vocab` path). The teacher generates
surface text greedily from `(context, reading)`; the student is then trained
with an additional CTC loss against teacher text via `loss/kd.py`.

If the teacher checkpoint is not present, KD must be left disabled in the
config — there is no auto-fallback because silently dropping KD masks the
intent of the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# Special token ids (matching pre-v2 ARCollator).
_PAD = 0
_SEP = 1
_OUT = 2
_EOS = 3
_UNK = 4
_VOCAB_OFFSET = 5


class ARTokenizer:
    """Frozen character-level tokenizer loaded from a `_vocab.json` mapping.

    The teacher vocab is fixed at checkpoint time; unknown chars fall back to
    `_UNK` so the embedding never sees an out-of-range id.
    """

    PAD = _PAD
    SEP = _SEP
    OUT = _OUT
    EOS = _EOS
    UNK = _UNK

    def __init__(self):
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}

    @classmethod
    def from_path(cls, path: str | Path) -> "ARTokenizer":
        tok = cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AR teacher vocab not found: {path}")
        tok._char_to_id = json.loads(path.read_text(encoding="utf-8"))
        tok._id_to_char = {v: k for k, v in tok._char_to_id.items()}
        return tok

    @property
    def vocab_size(self) -> int:
        if not self._char_to_id:
            return _VOCAB_OFFSET
        return max(self._char_to_id.values()) + 1

    def encode(self, text: str) -> list[int]:
        return [self._char_to_id.get(c, self.UNK) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(
            self._id_to_char.get(int(i), "?")
            for i in ids
            if int(i) >= _VOCAB_OFFSET
        )


class SimpleGPT2(nn.Module):
    """Minimal GPT-2 style decoder-only transformer (8cdf0df reference)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_positions: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_positions, hidden_size)
        self.drop = nn.Dropout(dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.embed_tokens(input_ids) + self.embed_positions(positions)
        x = self.drop(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


class ARTeacher(nn.Module):
    """Frozen AR teacher running batched greedy generation."""

    arch_tag = "ar-simplegpt2"

    def __init__(
        self,
        model: SimpleGPT2,
        tokenizer: ARTokenizer,
        *,
        device: torch.device,
        max_seq_len: int,
        max_new_tokens: int,
        max_context_chars: int = 40,
        fp16: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self.max_new_tokens = int(max_new_tokens)
        self.max_context_chars = int(max_context_chars)
        self.fp16 = bool(fp16) and device.type == "cuda"
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        if self.fp16:
            self.model = self.model.half()
        self.model.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        *,
        teacher_vocab_path: str | Path,
        device: torch.device,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        max_seq_len: int = 192,
        max_new_tokens: int = 48,
        max_context_chars: int = 40,
        fp16: bool = True,
    ) -> "ARTeacher":
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"AR teacher checkpoint not found: {ckpt_path}")
        tokenizer = ARTokenizer.from_path(teacher_vocab_path)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_vocab = int(blob.get("vocab_size", tokenizer.vocab_size))
        if ckpt_vocab != tokenizer.vocab_size:
            raise ValueError(
                f"AR teacher vocab mismatch: ckpt={ckpt_vocab} "
                f"vocab_json={tokenizer.vocab_size}"
            )
        model = SimpleGPT2(
            vocab_size=ckpt_vocab,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_positions=max_seq_len,
        )
        model.load_state_dict(blob["model_state_dict"])
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            max_context_chars=max_context_chars,
            fp16=fp16,
        )

    def train(self, mode: bool = True) -> "ARTeacher":  # type: ignore[override]
        super().train(False)
        self.model.eval()
        return self

    def _encode_prompt(self, context: str, reading: str) -> list[int]:
        ctx = (context or "")[-self.max_context_chars :]
        ctx_ids = self.tokenizer.encode(ctx)
        read_ids = self.tokenizer.encode(reading or "")
        return ctx_ids + [self.tokenizer.SEP] + read_ids + [self.tokenizer.OUT]

    @torch.no_grad()
    def generate(
        self,
        contexts: Sequence[str],
        readings: Sequence[str],
        *,
        max_new_tokens: int | None = None,
    ) -> tuple[list[str], torch.Tensor]:
        """Batched greedy generation.

        Returns:
            texts: teacher greedy output per row
            confidences: (B,) mean top-1 softmax probability
        """
        if len(contexts) != len(readings):
            raise ValueError("contexts and readings length mismatch")
        B = len(contexts)
        if B == 0:
            return [], torch.zeros((0,), device=self.device)

        max_new_tokens = max_new_tokens or self.max_new_tokens
        prompts = [self._encode_prompt(c, r) for c, r in zip(contexts, readings)]
        prompt_lengths = [len(p) for p in prompts]
        max_prompt = max(prompt_lengths) if prompt_lengths else 0
        if max_prompt >= self.max_seq_len:
            return ["" for _ in range(B)], torch.zeros((B,), device=self.device)

        total_len = min(max_prompt + max_new_tokens, self.max_seq_len)
        input_ids = torch.full(
            (B, total_len), self.tokenizer.PAD, dtype=torch.long, device=self.device
        )
        attn = torch.zeros((B, total_len), dtype=torch.long, device=self.device)
        for i, p in enumerate(prompts):
            p = p[: self.max_seq_len]
            input_ids[i, : len(p)] = torch.tensor(
                p, dtype=torch.long, device=self.device
            )
            attn[i, : len(p)] = 1

        lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        generated: list[list[int]] = [[] for _ in range(B)]
        conf_sum = torch.zeros(B, device=self.device)
        conf_cnt = torch.zeros(B, device=self.device)

        amp_dtype = torch.float16 if self.fp16 else torch.float32

        for _ in range(max_new_tokens):
            cur_max = int(lengths.max().item())
            if cur_max >= total_len or bool(finished.all().item()):
                break
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.fp16,
                dtype=amp_dtype,
            ):
                logits = self.model(input_ids[:, :cur_max], attn[:, :cur_max])
            last_idx = (lengths - 1).clamp(max=cur_max - 1)
            arange = torch.arange(B, device=self.device)
            step_logits = logits[arange, last_idx].float()
            probs = F.softmax(step_logits, dim=-1)
            top_probs, top_ids = probs.max(dim=-1)

            for i in range(B):
                if bool(finished[i].item()):
                    continue
                tok = int(top_ids[i].item())
                conf_sum[i] += float(top_probs[i].item())
                conf_cnt[i] += 1.0
                if tok == self.tokenizer.EOS or tok == self.tokenizer.PAD:
                    finished[i] = True
                    continue
                pos = int(lengths[i].item())
                if pos >= total_len:
                    finished[i] = True
                    continue
                input_ids[i, pos] = tok
                attn[i, pos] = 1
                lengths[i] += 1
                generated[i].append(tok)

        texts = [self.tokenizer.decode(ids) for ids in generated]
        confidences = (conf_sum / conf_cnt.clamp_min(1.0)).to(torch.float32)
        return texts, confidences
