"""Transformer encoder + AR decoder teacher model.

CTC student の弱点共有問題を避けるための teacher。AR 出力で CTC の blank
collapse / 候補多様性不足を持たず、bidirectional encoder で homophone の
文脈活用も可能。

ランタイム用途: KD teacher。推論には zenz/AR と同様の AR 生成を行う。
student との共有: tokenizer (SharedCharTokenizer)、vocab (4801)。

設計:
- Encoder: bidirectional Transformer, L=10, h=768, heads=12, ffn=3072
- Decoder: causal AR Transformer with cross-attention, L=8, h=768, heads=12
- Embedding: encoder input / decoder input / lm_head を全共有 (tied)

Specials:
- CLS (3) → decoder BOS (生成開始)
- SEP (2) → decoder EOS (生成終了)
- PAD (0) → padding、loss ignore
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.src.data.tokenizer import PAD_ID, SEP_ID, CLS_ID


@dataclass
class TeacherPreset:
    name: str
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    ffn_size: int
    max_positions: int
    dropout: float = 0.1


TEACHER_PRESETS: dict[str, TeacherPreset] = {
    "teacher_150m": TeacherPreset(
        name="teacher_150m",
        hidden_size=768,
        encoder_layers=10,
        decoder_layers=8,
        num_heads=12,
        ffn_size=3072,
        max_positions=192,
    ),
    # Smoke-test preset for local sanity checks
    "teacher_smoke": TeacherPreset(
        name="teacher_smoke",
        hidden_size=256,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=4,
        ffn_size=1024,
        max_positions=128,
    ),
}


class TeacherSeq2Seq(nn.Module):
    """Transformer encoder + AR decoder teacher, shared char vocab.

    Training: teacher forcing with NLL on decoder output (shifted right).
    Inference: autoregressive greedy / beam from BOS=CLS until EOS=SEP.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        encoder_layers: int,
        decoder_layers: int,
        num_heads: int,
        ffn_size: int,
        max_positions: int,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        bos_id: int = CLS_ID,
        eos_id: int = SEP_ID,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_positions = max_positions
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.pos_embedding = nn.Embedding(max_positions, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_scale = hidden_size ** 0.5

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.encoder_ln = nn.LayerNorm(hidden_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_layers)
        self.decoder_ln = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].zero_()

    @classmethod
    def from_preset(cls, preset_name: str, vocab_size: int) -> "TeacherSeq2Seq":
        p = TEACHER_PRESETS[preset_name]
        return cls(
            vocab_size=vocab_size,
            hidden_size=p.hidden_size,
            encoder_layers=p.encoder_layers,
            decoder_layers=p.decoder_layers,
            num_heads=p.num_heads,
            ffn_size=p.ffn_size,
            max_positions=p.max_positions,
            dropout=p.dropout,
        )

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.size(1)
        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0)
        x = self.token_embedding(ids) * self.embed_scale + self.pos_embedding(positions)
        return self.embed_dropout(x)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (encoder_out, src_key_padding_mask)."""
        enc_x = self._embed(input_ids)
        # Transformer expects True where padded (ignored positions).
        src_key_padding_mask = attention_mask.eq(0)
        enc_out = self.encoder(enc_x, src_key_padding_mask=src_key_padding_mask)
        enc_out = self.encoder_ln(enc_out)
        return enc_out, src_key_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_out: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dec_x = self._embed(decoder_input_ids)
        dec_len = decoder_input_ids.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            dec_len, device=dec_x.device
        )
        tgt_key_padding_mask = (
            decoder_attention_mask.eq(0) if decoder_attention_mask is not None else None
        )
        dec_out = self.decoder(
            dec_x,
            encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        dec_out = self.decoder_ln(dec_out)
        return self.lm_head(dec_out)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward. Returns {logits, loss?}.

        Args:
            input_ids: (B, T_src) encoder input (with [CLS][ctx][SEP][reading])
            attention_mask: (B, T_src) 1 for real, 0 for pad
            decoder_input_ids: (B, T_tgt) [BOS] + surface[:-1]
            decoder_attention_mask: (B, T_tgt) 1 for real, 0 for pad
            labels: (B, T_tgt) surface + [EOS], -100 for pad positions
        """
        enc_out, src_mask = self.encode(input_ids, attention_mask)
        logits = self.decode(
            decoder_input_ids, enc_out, src_mask, decoder_attention_mask
        )
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy AR generation. Returns (tokens, mean_top1_confidence)."""
        self.eval()
        enc_out, src_mask = self.encode(input_ids, attention_mask)
        batch = input_ids.size(0)
        device = input_ids.device

        gen = torch.full((batch, 1), self.bos_id, dtype=torch.long, device=device)
        conf_sum = torch.zeros(batch, device=device)
        conf_count = torch.zeros(batch, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits = self.decode(gen, enc_out, src_mask)
            step_logits = logits[:, -1, :]
            probs = F.softmax(step_logits.float(), dim=-1)
            top_p, top_id = probs.max(dim=-1)
            # Mask finished samples: keep emitting eos (so tensor shape stays)
            next_id = torch.where(finished, torch.full_like(top_id, self.eos_id), top_id)
            gen = torch.cat([gen, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == self.eos_id)
            conf_sum = conf_sum + torch.where(finished, torch.zeros_like(top_p), top_p)
            conf_count = conf_count + (~finished).float()
            if finished.all():
                break

        mean_conf = conf_sum / conf_count.clamp_min(1.0)
        return gen, mean_conf
