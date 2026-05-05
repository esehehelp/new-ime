"""NAT Decoder: bidirectional self-attention with optional FiLM conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class NATDecoderLayer(nn.Module):
    """Single NAT decoder layer with optional FiLM modulation."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        # Output dropout on attended representation before residual add. The
        # MHA's internal `dropout` only drops attention weights; the standard
        # transformer recipe (Vaswani 2017) also drops the attention output.
        # Without this, the only stochasticity on the attn path is on the
        # weight pattern itself — too weak as regularization.
        self.self_attn_out_dropout = nn.Dropout(dropout)

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        self.cross_attn_out_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_padding_mask: torch.Tensor | None = None,
        cross_attn_padding_mask: torch.Tensor | None = None,
        film_condition: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=self_attn_padding_mask)
        x = residual + self.self_attn_out_dropout(x)

        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            x, encoder_out, encoder_out, key_padding_mask=cross_attn_padding_mask
        )
        x = residual + self.cross_attn_out_dropout(x)

        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        if film_condition is not None:
            gamma, beta = film_condition
            x = gamma * x + beta

        return x


class NATDecoder(nn.Module):
    """Non-autoregressive Transformer decoder."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_size: int = 3072,
        dropout: float = 0.1,
        max_positions: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pos_embed = nn.Embedding(max_positions, hidden_size)
        self.input_projection = nn.Identity()
        self.layers = nn.ModuleList(
            [NATDecoderLayer(hidden_size, num_heads, ffn_size, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
        film_conditioning: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        capture_layers: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """
        capture_layers: when given, return `(final, [(idx, hidden_at_idx), ...])`
        where each captured hidden has been passed through `final_norm` (so the
        caller can apply the same head as the final output without an extra
        LayerNorm). When None, the original `final_only` return is preserved
        for backward compatibility with existing callers.
        """
        batch_size, seq_len, _ = encoder_out.shape
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_positions {self.pos_embed.num_embeddings}"
            )
        x = self.input_projection(encoder_out)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)

        capture_set = set(capture_layers) if capture_layers else set()
        captures: list[tuple[int, torch.Tensor]] = []

        decoder_padding_mask = encoder_padding_mask
        for layer_idx, layer in enumerate(self.layers):
            film_condition = None
            if film_conditioning is not None:
                film_condition = film_conditioning[layer_idx]
            x = layer(
                x,
                encoder_out,
                self_attn_padding_mask=decoder_padding_mask,
                cross_attn_padding_mask=encoder_padding_mask,
                film_condition=film_condition,
            )
            if layer_idx in capture_set:
                captures.append((layer_idx, self.final_norm(x)))

        final = self.final_norm(x)
        if capture_layers is None:
            return final
        return final, captures


class MaskCTCRefinementDecoder(nn.Module):
    """Masked-token decoder used by the dedicated Mask-CTC refinement branch."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_size: int = 3072,
        dropout: float = 0.1,
        max_positions: int = 512,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.token_embed = embedding or nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_positions, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [NATDecoderLayer(hidden_size, num_heads, ffn_size, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hypothesis_ids: torch.Tensor,
        hypothesis_padding_mask: torch.Tensor | None,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
        film_conditioning: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hypothesis_ids.shape
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_positions {self.pos_embed.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=hypothesis_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(hypothesis_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        for layer_idx, layer in enumerate(self.layers):
            film_condition = None
            if film_conditioning is not None:
                film_condition = film_conditioning[layer_idx]
            x = layer(
                x,
                encoder_out,
                self_attn_padding_mask=hypothesis_padding_mask,
                cross_attn_padding_mask=encoder_padding_mask,
                film_condition=film_condition,
            )
        return self.final_norm(x)
