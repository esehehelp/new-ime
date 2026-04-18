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

        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

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
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            x, encoder_out, encoder_out, key_padding_mask=cross_attn_padding_mask
        )
        x = residual + x

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
    ) -> torch.Tensor:
        batch_size, seq_len, _ = encoder_out.shape
        x = self.input_projection(encoder_out)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)

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

        return self.final_norm(x)
