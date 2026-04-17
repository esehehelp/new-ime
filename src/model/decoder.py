"""NAT Decoder: Non-autoregressive decoder with bidirectional self-attention.

Unlike autoregressive decoders, there is NO causal mask — all positions
can attend to all other positions. This is the key structural difference.

The decoder takes encoder hidden states via cross-attention and produces
output logits for all positions in parallel.

Reference:
    fairseq NATransformerDecoder — fairseq/models/nat/nonautoregressive_transformer.py:207-402
    Key difference: fairseq uses causal=False via enable_ensemble; we set it explicitly.

    DA-Transformer GlatLinkDecoder — fs_plugins/models/glat_decomposed_with_link.py:929-1049
    Uses segment embeddings and link prediction; we skip links and use CTC instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NATDecoderLayer(nn.Module):
    """Single NAT decoder layer: bidirectional self-attention + cross-attention + FFN."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, dropout: float = 0.1):
        super().__init__()
        # Bidirectional self-attention (no causal mask)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # Cross-attention to encoder output
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)

        # Feed-forward network
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
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len, hidden) decoder input
            encoder_out: (batch, src_len, hidden) encoder output
            self_attn_padding_mask: (batch, tgt_len) True=pad
            cross_attn_padding_mask: (batch, src_len) True=pad
        """
        # Self-attention (bidirectional — no causal mask)
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=self_attn_padding_mask)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(
            x, encoder_out, encoder_out, key_padding_mask=cross_attn_padding_mask
        )
        x = residual + x

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)

        return x


class NATDecoder(nn.Module):
    """Non-autoregressive Transformer decoder.

    For CTC-NAT, the decoder input length = encoder output length.
    CTC's blank tokens handle the length mismatch between input and output.

    The decoder input is initialized from encoder output (no separate
    embedding — the encoder hidden states serve as query tokens).

    Reference:
        fairseq NATransformerDecoder.extract_features (line 247-329)
        — but simplified: no length prediction, no copying source.
        We let CTC handle alignment.
    """

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

        # Positional embedding for decoder positions
        self.pos_embed = nn.Embedding(max_positions, hidden_size)

        # Projection from encoder hidden size to decoder hidden size
        # (identity if same size)
        self.input_projection = nn.Identity()

        # Decoder layers
        self.layers = nn.ModuleList([
            NATDecoderLayer(hidden_size, num_heads, ffn_size, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode in parallel.

        Args:
            encoder_out: (batch, src_len, hidden) encoder hidden states.
                         Also used as decoder input (query tokens).
            encoder_padding_mask: (batch, src_len) True=pad for encoder.

        Returns:
            Decoder features: (batch, src_len, hidden)
        """
        batch_size, seq_len, _ = encoder_out.shape

        # Initialize decoder input from encoder output
        x = self.input_projection(encoder_out)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)

        # Decoder padding mask = encoder padding mask
        # (same length, since CTC handles alignment)
        decoder_padding_mask = encoder_padding_mask

        # Run through decoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                self_attn_padding_mask=decoder_padding_mask,
                cross_attn_padding_mask=encoder_padding_mask,
            )

        x = self.final_norm(x)
        return x
