"""DAG transition matrix `E` for the DAT decoder.

Each vertex v_i emits one token; each path through the DAG corresponds to
one possible output sequence. The transition log-probability E[i, j] is
computed by a multi-head attention block over (feature + position):

    log_multi_content[b, i, j, h] = (Q_h(x_i) . K_h(x_j)) / sqrt(d_head)
    log_gate[b, i, h]             = log_softmax(W_gate(x_i))
    valid_mask[b, i, j]           = (i < j) & (j < seq_len[b])

Per-head softmax over j:
    log_attn[b, i, j, h] = log_softmax(log_multi_content[b, i, j, h] | valid_mask)

Marginalize heads with the gate:
    E[b, i, j] = logsumexp_h(log_attn[b, i, j, h] + log_gate[b, i, h])

Reference: `references/DA-Transformer/fs_plugins/models/glat_decomposed_with_link.py:351-413`
(thu-coai, Apache-2.0). We follow the `max_transition_length=-1` (full
square) regime; the CUDA-friendly compressed `[B, prelen, translen]`
format is intentionally not implemented here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# Mask sentinel used when zeroing out invalid transitions before softmax.
# A finite, large-negative value keeps backward gradients finite even where
# the resulting probability is numerically zero.
_LINK_MASK_VALUE = -1e9


class DagLinkExtractor(nn.Module):
    """Computes the DAG transition log-prob matrix from decoder features.

    Args:
        hidden_size:   feature dim of decoder output
        num_heads:     transition attention heads (separately log-softmaxed,
                       then mixed through the gate)
        link_pos_embedding: shared learnable position embedding indexed by
                       `[0, prelen)`. Concatenated to the decoder feature
                       (matching `links_feature="feature:position"` in the
                       reference). If None, only feature is used.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        link_pos_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.link_pos_embedding = link_pos_embedding

        feature_dim = hidden_size + (
            link_pos_embedding.embedding_dim if link_pos_embedding is not None else 0
        )
        self.query_linear = nn.Linear(feature_dim, hidden_size)
        self.key_linear = nn.Linear(feature_dim, hidden_size)
        self.gate_linear = nn.Linear(feature_dim, num_heads)

    def forward(
        self,
        features: Tensor,        # (B, prelen, hidden_size)
        valid_mask: Tensor,      # (B, prelen) bool — True at valid vertices (1=non-pad)
    ) -> Tensor:
        """Returns log-transition matrix of shape (B, prelen, prelen)."""
        batch_size, prelen, _ = features.shape
        device = features.device

        if self.link_pos_embedding is not None:
            positions = torch.arange(prelen, device=device).unsqueeze(0).expand(batch_size, -1)
            features_withpos = torch.cat([features, self.link_pos_embedding(positions)], dim=-1)
        else:
            features_withpos = features

        # Multi-head Q / K / gate.
        Q = self.query_linear(features_withpos).view(batch_size, prelen, self.num_heads, self.head_dim)
        K = self.key_linear(features_withpos).view(batch_size, prelen, self.num_heads, self.head_dim)
        log_gates = torch.log_softmax(self.gate_linear(features_withpos), dim=-1)  # (B, prelen, H)

        # Per-head attention scores: (B, i, j, H).
        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / math.sqrt(self.head_dim)

        # transition_valid_mask[b, i, j] = (i < j) & (j is non-pad in batch b)
        triu = torch.triu(
            torch.ones(prelen, prelen, dtype=torch.bool, device=device),
            diagonal=1,
        )
        transition_valid_mask = triu.unsqueeze(0) & valid_mask.unsqueeze(1)  # (B, prelen, prelen)

        # Rows with no valid successors (e.g., padding rows / terminal vertex)
        # would softmax to NaN. Mark them as "all-invalid" and re-enable the
        # row so softmax produces a uniform distribution we'll later ignore.
        row_dead = transition_valid_mask.sum(dim=2, keepdim=True) == 0  # (B, prelen, 1)
        transition_valid_mask = transition_valid_mask | row_dead

        # Mask scores BEFORE per-head log_softmax so that invalid successors
        # contribute zero probability.
        mask_4d = transition_valid_mask.unsqueeze(-1)  # (B, i, j, 1)
        scores = scores.masked_fill(~mask_4d, _LINK_MASK_VALUE)
        log_per_head = torch.log_softmax(scores, dim=2)  # (B, i, j, H)
        # Force the dead-row probability mass to the sentinel, so downstream
        # consumers see a clearly-invalid row.
        log_per_head = log_per_head.masked_fill(row_dead.unsqueeze(-1), _LINK_MASK_VALUE)

        # Mix heads via the gate: log P(j | i) = logsumexp_h(log_attn_h + log_gate_h).
        log_links = torch.logsumexp(log_per_head + log_gates.unsqueeze(2), dim=-1)
        return log_links  # (B, prelen, prelen)


def link_mask_value() -> float:
    """Public accessor for tests / model code that need the same sentinel."""
    return _LINK_MASK_VALUE
