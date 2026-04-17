"""Encoder: Japanese BERT wrapper for CTC-NAT.

Wraps cl-tohoku/bert-base-japanese-char-v3 as the encoder.
Input: [context] [SEP] [kana_input] token IDs
Output: hidden states (batch, seq_len, hidden_dim)

Reference:
    fairseq NATransformerEncoder — fairseq/models/nat/nonautoregressive_transformer.py
    But we use HuggingFace BERT instead of fairseq's TransformerEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Conditional import: transformers may not be installed in all environments
try:
    from transformers import BertConfig, BertModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class BertEncoder(nn.Module):
    """BERT-based encoder initialized from cl-tohoku/bert-base-japanese-char-v3.

    Can also be initialized with a custom config for testing without
    downloading the pretrained model.
    """

    def __init__(
        self,
        pretrained_name: str = "cl-tohoku/bert-base-japanese-char-v3",
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        from_pretrained: bool = True,
    ):
        super().__init__()
        if from_pretrained and HAS_TRANSFORMERS:
            self.bert = BertModel.from_pretrained(pretrained_name)
            self.hidden_size = self.bert.config.hidden_size
        else:
            config = BertConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=hidden_size * 4,
            )
            self.bert = BertModel(config)
            self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input tokens.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) mask (1=valid, 0=pad)

        Returns:
            Hidden states: (batch, seq_len, hidden_size)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def freeze(self) -> None:
        """Freeze all BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers_from_top: int = -1) -> None:
        """Unfreeze BERT parameters.

        Args:
            num_layers_from_top: Number of layers to unfreeze from top.
                -1 means unfreeze all.
        """
        if num_layers_from_top < 0:
            for param in self.bert.parameters():
                param.requires_grad = True
            return

        # Freeze everything first
        self.freeze()

        # Unfreeze pooler and top N layers
        total_layers = self.bert.config.num_hidden_layers
        for i in range(total_layers - num_layers_from_top, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True


class MockEncoder(nn.Module):
    """Lightweight encoder for testing without BERT dependency."""

    def __init__(self, vocab_size: int = 200, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        # TransformerEncoder expects src_key_padding_mask where True = pad
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        return self.layers(x, src_key_padding_mask=pad_mask)

    def freeze(self) -> None:
        pass

    def unfreeze(self, num_layers_from_top: int = -1) -> None:
        pass
