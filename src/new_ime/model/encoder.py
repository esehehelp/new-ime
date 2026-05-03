"""Encoders for CTC-NAT experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from transformers import BertConfig, BertModel

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class BertEncoder(nn.Module):
    """BERT-based encoder initialized from cl-tohoku/bert-base-japanese-char-v3."""

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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def get_input_embedding(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings

    def freeze(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers_from_top: int = -1) -> None:
        if num_layers_from_top < 0:
            for param in self.bert.parameters():
                param.requires_grad = True
            return

        self.freeze()
        total_layers = self.bert.config.num_hidden_layers
        for i in range(total_layers - num_layers_from_top, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True


class SmallEncoder(nn.Module):
    """Scratch transformer encoder for the ~30M / ~100M CTC-NAT variants."""

    def __init__(
        self,
        vocab_size: int = 6500,
        hidden_size: int = 640,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_size: int = 2560,
        max_positions: int = 128,
        dropout: float = 0.1,
        embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embedding = embedding or nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_positions, hidden_size)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.layers(x, src_key_padding_mask=pad_mask)
        return self.final_norm(x)

    def get_input_embedding(self) -> nn.Embedding:
        return self.token_embedding

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers_from_top: int = -1) -> None:
        if num_layers_from_top < 0:
            for param in self.parameters():
                param.requires_grad = True
            return

        self.freeze()
        for param in self.token_embedding.parameters():
            param.requires_grad = True
        for param in self.pos_embedding.parameters():
            param.requires_grad = True
        if num_layers_from_top <= 0:
            return
        layers = list(self.layers.layers)
        for layer in layers[-num_layers_from_top:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.final_norm.parameters():
            param.requires_grad = True


class MockEncoder(nn.Module):
    """Lightweight encoder for tests without heavy dependencies."""

    def __init__(
        self,
        vocab_size: int = 200,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = embedding or nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        pad_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.layers(x, src_key_padding_mask=pad_mask)
        return self.final_norm(x)

    def get_input_embedding(self) -> nn.Embedding:
        return self.embed

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self, num_layers_from_top: int = -1) -> None:
        for param in self.parameters():
            param.requires_grad = True
