"""1.58-bit research utilities.

This module implements the training-side fake quantization primitive only.
Runtime packing / bitnet.cpp integration remains a later step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_activation_absmax(x: torch.Tensor, clamp_min: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token int8 absmax quantization used for fake-quant activations."""

    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=clamp_min) / 127.0
    quantized = (x / scale).round().clamp(-128, 127)
    dequantized = quantized * scale
    return dequantized, scale


def quantize_weight_ternary(
    weight: torch.Tensor,
    clamp_min: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Median-scaled ternary quantization {-1, 0, +1}."""

    scale = weight.abs().median().clamp(min=clamp_min)
    ternary = (weight / scale).round().clamp(-1, 1)
    dequantized = ternary * scale
    return dequantized, scale


class BitLinear(nn.Linear):
    """Training-time fake-quant linear layer for the 1.58-bit research branch."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.last_activation_scale: torch.Tensor | None = None
        self.last_weight_scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = quantize_activation_absmax(x)
        w_q, w_scale = quantize_weight_ternary(self.weight)

        self.last_activation_scale = x_scale.detach()
        self.last_weight_scale = w_scale.detach()

        x_ste = x + (x_q - x).detach()
        w_ste = self.weight + (w_q - self.weight).detach()
        return F.linear(x_ste, w_ste, self.bias)


def replace_linears_with_bitlinear(module: nn.Module) -> nn.Module:
    """Recursively replace nn.Linear with BitLinear in-place."""

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BitLinear):
            replacement = BitLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
            )
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
        else:
            replace_linears_with_bitlinear(child)
    return module
