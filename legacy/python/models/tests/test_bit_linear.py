import torch
import torch.nn as nn

from models.src.model.bit_linear import (
    BitLinear,
    quantize_activation_absmax,
    quantize_weight_ternary,
    replace_linears_with_bitlinear,
)


def test_activation_quantization_shape():
    x = torch.randn(2, 4, 8)
    x_q, scale = quantize_activation_absmax(x)
    assert x_q.shape == x.shape
    assert scale.shape == (2, 4, 1)


def test_weight_quantization_returns_scalar_scale():
    w = torch.randn(16, 8)
    w_q, scale = quantize_weight_ternary(w)
    assert w_q.shape == w.shape
    assert scale.dim() == 0


def test_bitlinear_forward_and_backward():
    layer = BitLinear(8, 4)
    x = torch.randn(3, 8, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 4)
    y.sum().backward()
    assert layer.weight.grad is not None


def test_replace_linears_with_bitlinear():
    module = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    replace_linears_with_bitlinear(module)
    assert isinstance(module[0], BitLinear)
    assert isinstance(module[2], BitLinear)
