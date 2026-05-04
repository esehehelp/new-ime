"""VRAM / RAM accounting utilities.

`estimate_training_memory` returns a coarse breakdown (params, optimizer
state, activations) in GB based on parameter count and a typical
fwd+bwd activation budget for transformer training. The numbers are not
exact — they are a sanity check ("does this preset fit at this batch
size?") not a budget calculator.

`measure_peak_vram` runs a single dummy forward+backward and reports
the actual peak CUDA VRAM seen (skipped on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_BYTES_PER_GB = 1024 ** 3


@dataclass
class MemoryEstimate:
    params_gb: float
    optimizer_gb: float
    activations_gb: float
    total_gb: float


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_training_memory(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    bytes_per_param: int = 4,
    optimizer_factor: int = 2,
    activation_factor: float = 16.0,
) -> MemoryEstimate:
    """Coarse breakdown for AdamW + fp32 transformer training.

    bytes_per_param: 4 for fp32, 2 for fp16 (mixed-precision halves
        params + optimizer states).
    optimizer_factor: AdamW keeps 2 moments per param (m, v). 2 means
        each is the same dtype as params.
    activation_factor: heuristic GiB per (M params * batch * seq /
        1e6) — calibrated against pre-v2 measurements for transformer
        training at fp32.
    """
    n_params = _param_count(model)
    params_gb = n_params * bytes_per_param / _BYTES_PER_GB
    optimizer_gb = n_params * bytes_per_param * optimizer_factor / _BYTES_PER_GB
    activations_gb = (
        activation_factor * (n_params / 1e9) * batch_size * seq_len / 128
    )
    total_gb = params_gb + optimizer_gb + activations_gb
    return MemoryEstimate(
        params_gb=params_gb,
        optimizer_gb=optimizer_gb,
        activations_gb=activations_gb,
        total_gb=total_gb,
    )


def measure_peak_vram(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    """Run one fwd+bwd with synthetic input; return peak CUDA VRAM in GiB.

    Returns 0.0 on CPU (not supported there).
    """
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_ids=target_ids,
        target_lengths=target_lengths,
    )
    outputs["loss"].backward()
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / _BYTES_PER_GB


def format_memory_table(
    estimate: MemoryEstimate,
    peak_vram_gb: float | None = None,
) -> str:
    rows = [
        ("params", f"{estimate.params_gb:.2f} GiB"),
        ("optimizer state", f"{estimate.optimizer_gb:.2f} GiB"),
        ("activations (est.)", f"{estimate.activations_gb:.2f} GiB"),
        ("total (est.)", f"{estimate.total_gb:.2f} GiB"),
    ]
    if peak_vram_gb is not None and peak_vram_gb > 0:
        rows.append(("measured peak VRAM", f"{peak_vram_gb:.2f} GiB"))
    return "\n".join(f"  {name:<22} {value}" for name, value in rows)
