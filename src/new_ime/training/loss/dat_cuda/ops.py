"""autograd.Function wrappers around the DAT CUDA kernel.

The reference (`thu-coai/DA-Transformer`,
`references/DA-Transformer/fs_plugins/custom_ops/dag_loss.py`) provides
`DagLossFunc`, `DagBestAlignmentFunc`, and `DagLogsoftmaxGatherFunc`,
which expose the CUDA kernel as autograd-aware torch ops. This module
ports those wrappers to operate on the *full-square* `links` layout
this project uses (`[B, prelen, prelen]`), since the CUDA kernel itself
expects the *compressed* layout (`[B, prelen, translen]`) where
`links_compressed[b, i, j]` represents the transition `i → i + j + 1`.

A pair of helpers (`_compress_links` / `_uncompress_links_grad`) bridge
the two layouts on the dispatcher boundary. The conversion is
O(B · prelen²) and dwarfed by the kernel cost
(O(B · prelen² · tarlen)).

The dispatcher in `dat_dp.py` calls the public functions exposed here
when CUDA is available (see `loader.py`).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

from new_ime.training.loss.dat_cuda.loader import load_dat_kernel

# Same sentinel the pure-PyTorch DP uses for masked-out transitions.
_LINK_NEG_INF = float("-inf")

# Reference's `DagLossFunc.config*` selectors. Exposed as constants in case
# we ever want to surface them as tunable knobs; for now they match the
# reference defaults verbatim.
_DAG_LOSS_CONFIG = 1
_DAG_LOSS_CONFIG1 = 2
_DAG_LOSS_CONFIG2 = 2
_DAG_BEST_ALIGNMENT_CONFIG = 1


# ---------------------------------------------------------------------------
# Layout conversion: full-square ↔ reference-compressed
# ---------------------------------------------------------------------------


def _make_compress_index(prelen: int, translen: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Build the gather index that maps compressed[i, j] → full[i, i+j+1].

    Returns `(safe_index, invalid_mask)` both of shape `[prelen, translen]`.
    `safe_index` is clamped to 0 wherever the source index would be out of
    bounds; the caller masks those entries to `-inf` after the gather.
    """
    i = torch.arange(prelen, device=device).unsqueeze(1)        # [prelen, 1]
    j = torch.arange(translen, device=device).unsqueeze(0)      # [1, translen]
    src = i + j + 1                                             # [prelen, translen]
    invalid = src >= prelen
    safe = src.masked_fill(invalid, 0)
    return safe, invalid


def _compress_links(links_full: Tensor) -> Tensor:
    """Pack full-square links `[B, prelen, prelen]` into the reference
    `[B, prelen, prelen-1]` compressed layout. Out-of-range entries are
    filled with `-inf` (== `_LINK_NEG_INF`)."""
    batch_size, prelen, prelen2 = links_full.shape
    assert prelen == prelen2, f"expected square links, got {tuple(links_full.shape)}"
    translen = prelen - 1
    if translen <= 0:
        return links_full.new_full((batch_size, prelen, 0), _LINK_NEG_INF)
    safe, invalid = _make_compress_index(prelen, translen, links_full.device)
    safe_b = safe.unsqueeze(0).expand(batch_size, -1, -1)        # [B, prelen, translen]
    compressed = links_full.gather(2, safe_b)
    compressed = compressed.masked_fill(invalid.unsqueeze(0), _LINK_NEG_INF)
    return compressed.contiguous()


def _uncompress_links_grad(grad_compressed: Tensor, prelen: int) -> Tensor:
    """Scatter compressed gradient `[B, prelen, prelen-1]` back to full-square
    `[B, prelen, prelen]`. Out-of-range entries (which received `-inf` and
    therefore zero gradient) become zero."""
    batch_size = grad_compressed.shape[0]
    translen = grad_compressed.shape[2]
    grad_full = grad_compressed.new_zeros((batch_size, prelen, prelen))
    if translen <= 0:
        return grad_full
    safe, invalid = _make_compress_index(prelen, translen, grad_compressed.device)
    safe_b = safe.unsqueeze(0).expand(batch_size, -1, -1)
    masked_grad = grad_compressed.masked_fill(invalid.unsqueeze(0), 0.0)
    grad_full.scatter_(2, safe_b, masked_grad)
    return grad_full


# ---------------------------------------------------------------------------
# autograd.Function wrappers (compressed-layout, kernel-facing)
# ---------------------------------------------------------------------------


class _DagLossFunc(Function):
    """Forward/backward of the marginalized DAG log-likelihood.

    Mirrors `DagLossFunc` in the reference implementation; the only
    difference is that the layout adaptation (`_compress_links`) lives
    *outside* of this class so the kernel sees the same layout the
    reference does.
    """

    @staticmethod
    def forward(
        ctx,
        match_all: Tensor,           # [B, tarlen, prelen]   log P(y_i | v_j)
        links_compressed: Tensor,    # [B, prelen, translen] log transition probs
        output_length: Tensor,       # [B]
        target_length: Tensor,       # [B]
    ) -> Tensor:
        kernel = load_dat_kernel()
        if kernel is None:
            raise RuntimeError("CUDA kernel unavailable; dispatcher should not reach here")

        require_grad = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        match_all = match_all.contiguous()
        links_compressed = links_compressed.contiguous()
        alpha, beta = kernel.dag_loss(
            match_all, links_compressed, output_length, target_length,
            require_grad, _DAG_LOSS_CONFIG,
        )
        if require_grad:
            res = beta[:, 0, 0].clone()
        else:
            res = alpha[range(alpha.shape[0]), target_length - 1, output_length - 1]
        ctx.save_for_backward(alpha, beta, match_all, links_compressed, output_length, target_length)
        return res

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        alpha, beta, match_all, links_compressed, output_length, target_length = ctx.saved_tensors
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None, None, None
        kernel = load_dat_kernel()
        grad_match_all, grad_links_compressed = kernel.dag_loss_backward(
            grad_output, alpha, beta, match_all, links_compressed,
            output_length, target_length,
            _DAG_LOSS_CONFIG1, _DAG_LOSS_CONFIG2,
        )
        return grad_match_all, grad_links_compressed, None, None


class _DagBestAlignmentFunc(Function):
    """Oracle vertex assignment under max-marginalization (GLAT helper).

    Returns `[B, prelen]` long tensor where `path[b, j] >= 0` is the target
    token index aligned to vertex j on the best path, and `-1` means the
    vertex is off-path. No backward (path is a discrete decision).
    """

    @staticmethod
    def forward(
        ctx,
        match_all: Tensor,
        links_compressed: Tensor,
        output_length: Tensor,
        target_length: Tensor,
    ) -> Tensor:
        kernel = load_dat_kernel()
        if kernel is None:
            raise RuntimeError("CUDA kernel unavailable; dispatcher should not reach here")
        match_all = match_all.contiguous()
        links_compressed = links_compressed.contiguous()
        _alpha, path = kernel.dag_best_alignment(
            match_all, links_compressed, output_length, target_length,
            _DAG_BEST_ALIGNMENT_CONFIG,
        )
        path = path.to(torch.long)
        ctx.mark_non_differentiable(path)
        return path

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None


# ---------------------------------------------------------------------------
# Public wrappers (full-square layout, dispatcher-facing)
# ---------------------------------------------------------------------------


def cuda_dag_loss(
    match_all: Tensor,        # [B, tarlen, prelen]
    links_full: Tensor,       # [B, prelen, prelen]
    output_length: Tensor,    # [B]
    target_length: Tensor,    # [B]
) -> Tensor:
    """CUDA-accelerated marginalized DAG log-likelihood.

    Parity with `_pytorch_dag_loss` in `dat_dp.py`: returns per-sample
    `[B]` log P(target | DAG). Caller negates and divides by
    target_length for the per-sample loss (see `dat.py:_dag_loss`).

    The kernel only dispatches over fp32 / fp16; bf16 inputs (common
    under autocast on Ampere+) are promoted to fp32 here. The
    pure-PyTorch DP also runs `match_all` through fp32 by convention
    (`torch.log_softmax(..., dtype=fp32)`), so this matches the existing
    numerical contract.
    """
    match_all_f = match_all.float() if match_all.dtype == torch.bfloat16 else match_all
    links_full_f = links_full.float() if links_full.dtype == torch.bfloat16 else links_full
    links_compressed = _CompressLinksFunc.apply(links_full_f)
    return _DagLossFunc.apply(match_all_f, links_compressed, output_length, target_length)


def cuda_dag_best_alignment(
    match_all: Tensor,
    links_full: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """CUDA-accelerated oracle vertex assignment for GLAT.

    bf16 → fp32 promotion (see `cuda_dag_loss` docstring).
    """
    match_all_f = match_all.float() if match_all.dtype == torch.bfloat16 else match_all
    links_full_f = links_full.float() if links_full.dtype == torch.bfloat16 else links_full
    # `_compress_links` has no autograd dependency since the kernel doesn't
    # backprop through best-alignment, so we can call it directly without
    # the autograd-aware wrapper.
    with torch.no_grad():
        links_compressed = _compress_links(links_full_f)
    return _DagBestAlignmentFunc.apply(match_all_f, links_compressed, output_length, target_length)


def cuda_dag_logsoftmax_gather(
    word_ins_out: Tensor,     # [B, prelen, V]
    select_idx: Tensor,       # [B, prelen, T_tgt]
) -> tuple[Tensor, Tensor]:
    """CUDA-accelerated fused log_softmax + gather.

    Parity with `_pytorch_dag_logsoftmax_gather` in `dat_dp.py`: returns
    `(word_ins_out, match)` where `match[b, j, i] = log P(target[b,i] | v_j)`.
    The first return is the original `word_ins_out` unchanged (so the
    caller's `outputs["logits"]` reference stays valid).

    Implementation note: the underlying CUDA kernel modifies its input
    tensor in-place to save memory. To avoid leaking that side effect to
    the caller (who may still need `word_ins_out` for `outputs["logits"]`),
    we clone before invoking the kernel. The pure-PyTorch path also
    allocates a fresh fp32 log_probs buffer, so the memory cost is
    comparable.

    bf16 inputs are promoted to fp32 since the kernel only dispatches
    over fp32 / fp16. The pure-PyTorch path also computes log_softmax
    in fp32 explicitly, so `match` semantics are unchanged.
    """
    kernel = load_dat_kernel()
    if kernel is None:
        raise RuntimeError("CUDA kernel unavailable; dispatcher should not reach here")
    scratch = (
        word_ins_out.float()
        if word_ins_out.dtype == torch.bfloat16
        else word_ins_out.clone()
    )
    _modified, match = _DagLogsoftmaxGatherFunc.apply(scratch, select_idx)
    return word_ins_out, match


class _DagLogsoftmaxGatherFunc(Function):
    """In-place fused log_softmax + gather. Operates on the cloned scratch
    tensor passed in by `cuda_dag_logsoftmax_gather`, so the side effect
    is contained."""

    @staticmethod
    def forward(ctx, word_ins_out: Tensor, select_idx: Tensor) -> tuple[Tensor, Tensor]:
        kernel = load_dat_kernel()
        require_grad = ctx.needs_input_grad[0]
        selected_result = kernel.logsoftmax_gather(word_ins_out, select_idx, require_grad)
        ctx.mark_dirty(word_ins_out)
        ctx.set_materialize_grads(False)
        if require_grad:
            ctx.save_for_backward(word_ins_out, select_idx)
            ctx.has_backward = False
        return word_ins_out, selected_result

    @staticmethod
    def backward(ctx, grad_word_ins_out, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None
        assert grad_word_ins_out is None, (
            "Cannot reuse word_ins_out after logsoftmax_gather (kernel mutates it)"
        )
        if grad_output is None:
            return None, None
        assert not ctx.has_backward, "logsoftmax_gather backward called twice"
        ctx.has_backward = True
        grad_input, selected_idx = ctx.saved_tensors
        grad_input.mul_(grad_output.sum(-1, keepdim=True).neg_().to(grad_input.dtype))
        grad_input.scatter_add_(-1, selected_idx, grad_output.to(grad_input.dtype))
        return grad_input, None


# ---------------------------------------------------------------------------
# autograd-aware compress (gradient flows from compressed grad → full)
# ---------------------------------------------------------------------------


class _CompressLinksFunc(Function):
    """`_compress_links` plus its `_uncompress_links_grad` adjoint.

    We could rely on autograd's automatic differentiation through `gather`
    and `masked_fill`, but the explicit Function (a) avoids saving the
    large index tensors for backward, and (b) makes the layout transition
    explicit at a single place.
    """

    @staticmethod
    def forward(ctx, links_full: Tensor) -> Tensor:
        ctx.prelen = links_full.shape[1]
        return _compress_links(links_full)

    @staticmethod
    def backward(ctx, grad_compressed: Tensor) -> Tensor:
        return _uncompress_links_grad(grad_compressed.contiguous(), ctx.prelen)
