"""Dynamic programming primitives for the DAT (DA-Transformer) loss.

The pure-PyTorch implementations live in `_pytorch_*` private functions
and operate on full-square `links` (`[B, prelen, prelen]`). The public
`torch_dag_loss` / `torch_dag_best_alignment` / `torch_dag_logsoftmax_gather`
are *dispatchers* that route to the CUDA kernel
(`new_ime.training.loss.dat_cuda`) when it builds successfully and the
inputs are on a CUDA device, falling back to the PyTorch path otherwise.

Set `NEW_IME_DAT_FORCE_PYTORCH=1` to disable the CUDA path explicitly
(useful for parity testing and on environments where the kernel build
breaks). See `dat_cuda/loader.py` for build-time semantics.

The forward DP follows
    f[i, j] = log P(emit y[0..i], path ends at vertex j)
    f[0, 0] = match_all[:, 0, 0],   f[0, j>0] = -inf
    f[i, j] = logsumexp_k(f[i-1, k] + links[k, j]) + match_all[:, i, j]

Loss = -f[T_tgt-1, output_length-1] / T_tgt   (caller averages over batch).

Match scores are log-probabilities (negative numbers in well-defined
softmax space), NOT negative log-probabilities, despite what the
reference's docstring claims — this matches the additive accumulation
in the recurrence.

Reference: `references/DA-Transformer/fs_plugins/custom_ops/dag_loss.py`
(thu-coai, Apache-2.0). The PyTorch version there (`torch_dag_loss`,
`torch_dag_best_alignment`, `torch_dag_logsoftmax_gather_inplace`,
`logsumexp_keepdim`, `loop_function_noempty`) is the direct provenance.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _should_use_cuda(*tensors: Tensor) -> bool:
    """True iff every tensor is on CUDA AND the CUDA kernel built OK."""
    if not all(t.is_cuda for t in tensors):
        return False
    from new_ime.training.loss.dat_cuda.loader import load_dat_kernel
    return load_dat_kernel() is not None


def _logsumexp_keepdim(x: Tensor, dim: int) -> Tensor:
    """Numerically stable logsumexp that survives all-`-inf` slices.

    `torch.logsumexp` returns NaN when every input is `-inf`, which
    happens for unreachable DAG vertices early in the DP. We mask the
    max, run a normal logsumexp, then put the `-inf` back so gradients
    propagate as zero through dead branches.
    """
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float("inf")
    m = m.detach()
    s = (x - m.masked_fill(mask, 0)).exp().sum(dim=dim, keepdim=True)
    return s.masked_fill(mask, 1).log() + m.masked_fill(mask, -float("inf"))


def _step_marginal(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    """One DP step under sum-marginalization (logsumexp over predecessors).

    Args:
        last_f: (B, prelen, 1) — log P up to step k-1 ending at each vertex
        links:  (B, prelen, prelen) — log transition prob (predecessor -> successor)
        match:  (B, prelen, 1) — log P(y_k | v_j) for each vertex j

    Returns:
        (B, prelen, 1) — log P up to step k
    """
    f_next = _logsumexp_keepdim(last_f + links, 1)  # (B, 1, prelen)
    return f_next.transpose(1, 2) + match  # (B, prelen, 1)


def _step_max(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    """One DP step under max-marginalization (Viterbi over predecessors).

    Same shape contract as `_step_marginal`. Used by `torch_dag_best_alignment`
    to obtain the oracle vertex assignment for GLAT (Stage 4).
    """
    f_next = (last_f + links).max(dim=1)[0]  # (B, prelen)
    return f_next.unsqueeze(-1) + match  # (B, prelen, 1)


def _pytorch_dag_logsoftmax_gather(
    word_ins_out: Tensor,
    select_idx: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fused log_softmax + gather to build the per-target match scores.

    Args:
        word_ins_out: (B, prelen, V) — vertex token logits
        select_idx:   (B, prelen, T_tgt) — target_ids broadcast across vertices,
                      typically built as `target.unsqueeze(1).expand(-1, prelen, -1)`
                      so select_idx[b, j, i] = target[b, i] for every vertex j

    Returns:
        (logits, match) where:
            logits = the input (returned for API parity with the reference)
            match  = (B, prelen, T_tgt) — log P(y_i | v_j) in fp32

    Note: `torch_dag_loss` takes match in (B, T_tgt, prelen) layout, so the
    caller must `match.transpose(1, 2)` before feeding it to the DP. This
    layout split matches the reference (`thu-coai/DA-Transformer`).
    """
    log_probs = torch.log_softmax(word_ins_out, dim=-1, dtype=torch.float32)
    match = log_probs.gather(dim=-1, index=select_idx)
    return word_ins_out, match


def torch_dag_logsoftmax_gather(
    word_ins_out: Tensor,
    select_idx: Tensor,
) -> tuple[Tensor, Tensor]:
    """Dispatcher: CUDA kernel path when available, else pure-PyTorch."""
    if _should_use_cuda(word_ins_out, select_idx):
        from new_ime.training.loss.dat_cuda.ops import cuda_dag_logsoftmax_gather
        return cuda_dag_logsoftmax_gather(word_ins_out, select_idx)
    return _pytorch_dag_logsoftmax_gather(word_ins_out, select_idx)


def _pytorch_dag_loss(
    match_all: Tensor,
    links: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """Marginalized log-likelihood of the target under the DAG.

    Args:
        match_all:     (B, T_tgt, prelen) — log P(y_i | v_j) for each (i, j)
        links:         (B, prelen, prelen) — log transition prob (j_prev -> j_next),
                       must be left-to-right masked (lower triangle = -inf)
        output_length: (B,) — graph size per sample (vertices [output_length:] ignored)
        target_length: (B,) — reference length per sample

    Returns:
        (B,) — log P(target | DAG) per sample. Caller negates and divides by
        target_length for the per-sample loss; see `nat_dag_loss.py:_compute_dag_loss`.
    """
    match_all_t = match_all.transpose(1, 2)  # (B, prelen, T_tgt)
    batch_size, prelen, tarlen = match_all_t.shape
    assert links.shape[1] == links.shape[2], (
        f"links must be square (B, prelen, prelen); got {tuple(links.shape)}"
    )

    f_init = torch.full(
        (batch_size, prelen, 1),
        float("-inf"),
        dtype=match_all_t.dtype,
        device=match_all_t.device,
    )
    f_init[:, 0, 0] = match_all_t[:, 0, 0]
    f_arr = [f_init]

    match_chunks = torch.chunk(match_all_t, tarlen, dim=-1)  # tarlen × (B, prelen, 1)
    for k in range(1, tarlen):
        f_arr.append(_step_marginal(f_arr[-1], links, match_chunks[k]))

    # f_full[b, j, k] = log P(emit y[0..k], path ends at vertex j)
    f_full = torch.cat(f_arr, dim=-1)  # (B, prelen, T_tgt)
    return f_full[
        torch.arange(batch_size, device=match_all.device),
        output_length - 1,
        target_length - 1,
    ]


def _torch_dag_max_loss(
    match_all: Tensor,
    links: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """Same as `torch_dag_loss` but with max-marginalization (no logsumexp).

    Used as the inner objective for `torch_dag_best_alignment`. Returns the
    log-prob of the single best path through the DAG that emits the target.
    """
    match_all_t = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all_t.shape

    f_init = torch.full(
        (batch_size, prelen, 1),
        float("-inf"),
        dtype=match_all_t.dtype,
        device=match_all_t.device,
    )
    f_init[:, 0, 0] = match_all_t[:, 0, 0]
    f_arr = [f_init]
    match_chunks = torch.chunk(match_all_t, tarlen, dim=-1)
    for k in range(1, tarlen):
        f_arr.append(_step_max(f_arr[-1], links, match_chunks[k]))

    f_full = torch.cat(f_arr, dim=-1)
    return f_full[
        torch.arange(batch_size, device=match_all.device),
        output_length - 1,
        target_length - 1,
    ]


def _pytorch_dag_best_alignment(
    match_all: Tensor,
    links: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """Oracle vertex assignment for each target token (used by GLAT).

    Runs the max-version of the DAG DP and recovers the assignment via
    a backward-pass autograd trick: `d(best_logprob)/d(match[b,i,j])` is 1
    iff vertex j is on the best path emitting target token i. We then take
    argmax over target tokens for each vertex.

    Returns:
        (B, prelen) long tensor:
            output[b, j] = i  if vertex j emits target token i on the best path
            output[b, j] = -1 if vertex j is not on the best path
    """
    with torch.enable_grad():
        match_all_g = match_all.detach().clone().requires_grad_(True)
        all_logprob = _torch_dag_max_loss(match_all_g, links, output_length, target_length)
        match_grad = torch.autograd.grad(all_logprob.sum(), [match_all_g])[0]
    # match_grad: (B, T_tgt, prelen) — 1 at oracle (i, j) entries, 0 elsewhere
    path_value, path = match_grad.max(dim=1)  # (B, prelen) each
    path = path.masked_fill(path_value < 0.5, -1)
    return path


def torch_dag_loss(
    match_all: Tensor,
    links: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """Dispatcher: CUDA kernel path when available, else pure-PyTorch."""
    if _should_use_cuda(match_all, links):
        from new_ime.training.loss.dat_cuda.ops import cuda_dag_loss
        return cuda_dag_loss(match_all, links, output_length, target_length)
    return _pytorch_dag_loss(match_all, links, output_length, target_length)


def torch_dag_best_alignment(
    match_all: Tensor,
    links: Tensor,
    output_length: Tensor,
    target_length: Tensor,
) -> Tensor:
    """Dispatcher: CUDA kernel path when available, else pure-PyTorch."""
    if _should_use_cuda(match_all, links):
        from new_ime.training.loss.dat_cuda.ops import cuda_dag_best_alignment
        return cuda_dag_best_alignment(match_all, links, output_length, target_length)
    return _pytorch_dag_best_alignment(match_all, links, output_length, target_length)
