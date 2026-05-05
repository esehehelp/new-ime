"""Numerical verification of the PyTorch DAT DP primitives.

Three layers of confidence:
    1. Analytical hand-computed DAGs match closed-form expectations.
    2. Random DAGs match the upstream reference (`thu-coai/DA-Transformer`,
       imported via importlib so we don't need to install fairseq).
    3. Backward passes produce finite gradients at all *valid* link
       positions (the masked positions are saturated -1e9 sentinels, mirroring
       what `extract_links` produces in production: softmax-over-masked-attn).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from new_ime.training.loss.dat_dp import (
    _pytorch_dag_best_alignment,
    _pytorch_dag_logsoftmax_gather,
    _pytorch_dag_loss,
    torch_dag_best_alignment,
    torch_dag_logsoftmax_gather,
    torch_dag_loss,
)


REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "references"
    / "DA-Transformer"
    / "fs_plugins"
    / "custom_ops"
    / "dag_loss.py"
)

# Production `extract_links` masks invalid (i>=j) transitions by adding a
# large-negative bias *before* softmax, so the resulting log-prob entries are
# finite (≈ -1e9 / -1e4 depending on the bias). Strict -inf in links makes the
# DP backward NaN out — that's true for the reference impl as well — and never
# arises in real training. We use a finite mask sentinel throughout the tests.
LINK_MASK_VALUE = -1e9


def _load_reference():
    spec = importlib.util.spec_from_file_location("ref_dag_loss", REFERENCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_left_to_right_links(
    batch: int,
    prelen: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Random log-softmax-normalized link matrix with left-to-right DAG masking.

    Invalid (i>=j) positions get a large-negative finite sentinel so that
    backward passes through the DP stay finite (cf. LINK_MASK_VALUE comment).
    The terminal vertex's row is also finite (we never feed it to softmax over
    an all-masked row, which would be NaN).
    """
    if seed is not None:
        torch.manual_seed(seed)
    raw = torch.randn(batch, prelen, prelen, dtype=dtype)
    valid = torch.triu(torch.ones(prelen, prelen, dtype=dtype), diagonal=1).bool()
    raw = torch.where(
        valid.unsqueeze(0),
        raw,
        torch.full_like(raw, LINK_MASK_VALUE),
    )
    # log_softmax over the last dim; valid rows have at least one finite entry,
    # the all-masked terminal row gives a uniform distribution over the
    # sentinels (its values never enter the DP because the terminal vertex has
    # no outgoing transitions anyone consumes).
    return torch.log_softmax(raw, dim=-1)


# ---------------------------------------------------------------------------
# Analytical sanity
# ---------------------------------------------------------------------------


def test_torch_dag_loss_two_vertex_chain_exact():
    """B=1, prelen=2, tarlen=2, single path 0->1, uniform 2-class emit.

    P(target | DAG) = P(y0 | v0) * 1.0 * P(y1 | v1) = 0.5 * 1 * 0.5 = 0.25
    log = -2 log 2.
    """
    log_half = math.log(0.5)
    match_all = torch.full((1, 2, 2), log_half)
    # v0 -> v1 deterministic; v1 terminal (its row is unused).
    links = torch.tensor(
        [[[LINK_MASK_VALUE, 0.0], [LINK_MASK_VALUE, LINK_MASK_VALUE]]]
    )
    output_length = torch.tensor([2], dtype=torch.long)
    target_length = torch.tensor([2], dtype=torch.long)

    out = torch_dag_loss(match_all, links, output_length, target_length)

    expected = -2.0 * math.log(2.0)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(expected, abs=1e-6)


def test_torch_dag_loss_branching_marginalizes():
    """B=1, prelen=3, tarlen=2, output_length=3. Path must end at v2.

    Init: f[0, 0, 0] = log 0.5; others = -inf.
    Step 1, vertex 2: logsumexp_i(f[0, i] + links[i, 2]) + match[1, 2]
                    = logsumexp(log 0.5 + (-log 2), -inf, -inf) + log 0.5
                    = -log 2 - log 2 + (-log 2)
                    = -3 log 2
    The v0->v1->v2 branch can't reach v2 within target_length=2 so the
    branching is implicit (only one effective path lands on v2 at step 1).
    """
    log_half = math.log(0.5)
    match_all = torch.full((1, 2, 3), log_half)
    # v0: uniform out to v1, v2 (both -log 2). v1: deterministic to v2.
    # v2: terminal (row unused).
    neg_log2 = -math.log(2.0)
    links = torch.tensor(
        [[
            [LINK_MASK_VALUE, neg_log2, neg_log2],
            [LINK_MASK_VALUE, LINK_MASK_VALUE, 0.0],
            [LINK_MASK_VALUE, LINK_MASK_VALUE, LINK_MASK_VALUE],
        ]]
    )

    out = torch_dag_loss(
        match_all, links,
        output_length=torch.tensor([3]),
        target_length=torch.tensor([2]),
    )

    expected = -3.0 * math.log(2.0)
    assert out.item() == pytest.approx(expected, abs=1e-5)


def test_torch_dag_logsoftmax_gather_shape_and_values():
    """Gather follows the reference convention: (B, prelen, T_tgt) layout."""
    torch.manual_seed(0)
    B, prelen, V, T_tgt = 2, 5, 7, 3
    logits = torch.randn(B, prelen, V)
    target = torch.randint(0, V, (B, T_tgt))
    select = target.unsqueeze(1).expand(B, prelen, T_tgt)

    _, match = torch_dag_logsoftmax_gather(logits, select)

    assert match.shape == (B, prelen, T_tgt)
    expected = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    for b in range(B):
        for j in range(prelen):
            for i in range(T_tgt):
                assert match[b, j, i].item() == pytest.approx(
                    expected[b, j, target[b, i]].item(), abs=1e-6
                )


# ---------------------------------------------------------------------------
# Reference parity
# ---------------------------------------------------------------------------


def test_torch_dag_loss_matches_reference_random():
    """Random DAGs (B=3, prelen=8, tarlen=5) match thu-coai's torch_dag_loss."""
    ref = _load_reference()

    torch.manual_seed(42)
    B, prelen, T_tgt = 3, 8, 5
    match_all = torch.randn(B, T_tgt, prelen, dtype=torch.float64)
    links = _make_left_to_right_links(B, prelen, dtype=torch.float64, seed=43)
    output_length = torch.tensor([prelen, prelen - 1, prelen - 2], dtype=torch.long)
    target_length = torch.tensor([T_tgt, T_tgt - 1, T_tgt - 1], dtype=torch.long)

    ours = torch_dag_loss(match_all, links, output_length, target_length)
    theirs = ref.torch_dag_loss(match_all, links, output_length, target_length)

    assert torch.allclose(ours, theirs, atol=1e-8, rtol=1e-6), (
        f"mismatch: ours={ours.tolist()} theirs={theirs.tolist()}"
    )


def test_torch_dag_best_alignment_matches_reference_random():
    """Oracle assignment should match the reference for a random DAG."""
    ref = _load_reference()

    torch.manual_seed(7)
    B, prelen, T_tgt = 2, 6, 4
    match_all = torch.randn(B, T_tgt, prelen, dtype=torch.float64)
    links = _make_left_to_right_links(B, prelen, dtype=torch.float64, seed=8)
    output_length = torch.full((B,), prelen, dtype=torch.long)
    target_length = torch.full((B,), T_tgt, dtype=torch.long)

    ours = torch_dag_best_alignment(match_all, links, output_length, target_length)
    theirs = ref.torch_dag_best_alignment(match_all, links, output_length, target_length)
    assert torch.equal(ours, theirs), f"ours={ours} theirs={theirs}"


# ---------------------------------------------------------------------------
# Gradient sanity
# ---------------------------------------------------------------------------


def test_torch_dag_loss_gradient_finite():
    """Backward through the DP must not produce NaN/Inf for production-shaped links.

    `_make_left_to_right_links` mirrors the production `extract_links` mask
    pattern (large-negative finite sentinels, not strict -inf), so the
    gradient signal through the DP must be finite everywhere.
    """
    torch.manual_seed(1)
    B, prelen, T_tgt = 2, 6, 4
    match_all = torch.randn(B, T_tgt, prelen, requires_grad=True)
    links = _make_left_to_right_links(B, prelen, seed=2).requires_grad_()
    output_length = torch.full((B,), prelen, dtype=torch.long)
    target_length = torch.full((B,), T_tgt, dtype=torch.long)

    out = torch_dag_loss(match_all, links, output_length, target_length)
    loss = -out.mean()
    loss.backward()

    assert torch.isfinite(match_all.grad).all(), (
        f"match_all.grad has NaN/Inf: {match_all.grad}"
    )
    assert torch.isfinite(links.grad).all(), (
        f"links.grad has NaN/Inf: {links.grad}"
    )


# ---------------------------------------------------------------------------
# CUDA backend parity (skipped if CUDA / kernel build unavailable)
# ---------------------------------------------------------------------------


def _cuda_kernel_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from new_ime.training.loss.dat_cuda.loader import load_dat_kernel

    if load_dat_kernel() is None:
        pytest.skip("DAT CUDA kernel did not build (toolchain missing or mismatch)")


def _make_links_strict_terminal(
    batch: int,
    prelen: int,
    output_length: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
) -> torch.Tensor:
    """Like `_make_left_to_right_links` but with rows past each sample's terminal
    vertex set to *strict* `-inf`.

    Why this matters for CUDA parity: the pytorch DP visits all `prelen` rows
    regardless of `output_length`, so any spurious finite mass on rows
    `>= output_length-1` (created by `log_softmax` over an all-`LINK_MASK_VALUE`
    row → `log(1/prelen)`) leaks into the answer. The CUDA kernel correctly
    skips transitions from the terminal vertex onward, so the two diverge.
    Setting those rows to strict `-inf` makes both impls agree.

    Real `link_extractor` outputs are normalized over a `valid_mask` so this
    artifact doesn't arise in production.
    """
    links = _make_left_to_right_links(batch, prelen, dtype=dtype, seed=seed)
    for b in range(batch):
        ol = int(output_length[b].item())
        if ol > 0 and ol - 1 < prelen:
            links[b, ol - 1:, :] = float("-inf")
    return links


def test_cuda_dag_loss_matches_pytorch_random():
    """Dispatcher-on-CUDA should agree with the pure-PyTorch DP within fp32 tol.

    Note: `output_length < prelen` for every sample. With output_length == prelen,
    the pytorch DP reads `links[prelen-1, j]` (the all-masked last row), which
    has spurious finite values from `log_softmax`-over-all-`LINK_MASK_VALUE`.
    The CUDA kernel correctly ignores transitions from the terminal vertex
    (there are none), so the two diverge at exactly that corner case. Real
    training never hits this — `output_length` is always the actual sequence
    length, and the DAG decoder pads beyond it — so we test in the regime
    that matches production.
    """
    _cuda_kernel_or_skip()

    torch.manual_seed(123)
    B, prelen, T_tgt = 3, 16, 8
    match_all_cpu = torch.randn(B, T_tgt, prelen, dtype=torch.float32)
    output_length = torch.tensor([prelen - 1, prelen - 2, prelen - 4], dtype=torch.long)
    target_length = torch.tensor([T_tgt, T_tgt - 1, T_tgt - 2], dtype=torch.long)
    links_cpu = _make_links_strict_terminal(B, prelen, output_length, dtype=torch.float32, seed=124)

    ref = _pytorch_dag_loss(match_all_cpu, links_cpu, output_length, target_length)
    cuda_out = torch_dag_loss(
        match_all_cpu.cuda(), links_cpu.cuda(), output_length.cuda(), target_length.cuda(),
    )
    assert torch.allclose(cuda_out.cpu(), ref, rtol=1e-3, atol=1e-4), (
        f"CUDA mismatch: cuda={cuda_out.cpu().tolist()} ref={ref.tolist()}"
    )


def test_cuda_dag_best_alignment_matches_pytorch_random():
    """Same `output_length < prelen` caveat as the loss test above."""
    _cuda_kernel_or_skip()

    torch.manual_seed(7)
    B, prelen, T_tgt = 2, 12, 5
    match_all_cpu = torch.randn(B, T_tgt, prelen, dtype=torch.float32)
    output_length = torch.full((B,), prelen - 1, dtype=torch.long)
    target_length = torch.full((B,), T_tgt, dtype=torch.long)
    links_cpu = _make_links_strict_terminal(B, prelen, output_length, dtype=torch.float32, seed=8)

    ref = _pytorch_dag_best_alignment(match_all_cpu, links_cpu, output_length, target_length)
    cuda_out = torch_dag_best_alignment(
        match_all_cpu.cuda(), links_cpu.cuda(), output_length.cuda(), target_length.cuda(),
    )
    # Best-alignment paths can have ties; CUDA and the autograd-trick PyTorch
    # impl may pick different tie-breakers. For non-degenerate random seeds
    # the paths typically match exactly.
    assert torch.equal(cuda_out.cpu(), ref), (
        f"path mismatch: cuda={cuda_out.cpu().tolist()} ref={ref.tolist()}"
    )


def test_cuda_dag_loss_gradient_finite():
    _cuda_kernel_or_skip()

    torch.manual_seed(1)
    B, prelen, T_tgt = 2, 8, 4
    match_all = torch.randn(B, T_tgt, prelen, dtype=torch.float32, device="cuda", requires_grad=True)
    output_length = torch.full((B,), prelen - 1, dtype=torch.long, device="cuda")
    target_length = torch.full((B,), T_tgt, dtype=torch.long, device="cuda")
    links = _make_links_strict_terminal(
        B, prelen, output_length.cpu(), dtype=torch.float32, seed=2,
    ).cuda().requires_grad_()

    out = torch_dag_loss(match_all, links, output_length, target_length)
    loss = -out.mean()
    loss.backward()

    assert torch.isfinite(match_all.grad).all(), "match_all.grad has NaN/Inf on CUDA"
    assert torch.isfinite(links.grad).all(), "links.grad has NaN/Inf on CUDA"


def test_cuda_dag_loss_accepts_bfloat16():
    """bf16 inputs (autocast on Ampere+) must not crash; kernel only
    dispatches fp32/fp16 so the wrapper promotes silently."""
    _cuda_kernel_or_skip()

    torch.manual_seed(11)
    B, prelen, T_tgt = 2, 8, 4
    output_length = torch.full((B,), prelen - 1, dtype=torch.long, device="cuda")
    target_length = torch.full((B,), T_tgt, dtype=torch.long, device="cuda")
    match_all = torch.randn(B, T_tgt, prelen, dtype=torch.float32, device="cuda").bfloat16()
    links = (
        _make_links_strict_terminal(B, prelen, output_length.cpu(), dtype=torch.float32, seed=12)
        .cuda()
        .bfloat16()
    )

    # Should run without RuntimeError and produce finite per-sample log P.
    out = torch_dag_loss(match_all, links, output_length, target_length)
    assert out.shape == (B,)
    assert torch.isfinite(out).all(), f"non-finite loss: {out}"


def test_cuda_dag_logsoftmax_gather_accepts_bfloat16():
    _cuda_kernel_or_skip()

    torch.manual_seed(13)
    B, prelen, V, T_tgt = 2, 6, 16, 4
    logits = torch.randn(B, prelen, V, dtype=torch.bfloat16, device="cuda")
    target = torch.randint(0, V, (B, T_tgt), device="cuda")
    select = target.unsqueeze(1).expand(B, prelen, T_tgt)

    # Must not raise; match is fp32 by spec regardless of input dtype.
    _, match = torch_dag_logsoftmax_gather(logits, select)
    assert match.dtype == torch.float32
    assert torch.isfinite(match).all()


def test_cuda_dag_logsoftmax_gather_matches_pytorch():
    _cuda_kernel_or_skip()

    torch.manual_seed(0)
    B, prelen, V, T_tgt = 2, 8, 16, 5
    logits = torch.randn(B, prelen, V, dtype=torch.float32)
    target = torch.randint(0, V, (B, T_tgt))
    select = target.unsqueeze(1).expand(B, prelen, T_tgt)

    _, ref = _pytorch_dag_logsoftmax_gather(logits, select)
    _, cuda_match = torch_dag_logsoftmax_gather(logits.cuda(), select.cuda())

    assert torch.allclose(cuda_match.cpu(), ref, rtol=1e-3, atol=1e-4), (
        f"logsoftmax_gather CUDA mismatch (max diff: "
        f"{(cuda_match.cpu() - ref).abs().max().item()})"
    )
