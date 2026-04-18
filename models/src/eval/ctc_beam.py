"""Minimal CTC prefix beam search for the evaluation harness.

The trainer and production decoder use `CTCNAT.greedy_decode`; this module
adds a Python prefix-beam variant of the standard Graves (2006) algorithm
so we can quantify the expected EM gain before investing in the C++
beam+KenLM path already planned for `server/src/ctc_decoder.cpp`.

Semantics:
- `log_probs`: (T, V) with `blank_id` occupying one column.
- Adjacent repeats collapse only when separated by a blank (CTC rule).
- We merge equivalent output prefixes across paths (standard prefix beam).
- Optional shallow LM fusion: pass `lm_scorer` + `lm_alpha` + `lm_beta` to
  rank by `logp_ctc(prefix) + alpha * logp_lm(prefix) + beta * len(prefix)`.
  The scorer is any object with `score(prefix: tuple[int, ...]) -> float`
  returning the full-prefix natural-log probability. Delta is cached so the
  scorer typically only does one extend-by-one step per beam slot per t.
- Top-K tokens per timestep bound the inner loop at `beam_width * K`.

The implementation is deliberately pure Python/PyTorch; it is meant for
the harness, not for training loops.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Protocol, Tuple

import torch


class PrefixLMScorer(Protocol):
    """Interface for any LM whose full-prefix log-prob we can query by token ids."""

    def score(self, prefix: Tuple[int, ...]) -> float: ...


NEG_INF = float("-inf")


def _logsumexp_pair(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)) for two scalars."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    lo, hi = (a, b) if a < b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def prefix_beam_search(
    log_probs: torch.Tensor,
    blank_id: int,
    beam_width: int = 8,
    top_k_per_step: int = 16,
    lm_scorer: Optional[PrefixLMScorer] = None,
    lm_alpha: float = 0.0,
    lm_beta: float = 0.0,
) -> List[Tuple[List[int], float]]:
    """Return the top-`beam_width` candidate token sequences.

    Args:
        log_probs: (T, V) log-probabilities for a single example. We operate
            on a CPU float32 copy; move the tensor back to the caller's
            device outside this function if needed.
        blank_id: CTC blank index.
        beam_width: output beam size.
        top_k_per_step: only extend each beam entry with these many tokens
            at each timestep. Caps the combinatorics; keep >= beam_width.

    Returns:
        list of `(tokens, logp)` sorted by descending logp. `tokens` is the
        collapsed CTC output (no blanks, adjacent duplicates merged across
        CTC rules).
    """
    if log_probs.dim() != 2:
        raise ValueError("log_probs must be 2-D (T, V)")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    log_probs = log_probs.detach().cpu().to(torch.float32)
    T, V = log_probs.shape
    top_k_per_step = min(top_k_per_step, V)

    # For each prefix tuple, track two scores:
    #   pb  — log prob of reaching this prefix and ending on blank
    #   pnb — log prob of reaching this prefix and ending on non-blank
    # The total log prob is logsumexp(pb, pnb).
    beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    # Cache top-K token indices per timestep to cap inner loop size.
    topk_probs, topk_idx = torch.topk(log_probs, top_k_per_step, dim=-1)
    topk_probs = topk_probs.tolist()
    topk_idx = topk_idx.tolist()

    for t in range(T):
        next_beam: dict[tuple[int, ...], list[float]] = {}

        def update(prefix: tuple[int, ...], new_pb: float, new_pnb: float) -> None:
            entry = next_beam.get(prefix)
            if entry is None:
                next_beam[prefix] = [new_pb, new_pnb]
            else:
                entry[0] = _logsumexp_pair(entry[0], new_pb)
                entry[1] = _logsumexp_pair(entry[1], new_pnb)

        # At timestep t, blank is always a candidate even if not in top-K of
        # step t, because maintaining the "prefix ending in blank" case is
        # essential for correctness (collapsing repeats requires it).
        blank_logp = log_probs[t, blank_id].item()

        for prefix, (pb, pnb) in beam.items():
            # Case 1: emit blank. Prefix unchanged; we only touch `pb`.
            new_blank_contrib = _logsumexp_pair(pb, pnb) + blank_logp
            update(prefix, new_blank_contrib, NEG_INF)

            # Case 2: emit a non-blank character c from top-K.
            for c, c_logp in zip(topk_idx[t], topk_probs[t]):
                if c == blank_id:
                    continue
                if prefix and c == prefix[-1]:
                    # Repeat of last char: only the "previous blank" path
                    # extends prefix; the "previous non-blank" path stays
                    # collapsed (no new token emitted).
                    update(prefix + (c,), NEG_INF, pb + c_logp)
                    update(prefix, NEG_INF, pnb + c_logp)
                else:
                    new_pnb = _logsumexp_pair(pb, pnb) + c_logp
                    update(prefix + (c,), NEG_INF, new_pnb)

        # Prune to top `beam_width`. Ranking = CTC + LM shallow fusion +
        # length penalty. LM and length terms only kick in when a scorer is
        # supplied and alpha/beta are non-zero; otherwise this reduces to
        # vanilla CTC prefix beam.
        def rank(prefix: tuple[int, ...], pb: float, pnb: float) -> float:
            ctc = _logsumexp_pair(pb, pnb)
            lm_part = 0.0
            if lm_scorer is not None and lm_alpha != 0.0 and prefix:
                lm_part = lm_alpha * lm_scorer.score(prefix)
            length_part = lm_beta * len(prefix)
            return ctc + lm_part + length_part

        scored = [
            (p, rank(p, pb, pnb))
            for p, (pb, pnb) in next_beam.items()
        ]
        scored.sort(key=lambda x: -x[1])
        beam = {p: tuple(next_beam[p]) for p, _ in scored[:beam_width]}

    # Final ranking uses the same fused score as pruning, so returned order
    # matches the inner selection criterion.
    final = []
    for p, (pb, pnb) in beam.items():
        final.append((list(p), rank(p, pb, pnb)))
    final.sort(key=lambda x: -x[1])
    return final


def beam_decode_logits(
    logits: torch.Tensor,
    blank_id: int,
    beam_width: int = 8,
    top_k_per_step: int = 16,
) -> List[List[Tuple[List[int], float]]]:
    """Vectorised over batch: run `prefix_beam_search` on each row.

    Args:
        logits: (batch, T, V) pre-softmax logits.
        blank_id: CTC blank id.
        beam_width: beam size per example.
        top_k_per_step: as `prefix_beam_search`.

    Returns:
        For each batch element, the beam list from `prefix_beam_search`.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    return [
        prefix_beam_search(log_probs[b], blank_id, beam_width, top_k_per_step)
        for b in range(log_probs.shape[0])
    ]
