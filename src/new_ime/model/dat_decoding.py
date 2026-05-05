"""DAG decoding strategies for DAT.

All three strategies operate on `(logits, links, output_length)` and
return `list[list[int]]` (one token sequence per batch row, blank_id
removed, consecutive duplicates collapsed). Path always starts at
vertex 0 — DAT's decoder placeholder convention.

* greedy        : argmax token at each visited vertex; next vertex =
                  argmax(transition log-prob row).
* lookahead     : next vertex = argmax(transition + beta * next-vertex
                  argmax-token log-prob); decode token from chosen vertex.
                  Equivalent to greedy when beta=0.
* viterbi       : forward DP with max-marginalization over predecessors,
                  joint over (path, tokens). Length-normalized score
                  picks the best path length up to a configurable cap.

Reference: `references/DA-Transformer/fs_plugins/models/glat_decomposed_with_link.py`
(L639-777). Pure-PyTorch / pure-Python; no fairseq dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize_token_run(seq: list[int], blank_id: int) -> list[int]:
    """Drop blank ids and collapse consecutive duplicates (CTC-style)."""
    out: list[int] = []
    for tok in seq:
        if tok == blank_id:
            continue
        if out and out[-1] == tok:
            continue
        out.append(tok)
    return out


def greedy_decode(
    logits: Tensor,           # (B, prelen, V)
    links: Tensor,            # (B, prelen, prelen) log transition prob
    output_length: Tensor,    # (B,) effective vertices per sample
    *,
    blank_id: int,
) -> list[list[int]]:
    """Pure-greedy DAG traversal: argmax token + argmax transition."""
    batch_size, prelen, _ = logits.shape
    token_argmax = logits.argmax(dim=-1).cpu().tolist()        # B × prelen
    next_argmax = links.argmax(dim=-1).cpu().tolist()          # B × prelen
    lengths = output_length.cpu().tolist()

    results: list[list[int]] = []
    for b in range(batch_size):
        limit = int(lengths[b])
        seq: list[int] = []
        cur = 0
        steps = 0
        seq.append(int(token_argmax[b][cur]))
        while cur < limit - 1 and steps < prelen:
            nxt = int(next_argmax[b][cur])
            if nxt <= cur or nxt >= limit:
                break
            cur = nxt
            seq.append(int(token_argmax[b][cur]))
            steps += 1
        results.append(_normalize_token_run(seq, blank_id))
    return results


def lookahead_decode(
    logits: Tensor,
    links: Tensor,
    output_length: Tensor,
    *,
    blank_id: int,
    beta: float = 1.0,
) -> list[list[int]]:
    """Joint scoring of (next transition, next token argmax) at each step.

    For each vertex v we compute the joint score
        score[v, j] = links[v, j] + beta * max_token_logit[j]
    and pick j = argmax. Token at j is the argmax over the vocab logits
    at j. With beta=0 this reduces to greedy.
    """
    batch_size, prelen, _ = logits.shape
    max_logit, token_argmax = logits.max(dim=-1)              # (B, prelen) each
    joint = links + max_logit.unsqueeze(1) * beta             # (B, prelen, prelen)
    next_argmax = joint.argmax(dim=-1).cpu().tolist()
    token_argmax = token_argmax.cpu().tolist()
    lengths = output_length.cpu().tolist()

    results: list[list[int]] = []
    for b in range(batch_size):
        limit = int(lengths[b])
        seq: list[int] = [int(token_argmax[b][0])]
        cur = 0
        steps = 0
        while cur < limit - 1 and steps < prelen:
            nxt = int(next_argmax[b][cur])
            if nxt <= cur or nxt >= limit:
                break
            cur = nxt
            seq.append(int(token_argmax[b][cur]))
            steps += 1
        results.append(_normalize_token_run(seq, blank_id))
    return results


def viterbi_decode(
    logits: Tensor,
    links: Tensor,
    output_length: Tensor,
    *,
    blank_id: int,
    length_penalty: float = 1.0,
    max_length: int | None = None,
) -> list[list[int]]:
    """Joint Viterbi: best (path, tokens) under length-normalized score.

    Forward DP (path always starts at vertex 0; both v0's and the next
    vertex's token logits enter alpha at step 0):

        alpha[0, v] = link[0 -> v] + max_logit[0] + max_logit[v]
        alpha[t, v] = max_u(alpha[t-1, u] + link[u -> v]) + max_logit[v]

    After T iterations, alpha[T, v] scores a path of length T+2 vertices
    (v0 -> ... -> v) emitting T+2 tokens. We pick the best (length, v)
    pair under `score / length^length_penalty`. Backtrack via stored
    argmax pointers; prepend v0's token at the very end.
    """
    batch_size, prelen, _ = logits.shape
    device = logits.device
    max_logit, token_argmax = logits.max(dim=-1)  # (B, prelen) each

    if max_length is None:
        max_length = max(2, prelen)
    max_length = min(max_length, prelen)
    # max_length here = the longest "alpha index" t we'll compute (path len = t+2).
    iters = max(0, max_length - 1)  # number of post-init iterations

    # alpha_0[v] = link[0 -> v] + logit[0] + logit[v]; path length 2.
    v0_logit = max_logit[:, 0:1]                 # (B, 1)
    alpha_t = links[:, 0, :] + v0_logit + max_logit  # (B, prelen)
    alpha_history: list[Tensor] = [alpha_t]
    backpointers: list[Tensor] = []

    for _t in range(iters):
        candidates = alpha_t.unsqueeze(-1) + links  # (B, u, v)
        alpha_t, ptr = candidates.max(dim=1)
        alpha_t = alpha_t + max_logit
        alpha_history.append(alpha_t)
        backpointers.append(ptr)

    alpha_stack = torch.stack(alpha_history, dim=0)  # (T_alpha, B, prelen)
    t_alpha = alpha_stack.shape[0]

    # Path-length normalization. alpha_history[i] = path of length (i + 2).
    path_lengths = torch.arange(2, 2 + t_alpha, device=device, dtype=alpha_stack.dtype)
    penalty = path_lengths.pow(length_penalty).view(t_alpha, 1, 1)
    norm_scores = alpha_stack / penalty  # (T_alpha, B, prelen)

    flat = norm_scores.permute(1, 0, 2).reshape(batch_size, -1)
    best_flat = flat.argmax(dim=-1)
    best_t_idx = (best_flat // prelen).cpu().tolist()
    best_end_v = (best_flat % prelen).cpu().tolist()
    bp = [b.cpu().tolist() for b in backpointers]
    token_argmax_list = token_argmax.cpu().tolist()

    results: list[list[int]] = []
    for b in range(batch_size):
        t_idx = int(best_t_idx[b])  # alpha index → path length = t_idx + 2
        v = int(best_end_v[b])
        path_tokens: list[int] = [int(token_argmax_list[b][v])]
        # Walk back through backpointers[t_idx-1], ..., [0]: each step goes
        # back one vertex along the best alpha path.
        for t in range(t_idx, 0, -1):
            v = int(bp[t - 1][b][v])
            path_tokens.append(int(token_argmax_list[b][v]))
        # Finally prepend v0's token (the path always starts at vertex 0).
        path_tokens.append(int(token_argmax_list[b][0]))
        path_tokens.reverse()
        results.append(_normalize_token_run(path_tokens, blank_id))
    return results
