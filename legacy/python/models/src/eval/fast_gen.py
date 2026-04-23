"""Fast batched generation with KV cache for SimpleGPT2.

SimpleGPT2 uses nn.TransformerEncoderLayer which doesn't expose a KV cache.
Here we reimplement the per-layer forward by reaching into the layer's
sub-modules (LN, MHA, FFN), allowing us to cache K/V across decode steps.

Time complexity per generated token drops from O((L+n)^2) to O(L+n), and
all beams run in one batched forward per step.

Correctness check: numerical outputs match the original forward within
float32 rounding on a prefix-only pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _layer_heads(layer: nn.TransformerEncoderLayer) -> int:
    return layer.self_attn.num_heads


def _layer_forward_with_cache(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,                  # (B, T, H)
    past_k: torch.Tensor | None,       # (B, nh, S, hd)
    past_v: torch.Tensor | None,
    is_prefix: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one transformer-encoder layer with KV cache.

    Assumes pre-LN (norm_first=True), ReLU activation, no dropout (eval mode).
    """
    # === self-attention branch (pre-LN + residual) ===
    residual = x
    y = layer.norm1(x)

    # Project Q, K, V from the combined in-proj matrix.
    Wqkv = layer.self_attn.in_proj_weight  # (3H, H)
    bqkv = layer.self_attn.in_proj_bias    # (3H,)
    qkv = F.linear(y, Wqkv, bqkv)           # (B, T, 3H)
    H = y.shape[-1]
    q, k, v = qkv.split(H, dim=-1)

    nh = _layer_heads(layer)
    hd = H // nh
    B, T, _ = q.shape
    q = q.view(B, T, nh, hd).transpose(1, 2)  # (B, nh, T, hd)
    k = k.view(B, T, nh, hd).transpose(1, 2)
    v = v.view(B, T, nh, hd).transpose(1, 2)

    # Append cache
    if past_k is not None:
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    # Scaled dot-product attention.
    #   is_prefix=True:  full causal attention over T query positions (T>=1).
    #   is_prefix=False: T=1, attending over cached past + current (no mask needed).
    if is_prefix and T > 1:
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        att = F.scaled_dot_product_attention(q, k, v)

    att = att.transpose(1, 2).contiguous().view(B, T, H)
    att = F.linear(att, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
    x = residual + att

    # === FFN branch (pre-LN + residual) ===
    residual = x
    y = layer.norm2(x)
    y = layer.linear2(layer.activation(layer.linear1(y)))
    x = residual + y
    return x, k, v


class FastARGenerator:
    """Wraps a SimpleGPT2 instance to provide KV-cache-aware generation."""

    def __init__(self, model: nn.Module, device: torch.device, max_pos: int) -> None:
        self.model = model
        self.device = device
        self.max_pos = max_pos
        self.num_layers = len(list(model.transformer.layers))

    @torch.no_grad()
    def encode_prefix(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the prefix once and return last-token logits + per-layer KV cache.

        input_ids: (B, T)
        Returns:
          last_logits: (B, V)
          cache: list of (k, v) per layer, each (B, nh, T, hd)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
        x = self.model.embed_tokens(input_ids) + self.model.embed_positions(positions)

        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.model.transformer.layers:
            x, k, v = _layer_forward_with_cache(layer, x, None, None, is_prefix=True)
            cache.append((k, v))
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x[:, -1, :])  # (B, V)
        return logits, cache

    @torch.no_grad()
    def step(
        self,
        new_tokens: torch.Tensor,  # (B,) or (B, 1)
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        cur_len: int,  # the index of the new token (0-based) in the full sequence
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Append one token per batch element using the KV cache."""
        if new_tokens.dim() == 1:
            new_tokens = new_tokens.unsqueeze(-1)
        B = new_tokens.shape[0]
        pos = torch.full((B, 1), cur_len, dtype=torch.long, device=self.device)
        x = self.model.embed_tokens(new_tokens) + self.model.embed_positions(pos)

        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (pk, pv) in zip(self.model.transformer.layers, cache):
            x, k, v = _layer_forward_with_cache(layer, x, pk, pv, is_prefix=False)
            new_cache.append((k, v))
        x = self.model.ln_f(x)
        logits = self.model.lm_head(x[:, -1, :])  # (B, V)
        return logits, new_cache

    def expand_cache(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        beam_width: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Repeat cache along the batch dim: (1, nh, T, hd) -> (bw, nh, T, hd)."""
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in cache:
            out.append(
                (
                    k.expand(beam_width, -1, -1, -1).contiguous(),
                    v.expand(beam_width, -1, -1, -1).contiguous(),
                )
            )
        return out

    def reorder_cache(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        beam_indices: torch.Tensor,  # (bw,) pointing into current batch
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Reorder cache along batch dim to follow beam parent selection."""
        return [
            (k.index_select(0, beam_indices), v.index_select(0, beam_indices))
            for k, v in cache
        ]


@torch.no_grad()
def fast_greedy(
    gen: FastARGenerator,
    prefix_ids: list[int],
    max_new: int,
    eos_id: int,
    pad_id: int,
) -> list[int]:
    prefix_ids = prefix_ids[-(gen.max_pos - 2) :]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=gen.device)
    logits, cache = gen.encode_prefix(ids)
    out: list[int] = []
    cur_len = ids.shape[1]
    for _ in range(max_new):
        if cur_len >= gen.max_pos:
            break
        nid = int(logits[0].argmax().item())
        if nid == eos_id or nid == pad_id:
            break
        out.append(nid)
        token = torch.tensor([nid], dtype=torch.long, device=gen.device)
        logits, cache = gen.step(token, cache, cur_len)
        cur_len += 1
    return out


@torch.no_grad()
def fast_beam(
    gen: FastARGenerator,
    prefix_ids: list[int],
    max_new: int,
    eos_id: int,
    pad_id: int,
    beam_width: int,
    length_penalty: float,
    repetition_penalty: float,
) -> list[list[int]]:
    """Batched beam search with KV cache.

    Returns up to beam_width generated-token sequences (excluding prefix),
    ranked by length-normalised score.
    """
    prefix_ids = prefix_ids[-(gen.max_pos - 2) :]
    prefix_len = len(prefix_ids)
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=gen.device)
    logits, cache = gen.encode_prefix(ids)  # (1, V), cache=(1, ...)

    bw = beam_width
    V = logits.shape[-1]

    # Seed: pick top-bw from the first distribution.
    log_probs0 = F.log_softmax(logits[0], dim=-1)  # (V,)
    top0 = torch.topk(log_probs0, bw)
    beam_scores = top0.values.clone()                 # (bw,)
    beam_tokens = top0.indices.tolist()              # list[int] (generated seqs)
    beam_seqs: list[list[int]] = [[t] for t in beam_tokens]
    finished: list[tuple[list[int], float]] = []

    # Expand cache to bw.
    cache = gen.expand_cache(cache, bw)
    cur_len = prefix_len  # length before appending new beam tokens

    # Mark EOS in seed (rare, but honour it).
    keep_mask = torch.ones(bw, dtype=torch.bool, device=gen.device)
    for i, t in enumerate(beam_tokens):
        if t == eos_id or t == pad_id:
            finished.append((beam_seqs[i][:-1], float(beam_scores[i].item())))
            keep_mask[i] = False
    if not keep_mask.any():
        # Fall back: return whatever we collected.
        return [seq for seq, _ in finished] or [[]]

    # Prune seed to surviving beams; apply first step of KV cache.
    alive_idx = keep_mask.nonzero(as_tuple=True)[0]
    beam_seqs = [beam_seqs[int(i)] for i in alive_idx]
    beam_scores = beam_scores[alive_idx]
    cache = gen.reorder_cache(cache, alive_idx)
    active_tokens = torch.tensor(
        [seq[-1] for seq in beam_seqs], dtype=torch.long, device=gen.device
    )
    # Advance KV cache by one step using the seed tokens.
    logits, cache = gen.step(active_tokens, cache, cur_len)
    cur_len += 1

    for _ in range(max_new - 1):
        if cur_len >= gen.max_pos:
            break
        B = logits.shape[0]
        log_probs = F.log_softmax(logits, dim=-1)  # (B, V)

        # Repetition penalty on generated tokens so far (per-beam).
        if repetition_penalty != 1.0:
            for bi, seq in enumerate(beam_seqs):
                if not seq:
                    continue
                idx = torch.tensor(list(set(seq)), device=gen.device)
                log_probs[bi].index_copy_(
                    0, idx, log_probs[bi].index_select(0, idx) / repetition_penalty
                )

        # Score = current beam score + next-token log prob.
        total = beam_scores.unsqueeze(-1) + log_probs  # (B, V)
        flat = total.view(-1)
        topk = torch.topk(flat, min(bw * 2, flat.shape[0]))
        parent = topk.indices // V
        token = topk.indices % V
        new_scores = topk.values  # (bw*2,)

        kept_parents: list[int] = []
        kept_tokens: list[int] = []
        kept_scores: list[float] = []
        for p, t, s in zip(parent.tolist(), token.tolist(), new_scores.tolist()):
            seq = beam_seqs[p] + [t]
            if t == eos_id or t == pad_id:
                finished.append((beam_seqs[p], s))
                continue
            kept_parents.append(p)
            kept_tokens.append(t)
            kept_scores.append(s)
            if len(kept_parents) >= bw:
                break

        if not kept_parents:
            break

        # Reorder cache & state to the new beam ordering.
        parent_t = torch.tensor(kept_parents, dtype=torch.long, device=gen.device)
        cache = gen.reorder_cache(cache, parent_t)
        beam_seqs = [beam_seqs[p] + [t] for p, t in zip(kept_parents, kept_tokens)]
        beam_scores = torch.tensor(kept_scores, device=gen.device)
        active_tokens = torch.tensor(kept_tokens, dtype=torch.long, device=gen.device)
        logits, cache = gen.step(active_tokens, cache, cur_len)
        cur_len += 1

    # Combine finished + still-active, rank by length-normalised score.
    all_results = finished + [
        (seq, float(beam_scores[i].item())) for i, seq in enumerate(beam_seqs)
    ]

    def norm(item: tuple[list[int], float]) -> float:
        s, sc = item
        return sc / max(len(s), 1) ** length_penalty

    all_results.sort(key=norm, reverse=True)
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for seq, _ in all_results[: bw]:
        k = tuple(seq)
        if k in seen:
            continue
        seen.add(k)
        out.append(seq)
    return out
