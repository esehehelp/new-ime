"""ConversionBackend for CTC-NAT checkpoints.

Wraps a phase3_20m / phase3_30m / phase3_90m checkpoint for the same
`ConversionBackend.convert(reading, context) -> list[str]` contract used by
the AR and zenz backends, so `run_all_evals.py` can put all three in a
single comparison table.

Three decoding knobs:

* `beam_width` — 1 for greedy (matches the existing `CTCNAT.greedy_decode`),
  or >1 for prefix beam search over the CTC emission (implemented in
  `src.eval.ctc_beam`).
* `chunk_threshold`, `chunk_size` — when the reading is >= threshold,
  split it into `chunk_size`-character segments and convert each segment
  with the previously-converted surface as rolling context. This
  compensates for a training distribution that is ~60% short-chunk
  examples when we're asked to convert long-form inputs (e.g. general/dev
  Wikipedia sentences).

KenLM fusion lives in `server/src/ctc_decoder.cpp` and is not plumbed here
yet.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.src.data.tokenizer import SharedCharTokenizer
from models.src.eval.ctc_beam import prefix_beam_search
from models.src.eval.run_eval import ConversionBackend
from models.src.model.ctc_nat import CTCNAT, PRESETS


class CTCNATBackend(ConversionBackend):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        max_context: int = 40,
        max_seq_len: int = 128,
        beam_width: int = 1,
        beam_top_k: int = 16,
        chunk_threshold: int = 0,
        chunk_size: int = 8,
        chunk_stride: int | None = None,
        chunk_max_overlap: int = 20,
        lm_path: str | None = None,
        lm_alpha: float = 0.0,
        lm_beta: float = 0.0,
        lm_gate_min_conf: float = 0.0,
        diversity_lambda: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        mask_refine_k: int = 0,
        mask_refine_alt: int = 2,
        lm_paths_by_domain: dict[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        self.ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        preset_name = self.ckpt.get("preset")
        if preset_name not in PRESETS:
            raise ValueError(f"unknown preset in checkpoint: {preset_name!r}")

        tokenizer_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"missing tokenizer sidecar: {tokenizer_path}. "
                "ctc_nat checkpoints save a _tokenizer.json next to the .pt."
            )
        self.tokenizer = SharedCharTokenizer.load(str(tokenizer_path))
        vocab_size = int(
            self.ckpt.get("vocab_size") or self.tokenizer.vocab_size
        )

        self.model = CTCNAT.from_preset(
            preset_name,
            vocab_size=vocab_size,
            use_cvae=bool(self.ckpt.get("use_cvae", False)),
        )
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.max_context = max_context
        self.max_seq_len = int(self.ckpt.get("max_seq_len", max_seq_len))
        self.step = self.ckpt.get("step", "?")
        self.beam_width = max(1, int(beam_width))
        self.beam_top_k = max(self.beam_width, int(beam_top_k))
        self.chunk_threshold = int(chunk_threshold)
        self.chunk_size = max(1, int(chunk_size))
        # Stride defaults to chunk_size (= non-overlapping). Setting a smaller
        # stride enables sliding-window chunking with (chunk_size - stride)
        # characters of overlap per step, which we then re-assemble via
        # longest suffix/prefix match below.
        self.chunk_stride = max(1, int(chunk_stride if chunk_stride is not None else chunk_size))
        self.chunk_max_overlap = max(0, int(chunk_max_overlap))

        # KenLM shallow fusion is optional; only loaded if lm_path is given.
        # Confidence gate: when the greedy path's mean top-1 log-prob (over
        # non-blank emitted positions) is above `lm_gate_min_conf`, skip the
        # beam+LM pass entirely and return the greedy decode. 0.0 disables
        # the gate (LM fires on every call, the pre-gate default).
        self.lm_scorer = None
        self.lm_alpha = float(lm_alpha)
        self.lm_beta = float(lm_beta)
        self.lm_gate_min_conf = float(lm_gate_min_conf)
        self.diversity_lambda = float(diversity_lambda)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.mask_refine_k = max(0, int(mask_refine_k))
        self.mask_refine_alt = max(2, int(mask_refine_alt))
        self.lm_estimator = None
        if lm_paths_by_domain and (self.lm_alpha > 0.0 or self.lm_beta > 0.0):
            from models.src.eval.kenlm_mixture import CategoryEstimator, KenLMMixtureScorer
            self.lm_scorer = KenLMMixtureScorer(lm_paths_by_domain, self.tokenizer)
            self.lm_estimator = CategoryEstimator(available_domains=set(lm_paths_by_domain))
        elif lm_path and (self.lm_alpha > 0.0 or self.lm_beta > 0.0):
            from models.src.eval.kenlm_scorer import KenLMCharScorer
            self.lm_scorer = KenLMCharScorer(lm_path, self.tokenizer)

        ckpt_id = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        mode_bits = []
        mode_bits.append(f"beam{self.beam_width}" if self.beam_width > 1 else "greedy")
        if self.chunk_threshold > 0:
            if self.chunk_stride < self.chunk_size:
                mode_bits.append(f"chunk>={self.chunk_threshold}/{self.chunk_size}s{self.chunk_stride}")
            else:
                mode_bits.append(f"chunk>={self.chunk_threshold}/{self.chunk_size}")
        if self.lm_scorer is not None:
            gate = f",gate={self.lm_gate_min_conf:g}" if self.lm_gate_min_conf < 0.0 else ""
            mode_bits.append(f"lm(a={self.lm_alpha:g},b={self.lm_beta:g}{gate})")
        self._name = name or f"ctc_nat({preset_name},{ckpt_id}@step{self.step},{','.join(mode_bits)})"

    @property
    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def _decode_one(self, reading: str, context: str, top_k: int = 5) -> list[str]:
        ids = self.tokenizer.encode_with_special(context[-self.max_context :], reading)
        ids = ids[: self.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        if (
            self.beam_width <= 1
            and self.lm_scorer is None
            and self.mask_refine_k <= 0
        ):
            decoded = self.model.greedy_decode(input_ids, attention_mask)
            return [self.tokenizer.decode(decoded[0])]

        # Single forward pass; both the gate check and the beam need logits.
        result = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = result["logits"][0]  # (T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        top1_lp, top1_id = log_probs.max(dim=-1)

        def _collapse_greedy() -> str:
            # Reuse the argmax path we already have — avoids a second forward.
            input_len = int(attention_mask[0].sum().item())
            tokens: list[int] = []
            prev = -1
            blank = self.model.blank_id
            ids = top1_id.tolist()
            for t in range(input_len):
                tok = ids[t]
                if tok != blank and tok != prev:
                    tokens.append(tok)
                prev = tok
            return self.tokenizer.decode(tokens)

        if self.lm_scorer is not None and self.lm_gate_min_conf < 0.0:
            nonblank = top1_id != self.model.blank_id
            if nonblank.any():
                mean_conf = top1_lp[nonblank].mean().item()
            else:
                mean_conf = top1_lp.mean().item()
            if mean_conf >= self.lm_gate_min_conf:
                return [_collapse_greedy()]

        if self.beam_width <= 1 and self.mask_refine_k > 0:
            return self._mask_refine(
                log_probs, top1_id, top1_lp,
                input_len=int(attention_mask[0].sum().item()),
                k=self.mask_refine_k,
                top_k=top_k,
                alt_j=self.mask_refine_alt,
            )

        if self.beam_width <= 1:
            return [_collapse_greedy()]

        # KenLM-MoE: pick per-input mixture weights before beam search.
        if self.lm_estimator is not None and self.lm_scorer is not None:
            weights = self.lm_estimator.estimate(reading, context)
            self.lm_scorer.set_weights(weights)
        beam = prefix_beam_search(
            log_probs,
            blank_id=self.model.blank_id,
            beam_width=self.beam_width,
            top_k_per_step=self.beam_top_k,
            lm_scorer=self.lm_scorer,
            lm_alpha=self.lm_alpha,
            lm_beta=self.lm_beta,
            diversity_lambda=self.diversity_lambda,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if not beam:
            return []
        out: list[str] = []
        seen: set[str] = set()
        for tokens, _score in beam:
            s = self.tokenizer.decode(tokens)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= top_k:
                break
        return out

    def _mask_refine(
        self,
        log_probs: torch.Tensor,
        top1_id: torch.Tensor,
        top1_lp: torch.Tensor,
        input_len: int,
        k: int,
        top_k: int,
        alt_j: int = 2,
    ) -> list[str]:
        """Generate up to top_k surface candidates by perturbing the argmax path.

        For the k lowest-confidence non-blank frames (by top1 − top2 log-prob
        gap), substitute top-1 with each of the top-2 .. top-alt_j alternates
        and re-collapse via the CTC rule. Seeds with the plain greedy decode.
        alt_j=2 means single alt (top-2), alt_j=4 means top-2/3/4.
        """
        blank = self.model.blank_id
        base_ids = top1_id.tolist()[:input_len]

        def _collapse(seq: list[int]) -> list[int]:
            out: list[int] = []
            prev = -1
            for tok in seq:
                if tok != blank and tok != prev:
                    out.append(tok)
                prev = tok
            return out

        cands: list[str] = [self.tokenizer.decode(_collapse(base_ids))]
        seen = set(cands)

        topj_lp, topj_id = log_probs[:input_len].topk(alt_j, dim=-1)
        gap = (topj_lp[:, 0] - topj_lp[:, 1]).tolist()
        nonblank_frames = [
            t for t in range(input_len) if base_ids[t] != blank
        ]
        nonblank_frames.sort(key=lambda t: gap[t])

        for t in nonblank_frames[:k]:
            for j in range(1, alt_j):
                alt_id = int(topj_id[t, j].item())
                if alt_id == base_ids[t] or alt_id == blank:
                    continue
                swapped = list(base_ids)
                swapped[t] = alt_id
                s = self.tokenizer.decode(_collapse(swapped))
                if s and s not in seen:
                    seen.add(s)
                    cands.append(s)
                if len(cands) >= top_k:
                    return cands
        return cands

    @staticmethod
    def _merge_with_overlap(
        pieces: list[str],
        max_overlap: int,
    ) -> str:
        """Stitch sequentially-decoded chunks by longest suffix/prefix match.

        Greedy: for each new piece, find the longest suffix of the current
        result that equals the prefix of the new piece, up to
        ``max_overlap`` characters, then append only the non-overlapping
        tail. A match of zero length just concatenates (same as naive
        non-overlapping chunking).

        The algorithm is O(sum(len) * max_overlap). For our sizes this is
        negligible compared to the per-chunk decode.
        """
        if not pieces:
            return ""
        merged = pieces[0]
        for nxt in pieces[1:]:
            if not nxt:
                continue
            upper = min(len(merged), len(nxt), max_overlap)
            match = 0
            # Longest L such that merged endswith nxt[:L].
            for L in range(upper, 0, -1):
                if merged.endswith(nxt[:L]):
                    match = L
                    break
            merged += nxt[match:]
        return merged

    def convert(self, reading: str, context: str) -> list[str]:
        # Short inputs or no-chunk mode: single-shot decode.
        if self.chunk_threshold <= 0 or len(reading) < self.chunk_threshold:
            return self._decode_one(reading, context)

        # Chunk mode: top-1 per chunk (diversity via top-K currently only for
        # single-shot decode; chunk-mode merging across K hypotheses would be
        # combinatorial — not worth it for the long-form path).
        pieces: list[str] = []
        rolling_ctx = context
        start = 0
        while start < len(reading):
            chunk = reading[start : start + self.chunk_size]
            cands = self._decode_one(chunk, rolling_ctx, top_k=1)
            piece = cands[0] if cands else ""
            pieces.append(piece)
            rolling_ctx = (rolling_ctx + piece)[-self.max_context :]
            if start + self.chunk_size >= len(reading):
                break
            start += self.chunk_stride
        merged = self._merge_with_overlap(pieces, self.chunk_max_overlap)
        return [merged] if merged else []
