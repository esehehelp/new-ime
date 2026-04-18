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
  examples when we're asked to convert long-form inputs (e.g. eval_v3/dev
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
        if lm_path and (self.lm_alpha > 0.0 or self.lm_beta > 0.0):
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
    def _decode_one(self, reading: str, context: str) -> str:
        ids = self.tokenizer.encode_with_special(context[-self.max_context :], reading)
        ids = ids[: self.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        if self.beam_width <= 1 and self.lm_scorer is None:
            decoded = self.model.greedy_decode(input_ids, attention_mask)
            return self.tokenizer.decode(decoded[0])

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

        # Low-confidence gate: when greedy mean top-1 logp (over non-blank
        # emission positions) is above the threshold, skip the beam+LM pass.
        # Values are natural-log; gate=0.0 disables the gate, negative
        # thresholds enable it.
        if self.lm_scorer is not None and self.lm_gate_min_conf < 0.0:
            nonblank = top1_id != self.model.blank_id
            if nonblank.any():
                mean_conf = top1_lp[nonblank].mean().item()
            else:
                mean_conf = top1_lp.mean().item()
            if mean_conf >= self.lm_gate_min_conf:
                return _collapse_greedy()

        if self.beam_width <= 1:
            return _collapse_greedy()

        beam = prefix_beam_search(
            log_probs,
            blank_id=self.model.blank_id,
            beam_width=self.beam_width,
            top_k_per_step=self.beam_top_k,
            lm_scorer=self.lm_scorer,
            lm_alpha=self.lm_alpha,
            lm_beta=self.lm_beta,
        )
        if not beam:
            return ""
        tokens, _ = beam[0]
        return self.tokenizer.decode(tokens)

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
            text = self._decode_one(reading, context)
            return [text] if text else []

        # Chunk mode: slide a window of `chunk_size` over the reading by
        # `chunk_stride` characters. Each window is decoded independently
        # with the running surface as context, and overlapping outputs are
        # merged via the longest suffix/prefix match.
        pieces: list[str] = []
        rolling_ctx = context
        start = 0
        while start < len(reading):
            chunk = reading[start : start + self.chunk_size]
            piece = self._decode_one(chunk, rolling_ctx)
            pieces.append(piece)
            rolling_ctx = (rolling_ctx + piece)[-self.max_context :]
            if start + self.chunk_size >= len(reading):
                break
            start += self.chunk_stride
        merged = self._merge_with_overlap(pieces, self.chunk_max_overlap)
        return [merged] if merged else []
