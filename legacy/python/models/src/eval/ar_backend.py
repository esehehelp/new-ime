"""ConversionBackend for our own AR (decoder-only) checkpoints.

Reuses the architecture and beam-search logic from manual_test_beam.py.
Supports greedy and beam search modes.
"""

from __future__ import annotations

import torch

from models.src.data.dataset import ARCollator
from models.src.eval.fast_gen import FastARGenerator, fast_beam, fast_greedy
from models.src.eval.run_eval import ConversionBackend
from models.src.training.train_ar import SimpleGPT2


class ARCheckpointBackend(ConversionBackend):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        beam_width: int = 1,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.2,
        name: str | None = None,
    ) -> None:
        self.collator = ARCollator()
        self.collator.load_vocab(checkpoint_path.replace(".pt", "_vocab.json"))

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        hidden = state["embed_tokens.weight"].shape[1]
        max_pos = state["embed_positions.weight"].shape[0]
        n_layers = sum(
            1 for k in state if k.endswith(".self_attn.in_proj_weight")
        )
        self.model = SimpleGPT2(
            vocab_size=self.collator.vocab_size,
            hidden_size=hidden,
            num_layers=n_layers,
            num_heads=8,
            max_positions=max_pos,
        )
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.max_pos = max_pos
        self.step = ckpt["step"]
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.fast = FastARGenerator(self.model, self.device, max_pos)

        ckpt_id = checkpoint_path.split("/")[-2] + "/" + checkpoint_path.split("/")[-1]
        mode = "greedy" if beam_width <= 1 else f"beam{beam_width}"
        self._name = name or f"own({ckpt_id}@step{self.step},{mode})"

    @property
    def name(self) -> str:
        return self._name

    def _build_prefix(self, reading: str, context: str) -> list[int]:
        ctx_ids = self.collator.encode_text(context[-40:]) if context else []
        read_ids = self.collator.encode_text(reading)
        return ctx_ids + [self.collator.SEP] + read_ids + [self.collator.OUT]

    @torch.no_grad()
    def _greedy(self, prefix: list[int], max_new: int) -> str:
        prefix = prefix[-(self.max_pos - 2) :]
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        out: list[int] = []
        for _ in range(max_new):
            if ids.shape[1] >= self.max_pos:
                break
            mask = torch.ones_like(ids)
            logits = self.model(ids, mask)
            nid = logits[0, -1].argmax().item()
            if nid == self.collator.EOS or nid == self.collator.PAD:
                break
            out.append(nid)
            ids = torch.cat([ids, torch.tensor([[nid]], device=self.device)], dim=1)
        return self.collator.decode_ids(out)

    @torch.no_grad()
    def _beam(self, prefix: list[int], max_new: int) -> list[str]:
        """Batched beam search.

        At every step all active beams have identical length, so we can stack
        them into one (B, T) tensor and run a single forward pass — eliminating
        the per-beam Python loop that dominated cost in the naive version.
        """
        prefix = prefix[-(self.max_pos - 2) :]
        prefix_len = len(prefix)
        bw = self.beam_width
        rep = self.repetition_penalty
        lp = self.length_penalty

        # active_seqs[i]: token list, active_scores[i]: cumulative log prob.
        active_seqs: list[list[int]] = [prefix[:]]
        active_scores: list[float] = [0.0]
        finished: list[tuple[list[int], float]] = []

        for _ in range(max_new):
            # Build (B, T) batch from active beams (all same length here).
            cur_len = len(active_seqs[0])
            if cur_len >= self.max_pos:
                for seq, sc in zip(active_seqs, active_scores):
                    finished.append((seq, sc))
                break
            batch = torch.tensor(active_seqs, dtype=torch.long, device=self.device)
            mask = torch.ones_like(batch)
            logits = self.model(batch, mask)  # (B, T, V)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (B, V)

            if rep != 1.0:
                # Penalize tokens already produced in each beam's generated tail.
                for bi, seq in enumerate(active_seqs):
                    gen_tokens = seq[prefix_len:]
                    if not gen_tokens:
                        continue
                    idx = torch.tensor(list(set(gen_tokens)), device=self.device)
                    log_probs[bi].index_copy_(
                        0, idx, log_probs[bi].index_select(0, idx) / rep
                    )

            # Per-beam top-K candidates, then global pruning.
            topk = torch.topk(log_probs, k=min(bw * 2, log_probs.shape[1]), dim=-1)
            cand_tokens = topk.indices.tolist()
            cand_logp = topk.values.tolist()

            new_candidates: list[tuple[list[int], float]] = []
            for bi, (seq, sc) in enumerate(zip(active_seqs, active_scores)):
                for tok, lp_val in zip(cand_tokens[bi], cand_logp[bi]):
                    new_score = sc + lp_val
                    if tok == self.collator.EOS or tok == self.collator.PAD:
                        finished.append((seq, new_score))
                    else:
                        new_candidates.append((seq + [tok], new_score))

            if not new_candidates:
                break

            def norm(item: tuple[list[int], float]) -> float:
                s, sc = item
                return sc / max(len(s) - prefix_len, 1) ** lp

            new_candidates.sort(key=norm, reverse=True)
            kept = new_candidates[:bw]
            active_seqs = [s for s, _ in kept]
            active_scores = [sc for _, sc in kept]

        all_results = finished + list(zip(active_seqs, active_scores))

        def fnorm(item: tuple[list[int], float]) -> float:
            s, sc = item
            return sc / max(len(s) - prefix_len, 1) ** lp

        all_results.sort(key=fnorm, reverse=True)
        out_texts: list[str] = []
        seen_texts: set[str] = set()
        for seq, _ in all_results[:bw]:
            text = self.collator.decode_ids(seq[prefix_len:])
            if text and text not in seen_texts:
                seen_texts.add(text)
                out_texts.append(text)
        return out_texts

    def convert(self, reading: str, context: str) -> list[str]:
        prefix = self._build_prefix(reading, context)
        max_new = len(reading) + 20
        if self.beam_width <= 1:
            ids = fast_greedy(
                self.fast, prefix, max_new,
                eos_id=self.collator.EOS, pad_id=self.collator.PAD,
            )
            text = self.collator.decode_ids(ids)
            return [text] if text else []
        seqs = fast_beam(
            self.fast, prefix, max_new,
            eos_id=self.collator.EOS, pad_id=self.collator.PAD,
            beam_width=self.beam_width,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
        )
        out: list[str] = []
        seen: set[str] = set()
        for ids in seqs:
            t = self.collator.decode_ids(ids)
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out
