"""CTC-NAT model family for the Phase 3 research prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_ime.model.cvae import CVAEConditioner
from new_ime.model.decoder import MaskCTCRefinementDecoder, NATDecoder
from new_ime.model.encoder import BertEncoder, MockEncoder, SmallEncoder


@dataclass(frozen=True)
class CTCNATPreset:
    name: str
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    ffn_size: int
    max_positions: int


@dataclass(frozen=True)
class CTCAlignmentToken:
    token_id: int
    start_frame: int
    end_frame: int
    min_log_prob: float
    mean_log_prob: float
    min_margin: float
    mean_margin: float

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def confidence(self) -> float:
        return math.exp(self.min_log_prob)


PRESETS: dict[str, CTCNATPreset] = {
    "phase3_20m": CTCNATPreset(
        name="phase3_20m",
        hidden_size=320,
        encoder_layers=5,
        decoder_layers=5,
        num_heads=4,
        ffn_size=1280,
        max_positions=128,
    ),
    "phase3_30m": CTCNATPreset(
        name="phase3_30m",
        hidden_size=384,
        encoder_layers=6,
        decoder_layers=6,
        num_heads=6,
        ffn_size=1536,
        max_positions=128,
    ),
    "phase3_90m": CTCNATPreset(
        name="phase3_90m",
        hidden_size=640,
        encoder_layers=8,
        decoder_layers=8,
        num_heads=8,
        ffn_size=2560,
        max_positions=128,
    ),
}


class CTCHead(nn.Module):
    """Projection from decoder hidden states to vocabulary logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tied_embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)
        if tied_embedding is not None:
            if tied_embedding.weight.shape == self.projection.weight.shape:
                self.projection.weight = tied_embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CTCNAT(nn.Module):
    """CTC-based non-autoregressive Transformer for kana-kanji conversion."""

    def __init__(
        self,
        encoder: BertEncoder | MockEncoder | SmallEncoder,
        output_vocab_size: int,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ffn_size: int = 3072,
        dropout: float = 0.1,
        blank_id: int = 4,
        max_positions: int = 128,
        use_cvae: bool = False,
        latent_size: int = 64,
        tie_output_projection: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.hidden_size

        self.decoder = NATDecoder(
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_size=decoder_ffn_size,
            dropout=dropout,
            max_positions=max_positions,
        )

        tied_embedding = None
        if tie_output_projection:
            candidate = encoder.get_input_embedding()
            if candidate.num_embeddings == output_vocab_size:
                tied_embedding = candidate
        self.refine_decoder = MaskCTCRefinementDecoder(
            vocab_size=output_vocab_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_size=decoder_ffn_size,
            dropout=dropout,
            max_positions=max_positions,
            embedding=tied_embedding,
        )
        self.ctc_head = CTCHead(hidden_size, output_vocab_size, tied_embedding=tied_embedding)
        self.refine_head = CTCHead(hidden_size, output_vocab_size, tied_embedding=tied_embedding)
        # Per-token binary head: "should this position be re-masked for the
        # next refinement iteration?" Positive label = refined argmax != target.
        self.remask_head = nn.Linear(hidden_size, 1)
        # Sequence-level scalar: "refinement converged (argmax matches target
        # everywhere inside valid span)." Pooled via mean over valid positions.
        self.stop_head = nn.Linear(hidden_size, 1)
        self.blank_id = blank_id
        self.output_vocab_size = output_vocab_size
        self.cvae = (
            CVAEConditioner(hidden_size, decoder_layers, latent_size=latent_size)
            if use_cvae
            else None
        )

    @staticmethod
    def collapse_predictions(
        predictions: torch.Tensor,
        input_lengths: torch.Tensor,
        blank_id: int,
    ) -> list[list[int]]:
        """Collapse frame-level CTC predictions into token sequences."""
        decoded: list[list[int]] = []
        for b in range(predictions.shape[0]):
            tokens: list[int] = []
            prev_token = -1
            limit = int(input_lengths[b].item())
            for t in range(limit):
                token = int(predictions[b, t].item())
                if token != blank_id and token != prev_token:
                    tokens.append(token)
                prev_token = token
            decoded.append(tokens)
        return decoded

    @staticmethod
    def collapse_with_alignment(
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        blank_id: int,
    ) -> list[list[CTCAlignmentToken]]:
        """Collapse CTC frames and keep token-level confidence/alignment stats."""
        log_probs = F.log_softmax(logits, dim=-1)
        top2_log_probs, top2_ids = log_probs.topk(k=min(2, logits.shape[-1]), dim=-1)
        top2_log_probs_cpu = top2_log_probs.detach().cpu()
        top2_ids_cpu = top2_ids.detach().cpu()
        input_lengths_cpu = input_lengths.detach().cpu()

        collapsed: list[list[CTCAlignmentToken]] = []
        for b in range(logits.shape[0]):
            batch_tokens: list[CTCAlignmentToken] = []
            limit = int(input_lengths_cpu[b].item())
            current_id: int | None = None
            start_frame = 0
            log_prob_sum = 0.0
            margin_sum = 0.0
            min_log_prob = float("inf")
            min_margin = float("inf")
            count = 0
            prev_token = -1

            def flush() -> None:
                nonlocal current_id, start_frame, log_prob_sum, margin_sum
                nonlocal min_log_prob, min_margin, count
                if current_id is None or count <= 0:
                    return
                batch_tokens.append(
                    CTCAlignmentToken(
                        token_id=current_id,
                        start_frame=start_frame,
                        end_frame=start_frame + count - 1,
                        min_log_prob=min_log_prob,
                        mean_log_prob=log_prob_sum / count,
                        min_margin=min_margin,
                        mean_margin=margin_sum / count,
                    )
                )
                current_id = None
                log_prob_sum = 0.0
                margin_sum = 0.0
                min_log_prob = float("inf")
                min_margin = float("inf")
                count = 0

            for t in range(limit):
                token = int(top2_ids_cpu[b, t, 0])
                top1_lp = float(top2_log_probs_cpu[b, t, 0])
                top2_lp = (
                    float(top2_log_probs_cpu[b, t, 1])
                    if top2_log_probs_cpu.shape[-1] > 1
                    else top1_lp
                )
                margin = top1_lp - top2_lp

                if token == blank_id:
                    flush()
                    prev_token = token
                    continue

                if token == prev_token:
                    if current_id is None:
                        current_id = token
                        start_frame = t
                    log_prob_sum += top1_lp
                    margin_sum += margin
                    min_log_prob = min(min_log_prob, top1_lp)
                    min_margin = min(min_margin, margin)
                    count += 1
                else:
                    flush()
                    current_id = token
                    start_frame = t
                    log_prob_sum = top1_lp
                    margin_sum = margin
                    min_log_prob = top1_lp
                    min_margin = margin
                    count = 1
                prev_token = token

            flush()
            collapsed.append(batch_tokens)
        return collapsed

    @staticmethod
    def collapse_alignment_tensors(
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        blank_id: int,
    ) -> dict[str, torch.Tensor]:
        """Vectorized CTC collapse that returns tensors instead of dataclasses.

        Functionally equivalent to `collapse_with_alignment` for the fields the
        training loop actually uses (token_id, min_log_prob, min_margin,
        per-row collapsed length) but runs entirely on the device with no
        per-frame Python loop. For the training hot path (batch 256 × seq 128)
        this is roughly 100× faster than the dataclass-returning version.

        Returns keys:
            token_ids   (B, Tmax) long — collapsed token ids, padded with
                blank_id on the right
            min_log_prob (B, Tmax) float — lowest top1 log-prob across the
                frames that map to each collapsed token (+inf for padding)
            min_margin   (B, Tmax) float — lowest (top1 - top2) margin across
                those frames (+inf for padding)
            lengths      (B,) long — number of collapsed tokens per row
        """
        log_probs = F.log_softmax(logits, dim=-1)
        top2_log_probs, top2_ids = log_probs.topk(k=min(2, logits.shape[-1]), dim=-1)
        top1_ids = top2_ids[..., 0]
        top1_lp = top2_log_probs[..., 0]
        if top2_log_probs.shape[-1] > 1:
            top2_lp = top2_log_probs[..., 1]
        else:
            top2_lp = top1_lp
        margin = top1_lp - top2_lp

        B, T = top1_ids.shape
        device = top1_ids.device

        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        valid = arange < input_lengths.to(device).unsqueeze(1)
        is_nonblank = (top1_ids != blank_id) & valid

        # A new collapsed token starts at frame t iff the frame is non-blank
        # AND (t is first, or prev frame was blank, or top1 differs from prev)
        prev_ids = torch.cat(
            [
                torch.full((B, 1), blank_id, device=device, dtype=top1_ids.dtype),
                top1_ids[:, :-1],
            ],
            dim=1,
        )
        prev_nonblank = torch.cat(
            [
                torch.zeros(B, 1, device=device, dtype=torch.bool),
                is_nonblank[:, :-1],
            ],
            dim=1,
        )
        new_token = is_nonblank & ((~prev_nonblank) | (top1_ids != prev_ids))

        # Per-frame collapsed-token index (0-based within row); -1 for blank/pad.
        cum = (new_token.long().cumsum(dim=1) - 1).clamp(min=0)
        token_idx = torch.where(
            is_nonblank, cum, torch.full_like(cum, -1)
        )

        lengths = new_token.sum(dim=1)
        Tmax = int(lengths.max().item()) if B > 0 else 0

        if Tmax == 0:
            return {
                "token_ids": torch.full((B, 0), blank_id, dtype=top1_ids.dtype, device=device),
                "min_log_prob": torch.full((B, 0), float("inf"), device=device),
                "min_margin": torch.full((B, 0), float("inf"), device=device),
                "lengths": lengths.long(),
            }

        # Flat indexing into (B * Tmax) for scatter_reduce.
        out_token_ids = torch.full(
            (B * Tmax,), blank_id, dtype=top1_ids.dtype, device=device
        )
        out_min_lp = torch.full((B * Tmax,), float("inf"), device=device)
        out_min_margin = torch.full((B * Tmax,), float("inf"), device=device)

        mask = is_nonblank
        if mask.any():
            flat_b = (
                torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
            )[mask]
            flat_t = token_idx[mask]
            linear_idx = flat_b * Tmax + flat_t

            # token_ids: any frame in a token has the same id; assign on
            # first frame (new_token True).
            first = new_token
            if first.any():
                first_b = (
                    torch.arange(B, device=device).unsqueeze(1).expand(-1, T)
                )[first]
                first_t = token_idx[first]
                first_linear = first_b * Tmax + first_t
                out_token_ids.index_copy_(
                    0, first_linear, top1_ids[first]
                )

            out_min_lp.scatter_reduce_(
                0, linear_idx, top1_lp[mask], reduce="amin", include_self=True
            )
            out_min_margin.scatter_reduce_(
                0, linear_idx, margin[mask], reduce="amin", include_self=True
            )

        return {
            "token_ids": out_token_ids.view(B, Tmax),
            "min_log_prob": out_min_lp.view(B, Tmax),
            "min_margin": out_min_margin.view(B, Tmax),
            "lengths": lengths.long(),
        }

    @staticmethod
    def select_mask_spans(
        aligned_tokens: list[CTCAlignmentToken],
        confidence_threshold: float,
        max_masks: int = 0,
    ) -> list[int]:
        """Return token indices to mask, ordered by lowest confidence first."""
        ranked = [
            (idx, tok.confidence, tok.min_margin)
            for idx, tok in enumerate(aligned_tokens)
            if tok.confidence < confidence_threshold
        ]
        ranked.sort(key=lambda item: (item[1], item[2], item[0]))
        if max_masks > 0:
            ranked = ranked[:max_masks]
        return [idx for idx, _conf, _margin in ranked]

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        vocab_size: int,
        dropout: float = 0.1,
        use_cvae: bool = False,
        blank_id: int = 4,
        max_positions: int | None = None,
    ) -> CTCNAT:
        preset = PRESETS[preset_name]
        pos = max_positions if max_positions is not None else preset.max_positions
        encoder = SmallEncoder(
            vocab_size=vocab_size,
            hidden_size=preset.hidden_size,
            num_layers=preset.encoder_layers,
            num_heads=preset.num_heads,
            ffn_size=preset.ffn_size,
            max_positions=pos,
            dropout=dropout,
        )
        return cls(
            encoder=encoder,
            output_vocab_size=vocab_size,
            decoder_layers=preset.decoder_layers,
            decoder_heads=preset.num_heads,
            decoder_ffn_size=preset.ffn_size,
            dropout=dropout,
            blank_id=blank_id,
            max_positions=pos,
            use_cvae=use_cvae,
        )

    def _build_cvae_output(
        self,
        batch_size: int,
        device: torch.device,
        target_ids: torch.Tensor | None,
        target_padding_mask: torch.Tensor | None,
        writer_ids: torch.Tensor | None,
        domain_ids: torch.Tensor | None,
        source_ids: torch.Tensor | None,
        sample_posterior: bool,
    ):
        if self.cvae is None:
            return None

        target_embeddings = None
        if target_ids is not None:
            target_embeddings = self.encoder.get_input_embedding()(target_ids)

        return self.cvae(
            target_embeddings=target_embeddings,
            target_padding_mask=target_padding_mask,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
            batch_size=batch_size,
            device=device,
            sample_posterior=sample_posterior,
        )

    def proposal_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        writer_ids: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run the shared encoder/proposal decoder path and return raw logits."""
        encoder_out = self.encoder(input_ids, attention_mask)
        encoder_padding_mask = ~attention_mask.bool()

        target_padding_mask = None
        if target_ids is not None and target_lengths is not None:
            seq_len = target_ids.shape[1]
            positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
            target_padding_mask = positions >= target_lengths.unsqueeze(1)

        cvae_output = self._build_cvae_output(
            batch_size=input_ids.shape[0],
            device=input_ids.device,
            target_ids=target_ids,
            target_padding_mask=target_padding_mask,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
            sample_posterior=sample_posterior,
        )

        decoder_out = self.decoder(
            encoder_out,
            encoder_padding_mask,
            film_conditioning=cvae_output.film_conditioning if cvae_output else None,
        )
        logits = self.ctc_head(decoder_out)
        result = {
            "encoder_out": encoder_out,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_out": decoder_out,
            "logits": logits,
            "film_conditioning": cvae_output.film_conditioning if cvae_output else None,
        }
        if cvae_output is not None:
            result["latent"] = cvae_output.latent
            result["kl"] = cvae_output.kl
            result["posterior_mean"] = cvae_output.mean
            result["posterior_logvar"] = cvae_output.logvar
        return result

    def refine_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hypothesis_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        cvae_target_ids: torch.Tensor | None = None,
        cvae_target_attention_mask: torch.Tensor | None = None,
        writer_ids: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        sample_posterior: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the dedicated refinement branch on a masked hypothesis."""
        cvae_target_lengths = None
        if cvae_target_attention_mask is not None:
            cvae_target_lengths = cvae_target_attention_mask.sum(dim=1).long()
        proposal = self.proposal_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=cvae_target_ids,
            target_lengths=cvae_target_lengths,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
            sample_posterior=sample_posterior,
        )
        return self.refine_from_proposal(
            proposal=proposal,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )

    def refine_from_proposal(
        self,
        proposal: dict[str, torch.Tensor],
        hypothesis_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run refinement using a previously computed proposal result."""
        decoder_out = self.refine_decoder(
            hypothesis_ids=hypothesis_ids,
            hypothesis_padding_mask=~hypothesis_attention_mask.bool(),
            encoder_out=proposal["encoder_out"],
            encoder_padding_mask=proposal["encoder_padding_mask"],
            film_conditioning=proposal.get("film_conditioning"),
        )
        logits = self.refine_head(decoder_out)
        remask_logits = self.remask_head(decoder_out).squeeze(-1)
        valid = hypothesis_attention_mask.bool()
        pooled = (decoder_out * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
            dim=1
        ).clamp(min=1).unsqueeze(-1)
        stop_logit = self.stop_head(pooled).squeeze(-1)
        result = {
            "encoder_out": proposal["encoder_out"],
            "encoder_padding_mask": proposal["encoder_padding_mask"],
            "decoder_out": decoder_out,
            "logits": logits,
            "remask_logits": remask_logits,
            "stop_logit": stop_logit,
        }
        if "latent" in proposal:
            result["latent"] = proposal["latent"]
            result["kl"] = proposal["kl"]
            result["posterior_mean"] = proposal["posterior_mean"]
            result["posterior_logvar"] = proposal["posterior_logvar"]
        return result

    @staticmethod
    def apply_refinement_predictions(
        hypothesis_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
        refinement_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Replace masked positions with refinement argmax predictions."""
        updated = hypothesis_ids.clone()
        predicted = refinement_logits.argmax(dim=-1)
        valid_mask = mask_positions.bool() & hypothesis_attention_mask.bool()
        updated[valid_mask] = predicted[valid_mask]
        return updated

    @torch.no_grad()
    def iterative_refine(
        self,
        proposal: dict[str, torch.Tensor],
        hypothesis_ids: torch.Tensor,
        hypothesis_attention_mask: torch.Tensor,
        max_iterations: int = 2,
        stop_threshold: float = 0.5,
        remask_threshold: float = 0.5,
        confidence_fallback: float = 0.5,
        use_learned_remask: bool = True,
        use_learned_stop: bool = True,
        mask_token_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run up to `max_iterations` refinement passes on a cached proposal.

        At each iteration: run refine_from_proposal, argmax-fill the current
        mask positions, then decide which tokens (if any) should be remasked
        for the next round. Halts early when the stop head says "converged"
        for every row, or when no position meets the remask criterion.
        """
        valid = hypothesis_attention_mask.bool()
        current_ids = hypothesis_ids.clone()
        if mask_token_id is None:
            mask_token_id = getattr(self, "_mask_token_id", None)
        last_result: dict[str, torch.Tensor] | None = None
        done = torch.zeros(current_ids.shape[0], dtype=torch.bool, device=current_ids.device)
        for it in range(max_iterations):
            refine_result = self.refine_from_proposal(
                proposal=proposal,
                hypothesis_ids=current_ids,
                hypothesis_attention_mask=hypothesis_attention_mask,
            )
            last_result = refine_result
            logits = refine_result["logits"]
            argmax = logits.argmax(dim=-1)
            # Fill in mask-token positions with the argmax prediction; keep
            # non-mask positions untouched.
            if mask_token_id is not None:
                mask_here = (current_ids == mask_token_id) & valid
                current_ids = torch.where(mask_here, argmax, current_ids)
            else:
                current_ids = torch.where(valid, argmax, current_ids)

            if use_learned_stop:
                stop_prob = torch.sigmoid(refine_result["stop_logit"])
                done = done | (stop_prob >= stop_threshold)
                if bool(done.all().item()):
                    break

            if it == max_iterations - 1:
                break

            if use_learned_remask:
                remask_prob = torch.sigmoid(refine_result["remask_logits"])
                next_mask = (remask_prob >= remask_threshold) & valid & ~done.unsqueeze(1)
            else:
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values
                next_mask = (max_probs < confidence_fallback) & valid & ~done.unsqueeze(1)

            if not bool(next_mask.any().item()):
                break

            if mask_token_id is not None:
                current_ids = torch.where(
                    next_mask,
                    torch.full_like(current_ids, mask_token_id),
                    current_ids,
                )

        assert last_result is not None
        last_result["final_ids"] = current_ids
        last_result["stopped_rows"] = done
        return last_result

    def set_mask_token(self, mask_token_id: int) -> None:
        """Record the tokenizer-defined mask id so iterative_refine can
        re-apply it when it decides to mask a position for the next pass."""
        self._mask_token_id = int(mask_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        writer_ids: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        sample_posterior: bool = True,
    ) -> dict[str, torch.Tensor]:
        result = self.proposal_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_ids=target_ids,
            target_lengths=target_lengths,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
            sample_posterior=sample_posterior,
        )
        logits = result["logits"]
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        result["log_probs"] = log_probs

        if target_ids is not None and target_lengths is not None:
            input_lengths = attention_mask.sum(dim=1).long()
            loss = F.ctc_loss(
                log_probs=log_probs,
                targets=target_ids,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def greedy_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        writer_ids: torch.Tensor | None = None,
        domain_ids: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:
        result = self.forward(
            input_ids,
            attention_mask,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
            sample_posterior=False,
        )
        predictions = result["logits"].argmax(dim=-1)
        input_lengths = attention_mask.sum(dim=1).long()
        return self.collapse_predictions(predictions, input_lengths, self.blank_id)


class GLATSampler:
    """Glancing Language Model Training (GLAT) sampling."""

    def __init__(
        self,
        initial_ratio: float = 0.5,
        min_ratio: float = 0.1,
        anneal_steps: int = 200_000,
    ):
        self.initial_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.anneal_steps = anneal_steps
        self.current_step = 0

    @property
    def sampling_ratio(self) -> float:
        if self.anneal_steps <= 0:
            return self.initial_ratio
        progress = min(self.current_step / self.anneal_steps, 1.0)
        return self.initial_ratio - (self.initial_ratio - self.min_ratio) * progress

    def step(self) -> None:
        self.current_step += 1

    def compute_glat_mask(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = ~target_padding_mask
        correct = (predictions == targets) & valid_mask
        target_lengths = valid_mask.sum(dim=1).float()
        correct_count = correct.sum(dim=1).float()
        error_fraction = (target_lengths - correct_count) / target_lengths.clamp(min=1)
        reveal_fraction = error_fraction * self.sampling_ratio

        rand = torch.rand_like(targets.float())
        rand = rand.masked_fill(target_padding_mask, 1.0)
        return rand < reveal_fraction.unsqueeze(1)


class MaskCTCRefiner(nn.Module):
    """Legacy confidence heuristic retained for compatibility and ablations."""

    def __init__(self, mask_id: int = 5, confidence_threshold: float = 0.5):
        super().__init__()
        self.mask_id = mask_id
        self.confidence_threshold = confidence_threshold

    def identify_low_confidence(
        self,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values
        low_conf = max_probs < self.confidence_threshold
        for b in range(logits.shape[0]):
            low_conf[b, input_lengths[b] :] = False
        return low_conf
