"""CTC-NAT model family for the Phase 3 research prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cvae import CVAEConditioner
from src.model.decoder import NATDecoder
from src.model.encoder import BertEncoder, MockEncoder, SmallEncoder


@dataclass(frozen=True)
class CTCNATPreset:
    name: str
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    ffn_size: int
    max_positions: int


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
        blank_logit_bias: float = 0.0,
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
        self.ctc_head = CTCHead(hidden_size, output_vocab_size, tied_embedding=tied_embedding)
        self.blank_id = blank_id
        self.output_vocab_size = output_vocab_size
        self.blank_logit_bias = float(blank_logit_bias)
        self.cvae = (
            CVAEConditioner(hidden_size, decoder_layers, latent_size=latent_size)
            if use_cvae
            else None
        )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        vocab_size: int,
        dropout: float = 0.1,
        use_cvae: bool = False,
        blank_id: int = 4,
        blank_logit_bias: float = 0.0,
    ) -> CTCNAT:
        preset = PRESETS[preset_name]
        encoder = SmallEncoder(
            vocab_size=vocab_size,
            hidden_size=preset.hidden_size,
            num_layers=preset.encoder_layers,
            num_heads=preset.num_heads,
            ffn_size=preset.ffn_size,
            max_positions=preset.max_positions,
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
            max_positions=preset.max_positions,
            use_cvae=use_cvae,
            blank_logit_bias=blank_logit_bias,
        )

    def _apply_blank_logit_bias(self, logits: torch.Tensor) -> torch.Tensor:
        if self.blank_logit_bias <= 0.0:
            return logits
        logits = logits.clone()
        logits[..., self.blank_id] = logits[..., self.blank_id] - self.blank_logit_bias
        return logits

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
        logits = self._apply_blank_logit_bias(logits)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        result = {
            "logits": logits,
            "log_probs": log_probs,
        }

        if cvae_output is not None:
            result["latent"] = cvae_output.latent
            result["kl"] = cvae_output.kl
            result["posterior_mean"] = cvae_output.mean
            result["posterior_logvar"] = cvae_output.logvar

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

        decoded = []
        for b in range(predictions.shape[0]):
            tokens = []
            prev_token = -1
            for t in range(input_lengths[b]):
                token = predictions[b, t].item()
                if token != self.blank_id and token != prev_token:
                    tokens.append(token)
                prev_token = token
            decoded.append(tokens)
        return decoded


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
    """Single-pass Mask-CTC refinement."""

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
