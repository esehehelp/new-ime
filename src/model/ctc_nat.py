"""CTC-NAT: Main model combining encoder, decoder, and CTC head.

Architecture:
    [context + kana input] → Encoder (BERT) → Decoder (NAT) → CTC Head → output

CTC loss handles:
    - Variable-length output (no explicit length prediction needed)
    - Monotonic alignment between input and output
    - Blank tokens absorb length differences

Reference:
    NMLA-NAT CTC criterion — examples/speech_recognition/criterions/CTC_loss.py
    fairseq NATransformerModel — fairseq/models/nat/nonautoregressive_transformer.py:44-205
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.decoder import NATDecoder
from src.model.encoder import BertEncoder, MockEncoder


class CTCHead(nn.Module):
    """Linear projection from hidden states to vocabulary logits.

    Includes the CTC blank token at index 0.
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        # +1 for CTC blank token (index 0 in CTC convention)
        # But our tokenizer already includes BLANK at index 4.
        # We output over the full vocab (which includes BLANK).
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            Logits: (batch, seq_len, vocab_size)
        """
        return self.projection(x)


class CTCNAT(nn.Module):
    """CTC-based Non-Autoregressive Transformer for kana-kanji conversion.

    Training:
        1. Encode [context + kana] → hidden states
        2. Decode (parallel) → features
        3. Project → logits
        4. CTC loss against target surface text

    Inference:
        1-3 same as training
        4. CTC greedy/beam decode → candidates
    """

    def __init__(
        self,
        encoder: BertEncoder | MockEncoder,
        output_vocab_size: int,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ffn_size: int = 3072,
        dropout: float = 0.1,
        blank_id: int = 4,  # BLANK_ID from tokenizer
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
        )

        self.ctc_head = CTCHead(hidden_size, output_vocab_size)
        self.blank_id = blank_id
        self.output_vocab_size = output_vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (batch, src_len) encoder input token IDs
            attention_mask: (batch, src_len) 1=valid, 0=pad
            target_ids: (batch, tgt_len) target token IDs (for training)
            target_lengths: (batch,) actual length of each target (for CTC loss)

        Returns:
            dict with:
                "logits": (batch, src_len, vocab_size)
                "log_probs": (src_len, batch, vocab_size) — CTC format (time-first)
                "loss": scalar CTC loss (only if target_ids provided)
        """
        # Encode
        encoder_out = self.encoder(input_ids, attention_mask)

        # Decode (parallel)
        encoder_padding_mask = ~attention_mask.bool()  # True=pad for decoder
        decoder_out = self.decoder(encoder_out, encoder_padding_mask)

        # Project to vocab
        logits = self.ctc_head(decoder_out)  # (batch, src_len, vocab_size)

        # CTC needs log-probs in (time, batch, vocab) format
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (src_len, batch, vocab)

        result = {
            "logits": logits,
            "log_probs": log_probs,
        }

        # Compute CTC loss if targets provided
        if target_ids is not None and target_lengths is not None:
            # Input lengths: actual (non-padded) length of encoder output
            # = sum of attention_mask per batch element
            input_lengths = attention_mask.sum(dim=1).long()  # (batch,)

            # CTC loss
            # Reference: NMLA-NAT CTC_loss.py line 141
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
    ) -> list[list[int]]:
        """CTC greedy decoding: argmax → collapse blanks and repeats.

        Args:
            input_ids: (batch, src_len)
            attention_mask: (batch, src_len)

        Returns:
            List of decoded token ID sequences (one per batch element).
        """
        result = self.forward(input_ids, attention_mask)
        logits = result["logits"]  # (batch, src_len, vocab_size)

        # Argmax
        predictions = logits.argmax(dim=-1)  # (batch, src_len)
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
    """Glancing Language Model Training (GLAT) sampling.

    During training:
    1. First forward pass → get model's current predictions
    2. Compare predictions to reference → compute Hamming distance
    3. Reveal a fraction of reference tokens to the decoder
    4. Train on the remaining (unrevealed) positions

    The fraction revealed = (hamming_distance / length) * sampling_ratio
    This creates an automatic curriculum: easy→hard as the model improves.

    Reference:
        DA-Transformer nat_dag_loss.py lines 239-321
        GLAT paper (Qian et al. 2021)
    """

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
        """Current GLAT sampling ratio, linearly annealed."""
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
        """Compute which target positions to reveal to the decoder.

        Reference: DA-Transformer nat_dag_loss.py glat_function

        Args:
            predictions: (batch, seq_len) model's predicted token IDs
            targets: (batch, seq_len) reference token IDs
            target_padding_mask: (batch, seq_len) True=pad

        Returns:
            reveal_mask: (batch, seq_len) True=reveal this position to decoder
        """
        # Hamming distance: count positions where prediction != target
        valid_mask = ~target_padding_mask
        correct = (predictions == targets) & valid_mask
        target_lengths = valid_mask.sum(dim=1).float()  # (batch,)
        correct_count = correct.sum(dim=1).float()  # (batch,)

        # Fraction to reveal = (errors / length) * sampling_ratio
        # More errors → reveal more (easier task)
        error_fraction = (target_lengths - correct_count) / target_lengths.clamp(min=1)
        reveal_fraction = error_fraction * self.sampling_ratio  # (batch,)

        # Sample positions to reveal
        rand = torch.rand_like(targets.float())
        rand = rand.masked_fill(target_padding_mask, 1.0)  # Never reveal padding

        # Reveal positions where rand < reveal_fraction
        reveal_mask = rand < reveal_fraction.unsqueeze(1)

        return reveal_mask


class MaskCTCRefiner(nn.Module):
    """Mask-CTC refinement: re-predict low-confidence CTC output positions.

    After CTC greedy/beam decode:
    1. Identify positions with low confidence (below threshold)
    2. Replace those positions with [MASK]
    3. Run decoder again to re-predict masked positions
    4. Combine with original predictions

    This is a single refinement pass (not iterative).

    Reference:
        Higuchi et al. 2021 — Mask-CTC
        fairseq cmlm_transformer.py _skeptical_unmasking (line 18-24)
    """

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
        """Find positions where the model is not confident.

        Args:
            logits: (batch, seq_len, vocab_size)
            predictions: (batch, seq_len) argmax token IDs
            input_lengths: (batch,) valid lengths

        Returns:
            mask: (batch, seq_len) True = should be re-predicted
        """
        # Softmax confidence
        probs = torch.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # (batch, seq_len)

        # Mark low-confidence positions
        low_conf = max_probs < self.confidence_threshold

        # Don't mask padding
        for b in range(logits.shape[0]):
            low_conf[b, input_lengths[b]:] = False

        return low_conf
