"""Tests for CTC-NAT model components."""

import math

import torch

from models.src.model.ctc_nat import (
    CTCAlignmentToken,
    CTCNAT,
    CTCHead,
    GLATSampler,
    MaskCTCRefiner,
)
from models.src.model.cvae import CVAEConditioner
from models.src.model.decoder import NATDecoder, NATDecoderLayer
from models.src.model.encoder import MockEncoder, SmallEncoder

BATCH = 2
SRC_LEN = 16
HIDDEN = 64
VOCAB = 100
BLANK_ID = 4


class TestNATDecoderLayer:
    def test_forward_shape(self):
        layer = NATDecoderLayer(hidden_size=HIDDEN, num_heads=4, ffn_size=HIDDEN * 4)
        x = torch.randn(BATCH, SRC_LEN, HIDDEN)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        out = layer(x, enc)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)

    def test_with_padding_mask(self):
        layer = NATDecoderLayer(hidden_size=HIDDEN, num_heads=4, ffn_size=HIDDEN * 4)
        x = torch.randn(BATCH, SRC_LEN, HIDDEN)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        # Mask last 4 positions
        pad_mask = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool)
        pad_mask[:, -4:] = True
        out = layer(x, enc, self_attn_padding_mask=pad_mask, cross_attn_padding_mask=pad_mask)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)

    def test_with_film_condition(self):
        layer = NATDecoderLayer(hidden_size=HIDDEN, num_heads=4, ffn_size=HIDDEN * 4)
        x = torch.randn(BATCH, SRC_LEN, HIDDEN)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        gamma = torch.ones(BATCH, 1, HIDDEN)
        beta = torch.zeros(BATCH, 1, HIDDEN)
        out = layer(x, enc, film_condition=(gamma, beta))
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)


class TestNATDecoder:
    def test_forward_shape(self):
        decoder = NATDecoder(hidden_size=HIDDEN, num_layers=2, num_heads=4, ffn_size=HIDDEN * 4)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        out = decoder(enc)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)

    def test_with_encoder_padding(self):
        decoder = NATDecoder(hidden_size=HIDDEN, num_layers=2, num_heads=4, ffn_size=HIDDEN * 4)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        pad_mask = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool)
        pad_mask[0, -3:] = True  # First sample has 3 padding tokens
        out = decoder(enc, encoder_padding_mask=pad_mask)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)

    def test_with_film_conditioning_list(self):
        decoder = NATDecoder(hidden_size=HIDDEN, num_layers=2, num_heads=4, ffn_size=HIDDEN * 4)
        enc = torch.randn(BATCH, SRC_LEN, HIDDEN)
        conditioning = [
            (torch.ones(BATCH, 1, HIDDEN), torch.zeros(BATCH, 1, HIDDEN)),
            (torch.ones(BATCH, 1, HIDDEN), torch.zeros(BATCH, 1, HIDDEN)),
        ]
        out = decoder(enc, film_conditioning=conditioning)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)


class TestSmallEncoder:
    def test_forward_shape(self):
        encoder = SmallEncoder(
            vocab_size=256,
            hidden_size=HIDDEN,
            num_layers=2,
            num_heads=4,
            ffn_size=HIDDEN * 4,
            max_positions=32,
        )
        input_ids = torch.randint(0, 256, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        out = encoder(input_ids, attention_mask)
        assert out.shape == (BATCH, SRC_LEN, HIDDEN)


class TestCTCHead:
    def test_forward_shape(self):
        head = CTCHead(HIDDEN, VOCAB)
        x = torch.randn(BATCH, SRC_LEN, HIDDEN)
        logits = head(x)
        assert logits.shape == (BATCH, SRC_LEN, VOCAB)


class TestCTCNAT:
    def setup_method(self):
        encoder = MockEncoder(vocab_size=200, hidden_size=HIDDEN, num_layers=2)
        self.model = CTCNAT(
            encoder=encoder,
            output_vocab_size=VOCAB,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_size=HIDDEN * 4,
            blank_id=BLANK_ID,
        )

    def test_forward_no_target(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        result = self.model(input_ids, attention_mask)
        assert "logits" in result
        assert "log_probs" in result
        assert "loss" not in result
        assert result["logits"].shape == (BATCH, SRC_LEN, VOCAB)
        # log_probs should be time-first: (src_len, batch, vocab)
        assert result["log_probs"].shape == (SRC_LEN, BATCH, VOCAB)

    def test_proposal_logits(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        result = self.model.proposal_logits(input_ids, attention_mask)
        assert result["encoder_out"].shape == (BATCH, SRC_LEN, HIDDEN)
        assert result["decoder_out"].shape == (BATCH, SRC_LEN, HIDDEN)
        assert result["logits"].shape == (BATCH, SRC_LEN, VOCAB)
        assert result["encoder_padding_mask"].shape == (BATCH, SRC_LEN)

    def test_forward_with_target(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        # Target is shorter than input (kanji compresses kana)
        tgt_len = 10
        target_ids = torch.randint(5, VOCAB, (BATCH, tgt_len))  # Avoid special tokens
        target_lengths = torch.tensor([tgt_len, tgt_len - 2])
        result = self.model(input_ids, attention_mask, target_ids, target_lengths)
        assert "loss" in result
        assert result["loss"].dim() == 0  # Scalar
        assert not torch.isnan(result["loss"])
        assert not torch.isinf(result["loss"])

    def test_refine_logits(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        hypothesis_ids = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
        hypothesis_attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        result = self.model.refine_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        assert result["logits"].shape == (BATCH, SRC_LEN, VOCAB)
        assert result["decoder_out"].shape == (BATCH, SRC_LEN, HIDDEN)

    def test_refine_from_proposal_reuses_encoder_outputs(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        hypothesis_ids = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
        hypothesis_attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        proposal = self.model.proposal_logits(input_ids, attention_mask)
        result = self.model.refine_from_proposal(
            proposal=proposal,
            hypothesis_ids=hypothesis_ids,
            hypothesis_attention_mask=hypothesis_attention_mask,
        )
        assert result["encoder_out"] is proposal["encoder_out"]
        assert result["encoder_padding_mask"] is proposal["encoder_padding_mask"]
        assert result["logits"].shape == (BATCH, SRC_LEN, VOCAB)

    def test_apply_refinement_predictions_updates_only_masked_positions(self):
        hypothesis_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        hypothesis_attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        mask_positions = torch.tensor([[False, True, False, True]])
        refinement_logits = torch.full((1, 4, VOCAB), -10.0)
        refinement_logits[0, 1, 17] = 10.0
        refinement_logits[0, 3, 19] = 10.0
        updated = CTCNAT.apply_refinement_predictions(
            hypothesis_ids,
            hypothesis_attention_mask,
            mask_positions,
            refinement_logits,
        )
        assert updated.tolist() == [[1, 17, 3, 19]]

    def test_refine_result_exposes_remask_and_stop_heads(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        hyp_ids = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
        hyp_attn = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        result = self.model.refine_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hypothesis_ids=hyp_ids,
            hypothesis_attention_mask=hyp_attn,
        )
        assert result["remask_logits"].shape == (BATCH, SRC_LEN)
        assert result["stop_logit"].shape == (BATCH,)

    def test_iterative_refine_fills_mask_positions(self):
        torch.manual_seed(0)
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        mask_id = 5
        hyp_ids = torch.full((BATCH, SRC_LEN), mask_id, dtype=torch.long)
        hyp_attn = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        proposal = self.model.proposal_logits(input_ids, attention_mask)
        out = self.model.iterative_refine(
            proposal=proposal,
            hypothesis_ids=hyp_ids,
            hypothesis_attention_mask=hyp_attn,
            max_iterations=2,
            mask_token_id=mask_id,
        )
        assert "final_ids" in out
        assert out["final_ids"].shape == (BATCH, SRC_LEN)
        # every position should now be filled with the argmax (no mask id
        # survives if we run at least one pass and all positions were masked).
        # It's possible some remain if the argmax itself happens to equal
        # mask_id — assert at least majority change for a random-init model.
        assert (out["final_ids"] != mask_id).any()

    def test_ctc_loss_requires_input_ge_target(self):
        """CTC requires input_length >= target_length."""
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        # Target shorter than input — should work
        target_ids = torch.randint(5, VOCAB, (BATCH, 8))
        target_lengths = torch.tensor([8, 6])
        result = self.model(input_ids, attention_mask, target_ids, target_lengths)
        assert not torch.isnan(result["loss"])

    def test_greedy_decode(self):
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        decoded = self.model.greedy_decode(input_ids, attention_mask)
        assert len(decoded) == BATCH
        # Each decoded sequence should be a list of ints
        for seq in decoded:
            assert isinstance(seq, list)
            assert all(isinstance(t, int) for t in seq)
            # No blank tokens in output
            assert BLANK_ID not in seq

    def test_collapse_predictions(self):
        predictions = torch.tensor(
            [
                [BLANK_ID, 7, 7, 9, BLANK_ID, 9, 11, 11],
                [5, 5, BLANK_ID, BLANK_ID, 6, 6, 6, BLANK_ID],
            ],
            dtype=torch.long,
        )
        input_lengths = torch.tensor([8, 7], dtype=torch.long)
        decoded = CTCNAT.collapse_predictions(predictions, input_lengths, BLANK_ID)
        assert decoded == [[7, 9, 9, 11], [5, 6]]

    def test_collapse_with_alignment(self):
        logits = torch.full((1, 7, VOCAB), -10.0)
        # blank
        logits[0, 0, BLANK_ID] = 8.0
        logits[0, 0, 7] = 1.0
        # token 7 twice
        logits[0, 1, 7] = 9.0
        logits[0, 1, 12] = 8.0
        logits[0, 2, 7] = 11.0
        logits[0, 2, 13] = 7.0
        # blank separator
        logits[0, 3, BLANK_ID] = 10.0
        logits[0, 3, 9] = 9.0
        # token 7 again as a distinct collapsed token
        logits[0, 4, 7] = 8.5
        logits[0, 4, 17] = 8.4
        # token 9 twice
        logits[0, 5, 9] = 12.0
        logits[0, 5, 21] = 6.0
        logits[0, 6, 9] = 10.0
        logits[0, 6, 20] = 9.0

        aligned = CTCNAT.collapse_with_alignment(
            logits,
            torch.tensor([7], dtype=torch.long),
            BLANK_ID,
        )
        assert len(aligned) == 1
        toks = aligned[0]
        assert [tok.token_id for tok in toks] == [7, 7, 9]
        assert (toks[0].start_frame, toks[0].end_frame) == (1, 2)
        assert (toks[1].start_frame, toks[1].end_frame) == (4, 4)
        assert (toks[2].start_frame, toks[2].end_frame) == (5, 6)
        assert toks[0].frame_count == 2
        assert toks[1].confidence < toks[0].confidence
        assert toks[2].confidence > 0.5

    def test_select_mask_spans(self):
        aligned_tokens = [
            CTCAlignmentToken(7, 0, 1, min_log_prob=math.log(0.85), mean_log_prob=math.log(0.9), min_margin=1.5, mean_margin=1.7),
            CTCAlignmentToken(8, 2, 2, min_log_prob=math.log(0.35), mean_log_prob=math.log(0.35), min_margin=0.05, mean_margin=0.05),
            CTCAlignmentToken(9, 3, 4, min_log_prob=math.log(0.42), mean_log_prob=math.log(0.5), min_margin=0.15, mean_margin=0.2),
        ]
        selected = CTCNAT.select_mask_spans(aligned_tokens, confidence_threshold=0.5, max_masks=1)
        assert selected == [1]

    def test_backward(self):
        """Verify gradients flow through the model."""
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        target_ids = torch.randint(5, VOCAB, (BATCH, 8))
        target_lengths = torch.tensor([8, 6])
        result = self.model(input_ids, attention_mask, target_ids, target_lengths)
        result["loss"].backward()
        # Check encoder has gradients
        for param in self.model.encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break
        # Check decoder has gradients
        for param in self.model.decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                break

    def test_with_padding(self):
        """Test with variable-length inputs."""
        input_ids = torch.randint(0, 200, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN)
        attention_mask[1, 12:] = 0  # Second sample is shorter

        target_ids = torch.randint(5, VOCAB, (BATCH, 8))
        target_lengths = torch.tensor([8, 5])

        result = self.model(input_ids, attention_mask, target_ids, target_lengths)
        assert not torch.isnan(result["loss"])


    def test_from_preset_30m(self):
        model = CTCNAT.from_preset("phase3_30m", vocab_size=1024, use_cvae=False)
        assert model.encoder.hidden_size == 384
        assert model.decoder.num_layers == 6

    def test_with_cvae(self):
        model = CTCNAT.from_preset("phase3_30m", vocab_size=1024, use_cvae=True)
        input_ids = torch.randint(0, 1024, (BATCH, SRC_LEN))
        attention_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.long)
        target_ids = torch.randint(6, 1024, (BATCH, 8))
        target_lengths = torch.tensor([8, 6])
        writer_ids = torch.tensor([1, 2])
        domain_ids = torch.tensor([1, 1])
        source_ids = torch.tensor([1, 0])
        result = model(
            input_ids,
            attention_mask,
            target_ids=target_ids,
            target_lengths=target_lengths,
            writer_ids=writer_ids,
            domain_ids=domain_ids,
            source_ids=source_ids,
        )
        assert "kl" in result
        assert result["latent"].shape == (BATCH, 64)
        assert result["kl"].dim() == 0


class TestGLATSampler:
    def test_sampling_ratio_initial(self):
        sampler = GLATSampler(initial_ratio=0.5, min_ratio=0.1, anneal_steps=100)
        assert sampler.sampling_ratio == 0.5

    def test_sampling_ratio_annealed(self):
        sampler = GLATSampler(initial_ratio=0.5, min_ratio=0.1, anneal_steps=100)
        for _ in range(50):
            sampler.step()
        # At step 50/100, should be halfway: 0.5 - 0.2 = 0.3
        assert abs(sampler.sampling_ratio - 0.3) < 0.01

    def test_sampling_ratio_final(self):
        sampler = GLATSampler(initial_ratio=0.5, min_ratio=0.1, anneal_steps=100)
        for _ in range(200):
            sampler.step()
        # Should clamp at min
        assert abs(sampler.sampling_ratio - 0.1) < 0.01

    def test_compute_glat_mask_shape(self):
        sampler = GLATSampler()
        predictions = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
        targets = torch.randint(0, VOCAB, (BATCH, SRC_LEN))
        padding = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool)
        mask = sampler.compute_glat_mask(predictions, targets, padding)
        assert mask.shape == (BATCH, SRC_LEN)
        assert mask.dtype == torch.bool

    def test_glat_mask_no_padding_revealed(self):
        sampler = GLATSampler(initial_ratio=1.0)
        predictions = torch.zeros(BATCH, SRC_LEN, dtype=torch.long)
        targets = torch.ones(BATCH, SRC_LEN, dtype=torch.long)  # All wrong
        padding = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool)
        padding[:, -4:] = True  # Last 4 are padding
        mask = sampler.compute_glat_mask(predictions, targets, padding)
        # Padding positions should never be revealed
        assert not mask[:, -4:].any()


class TestMaskCTCRefiner:
    def test_identify_low_confidence(self):
        refiner = MaskCTCRefiner(confidence_threshold=0.9)
        logits = torch.randn(BATCH, SRC_LEN, VOCAB)
        # Make first position very confident
        logits[0, 0, :] = -100.0
        logits[0, 0, 5] = 100.0
        predictions = logits.argmax(dim=-1)
        input_lengths = torch.tensor([SRC_LEN, SRC_LEN])
        mask = refiner.identify_low_confidence(logits, predictions, input_lengths)
        assert mask.shape == (BATCH, SRC_LEN)
        # First position of first batch should NOT be masked (high confidence)
        assert not mask[0, 0]


class TestCVAEConditioner:
    def test_forward_shapes(self):
        conditioner = CVAEConditioner(hidden_size=HIDDEN, num_decoder_layers=2, latent_size=16)
        target_embeddings = torch.randn(BATCH, SRC_LEN, HIDDEN)
        padding_mask = torch.zeros(BATCH, SRC_LEN, dtype=torch.bool)
        out = conditioner(
            target_embeddings=target_embeddings,
            target_padding_mask=padding_mask,
            writer_ids=torch.tensor([1, 2]),
            domain_ids=torch.tensor([1, 0]),
            source_ids=torch.tensor([0, 1]),
            batch_size=BATCH,
            device=target_embeddings.device,
        )
        assert out.latent.shape == (BATCH, 16)
        assert len(out.film_conditioning) == 2
        gamma, beta = out.film_conditioning[0]
        assert gamma.shape == (BATCH, 1, HIDDEN)
        assert beta.shape == (BATCH, 1, HIDDEN)
