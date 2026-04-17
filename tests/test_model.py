"""Tests for CTC-NAT model components."""

import torch

from src.model.ctc_nat import CTCNAT, CTCHead, GLATSampler, MaskCTCRefiner
from src.model.decoder import NATDecoder, NATDecoderLayer
from src.model.encoder import MockEncoder

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
