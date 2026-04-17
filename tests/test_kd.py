"""Tests for online KD (src/training/kd.py)."""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
import pytest

from src.data.dataset import ARCollator
from src.data.tokenizer import BLANK_ID, SharedCharTokenizer
from src.training.kd import (
    ARTeacher,
    KDConfig,
    TeacherConfig,
    compute_kd_ctc_loss,
    encode_texts_for_student,
    hard_example_mask,
)
from src.training.train_ar import SimpleGPT2


VOCAB_SIZE = 20
HIDDEN = 32


def _build_tiny_teacher(tmp_path, device: torch.device) -> ARTeacher:
    """Construct a small SimpleGPT2-based teacher with a deterministic vocab."""
    collator = ARCollator(max_seq_len=32)
    # Populate a minimal character vocab so decode_ids works.
    for ch in "あいうえおかきくけこさしすせそ":
        collator.encode_text(ch)

    model = SimpleGPT2(
        vocab_size=collator.vocab_size,
        hidden_size=HIDDEN,
        num_layers=2,
        num_heads=4,
        max_positions=32,
    )
    ckpt_path = tmp_path / "tiny_teacher.pt"
    vocab_path = tmp_path / "tiny_teacher_vocab.json"
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": collator.vocab_size}, ckpt_path)
    vocab_path.write_text(
        json.dumps(collator._char_to_id, ensure_ascii=False), encoding="utf-8"
    )

    config = TeacherConfig(
        checkpoint_path=str(ckpt_path),
        vocab_path=str(vocab_path),
        hidden_size=HIDDEN,
        num_layers=2,
        num_heads=4,
        max_seq_len=32,
        max_new_tokens=8,
        fp16=False,
    )
    return ARTeacher.from_checkpoint(config, device=device)


class TestKDConfig:
    def test_alpha_at_returns_zero_before_start(self):
        cfg = KDConfig(alpha=0.3, start_step=1000, warmup_steps=0)
        assert cfg.alpha_at(500) == 0.0
        assert cfg.alpha_at(1000) == pytest.approx(0.3)

    def test_alpha_at_linear_warmup(self):
        cfg = KDConfig(alpha=0.4, start_step=100, warmup_steps=100)
        assert cfg.alpha_at(100) == pytest.approx(0.0)
        assert cfg.alpha_at(150) == pytest.approx(0.2)
        assert cfg.alpha_at(200) == pytest.approx(0.4)
        assert cfg.alpha_at(10_000) == pytest.approx(0.4)

    def test_active_respects_alpha_zero(self):
        cfg = KDConfig(alpha=0.0)
        assert not cfg.active(1000)

    def test_active_respects_every(self):
        cfg = KDConfig(alpha=0.3, start_step=0, every=4)
        assert cfg.active(0)
        assert not cfg.active(1)
        assert cfg.active(4)


class TestHardExampleMask:
    def test_threshold_boundary(self):
        confs = torch.tensor([0.1, 0.5, 0.6, 0.9])
        mask = hard_example_mask(confs, threshold=0.6)
        assert mask.tolist() == [True, True, False, False]

    def test_empty(self):
        mask = hard_example_mask(torch.empty(0), threshold=0.6)
        assert mask.numel() == 0


class TestEncodeTextsForStudent:
    def test_roundtrip(self):
        tok = SharedCharTokenizer(max_kanji=200)
        texts = ["あい", "うえお", ""]
        ids, lengths = encode_texts_for_student(texts, tok, max_len=16)
        assert ids.shape[0] == 3
        assert lengths.tolist() == [2, 3, 0]

    def test_drops_blank_ids(self):
        tok = SharedCharTokenizer(max_kanji=50)
        ids, lengths = encode_texts_for_student(["ああ"], tok, max_len=8)
        # encoded IDs must not contain BLANK_ID
        for row_idx, length in enumerate(lengths.tolist()):
            row = ids[row_idx, :length].tolist()
            assert BLANK_ID not in row


class TestComputeKDCTCLoss:
    def _make_inputs(self, batch=4, time=12, vocab=50):
        logits = torch.randn(batch, time, vocab)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((batch,), time, dtype=torch.long)
        target_lengths = torch.tensor([3, 4, 5, 2], dtype=torch.long)
        targets = torch.randint(6, vocab, (batch, int(target_lengths.max())))
        return log_probs, input_lengths, targets, target_lengths

    def test_zero_when_mask_empty(self):
        log_probs, input_lengths, targets, target_lengths = self._make_inputs()
        mask = torch.zeros(4, dtype=torch.bool)
        loss, n = compute_kd_ctc_loss(
            log_probs, input_lengths, targets, target_lengths, mask
        )
        assert n == 0
        assert loss.item() == 0.0

    def test_nonzero_when_hard(self):
        log_probs, input_lengths, targets, target_lengths = self._make_inputs()
        mask = torch.tensor([True, True, False, False])
        loss, n = compute_kd_ctc_loss(
            log_probs, input_lengths, targets, target_lengths, mask
        )
        assert n == 2
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_skips_zero_length_targets(self):
        log_probs, input_lengths, targets, target_lengths = self._make_inputs()
        target_lengths[0] = 0
        target_lengths[1] = 0
        mask = torch.tensor([True, True, False, False])
        loss, n = compute_kd_ctc_loss(
            log_probs, input_lengths, targets, target_lengths, mask
        )
        assert n == 0
        assert loss.item() == 0.0

    def test_skips_when_input_shorter_than_target(self):
        log_probs, input_lengths, targets, target_lengths = self._make_inputs(
            batch=2, time=4, vocab=50
        )
        target_lengths = torch.tensor([6, 6], dtype=torch.long)  # longer than inputs
        targets = torch.randint(6, 50, (2, 6))
        mask = torch.tensor([True, True])
        loss, n = compute_kd_ctc_loss(
            log_probs, input_lengths, targets, target_lengths, mask
        )
        assert n == 0
        assert loss.item() == 0.0


class TestARTeacher:
    def test_from_checkpoint_and_generate_shapes(self, tmp_path):
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        texts, confs = teacher.generate(
            contexts=["", ""],
            readings=["あい", "うえ"],
            max_new_tokens=6,
        )
        assert len(texts) == 2
        assert len(confs) == 2
        for c in confs:
            assert 0.0 <= c <= 1.0

    def test_teacher_params_frozen(self, tmp_path):
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        for p in teacher.model.parameters():
            assert not p.requires_grad

    def test_teacher_stays_eval_after_train_call(self, tmp_path):
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        teacher.train(True)
        assert not teacher.model.training

    def test_vocab_mismatch_raises(self, tmp_path):
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        ckpt = torch.load(teacher.config.checkpoint_path, map_location="cpu", weights_only=False)
        ckpt["vocab_size"] = teacher.collator.vocab_size + 10
        torch.save(ckpt, teacher.config.checkpoint_path)
        with pytest.raises(ValueError):
            ARTeacher.from_checkpoint(teacher.config, device=device)

    def test_empty_batch(self, tmp_path):
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        texts, confs = teacher.generate(contexts=[], readings=[])
        assert texts == []
        assert confs == []

    def test_unknown_chars_do_not_extend_vocab(self, tmp_path):
        """Regression: ARTeacher must map unknown chars to UNK without mutating vocab.

        Previously used collator.encode_text which side-effects the vocab; that
        produces IDs beyond the frozen teacher embedding range at runtime.
        """
        device = torch.device("cpu")
        teacher = _build_tiny_teacher(tmp_path, device)
        teacher_vocab_size_before = teacher.collator.vocab_size
        # "〒" and "漢" are not in the tiny test vocab.
        texts, confs = teacher.generate(
            contexts=["〒漢字"],
            readings=["あ漢〒い"],
            max_new_tokens=4,
        )
        assert teacher.collator.vocab_size == teacher_vocab_size_before
        # All prompt IDs must be < embedding size; generation must not crash.
        assert len(texts) == 1
        assert 0.0 <= confs[0] <= 1.0
