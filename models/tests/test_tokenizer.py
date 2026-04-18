"""Tests for input/output tokenizers."""

import tempfile
from pathlib import Path

from src.data.tokenizer import (
    BLANK_ID,
    CLS_ID,
    INVALID_BYTE_TOKEN,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    UNK_ID,
    InputTokenizer,
    OutputTokenizer,
    SharedCharTokenizer,
)


class TestInputTokenizer:
    def setup_method(self):
        self.tok = InputTokenizer()

    def test_vocab_size(self):
        # Special(6) + Hiragana(86) + Katakana(90) + marks + punctuation
        assert self.tok.vocab_size > 180

    def test_special_token_ids(self):
        assert self.tok.token_to_id["[PAD]"] == PAD_ID
        assert self.tok.token_to_id["[UNK]"] == UNK_ID
        assert self.tok.token_to_id["[SEP]"] == SEP_ID
        assert self.tok.token_to_id["[CLS]"] == CLS_ID

    def test_encode_hiragana(self):
        ids = self.tok.encode("かんじ")
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)
        # All should be valid (not UNK)
        assert UNK_ID not in ids

    def test_encode_katakana(self):
        ids = self.tok.encode("カンジ")
        assert len(ids) == 3
        assert UNK_ID not in ids

    def test_encode_unknown(self):
        # Kanji is not in input vocab
        ids = self.tok.encode("漢")
        assert ids == [UNK_ID]

    def test_decode_roundtrip(self):
        text = "こんにちは"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_encode_with_special(self):
        ids = self.tok.encode_with_special("まえのぶん", "かんじ")
        assert ids[0] == CLS_ID
        # Find SEP
        sep_pos = ids.index(SEP_ID)
        assert sep_pos == 6  # CLS + 5 context chars
        assert len(ids) == 6 + 1 + 3  # CLS + context + SEP + input

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input_tokenizer.json"
            self.tok.save(path)
            loaded = InputTokenizer.load(path)
            assert loaded.vocab_size == self.tok.vocab_size
            text = "てすと"
            assert loaded.encode(text) == self.tok.encode(text)


class TestOutputTokenizer:
    def setup_method(self):
        self.tok = OutputTokenizer()

    def test_vocab_size(self):
        # Should include kanji, kana, ASCII, symbols, byte fallback
        assert self.tok.vocab_size > 6000

    def test_special_tokens(self):
        assert self.tok.token_to_id["[BLANK]"] == BLANK_ID
        assert self.tok.token_to_id["[MASK]"] == MASK_ID

    def test_encode_kanji(self):
        ids = self.tok.encode("漢字")
        assert len(ids) == 2
        assert UNK_ID not in ids

    def test_encode_mixed(self):
        ids = self.tok.encode("漢字かな混じり")
        assert len(ids) == 7
        assert UNK_ID not in ids

    def test_decode_roundtrip(self):
        text = "東京都渋谷区"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_decode_skips_special_tokens(self):
        ids = [PAD_ID, CLS_ID, BLANK_ID, MASK_ID]
        ids.extend(self.tok.encode("漢字"))
        decoded = self.tok.decode(ids)
        assert decoded == "漢字"

    def test_byte_fallback_encode(self):
        # Emoji is not in default vocab, should fall back to bytes
        text = "\U0001f600"  # 😀
        ids = self.tok.encode(text)
        # UTF-8 encoding of 😀 is 4 bytes: F0 9F 98 80
        assert len(ids) == 4

    def test_byte_fallback_roundtrip(self):
        text = "テスト😀です"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        assert decoded == text

    def test_invalid_byte_sequence_never_emits_replacement_char(self):
        ids = [
            self.tok.token_to_id["<0xE3>"],
            self.tok.token_to_id["<0x81>"],
        ]
        decoded = self.tok.decode(ids)
        assert "\ufffd" not in decoded
        assert decoded == INVALID_BYTE_TOKEN

    def test_ascii(self):
        ids = self.tok.encode("Hello")
        assert len(ids) == 5
        assert UNK_ID not in ids

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output_tokenizer.json"
            self.tok.save(path)
            loaded = OutputTokenizer.load(path)
            assert loaded.vocab_size == self.tok.vocab_size
            text = "漢字テスト"
            assert loaded.encode(text) == self.tok.encode(text)

    def test_ctc_blank_in_decode(self):
        """CTC blank tokens should be stripped during decode."""
        kanji_ids = self.tok.encode("変換")
        # Simulate CTC output with blanks between characters
        ctc_output = [BLANK_ID, kanji_ids[0], BLANK_ID, kanji_ids[1], BLANK_ID]
        decoded = self.tok.decode(ctc_output)
        assert decoded == "変換"


class TestSharedCharTokenizer:
    def setup_method(self):
        self.tok = SharedCharTokenizer(max_kanji=4000)

    def test_special_token_ids_stable(self):
        assert self.tok.token_to_id["[PAD]"] == PAD_ID
        assert self.tok.token_to_id["[SEP]"] == SEP_ID
        assert self.tok.token_to_id["[BLANK]"] == BLANK_ID
        assert self.tok.token_to_id["[MASK]"] == MASK_ID

    def test_context_can_contain_kanji(self):
        ids = self.tok.encode_with_special("東京都", "かな")
        assert ids[0] == CLS_ID
        assert SEP_ID in ids
        assert UNK_ID not in ids[:4]

    def test_vocab_is_phase3_sized(self):
        assert 4000 < self.tok.vocab_size < 9000

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "shared_tokenizer.json"
            self.tok.save(path)
            loaded = SharedCharTokenizer.load(path)
            text = "漢字かな😀"
            assert loaded.encode(text) == self.tok.encode(text)
            assert loaded.decode(self.tok.encode(text)) == text

    def test_invalid_byte_sequence_uses_safe_sentinel(self):
        ids = [
            self.tok.token_to_id["<0xE3>"],
            self.tok.token_to_id["<0x81>"],
            self.tok.token_to_id["A"],
        ]
        decoded = self.tok.decode(ids)
        assert "\ufffd" not in decoded
        assert decoded == INVALID_BYTE_TOKEN + "A"

    def test_fullwidth_space_is_direct_token(self):
        ids = self.tok.encode("\u3000")
        assert len(ids) == 1

    def test_byte_fallback_ratio_counts_unknown_chars(self):
        ratio = self.tok.byte_fallback_ratio(["漢字", "😀"])
        assert 0.0 < ratio < 1.0
