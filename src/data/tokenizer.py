"""Tokenizers for CTC-NAT kana-kanji conversion.

Input tokenizer: character-level hiragana/katakana (small vocab ~180)
Output tokenizer: character-level kanji-kana mixed (vocab ~6500)
"""

from __future__ import annotations

import json
from pathlib import Path

# Hiragana: U+3041 - U+3096
HIRAGANA_START = 0x3041
HIRAGANA_END = 0x3096

# Katakana: U+30A1 - U+30FA
KATAKANA_START = 0x30A1
KATAKANA_END = 0x30FA

# Special tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
SEP_TOKEN = "[SEP]"
CLS_TOKEN = "[CLS]"
BLANK_TOKEN = "[BLANK]"  # CTC blank
MASK_TOKEN = "[MASK]"  # Mask-CTC refinement

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, CLS_TOKEN, BLANK_TOKEN, MASK_TOKEN]

PAD_ID = 0
UNK_ID = 1
SEP_ID = 2
CLS_ID = 3
BLANK_ID = 4
MASK_ID = 5


class InputTokenizer:
    """Character-level tokenizer for kana input.

    Vocabulary: special tokens + hiragana + katakana + common punctuation.
    Designed for the encoder side (reading input).
    """

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        idx = 0
        # Special tokens
        for token in SPECIAL_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Hiragana
        for cp in range(HIRAGANA_START, HIRAGANA_END + 1):
            char = chr(cp)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Katakana
        for cp in range(KATAKANA_START, KATAKANA_END + 1):
            char = chr(cp)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Prolonged sound mark, voiced/semi-voiced marks
        for char in "ー・ヽヾゝゞ":
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # Common punctuation that may appear in readings
        for char in "、。！？「」（）":
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        self._vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(c, UNK_ID) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.id_to_token.get(i, UNK_TOKEN)
            for i in ids
            if i not in (PAD_ID, CLS_ID, SEP_ID, BLANK_ID, MASK_ID)
        )

    def encode_with_special(self, context: str, kana_input: str) -> list[int]:
        """Encode with context: [CLS] context [SEP] kana_input"""
        tokens = [CLS_ID]
        tokens.extend(self.encode(context))
        tokens.append(SEP_ID)
        tokens.extend(self.encode(kana_input))
        return tokens

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "input",
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InputTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tokenizer._vocab_size = len(tokenizer.token_to_id)
        return tokenizer


class OutputTokenizer:
    """Character-level tokenizer for kanji-kana mixed output.

    Vocabulary: special tokens + JIS X 0208 kanji + hiragana + katakana +
    ASCII + common symbols. ~6500 tokens.
    Includes byte fallback for rare characters.
    """

    # JIS X 0208 Level 1 kanji: U+4E00 - U+9FFF (CJK Unified Ideographs)
    # We select the most common ~4000 kanji + all kana + ASCII + symbols
    # Full vocab is built from frequency data; this is the static fallback

    NUM_BYTE_TOKENS = 256  # UTF-8 byte fallback tokens <0x00> - <0xFF>

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is not None:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
            self._vocab_size = len(vocab)
        else:
            self.token_to_id: dict[str, int] = {}
            self.id_to_token: dict[int, str] = {}
            self._build_default_vocab()

    def _build_default_vocab(self) -> None:
        idx = 0

        # Special tokens
        for token in SPECIAL_TOKENS:
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Byte fallback tokens <0x00> - <0xFF>
        for b in range(self.NUM_BYTE_TOKENS):
            token = f"<0x{b:02X}>"
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            idx += 1

        # Hiragana
        for cp in range(HIRAGANA_START, HIRAGANA_END + 1):
            char = chr(cp)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # Katakana
        for cp in range(KATAKANA_START, KATAKANA_END + 1):
            char = chr(cp)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        # ASCII printable (0x20 - 0x7E)
        for cp in range(0x20, 0x7F):
            char = chr(cp)
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # Fullwidth ASCII (Ａ-Ｚ, ａ-ｚ, ０-９)
        for cp in range(0xFF01, 0xFF5F):
            char = chr(cp)
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # Common Japanese punctuation and symbols
        for char in "、��！？「」『』（）【】〔〕｛｝〈〉《》・ー～…‥々〇〻ヶヵ":
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1

        # CJK Unified Ideographs - most common kanji
        # JIS X 0208 Level 1: U+4E00 to U+9FFF
        # We include all CJK unified ideographs in the basic block
        # This gives ~20k characters; in practice we'd prune by frequency
        # For now include the full range; actual vocab will be built from data
        for cp in range(0x4E00, 0x9FFF + 1):
            char = chr(cp)
            self.token_to_id[char] = idx
            self.id_to_token[idx] = char
            idx += 1

        self._vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char in self.token_to_id:
                ids.append(self.token_to_id[char])
            else:
                # Byte fallback: encode character as UTF-8 bytes
                for b in char.encode("utf-8"):
                    ids.append(self.token_to_id.get(f"<0x{b:02X}>", UNK_ID))
        return ids

    def decode(self, ids: list[int]) -> str:
        result: list[str] = []
        byte_buffer: list[int] = []

        for token_id in ids:
            if token_id in (PAD_ID, CLS_ID, SEP_ID, BLANK_ID, MASK_ID):
                continue

            token = self.id_to_token.get(token_id, UNK_TOKEN)

            if token.startswith("<0x") and token.endswith(">"):
                # Byte token - accumulate for UTF-8 decoding
                byte_val = int(token[3:5], 16)
                byte_buffer.append(byte_val)
            else:
                # Flush byte buffer if any
                if byte_buffer:
                    result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                    byte_buffer.clear()
                if token not in SPECIAL_TOKENS:
                    result.append(token)

        # Flush remaining bytes
        if byte_buffer:
            result.append(bytes(byte_buffer).decode("utf-8", errors="replace"))

        return "".join(result)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "output",
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> OutputTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=data["token_to_id"])

    @classmethod
    def from_frequency_file(cls, freq_path: str | Path, max_kanji: int = 4000) -> OutputTokenizer:
        """Build vocabulary from character frequency data.

        The frequency file should have one character per line, sorted by
        frequency (most frequent first). Only kanji are filtered by max_kanji;
        kana, ASCII, and symbols are always included.
        """
        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = {}
        tokenizer.id_to_token = {}
        idx = 0

        # Special tokens
        for token in SPECIAL_TOKENS:
            tokenizer.token_to_id[token] = idx
            tokenizer.id_to_token[idx] = token
            idx += 1

        # Byte fallback
        for b in range(cls.NUM_BYTE_TOKENS):
            token = f"<0x{b:02X}>"
            tokenizer.token_to_id[token] = idx
            tokenizer.id_to_token[idx] = token
            idx += 1

        # Read frequency file and add characters
        kanji_count = 0
        freq_data = Path(freq_path).read_text(encoding="utf-8")
        for line in freq_data.strip().split("\n"):
            char = line.strip().split()[0] if line.strip() else ""
            if not char or char in tokenizer.token_to_id:
                continue

            is_cjk = 0x4E00 <= ord(char) <= 0x9FFF
            if is_cjk:
                if kanji_count >= max_kanji:
                    continue
                kanji_count += 1

            tokenizer.token_to_id[char] = idx
            tokenizer.id_to_token[idx] = char
            idx += 1

        tokenizer._vocab_size = idx
        return tokenizer
