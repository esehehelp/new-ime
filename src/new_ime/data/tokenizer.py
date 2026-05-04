"""Tokenizers for CTC-NAT kana-kanji conversion.

InputTokenizer is kept for backward compatibility with the Phase 2 / early
Phase 3 experiments. SharedCharTokenizer is the new default for the research
prototype: encoder and decoder share one character vocabulary so left context
can contain kanji without falling back to UNK.
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

KANA_MARKS = "ー・ヽヾゝゞ"
READING_PUNCT = "、。！？「」（）"
JP_SYMBOLS = "　、。！？「」『』（）【】〔〕｛｝〈〉《》・ー～…‥々〇〻ヶヵ"
INVALID_BYTE_TOKEN = "〓"


def _init_vocab() -> tuple[dict[str, int], dict[int, str], int]:
    token_to_id: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    idx = 0
    for token in SPECIAL_TOKENS:
        token_to_id[token] = idx
        id_to_token[idx] = token
        idx += 1
    return token_to_id, id_to_token, idx


def _append_chars(
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    idx: int,
    chars,
) -> int:
    for char in chars:
        if char not in token_to_id:
            token_to_id[char] = idx
            id_to_token[idx] = char
            idx += 1
    return idx


class InputTokenizer:
    """Legacy character-level tokenizer for kana-only encoder input."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        self.token_to_id, self.id_to_token, idx = _init_vocab()

        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(HIRAGANA_START, HIRAGANA_END + 1)),
        )
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(KATAKANA_START, KATAKANA_END + 1)),
        )
        idx = _append_chars(self.token_to_id, self.id_to_token, idx, KANA_MARKS)
        idx = _append_chars(self.token_to_id, self.id_to_token, idx, READING_PUNCT)
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


class SharedCharTokenizer:
    """Unified character tokenizer for both encoder and decoder.

    The default vocabulary is intentionally capped to roughly the Phase 3
    target regime: specials + byte fallback + kana + ASCII/symbols + the first
    `max_kanji` CJK code points. For production training this should be rebuilt
    from corpus frequency, but the static fallback keeps ids stable for tests
    and local experiments.
    """

    NUM_BYTE_TOKENS = 256

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        max_kanji: int = 6000,
    ) -> None:
        self.max_kanji = max_kanji
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.sep_id = SEP_ID
        self.cls_id = CLS_ID
        self.blank_id = BLANK_ID
        self.mask_id = MASK_ID
        if vocab is not None:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
            self._vocab_size = len(vocab)
        else:
            self.token_to_id: dict[str, int] = {}
            self.id_to_token: dict[int, str] = {}
            self._build_default_vocab(max_kanji=max_kanji)

    def _build_default_vocab(self, max_kanji: int) -> None:
        self.token_to_id, self.id_to_token, idx = _init_vocab()

        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (f"<0x{b:02X}>" for b in range(self.NUM_BYTE_TOKENS)),
        )
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(HIRAGANA_START, HIRAGANA_END + 1)),
        )
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(KATAKANA_START, KATAKANA_END + 1)),
        )
        idx = _append_chars(self.token_to_id, self.id_to_token, idx, KANA_MARKS)
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(0x20, 0x7F)),
        )
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(0xFF01, 0xFF5F)),
        )
        idx = _append_chars(self.token_to_id, self.id_to_token, idx, JP_SYMBOLS)

        # Static fallback: contiguous kanji slice. Real training vocab should
        # prefer corpus frequency, but this keeps the shared tokenizer bounded.
        idx = _append_chars(
            self.token_to_id,
            self.id_to_token,
            idx,
            (chr(cp) for cp in range(0x4E00, 0x4E00 + max_kanji)),
        )

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
                for b in char.encode("utf-8"):
                    ids.append(self.token_to_id.get(f"<0x{b:02X}>", UNK_ID))
        return ids

    def count_byte_fallbacks(self, text: str) -> tuple[int, int]:
        """Return (fallback_chars, total_chars) for one string."""

        total_chars = len(text)
        fallback_chars = sum(1 for char in text if char not in self.token_to_id)
        return fallback_chars, total_chars

    def byte_fallback_ratio(self, texts: list[str]) -> float:
        fallback = 0
        total = 0
        for text in texts:
            part_fallback, part_total = self.count_byte_fallbacks(text)
            fallback += part_fallback
            total += part_total
        if total == 0:
            return 0.0
        return fallback / total

    def decode(self, ids: list[int]) -> str:
        result: list[str] = []
        byte_buffer: list[int] = []

        for token_id in ids:
            if token_id in (PAD_ID, CLS_ID, SEP_ID, BLANK_ID, MASK_ID):
                continue

            token = self.id_to_token.get(token_id, UNK_TOKEN)
            if token.startswith("<0x") and token.endswith(">"):
                byte_buffer.append(int(token[3:5], 16))
                continue

            if byte_buffer:
                result.append(self._flush_byte_buffer(byte_buffer))
                byte_buffer.clear()
            if token not in SPECIAL_TOKENS:
                result.append(token)

        if byte_buffer:
            result.append(self._flush_byte_buffer(byte_buffer))

        return "".join(result)

    @staticmethod
    def _flush_byte_buffer(byte_buffer: list[int]) -> str:
        """Decode byte fallback without ever emitting U+FFFD.

        CTC collapse can truncate UTF-8 byte runs during early training. Those
        sequences must still decode to valid Unicode text, even if semantically
        meaningless. We therefore replace invalid byte runs with a dedicated
        sentinel instead of the replacement character.
        """

        if not byte_buffer:
            return ""
        try:
            return bytes(byte_buffer).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return INVALID_BYTE_TOKEN

    def encode_with_special(self, context: str, text: str) -> list[int]:
        tokens = [CLS_ID]
        tokens.extend(self.encode(context))
        tokens.append(SEP_ID)
        tokens.extend(self.encode(text))
        return tokens

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "shared",
            "max_kanji": self.max_kanji,
            "token_to_id": self.token_to_id,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SharedCharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=data["token_to_id"], max_kanji=data.get("max_kanji", 6000))

    @classmethod
    def from_frequency_file(
        cls,
        freq_path: str | Path,
        max_kanji: int = 6000,
    ) -> SharedCharTokenizer:
        tokenizer = cls.__new__(cls)
        tokenizer.max_kanji = max_kanji
        tokenizer.pad_id = PAD_ID
        tokenizer.unk_id = UNK_ID
        tokenizer.sep_id = SEP_ID
        tokenizer.cls_id = CLS_ID
        tokenizer.blank_id = BLANK_ID
        tokenizer.mask_id = MASK_ID
        tokenizer.token_to_id, tokenizer.id_to_token, idx = _init_vocab()

        idx = _append_chars(
            tokenizer.token_to_id,
            tokenizer.id_to_token,
            idx,
            (f"<0x{b:02X}>" for b in range(cls.NUM_BYTE_TOKENS)),
        )
        idx = _append_chars(
            tokenizer.token_to_id,
            tokenizer.id_to_token,
            idx,
            (chr(cp) for cp in range(HIRAGANA_START, HIRAGANA_END + 1)),
        )
        idx = _append_chars(
            tokenizer.token_to_id,
            tokenizer.id_to_token,
            idx,
            (chr(cp) for cp in range(KATAKANA_START, KATAKANA_END + 1)),
        )
        idx = _append_chars(tokenizer.token_to_id, tokenizer.id_to_token, idx, KANA_MARKS)
        idx = _append_chars(
            tokenizer.token_to_id,
            tokenizer.id_to_token,
            idx,
            (chr(cp) for cp in range(0x20, 0x7F)),
        )
        idx = _append_chars(tokenizer.token_to_id, tokenizer.id_to_token, idx, JP_SYMBOLS)

        kanji_count = 0
        freq_data = Path(freq_path).read_text(encoding="utf-8")
        for line in freq_data.strip().split("\n"):
            char = line.strip().split()[0] if line.strip() else ""
            if not char or char in tokenizer.token_to_id:
                continue
            is_cjk = len(char) == 1 and 0x4E00 <= ord(char) <= 0x9FFF
            if is_cjk:
                if kanji_count >= max_kanji:
                    continue
                kanji_count += 1
            tokenizer.token_to_id[char] = idx
            tokenizer.id_to_token[idx] = char
            idx += 1

        tokenizer._vocab_size = idx
        return tokenizer


class OutputTokenizer(SharedCharTokenizer):
    """Backward-compatible tokenizer with the old broad default kanji coverage."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        max_kanji: int = (0x9FFF - 0x4E00 + 1),
    ) -> None:
        super().__init__(vocab=vocab, max_kanji=max_kanji)
