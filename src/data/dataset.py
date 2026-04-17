"""PyTorch Dataset for kana-kanji conversion training.

Each sample: (input_ids, target_ids)
  input_ids:  [CLS] context [SEP] reading [EOS]  (for AR: causal LM)
  target_ids: surface text tokens

For AR baseline (Phase 2), we concatenate as:
  [CLS] context [SEP] reading [OUT] surface [EOS]
and train with causal LM loss on the surface portion.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class KanaKanjiDataset(Dataset):
    """Dataset for kana-kanji conversion from JSONL files.

    Each line: {"reading": "...", "surface": "...", "context": "..."}
    """

    def __init__(
        self,
        jsonl_path: str,
        max_samples: int = 0,
        max_seq_len: int = 256,
        seed: int = 42,
    ):
        self.data: list[dict] = []
        self.max_seq_len = max_seq_len

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        if max_samples and max_samples < len(self.data):
            rng = random.Random(seed)
            self.data = rng.sample(self.data, max_samples)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class ARCollator:
    """Collate function for autoregressive (causal LM) training.

    Builds sequences: context + [SEP] + reading + [OUT] + surface + [EOS]
    Labels: -100 for context+reading portion, surface token IDs for loss.

    Uses a simple character-level tokenizer shared between input and output
    (since AR model is decoder-only, single vocabulary).
    """

    # Special tokens for AR model
    PAD = 0
    SEP = 1
    OUT = 2  # Marks start of output (surface)
    EOS = 3
    UNK = 4
    VOCAB_OFFSET = 5

    def __init__(self, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = self.VOCAB_OFFSET

    def _get_char_id(self, char: str) -> int:
        if char not in self._char_to_id:
            self._char_to_id[char] = self._next_id
            self._id_to_char[self._next_id] = char
            self._next_id += 1
        return self._char_to_id[char]

    def encode_text(self, text: str) -> list[int]:
        return [self._get_char_id(c) for c in text]

    def decode_ids(self, ids: list[int]) -> str:
        return "".join(
            self._id_to_char.get(i, "?")
            for i in ids
            if i >= self.VOCAB_OFFSET
        )

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def save_vocab(self, path: str) -> None:
        import json as _json
        Path(path).write_text(
            _json.dumps(self._char_to_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_vocab(self, path: str) -> None:
        import json as _json
        self._char_to_id = _json.loads(Path(path).read_text(encoding="utf-8"))
        self._id_to_char = {v: k for k, v in self._char_to_id.items()}
        if self._char_to_id:
            self._next_id = max(self._char_to_id.values()) + 1
        else:
            self._next_id = self.VOCAB_OFFSET

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of samples into padded tensors.

        Returns:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) — -100 for non-target positions
            attention_mask: (batch, seq_len)
        """
        all_input_ids = []
        all_labels = []

        for sample in batch:
            context = sample.get("context", "")
            reading = sample["reading"]
            surface = sample["surface"]

            # Truncate context to fit
            max_context = 40
            context = context[-max_context:] if context else ""

            # Build sequence: context [SEP] reading [OUT] surface [EOS]
            ctx_ids = self.encode_text(context)
            read_ids = self.encode_text(reading)
            surf_ids = self.encode_text(surface)

            seq = ctx_ids + [self.SEP] + read_ids + [self.OUT] + surf_ids + [self.EOS]

            # Labels: -100 for context+reading+SEP+OUT, actual IDs for surface+EOS
            prefix_len = len(ctx_ids) + 1 + len(read_ids) + 1  # context + SEP + reading + OUT
            labels = [-100] * prefix_len + surf_ids + [self.EOS]

            # Truncate to max_seq_len
            seq = seq[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

            all_input_ids.append(seq)
            all_labels.append(labels)

        # Pad to max length in batch
        max_len = max(len(s) for s in all_input_ids)
        padded_ids = []
        padded_labels = []
        attention_masks = []

        for seq, lab in zip(all_input_ids, all_labels):
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [self.PAD] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
