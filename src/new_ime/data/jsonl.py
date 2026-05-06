"""JSONL kana-kanji corpus loader for v2.5 training.

This is the primary training data path. Pre-tokenized binary `.kkc` shards
(`data/shards.py`) are no longer used by the training loop; the shard
infrastructure is retained for tools that benchmark the legacy mmap path
but is decoupled from training.

JSONL schema:
    {"reading": "...", "surface": "...", "context": "...",
     "writer_id"?: int, "domain_id"?: int, "source_id"?: int}

Two load modes:
    - max_samples == 0 → eager full load into RAM (small files only)
    - max_samples > 0  → Algorithm-R reservoir sampling, RAM bounded at
      max_samples regardless of source size
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from new_ime.data.tokenizer import BLANK_ID, CLS_ID, PAD_ID, SEP_ID, SharedCharTokenizer


class KanaKanjiJsonlDataset(Dataset):
    """Map-style Dataset over a JSONL kana-kanji corpus.

    With `max_samples=0` the entire file is loaded into RAM. With
    `max_samples>0` the dataset uses Algorithm R reservoir sampling so
    RAM is bounded irrespective of source-file size.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: SharedCharTokenizer,
        *,
        max_context_chars: int = 32,
        max_samples: int = 0,
        seed: int = 0,
    ):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.path}")
        self.tokenizer = tokenizer
        self.max_context_chars = int(max_context_chars)

        self._rows: list[dict] = self._load(int(max_samples), int(seed))

    def _load(self, max_samples: int, seed: int) -> list[dict]:
        with self.path.open("r", encoding="utf-8") as fh:
            if max_samples <= 0:
                return [json.loads(line) for line in fh if line.strip()]

            rng = np.random.default_rng(seed)
            reservoir: list[dict] = []
            i = 0
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if i < max_samples:
                    reservoir.append(row)
                else:
                    j = int(rng.integers(0, i + 1))
                    if j < max_samples:
                        reservoir[j] = row
                i += 1
            return reservoir

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        # Two corpus schemas coexist in our mixes:
        #   legacy pool (zenz / fineweb2 / hplt / wiki / aozora):
        #     `context` field carries already-converted kanji text.
        #   bunsetsu pool (wikibooks / wiktionary / wikinews / aozora_dialogue
        #                  / tatoeba):
        #     `left_context_surface` carries the kanji prefix instead.
        # The original reader only honoured `context`, silently dropping
        # ~10 % of context-bearing training rows. Fall back to
        # left_context_surface when `context` is empty.
        context = row.get("context") or row.get("left_context_surface") or ""
        context = context[-self.max_context_chars :]
        reading = row.get("reading") or ""
        surface = row.get("surface") or ""
        return {
            "reading_ids": np.asarray(self.tokenizer.encode(reading), dtype="<u4"),
            "surface_ids": np.asarray(self.tokenizer.encode(surface), dtype="<u4"),
            "context_ids": np.asarray(self.tokenizer.encode(context), dtype="<u4"),
            "writer_id": int(row.get("writer_id", 0)),
            "domain_id": int(row.get("domain_id", 0)),
            "source_id": int(row.get("source_id", 0)),
            # Raw strings preserved for KD teacher (text round-trip).
            "context_text": context,
            "reading_text": reading,
            "surface_text": surface,
        }


class JsonlCollator:
    """Pad token arrays into CTC-shaped tensor batches.

    Input layout: [CLS] + context_ids + [SEP] + reading_ids, truncated
    to `max_seq_len`. Target layout: surface_ids truncated to
    `max_seq_len`, with BLANK_ID stripped.

    Raw text fields (`context_text`, `reading_text`, `surface_text`) are
    aggregated into list[str] so KD teachers can text-roundtrip without
    decoding back from token ids.
    """

    PAD_ID = PAD_ID
    SEP_ID = SEP_ID
    CLS_ID = CLS_ID
    BLANK_ID = BLANK_ID

    def __init__(self, max_seq_len: int = 128):
        self.max_seq_len = int(max_seq_len)

    def _encode_input(
        self, context_ids: np.ndarray, reading_ids: np.ndarray
    ) -> list[int]:
        ids = [self.CLS_ID]
        ids.extend(int(x) for x in context_ids.tolist())
        ids.append(self.SEP_ID)
        ids.extend(int(x) for x in reading_ids.tolist())
        return ids[: self.max_seq_len]

    def _encode_target(self, surface_ids: np.ndarray) -> list[int]:
        return [
            int(x)
            for x in surface_ids[: self.max_seq_len].tolist()
            if int(x) != self.BLANK_ID
        ]

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        encoded_inputs: list[list[int]] = []
        encoded_targets: list[list[int]] = []
        target_lengths: list[int] = []
        writer_ids: list[int] = []
        domain_ids: list[int] = []
        source_ids: list[int] = []
        context_texts: list[str] = []
        reading_texts: list[str] = []
        surface_texts: list[str] = []

        for sample in batch:
            inp = self._encode_input(sample["context_ids"], sample["reading_ids"])
            tgt = self._encode_target(sample["surface_ids"])
            if len(tgt) == 0:
                tgt = [self.PAD_ID]
            encoded_inputs.append(inp)
            encoded_targets.append(tgt)
            target_lengths.append(len(tgt))
            writer_ids.append(sample["writer_id"])
            domain_ids.append(sample["domain_id"])
            source_ids.append(sample["source_id"])
            context_texts.append(sample.get("context_text", ""))
            reading_texts.append(sample.get("reading_text", ""))
            surface_texts.append(sample.get("surface_text", ""))

        max_input_len = max(len(x) for x in encoded_inputs)
        max_target_len = max(target_lengths)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        target_ids: list[list[int]] = []
        for inp, tgt in zip(encoded_inputs, encoded_targets, strict=True):
            input_pad = max_input_len - len(inp)
            target_pad = max_target_len - len(tgt)
            input_ids.append(inp + [self.PAD_ID] * input_pad)
            attention_mask.append([1] * len(inp) + [0] * input_pad)
            target_ids.append(tgt + [self.PAD_ID] * target_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "writer_ids": torch.tensor(writer_ids, dtype=torch.long),
            "domain_ids": torch.tensor(domain_ids, dtype=torch.long),
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "context_texts": context_texts,
            "reading_texts": reading_texts,
            "surface_texts": surface_texts,
        }


__all__ = ["KanaKanjiJsonlDataset", "JsonlCollator"]
