"""JSONL fallback Dataset for kana-kanji corpora.

Pre-tokenized binary `.kkc` shards (data/shards.py) are the primary path.
This module exists for ad-hoc experiments where producing a shard is
overhead — eval set sweeps, smoke tests on raw JSONL, debugging single
malformed lines. Hot training paths should always use the shard reader.

The JSONL schema matches the legacy KanaKanjiDataset:
    {"reading": "...", "surface": "...", "context": "...",
     "writer_id"?: int, "domain_id"?: int, "source_id"?: int}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from new_ime.data.shards import CTCShardCollator
from new_ime.data.tokenizer import SharedCharTokenizer


class KanaKanjiJsonlDataset(Dataset):
    """Map-style Dataset that reads one JSON object per line.

    Lines are read into RAM eagerly. JSONL fallback is reserved for small
    sets (eval, smoke, debug); for full training mixes always use shards.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: SharedCharTokenizer,
        max_context_chars: int = 32,
    ):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.path}")
        self.tokenizer = tokenizer
        self.max_context_chars = int(max_context_chars)
        with self.path.open("r", encoding="utf-8") as fh:
            self._rows = [json.loads(line) for line in fh if line.strip()]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        context = (row.get("context") or "")[-self.max_context_chars :]
        reading = row.get("reading") or ""
        surface = row.get("surface") or ""
        return {
            "reading_ids": np.asarray(self.tokenizer.encode(reading), dtype="<u4"),
            "surface_ids": np.asarray(self.tokenizer.encode(surface), dtype="<u4"),
            "context_ids": np.asarray(self.tokenizer.encode(context), dtype="<u4"),
            "writer_id": int(row.get("writer_id", 0)),
            "domain_id": int(row.get("domain_id", 0)),
            "source_id": int(row.get("source_id", 0)),
        }


__all__ = ["KanaKanjiJsonlDataset", "CTCShardCollator"]
