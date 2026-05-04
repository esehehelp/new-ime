"""mmap reader for `.kkc` shards produced by `cargo run -p rust-data -- compile`.

KKCSHRD1 V2 layout (frozen, see crates/rust-data/src/shard.rs):
    Header (36 bytes, little-endian):
        magic[8] = b"KKCSHRD1"
        version u32, row_count u64, payload_offset u64, index_offset u64
    Payload (per row, V2):
        reading_len u32, surface_len u32, context_len u32,
        writer_id u32, domain_id u32, source_id u32,
        reading[reading_len] u32,
        surface[surface_len] u32,
        context[context_len] u32
    Index: row_count u64, each entry = row byte offset within payload region.

Shard mode pairs with `CTCShardCollator` to emit padded tensor batches with
zero per-step Python tokenize / JSON parse on the dataloader hot path.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from new_ime.data.tokenizer import BLANK_ID, CLS_ID, PAD_ID, SEP_ID

_SHARD_MAGIC = b"KKCSHRD1"
_SHARD_VERSION_MIN = 1
_SHARD_VERSION_MAX = 2
_SHARD_HEADER_LEN = 36


class KanaKanjiShardDataset(Dataset):
    """Map-style Dataset over a `rust-data` compiled shard.

    `expected_vocab_size` (when set) validates the sidecar `*.meta.json`
    so a mismatched tokenizer / shard pair fails loudly at startup.
    """

    def __init__(
        self,
        shard_path: str | Path,
        expected_vocab_size: int | None = None,
    ):
        self.path = Path(shard_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Shard not found: {self.path}")

        self._fh = None
        self._mm: mmap.mmap | None = None
        self._open_mmap()

        if bytes(self._mm[:8]) != _SHARD_MAGIC:
            raise ValueError(f"Not a rust-data shard (bad magic): {self.path}")
        version, row_count, payload_offset, index_offset = struct.unpack_from(
            "<IQQQ", self._mm, 8
        )
        if not (_SHARD_VERSION_MIN <= version <= _SHARD_VERSION_MAX):
            raise ValueError(
                f"Unsupported shard version {version} "
                f"(supported {_SHARD_VERSION_MIN}..{_SHARD_VERSION_MAX})"
            )
        self.version = int(version)
        self.row_count = int(row_count)
        self.payload_offset = int(payload_offset)
        self.index_offset = int(index_offset)

        index_bytes = self._mm[
            self.index_offset : self.index_offset + self.row_count * 8
        ]
        self.index = np.frombuffer(index_bytes, dtype="<u8")
        if self.index.shape[0] != self.row_count:
            raise ValueError(
                f"Shard index length {self.index.shape[0]} != row_count {self.row_count}"
            )

        self.meta: dict | None = None
        meta_path = Path(str(self.path) + ".meta.json")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                expected_vocab_size is not None
                and self.meta.get("vocab_size") != expected_vocab_size
            ):
                raise ValueError(
                    "Shard / tokenizer vocab mismatch: "
                    f"shard={self.meta.get('vocab_size')} "
                    f"tokenizer={expected_vocab_size} (meta={meta_path})"
                )

    def _open_mmap(self) -> None:
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.row_count

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_mm"] = None
        state["index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_mmap()
        index_bytes = self._mm[
            self.index_offset : self.index_offset + self.row_count * 8
        ]
        self.index = np.frombuffer(index_bytes, dtype="<u8")

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.row_count:
            raise IndexError(idx)
        base = self.payload_offset + int(self.index[idx])
        reading_len, surface_len, context_len = struct.unpack_from(
            "<III", self._mm, base
        )
        if self.version == 1:
            writer_id = 0
            domain_id = 0
            (source_id,) = struct.unpack_from("<I", self._mm, base + 12)
            cur = base + 16
        else:
            writer_id, domain_id, source_id = struct.unpack_from(
                "<III", self._mm, base + 12
            )
            cur = base + 24
        rbytes = reading_len * 4
        sbytes = surface_len * 4
        cbytes = context_len * 4
        reading_ids = np.frombuffer(self._mm[cur : cur + rbytes], dtype="<u4")
        cur += rbytes
        surface_ids = np.frombuffer(self._mm[cur : cur + sbytes], dtype="<u4")
        cur += sbytes
        context_ids = np.frombuffer(self._mm[cur : cur + cbytes], dtype="<u4")
        return {
            "reading_ids": reading_ids,
            "surface_ids": surface_ids,
            "context_ids": context_ids,
            "writer_id": int(writer_id),
            "domain_id": int(domain_id),
            "source_id": int(source_id),
        }


class KanaKanjiShardIterable(IterableDataset):
    """Block-shuffled streaming view.

    Random per-row shuffling on a multi-GiB mmap touches fresh 4 KiB pages
    every sample and on Windows pushes the working set into the pagefile.
    Iterating by block (block order shuffled, rows inside sequential)
    keeps page access sequential while preserving enough randomness for SGD.
    """

    def __init__(
        self,
        shard_path: str | Path,
        *,
        block_size: int = 1024,
        shuffle: bool = True,
        seed: int = 0,
        expected_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self._base = KanaKanjiShardDataset(
            shard_path, expected_vocab_size=expected_vocab_size
        )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.path = self._base.path
        self.meta = self._base.meta
        self.row_count = self._base.row_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._base.row_count

    def __iter__(self):
        total = self._base.row_count
        n_blocks = (total + self.block_size - 1) // self.block_size

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, nw = 0, 1
        else:
            wid, nw = worker_info.id, worker_info.num_workers

        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(n_blocks, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(order)
        my_blocks = order[wid::nw]

        for b in my_blocks:
            start = int(b) * self.block_size
            stop = min(start + self.block_size, total)
            for idx in range(start, stop):
                yield self._base[idx]


class CTCShardCollator:
    """Pad u32 arrays from the shard into CTC-shaped tensor batches.

    Input layout (matches legacy CTCCollator._encode_input):
        [CLS_ID] + context_ids + [SEP_ID] + reading_ids
    truncated to `max_seq_len`. context_ids are already capped at
    `max_context_chars` during shard compile.

    Target layout: surface_ids truncated to `max_seq_len`. BLANK_ID is
    stripped (V2 strips at compile; we filter defensively for V1).

    `short_sample_max_chars > 0` filters rows whose reading_ids or
    surface_ids exceeds the cap (curriculum warmup).
    """

    PAD_ID = PAD_ID
    SEP_ID = SEP_ID
    CLS_ID = CLS_ID
    BLANK_ID = BLANK_ID

    def __init__(
        self,
        max_seq_len: int = 128,
        short_sample_max_chars: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.short_sample_max_chars = short_sample_max_chars

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

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        if self.short_sample_max_chars > 0:
            cap = self.short_sample_max_chars
            filtered = [
                s
                for s in batch
                if len(s["reading_ids"]) <= cap and len(s["surface_ids"]) <= cap
            ]
            if filtered:
                batch = filtered

        encoded_inputs: list[list[int]] = []
        encoded_targets: list[list[int]] = []
        target_lengths: list[int] = []
        writer_ids: list[int] = []
        domain_ids: list[int] = []
        source_ids: list[int] = []

        for sample in batch:
            inp = self._encode_input(sample["context_ids"], sample["reading_ids"])
            tgt = self._encode_target(sample["surface_ids"])
            encoded_inputs.append(inp)
            encoded_targets.append(tgt)
            target_lengths.append(len(tgt))
            writer_ids.append(sample["writer_id"])
            domain_ids.append(sample["domain_id"])
            source_ids.append(sample["source_id"])

        max_input_len = max(len(x) for x in encoded_inputs)
        max_target_len = max(max(target_lengths), 1)

        input_ids = []
        attention_mask = []
        target_ids = []
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
        }


def write_shard(
    path: str | Path,
    rows: list[dict],
    *,
    version: int = 2,
) -> None:
    """Write a minimal V2 .kkc shard. Test-only helper.

    Each row dict needs keys: reading_ids, surface_ids, context_ids
    (sequences of u32-compatible ints), and optionally
    writer_id / domain_id / source_id (default 0).
    """
    if version != 2:
        raise NotImplementedError("only V2 supported by write_shard")
    path = Path(path)
    payload_chunks: list[bytes] = []
    offsets: list[int] = []
    cur = 0
    for row in rows:
        reading = np.asarray(row["reading_ids"], dtype="<u4")
        surface = np.asarray(row["surface_ids"], dtype="<u4")
        context = np.asarray(row["context_ids"], dtype="<u4")
        header = struct.pack(
            "<IIIIII",
            int(reading.shape[0]),
            int(surface.shape[0]),
            int(context.shape[0]),
            int(row.get("writer_id", 0)),
            int(row.get("domain_id", 0)),
            int(row.get("source_id", 0)),
        )
        body = header + reading.tobytes() + surface.tobytes() + context.tobytes()
        offsets.append(cur)
        payload_chunks.append(body)
        cur += len(body)

    payload = b"".join(payload_chunks)
    payload_offset = _SHARD_HEADER_LEN
    index_offset = payload_offset + len(payload)
    header = (
        _SHARD_MAGIC
        + struct.pack("<IQQQ", 2, len(rows), payload_offset, index_offset)
    )
    index_bytes = np.asarray(offsets, dtype="<u8").tobytes()
    path.write_bytes(header + payload + index_bytes)
