"""PyTorch Dataset for kana-kanji conversion training.

Each sample: (input_ids, target_ids)
  input_ids:  [CLS] context [SEP] reading [EOS]  (for AR: causal LM)
  target_ids: surface text tokens

For AR baseline (Phase 2), we concatenate as:
  [CLS] context [SEP] reading [OUT] surface [EOS]
and train with causal LM loss on the surface portion.

## Shard mode (dev-branch Phase C)

`KanaKanjiShardDataset` + `CTCShardCollator` read pre-tokenized binary
shards produced by `cargo run -p rust-data -- compile ...`. Tokenize +
JSON parse run in Rust at compile time, so the dataloader hot path is
reduced to mmap slicing + numpy → tensor conversion. Format must stay
in sync with `crates/rust-data/src/shard.rs` (magic `KKCSHRD1`, V2).
"""

from __future__ import annotations

import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_OFFSETS_SUFFIX = ".offsets.npy"


def _find_rust_indexer() -> Path | None:
    """Locate the Rust `offset-index` binary if built via `cargo build -p offset-index`.

    Checks the repo-local cargo output dir under `build/`, then $PATH.
    Returns None if neither is available (caller falls back to the Python
    scanner — correct but 5-10× slower on 40+ GiB jsonl).
    """
    # repo root = 3 parents up from this file (models/src/data/dataset.py)
    repo_root = Path(__file__).resolve().parents[3]
    for rel in ("build/release/offset-index", "build/release/offset-index.exe"):
        cand = repo_root / rel
        if cand.exists():
            return cand
    path_hit = shutil.which("offset-index")
    return Path(path_hit) if path_hit else None


def _build_offset_index(jsonl_path: Path, index_path: Path) -> None:
    """One-time sequential scan: record byte offset of every JSONL line.

    Prefers the Rust `offset-index` binary (30-60 s on 46 GiB jsonl) and
    falls back to a Python scanner (3-5 min) if the binary is absent.
    Idempotent at caller level (caller checks index mtime vs jsonl mtime).
    """
    rust_bin = _find_rust_indexer()
    if rust_bin is not None:
        print(f"[dataset] invoking Rust indexer: {rust_bin}", flush=True)
        try:
            subprocess.run(
                [str(rust_bin), "--input", str(jsonl_path), "--output", str(index_path)],
                check=True,
                stdout=sys.stderr,   # keep offset-index progress visible
                stderr=sys.stderr,
            )
            return
        except subprocess.CalledProcessError as e:
            print(
                f"[dataset] Rust indexer failed ({e.returncode}); "
                "falling back to Python scanner",
                flush=True,
            )
            # fall through to Python

    # Python fallback.
    offsets: list[int] = []
    with open(jsonl_path, "rb") as f:
        pos = 0
        for line in f:
            if line.strip():
                offsets.append(pos)
            pos += len(line)
    arr = np.asarray(offsets, dtype=np.uint64)
    # np.save auto-appends ".npy" if missing, so round-trip through a tmp path
    # that we then rename to the canonical `<jsonl>.offsets.npy` target.
    tmp_stem = str(index_path) + ".tmp"   # e.g. "foo.offsets.npy.tmp"
    np.save(tmp_stem, arr)                # writes "foo.offsets.npy.tmp.npy"
    os.replace(tmp_stem + ".npy", index_path)


class KanaKanjiDataset(Dataset):
    """Disk-backed Map-style Dataset for kana-kanji JSONL.

    Implementation:
      1. On first use, build a line-offset index next to the source file as
         `<jsonl_path>.offsets.npy` (one-time O(file size) scan).
      2. `__getitem__(idx)` seeks the file to `offsets[idx]` and parses one
         line. RAM stays ~constant regardless of file size.
      3. `max_samples > 0` materialises a deterministic sorted subset of the
         full offsets (via `np.random.default_rng(seed).choice`) — no bytes
         beyond the subset array live in RAM.
      4. Each DataLoader worker opens its own file descriptor lazily on the
         first `__getitem__` call (safe across `fork()`; no shared seek
         position).

    Memory footprint:
        full 200M-row mix    : offset mmap ≈ 1.6 GiB (kernel page cache, not
                               counted as Python RSS)
        60M subset           : offsets ~500 MiB materialised (kept alive
                               through training; counted as RSS)
        20M subset           : ~160 MiB
    Payload bytes (the actual JSONL content) are never held in Python lists —
    each row is re-parsed on demand and GC'd after the collator copies it.
    """

    def __init__(
        self,
        jsonl_path: str,
        max_samples: int = 0,
        max_seq_len: int = 256,
        seed: int = 42,
        preload: bool = False,
    ):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset jsonl not found: {self.path}")
        self.max_seq_len = max_seq_len

        index_path = Path(str(self.path) + _OFFSETS_SUFFIX)
        need_build = (
            not index_path.exists()
            or index_path.stat().st_mtime < self.path.stat().st_mtime
        )
        if need_build:
            print(
                f"[dataset] building offset index for {self.path.name} "
                f"→ {index_path.name} (one-time)",
                flush=True,
            )
            _build_offset_index(self.path, index_path)

        all_offsets = np.load(index_path, mmap_mode="r")

        if max_samples and 0 < max_samples < len(all_offsets):
            rng = np.random.default_rng(seed)
            pick = rng.choice(
                len(all_offsets), size=int(max_samples), replace=False
            )
            pick.sort()   # sequential disk reads
            self.offsets = np.asarray(all_offsets[pick], dtype=np.uint64)
        else:
            self.offsets = all_offsets  # mmap'd view

        self._fh = None  # lazy per-worker file descriptor
        # When `preload` is set, read + json.loads every sample into a Python
        # list at __init__ time. Eliminates per-step disk seeks and JSON parse
        # from the dataloader hot path. Costs ~1 KiB per sample (Python dict
        # overhead), so 30M samples ≈ 30 GiB RAM — only turn this on when the
        # cgroup budget explicitly permits it.
        self._preloaded: list[dict] | None = None
        if preload:
            print(
                f"[dataset] preloading {len(self.offsets)} samples into RAM "
                f"(streaming disabled, ~{len(self.offsets) * 1000 // (1024**3)} GiB estimated)",
                flush=True,
            )
            t0 = time.time()
            fh = open(self.path, "rb")
            preloaded: list[dict] = [None] * len(self.offsets)  # type: ignore[list-item]
            for i, off in enumerate(self.offsets):
                fh.seek(int(off))
                preloaded[i] = json.loads(fh.read(4096).split(b"\n", 1)[0])
                if (i + 1) % 1_000_000 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-6)
                    print(
                        f"[dataset]   preloaded {i + 1}/{len(self.offsets)} "
                        f"({rate / 1000:.0f}k rows/s)",
                        flush=True,
                    )
            fh.close()
            self._preloaded = preloaded
            print(
                f"[dataset] preload done in {time.time() - t0:.1f}s",
                flush=True,
            )

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # don't pickle fd across fork
        return state

    def __getitem__(self, idx: int) -> dict:
        if self._preloaded is not None:
            return self._preloaded[idx]
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(int(self.offsets[idx]))
        line = self._fh.readline()
        return json.loads(line)


# -----------------------------------------------------------------------------
# Shard mode: KKCSHRD1 V2 binary layout produced by `rust-data compile`.
# -----------------------------------------------------------------------------
# Header (36 bytes, little-endian):
#   magic[8] = b"KKCSHRD1"
#   version  u32
#   row_count u64
#   payload_offset u64
#   index_offset u64
# Payload (V2 per row):
#   reading_len u32, surface_len u32, context_len u32,
#   writer_id u32, domain_id u32, source_id u32,
#   reading[reading_len] u32,
#   surface[surface_len] u32,
#   context[context_len] u32
# Index: row_count × u64 (each = row byte offset within payload region).
_SHARD_MAGIC = b"KKCSHRD1"
_SHARD_VERSION_MIN = 1
_SHARD_VERSION_MAX = 2
_SHARD_HEADER_LEN = 36


class KanaKanjiShardDataset(Dataset):
    """Map-style Dataset over a `rust-data` compiled shard.

    Zero Python-side tokenize / JSON parse: `__getitem__` returns
    pre-tokenized u32 arrays sliced from an mmap'd shard. Pair with
    `CTCShardCollator` to build padded tensor batches.

    Parameters
    ----------
    shard_path:
        Path to a `.kkc` shard written by `rust-data compile`.
    expected_vocab_size:
        When set, validates the sidecar `*.meta.json` `vocab_size` matches
        so a mismatched tokenizer / shard pair fails loudly at startup.
    """

    def __init__(
        self,
        shard_path: str,
        expected_vocab_size: int | None = None,
    ):
        self.path = Path(shard_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Shard not found: {self.path}")

        self._fh = None
        self._mm: mmap.mmap | None = None
        self._open_mmap()

        magic = self._mm[:8]
        if bytes(magic) != _SHARD_MAGIC:
            raise ValueError(f"Not a rust-data shard (bad magic): {self.path}")
        version, row_count, payload_offset, index_offset = struct.unpack_from(
            "<IQQQ", self._mm, 8
        )
        if not (_SHARD_VERSION_MIN <= version <= _SHARD_VERSION_MAX):
            raise ValueError(
                f"Unsupported shard version {version} (supported {_SHARD_VERSION_MIN}..{_SHARD_VERSION_MAX})"
            )
        self.version = int(version)
        self.row_count = int(row_count)
        self.payload_offset = int(payload_offset)
        self.index_offset = int(index_offset)

        # Materialize the index table as a numpy u64 view (zero-copy on mmap).
        index_bytes = self._mm[
            self.index_offset : self.index_offset + self.row_count * 8
        ]
        self.index = np.frombuffer(index_bytes, dtype="<u8")
        if self.index.shape[0] != self.row_count:
            raise ValueError(
                f"Shard index length {self.index.shape[0]} != row_count {self.row_count}"
            )

        # Optional sidecar validation. Silent if absent (for interop with
        # shards that predate the sidecar emitter).
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
                    f"tokenizer={expected_vocab_size} "
                    f"(meta={meta_path})"
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
        header = self._mm[base : base + 24]
        reading_len, surface_len, context_len, writer_id, domain_id, source_id = (
            struct.unpack_from("<IIIIII", header)
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


class CTCShardCollator:
    """Batch collator for `KanaKanjiShardDataset`.

    Mirrors `CTCCollator` from `train_ctc_nat.py` but operates on
    pre-tokenized u32 arrays rather than JSON dicts, so no per-step
    Python tokenize loop runs on the dataloader hot path.

    Input layout (matches `CTCCollator._encode_input`):
        [CLS_ID] + context_ids + [SEP_ID] + reading_ids
      truncated to `max_seq_len`. `context_ids` are already capped at
      `max_context_chars` during shard compile.

    Target layout (matches `CTCCollator._encode_target`):
        surface_ids truncated to `max_seq_len`. `BLANK_ID` is stripped
      during shard compile (see `compile.rs`).

    `short_sample_max_chars > 0` filters rows whose reading_ids or
    surface_ids length exceeds the cap — semantically equivalent to the
    character-count filter in `CTCCollator.__call__` for char-level
    tokenizers. Rows with no byte-fallback will match exactly.
    """

    # Align with legacy/python/models/src/data/tokenizer.py and
    # crates/rust-tokenizer/src/lib.rs: PAD=0 UNK=1 SEP=2 CLS=3 BLANK=4 MASK=5.
    PAD_ID = 0
    SEP_ID = 2
    CLS_ID = 3

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
        # BLANK_ID is already stripped at compile time; just truncate.
        return [int(x) for x in surface_ids[: self.max_seq_len].tolist()]

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
            # Raw string fields are not stored in the shard (until Phase E),
            # so KD teacher round-trips must stay on the JSONL path for now.
            "_contexts": [""] * len(batch),
            "_readings": [""] * len(batch),
            "_surfaces": [""] * len(batch),
        }


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
