"""Parity + smoke tests for shard-backed dataloader.

Ensures `KanaKanjiShardDataset` + `CTCShardCollator` produce byte-identical
`input_ids`/`attention_mask`/`target_ids`/`target_lengths` as the legacy
`KanaKanjiDataset` + `CTCCollator` path for the same JSONL input.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from models.src.data.dataset import (
    CTCShardCollator,
    KanaKanjiDataset,
    KanaKanjiShardDataset,
)
from models.src.data.tokenizer import SharedCharTokenizer
from models.src.training.train_ctc_nat import CTCCollator


REPO_ROOT = Path(__file__).resolve().parents[3]


def _find_rust_data_bin() -> Path | None:
    for rel in (
        "build/debug/rust-data.exe",
        "build/debug/rust-data",
        "build/release/rust-data.exe",
        "build/release/rust-data",
    ):
        cand = REPO_ROOT / rel
        if cand.exists():
            return cand
    path_hit = shutil.which("rust-data")
    return Path(path_hit) if path_hit else None


def _write_fixture(tmp: Path) -> tuple[Path, Path, SharedCharTokenizer]:
    rows = [
        {
            "reading": "あしたはてんき",
            "surface": "明日は天気",
            "context": "今日の話",
            "writer": "W1",
            "domain": "D1",
            "source": "S1",
        },
        {
            "reading": "おはよう",
            "surface": "おはよう",
            "context": "",
            "writer": "W2",
            "domain": "D1",
            "source": "S2",
        },
        {
            "reading": "ねこがすき",
            "surface": "猫が好き",
            "context": "動物",
            "writer": "W1",
            "domain": "D2",
            "source": "S1",
        },
    ]
    jsonl = tmp / "fixture.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    # Minimal tokenizer covering every char in the fixture plus specials.
    chars = set()
    for r in rows:
        chars.update(r["reading"])
        chars.update(r["surface"])
        chars.update(r["context"])
    tokenizer_json = tmp / "tokenizer.json"
    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[SEP]": 2,
        "[CLS]": 3,
        "[BLANK]": 4,
        "[MASK]": 5,
    }
    for c in sorted(chars):
        token_to_id[c] = len(token_to_id)
    tokenizer_json.write_text(
        json.dumps({"type": "input", "token_to_id": token_to_id}, ensure_ascii=False),
        encoding="utf-8",
    )
    tokenizer = SharedCharTokenizer.load(str(tokenizer_json))
    return jsonl, tokenizer_json, tokenizer


def _compile_shard(jsonl: Path, tokenizer_json: Path, shard: Path) -> None:
    bin_path = _find_rust_data_bin()
    if bin_path is None:
        pytest.skip("rust-data binary not built (run `cargo build -p rust-data`)")
    subprocess.run(
        [
            str(bin_path),
            "compile",
            "--input",
            str(jsonl),
            "--output",
            str(shard),
            "--tokenizer",
            str(tokenizer_json),
            "--max-context-chars",
            "40",
            "--max-reading-tokens",
            "128",
            "--max-surface-tokens",
            "128",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )


def _collate_via_shard(shard: Path, batch_size: int, max_seq_len: int, short_sample_max_chars: int):
    dataset = KanaKanjiShardDataset(str(shard))
    collator = CTCShardCollator(
        max_seq_len=max_seq_len, short_sample_max_chars=short_sample_max_chars
    )
    rows = [dataset[i] for i in range(len(dataset))][:batch_size]
    return collator(rows)


def _collate_via_jsonl(
    jsonl: Path,
    tokenizer: SharedCharTokenizer,
    batch_size: int,
    max_seq_len: int,
    short_sample_max_chars: int,
    max_context: int,
):
    dataset = KanaKanjiDataset(str(jsonl))
    collator = CTCCollator(
        tokenizer,
        max_seq_len=max_seq_len,
        max_context=max_context,
        short_sample_max_chars=short_sample_max_chars,
    )
    rows = [dataset[i] for i in range(len(dataset))][:batch_size]
    return collator(rows)


def test_shard_dataset_matches_jsonl_collator_output(tmp_path: Path):
    jsonl, tokenizer_json, tokenizer = _write_fixture(tmp_path)
    shard = tmp_path / "fixture.kkc"
    _compile_shard(jsonl, tokenizer_json, shard)

    batch_size, max_seq_len = 3, 128
    shard_batch = _collate_via_shard(
        shard, batch_size=batch_size, max_seq_len=max_seq_len, short_sample_max_chars=0
    )
    jsonl_batch = _collate_via_jsonl(
        jsonl,
        tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        short_sample_max_chars=0,
        max_context=40,
    )

    for key in ("input_ids", "attention_mask", "target_ids", "target_lengths"):
        assert torch.equal(
            shard_batch[key], jsonl_batch[key]
        ), f"mismatch on '{key}'\n  shard={shard_batch[key]}\n  jsonl={jsonl_batch[key]}"

    # Clean up the offset index file KanaKanjiDataset writes next to the JSONL.
    for artifact in (jsonl.with_suffix(".jsonl.offsets.npy"),):
        if artifact.exists():
            os.remove(artifact)


def test_shard_dataset_sidecar_vocab_mismatch_fails(tmp_path: Path):
    jsonl, tokenizer_json, tokenizer = _write_fixture(tmp_path)
    shard = tmp_path / "fixture.kkc"
    _compile_shard(jsonl, tokenizer_json, shard)

    # Sidecar is emitted by `compile`; reject a mismatched expected size.
    with pytest.raises(ValueError, match="vocab mismatch"):
        KanaKanjiShardDataset(str(shard), expected_vocab_size=9999)
    # Correct size loads cleanly.
    loaded = KanaKanjiShardDataset(
        str(shard), expected_vocab_size=tokenizer.vocab_size
    )
    assert len(loaded) == 3


def test_shard_short_sample_filter(tmp_path: Path):
    jsonl, tokenizer_json, tokenizer = _write_fixture(tmp_path)
    shard = tmp_path / "fixture.kkc"
    _compile_shard(jsonl, tokenizer_json, shard)

    # 5-char cap: "あしたはてんき"(7) and "ねこがすき"(5) filtered vs kept.
    # Applied on reading_ids length (approx char count for in-vocab chars).
    dataset = KanaKanjiShardDataset(str(shard))
    rows = [dataset[i] for i in range(len(dataset))]
    collator = CTCShardCollator(max_seq_len=128, short_sample_max_chars=5)
    batch = collator(rows)
    # With cap=5, row 0 (reading len 7) drops; rows 1,2 remain (lens 4, 5).
    assert batch["input_ids"].shape[0] == 2
