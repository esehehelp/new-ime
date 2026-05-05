"""Shared fixtures for training acceptance tests.

Builds a tiny CTCNAT (≈ thousands of params, runs in seconds on CPU) and a
small `.kkc` shard backed by the default SharedCharTokenizer so loop tests
do not need a real corpus or GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from new_ime.data.shards import write_shard
from new_ime.data.tokenizer import SharedCharTokenizer
from new_ime.model.ctc_nat import CTCNAT
from new_ime.model.dat import DAT
from new_ime.model.encoder import SmallEncoder

_SAMPLE_PAIRS = [
    ("あいうえお", "愛上絵"),
    ("かきくけこ", "書く来る"),
    ("さしすせそ", "差し背"),
    ("たちつてと", "立ち手"),
    ("なにぬねの", "何の根"),
    ("はひふへほ", "歯人辺"),
    ("まみむめも", "見目元"),
    ("やゆよ", "矢由世"),
]


def _tiny_model(vocab_size: int, max_positions: int = 64) -> CTCNAT:
    encoder = SmallEncoder(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=128,
        max_positions=max_positions,
        dropout=0.0,
    )
    model = CTCNAT(
        encoder=encoder,
        output_vocab_size=vocab_size,
        decoder_layers=2,
        decoder_heads=4,
        decoder_ffn_size=128,
        dropout=0.0,
        blank_id=4,
        max_positions=max_positions,
    )
    model._preset_name = "tiny_test"
    model._max_seq_len = max_positions
    return model


@pytest.fixture
def tiny_tokenizer() -> SharedCharTokenizer:
    return SharedCharTokenizer()


@pytest.fixture
def tiny_model_factory(tiny_tokenizer):
    def _factory(seed: int = 0) -> CTCNAT:
        torch.manual_seed(seed)
        return _tiny_model(tiny_tokenizer.vocab_size)

    return _factory


def _tiny_dat_model(vocab_size: int, max_positions: int = 64, upsample_scale: int = 2) -> DAT:
    """Builds a tiny DAT (~thousands of params) for fast CPU smoke tests.

    Mirrors `_tiny_model` so DAT and CTC-NAT share the same shard / tokenizer
    fixtures. `upsample_scale=2` keeps `T_up = 2 * T_in` small, which is the
    minimum value that still exercises the multi-vertex DAG path.
    """
    encoder = SmallEncoder(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        ffn_size=128,
        max_positions=max_positions,
        dropout=0.0,
    )
    model = DAT(
        encoder=encoder,
        output_vocab_size=vocab_size,
        decoder_layers=2,
        decoder_heads=4,
        decoder_ffn_size=128,
        upsample_scale=upsample_scale,
        num_link_heads=2,
        dropout=0.0,
        max_positions=max_positions,
        blank_id=4,
    )
    model._preset_name = "tiny_dat_test"
    model._max_seq_len = max_positions
    return model


@pytest.fixture
def tiny_dat_factory(tiny_tokenizer):
    def _factory(seed: int = 0, upsample_scale: int = 2) -> DAT:
        torch.manual_seed(seed)
        return _tiny_dat_model(tiny_tokenizer.vocab_size, upsample_scale=upsample_scale)

    return _factory


@pytest.fixture
def mock_shard(tmp_path: Path, tiny_tokenizer: SharedCharTokenizer) -> Path:
    rows = []
    for reading, surface in _SAMPLE_PAIRS:
        rows.append(
            {
                "reading_ids": np.asarray(
                    tiny_tokenizer.encode(reading), dtype="<u4"
                ),
                "surface_ids": np.asarray(
                    tiny_tokenizer.encode(surface), dtype="<u4"
                ),
                "context_ids": np.asarray([], dtype="<u4"),
                "writer_id": 0,
                "domain_id": 0,
                "source_id": 0,
            }
        )
    shard_path = tmp_path / "tiny.kkc"
    write_shard(shard_path, rows)
    return shard_path
