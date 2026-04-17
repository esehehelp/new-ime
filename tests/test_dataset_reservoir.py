"""Tests for the reservoir-sampling path in KanaKanjiDataset."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.dataset import KanaKanjiDataset


def _write_jsonl(path: Path, n: int) -> None:
    """Create a JSONL with n rows. Each row encodes its index so we can
    check which rows survived sampling."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "reading": f"よみ{i:04d}",
                        "surface": f"表層{i:04d}",
                        "context": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def test_full_load_when_max_samples_zero(tmp_path: Path) -> None:
    path = tmp_path / "small.jsonl"
    _write_jsonl(path, 50)
    ds = KanaKanjiDataset(str(path), max_samples=0)
    assert len(ds) == 50
    # Full-load preserves order.
    assert ds[0]["surface"] == "表層0000"
    assert ds[-1]["surface"] == "表層0049"


def test_reservoir_reduces_to_max_samples(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    _write_jsonl(path, 10_000)
    ds = KanaKanjiDataset(str(path), max_samples=100, seed=42)
    assert len(ds) == 100
    # Every row should be a valid row from the source file.
    for row in ds.data:
        idx = int(row["surface"].removeprefix("表層"))
        assert 0 <= idx < 10_000


def test_reservoir_is_deterministic_with_seed(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    _write_jsonl(path, 2_000)
    a = KanaKanjiDataset(str(path), max_samples=50, seed=7)
    b = KanaKanjiDataset(str(path), max_samples=50, seed=7)
    assert [r["surface"] for r in a.data] == [r["surface"] for r in b.data]


def test_reservoir_different_seeds_give_different_samples(tmp_path: Path) -> None:
    path = tmp_path / "big.jsonl"
    _write_jsonl(path, 2_000)
    a = KanaKanjiDataset(str(path), max_samples=50, seed=7)
    b = KanaKanjiDataset(str(path), max_samples=50, seed=13)
    assert [r["surface"] for r in a.data] != [r["surface"] for r in b.data]


def test_reservoir_handles_empty_lines_and_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write('{"reading": "a", "surface": "A", "context": ""}\n')
        f.write("\n")  # empty
        f.write("not-json\n")
        f.write('{"reading": "b", "surface": "B", "context": ""}\n')
    ds = KanaKanjiDataset(str(path), max_samples=10, seed=1)
    assert len(ds) == 2
    surfaces = {r["surface"] for r in ds.data}
    assert surfaces == {"A", "B"}


def test_reservoir_smaller_file_returns_all(tmp_path: Path) -> None:
    path = tmp_path / "tiny.jsonl"
    _write_jsonl(path, 5)
    ds = KanaKanjiDataset(str(path), max_samples=100, seed=1)
    assert len(ds) == 5
