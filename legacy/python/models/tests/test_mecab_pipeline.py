"""Tests for the shared corpus processing helpers (no MeCab required for
the pure-Python filters)."""

from __future__ import annotations

import pytest

from models.src.data.mecab_pipeline import (
    MAX_SENTENCE_LEN,
    MIN_SENTENCE_LEN,
    attach_context,
    reading_from_mecab,
    sentence_acceptable,
    split_sentences,
)


def test_split_sentences_boundaries():
    text = "これは文1です。ここが文2！そして文3？\n最終文。"
    parts = split_sentences(text)
    assert len(parts) == 4
    assert parts[0].endswith("。")
    assert parts[1].endswith("！")
    assert parts[2].endswith("？")


def test_split_sentences_ignores_blank():
    assert split_sentences("\n\n\n") == []


def test_sentence_acceptable_length_bounds():
    assert not sentence_acceptable("短い")
    assert sentence_acceptable("十分な長さの日本語文章です。")
    too_long = "あ" * (MAX_SENTENCE_LEN + 1)
    assert not sentence_acceptable(too_long)
    just_long_enough = "あ" * MAX_SENTENCE_LEN
    assert sentence_acceptable(just_long_enough)
    just_short_enough = "あ" * MIN_SENTENCE_LEN
    assert sentence_acceptable(just_short_enough)


def test_sentence_acceptable_rejects_url():
    assert not sentence_acceptable("http://example.com で確認してください。")


def test_sentence_acceptable_rejects_markup():
    assert not sentence_acceptable("文中に [ブラケット] があるとダメです。")


def test_sentence_acceptable_rejects_ascii_heavy():
    # >30% ASCII → reject
    assert not sentence_acceptable("ABC DEF GHI JKLMN 短い。")


def test_attach_context_threads_previous_surface():
    pairs = [
        {"reading": "あ", "surface": "ア" * 30},
        {"reading": "い", "surface": "イ" * 30},
        {"reading": "う", "surface": "ウ" * 30},
    ]
    tail = attach_context(pairs, prev_tail="", max_context=8)
    assert pairs[0]["context"] == ""
    assert pairs[1]["context"] == "ア" * 8
    assert pairs[2]["context"] == "イ" * 8
    assert tail == pairs[-1]["surface"]


def test_reading_from_mecab_requires_worker_init():
    """reading_from_mecab without worker_init must raise, not silently no-op."""
    import models.src.data.mecab_pipeline as mp

    original = mp._tagger
    mp._tagger = None
    try:
        with pytest.raises(RuntimeError, match="worker_init"):
            reading_from_mecab("テスト")
    finally:
        mp._tagger = original
