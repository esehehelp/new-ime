"""Shared MeCab reading-extraction pipeline for corpus processing.

All corpora that start from plain Japanese text (Wikipedia, HPLT, CulturaX,
政府白書, 国会会議録, Wikinews, …) go through the same sentence → reading
extraction path. This module centralises that logic so individual
``scripts/process_*.py`` only own the *source-specific* parts (how to stream
raw text out of the upstream artefact, any per-source filtering).

Design
------
``worker_init`` is called once per ``multiprocessing.Pool`` worker to
instantiate the MeCab tagger (expensive; reused across calls). ``text_to_pairs``
is the per-worker entry point: it takes **plain text** and returns a list of
``{"reading", "surface"}`` dicts with hiragana readings derived from MeCab
unidic-lite ``features[17]`` (仮名形出現形, the only feature confirmed correct
for IME training — see ``docs/data_pipeline.md``).

The heavy imports (``MeCab``, ``jaconv``) happen inside the module but at
function call time so tests can import the module without a MeCab dictionary
present. Worker processes must call ``worker_init()`` before calling any of
the pair-producing functions.
"""

from __future__ import annotations

import re


SENTENCE_SPLIT = re.compile(r"(?<=[。！？\n])")
RE_URL = re.compile(r"https?://")
RE_MARKUP = re.compile(r"[{}\[\]|<>]")

MIN_SENTENCE_LEN = 5
MAX_SENTENCE_LEN = 100
ASCII_RATIO_CUTOFF = 0.3

_tagger = None  # set by worker_init in each worker process


def worker_init() -> None:
    """Initialise the per-process MeCab tagger.

    Must be called once per worker process. Safe to call multiple times but
    only the first call creates the tagger.
    """
    global _tagger
    if _tagger is not None:
        return
    import MeCab

    _tagger = MeCab.Tagger()
    _tagger.parse("")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def sentence_acceptable(sentence: str) -> bool:
    """Reject sentences that wiki/HPLT/CulturaX share as unusable."""
    if len(sentence) < MIN_SENTENCE_LEN or len(sentence) > MAX_SENTENCE_LEN:
        return False
    if RE_URL.search(sentence):
        return False
    if RE_MARKUP.search(sentence):
        return False
    ascii_count = sum(1 for c in sentence if ord(c) < 0x80)
    if ascii_count / len(sentence) > ASCII_RATIO_CUTOFF:
        return False
    return True


def reading_from_mecab(sentence: str) -> str | None:
    """Return the hiragana reading for a sentence, or ``None`` if unreliable.

    Uses unidic-lite ``features[17]`` (仮名形出現形). See
    ``docs/data_pipeline.md`` for the full feature-index rationale.
    """
    global _tagger
    if _tagger is None:
        raise RuntimeError("worker_init() must be called before reading_from_mecab()")
    import jaconv

    node = _tagger.parseToNode(sentence)
    readings: list[str] = []

    while node:
        surface = node.surface
        if not surface:
            node = node.next
            continue

        features = node.feature.split(",")
        if features[0] == "未知語":
            return None

        reading: str | None = None
        if len(features) >= 18 and features[17] != "*":
            reading = features[17]
        elif len(features) >= 7 and features[6] != "*":
            reading = features[6]

        if reading:
            hira = jaconv.kata2hira(reading)
            if any(c.isascii() and c.isalpha() for c in hira):
                return None
            readings.append(hira)
        elif all("\u3040" <= c <= "\u309f" or not c.strip() for c in surface):
            readings.append(surface)
        elif all(c in "、。！？「」『』（）・…―─　\n\r\t " for c in surface):
            readings.append(surface)
        else:
            return None

        node = node.next

    return "".join(readings)


def text_to_pairs(text: str) -> list[dict]:
    """Convert a plain-text chunk (document / article / utterance) into clean
    kana-kanji sentence pairs. Returns an empty list on any failure; callers
    chain context across the returned pairs if desired.
    """
    if not text:
        return []
    pairs: list[dict] = []
    for sentence in split_sentences(text):
        if not sentence_acceptable(sentence):
            continue
        reading = reading_from_mecab(sentence)
        if reading is None:
            continue
        pairs.append({"reading": reading, "surface": sentence})
    return pairs


def attach_context(pairs: list[dict], prev_tail: str = "", max_context: int = 40) -> str:
    """In-place: attach ``context`` to each pair as the trailing ``max_context``
    characters of the previous surface. Returns the new trailing surface.
    """
    tail = prev_tail
    for pair in pairs:
        pair["context"] = tail[-max_context:] if tail else ""
        tail = pair["surface"]
    return tail
