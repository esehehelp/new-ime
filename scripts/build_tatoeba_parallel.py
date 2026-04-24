"""Build a Tatoeba JA-EN parallel JSONL pool (quality-first).

Inputs (from Tatoeba exports, downloaded under datasets/raw/current/tatoeba):
- sentences.csv            # id<TAB>lang<TAB>text, all languages
- jpn-eng_links.tsv.bz2    # ja_id<TAB>en_id (JP-EN pairings only)

Output:
- datasets/corpus/parallel/tatoeba-ja-en.jsonl

Schema per row:
    {"reading": <hira>, "surface": <ja_text>, "context": "",
     "english": <en_text>, "ja_id": <int>, "en_id": <int>,
     "source": "tatoeba_ja_en"}

Quality filter (before MeCab reading):
- NFKC-normalise both sides
- ja: 3 <= len <= 120, has kanji, no URL
- en: 5 <= len <= 300, >= 2 words, has ASCII alpha, no URL
- reading non-empty after MeCab

Post-hoc (applied by Phase 4 when compiling the mix):
- rules_v3 gate on {reading, surface}. English side is metadata
  and not subject to rules_v3 (which target JA conventions).
"""
from __future__ import annotations
import argparse
import bz2
import json
import re
import sys
import unicodedata
from pathlib import Path

import MeCab

REPO = Path(__file__).resolve().parents[1]

KANJI = re.compile(r"[一-鿿㐀-䶿]")
URL = re.compile(r"https?://|www\.|\.com/|\.org/|\.net/|\.jp/", re.IGNORECASE)
EN_WORDS = re.compile(r"[A-Za-z]+")


def kata_to_hira(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
        for c in s
    )


# Long-vowel ー expansion: for IME training the typed form matters.
# Users type おとうと, not おとーと. Expand trailing ー after hira vowel.
_CHOUON_EXPAND = {
    "あ": "あ",
    "い": "い",
    "う": "う",
    "え": "い",   # most common Japanese long-vowel convention
    "お": "う",
    "ゃ": "あ",
    "ゅ": "う",
    "ょ": "う",
    "か": "あ", "が": "あ", "さ": "あ", "ざ": "あ", "た": "あ", "だ": "あ",
    "な": "あ", "は": "あ", "ば": "あ", "ぱ": "あ", "ま": "あ", "や": "あ",
    "ら": "あ", "わ": "あ",
    "き": "い", "ぎ": "い", "し": "い", "じ": "い", "ち": "い", "ぢ": "い",
    "に": "い", "ひ": "い", "び": "い", "ぴ": "い", "み": "い", "り": "い",
    "く": "う", "ぐ": "う", "す": "う", "ず": "う", "つ": "う", "づ": "う",
    "ぬ": "う", "ふ": "う", "ぶ": "う", "ぷ": "う", "む": "う", "ゆ": "う", "る": "う",
    "け": "い", "げ": "い", "せ": "い", "ぜ": "い", "て": "い", "で": "い",
    "ね": "い", "へ": "い", "べ": "い", "ぺ": "い", "め": "い", "れ": "い",
    "こ": "う", "ご": "う", "そ": "う", "ぞ": "う", "と": "う", "ど": "う",
    "の": "う", "ほ": "う", "ぼ": "う", "ぽ": "う", "も": "う", "よ": "う", "ろ": "う",
    "を": "う",
}


def normalize_chouon(s: str) -> str:
    """Expand `ー` after a hiragana into the proper long-vowel.

    Katakana-origin long vowels (e.g. コーヒー) stay as ー in IME input;
    but when we've already kata→hira converted and left `ー` in the
    middle of a hiragana run, the user would have typed the vowel
    explicitly. Only expand when the preceding char is hira (kata was
    already converted upstream).
    """
    if "ー" not in s:
        return s
    out = []
    prev = ""
    for c in s:
        if c == "ー" and prev in _CHOUON_EXPAND:
            out.append(_CHOUON_EXPAND[prev])
        else:
            out.append(c)
        prev = c
    return "".join(out)


def make_tagger() -> MeCab.Tagger:
    # unidic-lite is bundled in dev deps. Default tagger picks it up.
    return MeCab.Tagger()


HAS_KANJI = re.compile(r"[一-鿿㐀-䶿]")


def reading_for(tagger: MeCab.Tagger, surface: str) -> str:
    """Orthographic reading: what the user types on an IME keyboard.

    - node.surface contains kanji: use MeCab kana column (pronunciation),
      converted hira. This is the right thing for kanji→reading.
    - node.surface is pure hira/kata/symbol: pass through (katakana →
      hira). MeCab unidic's `kana` column is PRONUNCIATION (は→ワ,
      いい→イー) which is *wrong* for IME training — users type は as は
      not わ, and いい as いい not いー.
    """
    node = tagger.parseToNode(surface)
    parts: list[str] = []
    while node:
        if node.surface:
            s = node.surface
            if HAS_KANJI.search(s):
                cols = (node.feature or "").split(",")
                kana = cols[9] if len(cols) > 9 else "*"
                if kana and kana != "*":
                    # Kanji reading from unidic uses ー for long vowels;
                    # expand it for IME-typed form.
                    parts.append(normalize_chouon(kata_to_hira(kana)))
                else:
                    parts.append(kata_to_hira(s))
            else:
                # hira / kata / punct / digits → orth pass-through.
                # Do NOT expand ー here: a lone ー in loan-word reading
                # (like "コーヒー"→"こーひー") is how the user types it.
                parts.append(kata_to_hira(s))
        node = node.next
    return "".join(parts)


def ja_ok(text: str) -> bool:
    if not 3 <= len(text) <= 120:
        return False
    if not KANJI.search(text):
        return False
    if URL.search(text):
        return False
    return True


def en_ok(text: str) -> bool:
    if not 5 <= len(text) <= 300:
        return False
    if URL.search(text):
        return False
    words = EN_WORDS.findall(text)
    if len(words) < 2:
        return False
    return True


def load_sentences(path: Path, want_langs: set[str]) -> dict[int, tuple[str, str]]:
    """id -> (lang, text) for rows whose lang is in want_langs."""
    out: dict[int, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sid, lang, text = parts
            if lang not in want_langs:
                continue
            try:
                out[int(sid)] = (lang, text)
            except ValueError:
                continue
    return out


def load_links(path: Path) -> list[tuple[int, int]]:
    """JA_id, EN_id pairs (order per links file — Tatoeba exports this as
    jpn-side first)."""
    pairs: list[tuple[int, int]] = []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for raw in f:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            try:
                pairs.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sentences",
        default=str(REPO / "datasets/raw/current/tatoeba/sentences.csv"),
    )
    ap.add_argument(
        "--links",
        default=str(REPO / "datasets/raw/current/tatoeba/jpn-eng_links.tsv.bz2"),
    )
    ap.add_argument(
        "--out",
        default=str(REPO / "datasets/corpus/parallel/tatoeba-ja-en.jsonl"),
    )
    args = ap.parse_args()

    sentences_path = Path(args.sentences)
    links_path = Path(args.links)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[parallel] loading sentences from {sentences_path}", flush=True)
    sents = load_sentences(sentences_path, {"jpn", "eng"})
    print(f"[parallel]   loaded {len(sents):,} ja/en sentences", flush=True)

    print(f"[parallel] loading links from {links_path}", flush=True)
    pairs = load_links(links_path)
    print(f"[parallel]   {len(pairs):,} ja-en pairs", flush=True)

    tagger = make_tagger()

    n_in = n_ja_fail = n_en_fail = n_read_fail = n_out = 0
    with out_path.open("w", encoding="utf-8") as g:
        for ja_id, en_id in pairs:
            n_in += 1
            ja_row = sents.get(ja_id)
            en_row = sents.get(en_id)
            if ja_row is None or en_row is None:
                continue
            _, ja_text = ja_row
            _, en_text = en_row
            ja_text = unicodedata.normalize("NFKC", ja_text).strip()
            en_text = unicodedata.normalize("NFKC", en_text).strip()
            if not ja_ok(ja_text):
                n_ja_fail += 1
                continue
            if not en_ok(en_text):
                n_en_fail += 1
                continue
            try:
                yomi = reading_for(tagger, ja_text)
            except Exception:
                n_read_fail += 1
                continue
            if not yomi or not 2 <= len(yomi) <= 200:
                n_read_fail += 1
                continue
            rec = {
                "reading": yomi,
                "surface": ja_text,
                "context": "",
                "english": en_text,
                "ja_id": ja_id,
                "en_id": en_id,
                "source": "tatoeba_ja_en",
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            if n_in % 50_000 == 0:
                print(
                    f"  [{n_in:,}/{len(pairs):,}]  out={n_out:,}"
                    f"  ja_fail={n_ja_fail:,} en_fail={n_en_fail:,} read_fail={n_read_fail:,}",
                    flush=True,
                )

    print(
        f"[parallel] done  pairs={n_in:,}  out={n_out:,}"
        f"  ja_fail={n_ja_fail:,} en_fail={n_en_fail:,} read_fail={n_read_fail:,}"
        f"  -> {out_path}"
    )


if __name__ == "__main__":
    main()
