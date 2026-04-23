"""Second-pass cleanup for corpus_v2 JSONL files.

Stage 1 — STRIP
    Strip the leading MediaWiki markup (``#``, ``##``, ``*``, ``**``,
    ``#*``, ``#;``, ``:*``, ``*:``, ``=``+) from both surface and
    reading. The marker is purely a formatting hint; the content that
    follows is the real sentence, and we've been throwing it away.

Stage 2 — DROP
    After strip, reject lines that match any of:

        * foreign_script — Han without kana, or Hangul / Cyrillic / Arabic
        * heavy_ascii    — ASCII codepoints > 40 %
        * symbol_heavy   — Japanese codepoints < 40 %
        * empty_quote    — surface still contains 「」 empty
        * empty_reading
        * numeric_form   — all digits / brackets
        * short          — < 6 codepoints after strip
        * wiki_meta      — starts with カテゴリ: / Category: / テンプレート:
        * template_artifact — ``* 異体字 : (...)`` metadata pattern

Input and output are the same reading/surface/context/source schema;
this step is idempotent (re-running on an already clean file is a
no-op).

Usage:
    uv run python -m datasets.tools.corpus.clean_v2 \
        --src  datasets/v2/wiktionary_v2.jsonl \
        --out  datasets/v2/wiktionary_v2.clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Leading MediaWiki markup tokens. Order matters — longer first so
# ``##`` wins over ``#`` for the prefix that actually appears.
LEADING_RE = re.compile(r"^\s*(?:[#*:=]+\s*[*#;:]*\s*|[*#]+\s*)")

# Genre/topic tag prefixes that Wikinews sprinkles into article body
# bullets: ``(政治)`` ``(社会)`` ``(スポーツ)`` ``(賛成)`` etc. Also
# ``（...）`` fullwidth. Stripped after the Markdown markers so e.g.
# ``* (政治) 首相が…`` becomes ``首相が…``.
GENRE_TAG_RE = re.compile(
    r"^\s*[(（]\s*(?:政治|社会|スポーツ|経済|国際|文化|科学|技術|賛成|反対|"
    r"娯楽|教育|論説|コラム|連載|話題|時事)\s*[)）]\s*"
)

# Any leftover wiki-link / template residue in the middle.
MARKUP_MID = re.compile(r"\[\[|\]\]|\{\{|\}\}|<ref|</ref|<br|&nbsp;|&amp;")

HAN = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
HIRA = re.compile(r"[\u3041-\u309F]")
KATA = re.compile(r"[\u30A1-\u30FA\u30FC]")
HANGUL = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
HEBREW = re.compile(r"[\u0590-\u05FF]")

EMPTY_QUOTE = re.compile(r"「\s*」|『\s*』")
DIGIT_OR_BRACKET_ONLY = re.compile(r"^[\d\s.()（）\[\]、。]+$")

WIKI_META_PREFIX = re.compile(
    r"^(?:カテゴリ|Category|テンプレート|Template|ファイル|File|"
    r"ヘルプ|Help|モジュール|Module|Wikipedia|ウィキペディア)[:：]"
)
META_ARTIFACT = re.compile(
    r"(?:異体字|同義字|対義字|繁体字|簡体字|字源|部首|画数|音読み|訓読み)\s*[:：]"
)


def strip_leading(s: str) -> str:
    """Remove leading wiki markup tokens + genre tags. Whitespace-safe."""
    prev = None
    out = s
    while prev != out:
        prev = out
        m = LEADING_RE.match(out)
        if m:
            out = out[m.end():]
        m = GENRE_TAG_RE.match(out)
        if m:
            out = out[m.end():]
    return out.lstrip()


def clean_record(rec: dict) -> dict | None:
    surface = rec.get("surface", "")
    reading = rec.get("reading", "")
    if not surface or not reading:
        return None

    # Stage 1 — strip the MediaWiki bullet / heading markers off the
    # front of both sides. We strip independently so a reading that was
    # generated from the original surface (which has markup) still
    # aligns once the marker chars are removed.
    s = strip_leading(surface)
    r = strip_leading(reading)

    if not s or not r:
        return None
    # Sometimes strip leaves a stray single leading symbol.
    for sym in ("#", "*", ":", ";", "="):
        if s.startswith(sym):
            s = s[1:].lstrip()
        if r.startswith(sym):
            r = r[1:].lstrip()
    if not s or not r:
        return None

    n = len(s)

    # Stage 2 — drop rules.
    if n < 6:
        return None
    if WIKI_META_PREFIX.match(s):
        return None
    if META_ARTIFACT.search(s):
        return None
    if MARKUP_MID.search(s):
        return None
    if EMPTY_QUOTE.search(s):
        return None
    if DIGIT_OR_BRACKET_ONLY.match(s):
        return None

    has_kanji = bool(HAN.search(s))
    has_kana = bool(HIRA.search(s) or KATA.search(s))
    if has_kanji and not has_kana:
        return None  # pure Chinese
    if HANGUL.search(s) or CYRILLIC.search(s) or ARABIC.search(s) or HEBREW.search(s):
        return None

    ascii_non_space = sum(1 for c in s if ord(c) < 128 and not c.isspace())
    if ascii_non_space / n > 0.4:
        return None

    jp_chars = sum(1 for c in s if HIRA.match(c) or KATA.match(c) or HAN.match(c))
    if jp_chars / n < 0.4:
        return None

    reading_jp = sum(1 for c in r if HIRA.match(c) or KATA.match(c))
    if reading_jp == 0:
        return None

    out = dict(rec)
    out["surface"] = s
    out["reading"] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cleaned = clean_record(rec)
            if cleaned is None:
                continue
            g.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
            if total % 500_000 == 0:
                print(f"  scanned {total:,} kept {kept:,}", flush=True)
    retention = kept / total * 100 if total else 0
    print(f"done: scanned {total:,} kept {kept:,} ({retention:.1f}%) -> {out}")


if __name__ == "__main__":
    main()
