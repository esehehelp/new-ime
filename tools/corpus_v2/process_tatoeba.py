"""Tatoeba full sentences.csv -> yomi'd JSONL.

Tatoeba dumps carry `id<TAB>lang<TAB>text`. Filter to `jpn`, run through
fugashi, emit the same schema the other corpus-v2 pools use.

Usage:
    uv run python -m tools.corpus_v2.process_tatoeba \
        --src datasets/raw_current/tatoeba/sentences.csv \
        --out datasets/v2/tatoeba_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import fugashi

sys.stdout.reconfigure(encoding="utf-8")

TAGGER = fugashi.Tagger()
KANJI = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


def kata_to_hira(s: str) -> str:
    return "".join(
        chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
        for c in s
    )


def reading_for(surface: str) -> str:
    parts = []
    for w in TAGGER(surface):
        feat = w.feature
        kana = getattr(feat, "kana", None) or getattr(feat, "pron", None)
        parts.append(kana if kana and kana != "*" else w.surface)
    return kata_to_hira("".join(parts))


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
    with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
        for raw in f:
            total += 1
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            _, lang, text = parts
            if lang != "jpn":
                continue
            text = unicodedata.normalize("NFKC", text).strip()
            if not 3 <= len(text) <= 120:
                continue
            if not KANJI.search(text):
                continue
            try:
                yomi = reading_for(text)
            except Exception:
                continue
            if not yomi:
                continue
            rec = {"reading": yomi, "surface": text, "context": "", "source": "tatoeba_v2"}
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if total % 500_000 == 0:
                print(f"  scanned {total:,} kept {kept:,}", flush=True)
    print(f"done: scanned {total:,} kept {kept:,} -> {out}")


if __name__ == "__main__":
    main()
