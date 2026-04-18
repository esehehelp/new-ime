"""Wikimedia XML dump -> sentences -> yomi'd JSONL.

Given a pages-articles.xml file, produce
    {"reading": ..., "surface": ..., "context": "", "source": <tag>}
lines. Pipeline:

    <text> extract  ->  wikitextparser.plain_text()  ->  regex sentence split
    ->  per-sentence fugashi (unidic-lite) reading  ->  katakana->hiragana.

The whole thing streams — we never hold more than one <page> in memory
and we emit one output line per accepted sentence.

Usage:
    uv run python -m tools.corpus_v2.process_wikimedia \
        --xml datasets/raw_v2/jawikibooks-latest-pages-articles.xml \
        --source wikibooks_v2 \
        --out datasets/v2/wikibooks_v2.jsonl

Quality filters (applied per sentence, before writing):
    - length within [5, 120] Japanese characters
    - must contain at least one kanji (guards against pure-kana / ASCII)
    - kana ratio >= 0.04 (catches templates / URLs that slipped through)
    - unicode category drops control chars, most symbols
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import fugashi
import wikitextparser as wtp

sys.stdout.reconfigure(encoding="utf-8")

TAGGER = fugashi.Tagger()

# Wikipedia page XML uses this namespace; strip it when matching tags.
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

# Sentence split — Japanese text usually terminates on 。, with some ! ? variants.
# We also split on newlines so bulleted lists become separate sentences.
SENT_SPLIT = re.compile(r"(?<=[。！？\!\?])\s*|[\r\n]+")

HIRA = re.compile(r"[\u3041-\u309F]")
KATA = re.compile(r"[\u30A1-\u30FA\u30FC]")
KANJI = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")


def kata_to_hira(s: str) -> str:
    out = []
    for c in s:
        code = ord(c)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(c)
    return "".join(out)


def keep_sentence(s: str) -> bool:
    if not 5 <= len(s) <= 120:
        return False
    if not KANJI.search(s):
        return False
    # Calm filter — avoid all-symbol residue of unparsed markup.
    non_space = sum(1 for c in s if not c.isspace())
    if non_space == 0:
        return False
    ascii_ratio = sum(1 for c in s if ord(c) < 0x80) / max(1, len(s))
    if ascii_ratio > 0.6:
        return False
    return True


def reading_for(surface: str) -> str:
    parts = []
    for w in TAGGER(surface):
        feat = w.feature
        kana = getattr(feat, "kana", None) or getattr(feat, "pron", None)
        if kana and kana != "*":
            parts.append(kana)
        else:
            # Unknown word (usually latin or digits) — pass through.
            parts.append(w.surface)
    return kata_to_hira("".join(parts))


def iter_pages(xml_path: Path):
    """Yield (title, text) from a MediaWiki pages-articles XML stream."""
    # Stream-parse so we never hold the whole 400 MB tree.
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != NS + "page":
            continue
        title_el = elem.find(NS + "title")
        rev = elem.find(NS + "revision")
        if rev is None:
            elem.clear()
            continue
        text_el = rev.find(NS + "text")
        title = title_el.text if title_el is not None and title_el.text else ""
        text = text_el.text if text_el is not None and text_el.text else ""
        yield title, text
        elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--source", required=True, help="source tag written into each output record")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pages", type=int, default=0, help="0 = no cap")
    parser.add_argument("--report-every", type=int, default=1000)
    args = parser.parse_args()

    xml = Path(args.xml)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen_pages = 0
    t0 = time.perf_counter()

    with out.open("w", encoding="utf-8") as g:
        for title, text in iter_pages(xml):
            seen_pages += 1
            if args.max_pages and seen_pages > args.max_pages:
                break
            if title.startswith(("特別:", "MediaWiki:", "Category:", "File:",
                                 "Template:", "Wikipedia:", "Help:", "Portal:",
                                 "Module:", "モジュール:", "テンプレート:",
                                 "カテゴリ:", "ファイル:", "ヘルプ:")):
                continue
            if not text:
                continue

            try:
                plain = wtp.parse(text).plain_text(replace_templates=False)
            except Exception:
                continue
            plain = unicodedata.normalize("NFKC", plain)
            # Strip leftover markup that plain_text left through.
            plain = re.sub(r"<[^>]+>", " ", plain)
            plain = re.sub(r"\[[^\]]*\]", " ", plain)
            plain = re.sub(r"\{\{[^}]*\}\}", " ", plain)
            plain = re.sub(r"=+\s*[^=]+\s*=+", " ", plain)
            plain = re.sub(r"[ \t]+", " ", plain)

            for raw_sent in SENT_SPLIT.split(plain):
                sent = raw_sent.strip()
                if not keep_sentence(sent):
                    continue
                try:
                    yomi = reading_for(sent)
                except Exception:
                    continue
                if not yomi or len(yomi) < 3:
                    continue
                rec = {
                    "reading": yomi,
                    "surface": sent,
                    "context": "",
                    "source": args.source,
                }
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

            if seen_pages % args.report_every == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  pages={seen_pages:,} kept={kept:,} "
                    f"rate={seen_pages / elapsed:.1f}/s",
                    flush=True,
                )

    print(f"done: pages={seen_pages:,} kept={kept:,} -> {out}")


if __name__ == "__main__":
    main()
