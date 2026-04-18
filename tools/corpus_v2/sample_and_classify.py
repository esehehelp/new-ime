"""Random-sample each v2 corpus, surface concrete noise classes.

Rather than adding regexes one by one based on what caught my eye in a
small sample, pull 200 lines per corpus and bucket each line by a
handful of candidate failure signals. The counts tell us which filters
will actually move the needle.

Signals (non-exclusive — a line can match several):
    * ``markup_prefix``: starts with ``*`` / ``#`` / ``**`` / ``=`` / ``:``
      (any leading wikitext control character)
    * ``markup_mid``: contains unbalanced wiki / mediawiki artifacts
      (``[[``, ``]]``, ``{{``, ``}}``, ``[[File:``, ``[[Category:``)
    * ``foreign_script``: at least one codepoint in Han (simplified Chinese
      without Japanese kanji context), Hangul, Cyrillic, Arabic, Hebrew
    * ``heavy_ascii``: > 40 % ASCII / latin characters
    * ``short``: < 8 graphs
    * ``empty_quote``: surface contains ``「」`` empty
    * ``template_artifact``: contains ``(政治)`` / ``(社会)`` / ``*(`` style
      template leakage from wikimedia bullets
    * ``symbol_heavy``: < 40 % hiragana+katakana+kanji; too many symbols
    * ``numeric_form``: surface is entirely numeric + bracket (leftover
      enumerations like ``1.`` or ``(1)``)
    * ``empty_reading``: reading is empty or all-latin (yomi failed)

Output:
    results/corpus_v2_sample/<pool>.sample.txt — 200 random lines with
        per-line signal tags
    results/corpus_v2_sample/summary.json — counts per signal per pool

Usage:
    uv run python -m tools.corpus_v2.sample_and_classify \
        --sample-size 200 \
        --out-dir results/corpus_v2_sample
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

POOLS = [
    ("wikinews_clean",   "datasets/v2/wikinews_v2.clean.jsonl"),
    ("wikibooks_clean",  "datasets/v2/wikibooks_v2.clean.jsonl"),
    ("wiktionary_clean", "datasets/v2/wiktionary_v2.clean.jsonl"),
    ("tatoeba_v2",       "datasets/v2/tatoeba_v2.jsonl"),
    ("aozora_dialogue",  "datasets/v2/aozora_dialogue.jsonl"),
]

MARKUP_PREFIX = re.compile(r"^\s*[#*=:]|^\s*\*\*|^\s*##")
MARKUP_MID = re.compile(r"\[\[|\]\]|\{\{|\}\}|<ref|</ref|<br|&nbsp;")
TEMPLATE_ARTIFACT = re.compile(r"^\*?\([^)]*\)|(\s|^)[（(][^)）]*[）)]\s*(ぶるがりあ|せいじ|しゃかい|すぽーつ|さんせい|こくさい)")
HAN = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
HIRA = re.compile(r"[\u3041-\u309F]")
KATA = re.compile(r"[\u30A1-\u30FA\u30FC]")
HANGUL = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")
CYRILLIC = re.compile(r"[\u0400-\u04FF]")
ARABIC = re.compile(r"[\u0600-\u06FF]")
HEBREW = re.compile(r"[\u0590-\u05FF]")
DIGIT_OR_BRACKET_ONLY = re.compile(r"^[\d\s.()（）\[\]]+$")
EMPTY_QUOTE = re.compile(r"「\s*」|『\s*』")


def classify(rec: dict) -> list[str]:
    surface = rec.get("surface", "")
    reading = rec.get("reading", "")
    tags = []

    if not surface or not reading:
        tags.append("empty_reading")
        return tags

    nfc = unicodedata.normalize("NFKC", surface)
    n = len(surface)

    if MARKUP_PREFIX.match(surface):
        tags.append("markup_prefix")
    if MARKUP_MID.search(surface):
        tags.append("markup_mid")
    if TEMPLATE_ARTIFACT.search(surface) or TEMPLATE_ARTIFACT.search(reading):
        tags.append("template_artifact")
    if EMPTY_QUOTE.search(surface):
        tags.append("empty_quote")

    # Script mixing — Han without hiragana/katakana => likely pure Chinese.
    has_kanji = bool(HAN.search(surface))
    has_kana = bool(HIRA.search(surface) or KATA.search(surface))
    if has_kanji and not has_kana:
        tags.append("foreign_script")
    if HANGUL.search(surface) or CYRILLIC.search(surface) or \
       ARABIC.search(surface) or HEBREW.search(surface):
        tags.append("foreign_script")

    ascii_chars = sum(1 for c in surface if ord(c) < 128 and not c.isspace())
    if n and ascii_chars / n > 0.4:
        tags.append("heavy_ascii")

    jp_chars = sum(1 for c in surface if HIRA.match(c) or KATA.match(c) or HAN.match(c))
    if n and jp_chars / n < 0.4:
        tags.append("symbol_heavy")

    if n < 8:
        tags.append("short")

    if DIGIT_OR_BRACKET_ONLY.match(surface):
        tags.append("numeric_form")

    # Yomi check: reading is latin-only / empty => mecab failed
    reading_jp = sum(1 for c in reading if HIRA.match(c) or KATA.match(c))
    if reading_jp == 0 and reading:
        tags.append("empty_reading")

    if not tags:
        tags.append("OK")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--out-dir", default="results/corpus_v2_sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for pool, path in POOLS:
        p = Path(path)
        if not p.exists():
            print(f"skip {pool}: {path} not found")
            continue

        sample = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(sample) < args.sample_size:
                    sample.append(line)
                else:
                    j = rng.randrange(i + 1)
                    if j < args.sample_size:
                        sample[j] = line

        per_pool_counts: dict[str, int] = {}
        dump_lines: list[str] = []
        for s in sample:
            try:
                rec = json.loads(s)
            except Exception:
                continue
            tags = classify(rec)
            for t in tags:
                per_pool_counts[t] = per_pool_counts.get(t, 0) + 1
            tag_str = ",".join(tags)
            surface = rec.get("surface", "")[:100]
            reading = rec.get("reading", "")[:100]
            dump_lines.append(f"[{tag_str}]\n  s: {surface}\n  r: {reading}\n")

        # Sort counts by freq descending
        sorted_counts = dict(sorted(per_pool_counts.items(), key=lambda kv: -kv[1]))
        summary[pool] = {
            "n": len(sample),
            "counts": sorted_counts,
        }

        (out_dir / f"{pool}.sample.txt").write_text(
            "\n".join(dump_lines), encoding="utf-8"
        )
        print(f"\n== {pool} ({len(sample)} sampled) ==")
        for tag, c in sorted_counts.items():
            pct = c / len(sample) * 100
            print(f"  {tag:<22} {c:>4} ({pct:4.1f}%)")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSamples written to {out_dir}")


if __name__ == "__main__":
    main()
