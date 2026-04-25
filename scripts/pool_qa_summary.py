"""Aggregate quality flags across all round_1 samples for quick user review.

Reads datasets/audits/pool-qa/<pool>/round_1/samples.jsonl and counts
obvious issues (missing fields, unicode classes, length, symbol-lead).
Prints a sortable summary table so the user can spot which pools need
the most attention first.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

AUDIT = Path(r"D:/Dev/new-ime/datasets/audits/pool-qa")

# Unicode classes
HIRA = re.compile(r"[ぁ-ゖ]")
KATA = re.compile(r"[ァ-ヺ]")
KANJI = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9０-９]")

# Symbol/punct start (rejected per user: 句読点、記号始まり)
# EXCEPTION: 「 / 」 remain allowed — natural for colloquial dialogue.
# Leading punctuation: 、。 !？ , . ! ? 『』（）〔〕【】〈〉《》・ー… and ASCII punct
PUNCT_START = re.compile(r"^[\s　、。！？,.!?『』（）()\[\]【】〔〕〈〉《》・ーー…‥々〇〻※\-_:;'\"”“‘’]")

# Reading should be mostly hiragana (+ optional ー and symbols). Katakana in
# reading signals raw tokenization (should have been katakana->hiragana).
# Kanji in reading is almost always wrong.
READING_KANJI = re.compile(r"[一-鿿]")
READING_LATIN = re.compile(r"[A-Za-z]")


def check_row(d: dict) -> list[str]:
    flags: list[str] = []
    reading = d.get("reading", "") or ""
    surface = d.get("surface", "") or ""
    if not reading:
        flags.append("empty_reading")
    if not surface:
        flags.append("empty_surface")
    if reading and PUNCT_START.match(reading):
        flags.append("reading_symbol_start")
    if surface and PUNCT_START.match(surface):
        flags.append("surface_symbol_start")
    if reading and READING_KANJI.search(reading):
        flags.append("reading_has_kanji")
    if reading and KATA.search(reading):
        flags.append("reading_has_katakana")
    if reading and READING_LATIN.search(reading):
        flags.append("reading_has_latin")
    # length-based
    if reading and len(reading) < 2:
        flags.append("reading_too_short")
    if surface and len(surface) < 1:
        flags.append("surface_too_short")
    if reading and len(reading) > 200:
        flags.append("reading_too_long")
    if surface and len(surface) > 200:
        flags.append("surface_too_long")
    # length ratio heuristic: reading is usually ~same len or longer than surface
    if reading and surface:
        r = len(reading) / max(len(surface), 1)
        if r < 0.4 or r > 3.0:
            flags.append("length_ratio_off")
    return flags


def main() -> None:
    pools = sorted(p for p in AUDIT.iterdir() if p.is_dir())
    if not pools:
        print("No pools found under", AUDIT)
        sys.exit(1)

    headers = [
        "empty_reading",
        "empty_surface",
        "reading_symbol_start",
        "surface_symbol_start",
        "reading_has_kanji",
        "reading_has_katakana",
        "reading_has_latin",
        "reading_too_short",
        "reading_too_long",
        "surface_too_long",
        "length_ratio_off",
    ]
    header_row = f"{'pool':<32} {'n':>4} " + " ".join(f"{h[:12]:>13}" for h in headers)
    print(header_row)
    print("-" * len(header_row))

    rows = []
    for p in pools:
        samples = p / "round_1" / "samples.jsonl"
        if not samples.exists():
            continue
        counts = {h: 0 for h in headers}
        n = 0
        with open(samples, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                n += 1
                for flag in check_row(d):
                    if flag in counts:
                        counts[flag] += 1
        rows.append((p.name, n, counts))
        row_str = f"{p.name:<32} {n:>4} " + " ".join(f"{counts[h]:>13}" for h in headers)
        print(row_str)


if __name__ == "__main__":
    main()
