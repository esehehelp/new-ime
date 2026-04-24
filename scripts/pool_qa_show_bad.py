"""Extract bad-looking rows from round_1 samples for user review.

For each pool, prints up to N rows that trip any of the current QA flags.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

AUDIT = Path(r"D:/Dev/new-ime/datasets/audits/pool-qa")
N_PER_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 5

HIRA = re.compile(r"[ぁ-ゖ]")
KATA = re.compile(r"[ァ-ヺ]")
KANJI = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")
# Excludes 「」 per user directive.
PUNCT_START = re.compile(r"^[\s　、。！？,.!?『』（）()\[\]【】〔〕〈〉《》・ー…‥々〇〻※\-_:;'\"”“‘’]")

def flags(d):
    out = []
    r = d.get("reading", "") or ""
    s = d.get("surface", "") or ""
    if PUNCT_START.match(r):
        out.append("r-sym")
    if PUNCT_START.match(s):
        out.append("s-sym")
    if KANJI.search(r):
        out.append("r-kanji")
    if KATA.search(r):
        out.append("r-kata")
    if LATIN.search(r):
        out.append("r-latin")
    if r and len(r) < 2:
        out.append("r-short")
    return out


def main():
    for p in sorted(AUDIT.iterdir()):
        if not p.is_dir():
            continue
        samples = p / "round_1" / "samples.jsonl"
        if not samples.exists():
            continue
        bads = []
        with open(samples, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                fl = flags(d)
                if fl:
                    bads.append((fl, d))
                    if len(bads) >= N_PER_POOL:
                        break
        if not bads:
            continue
        print(f"\n=== {p.name} ({len(bads)} bad shown) ===")
        for fl, d in bads:
            print(f"  [{','.join(fl):<20}]  r={d.get('reading','')[:50]!r}")
            print(f"                          s={d.get('surface','')[:50]!r}")


if __name__ == "__main__":
    main()
