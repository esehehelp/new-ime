"""Extract dialogue lines from the existing aozora corpus.

The raw `datasets/aozora_clean.jsonl` is full sentences from literary
works; Japanese fiction uses 「...」 quotes for spoken dialogue. Pulling
just those lines gives a colloquial register corpus with the phonetic
readings already computed — no mecab re-pass needed.

Output: `datasets/v2/aozora_dialogue.jsonl` in the same
reading/surface/context schema as the other training pools.

We keep a line when BOTH the surface AND the reading appear to be a
single 「...」 utterance (or contain «spoken quote»-style content).
Narrator lines that happen to contain an embedded quote are dropped;
they mix registers and confuse the training signal.

Usage:
    uv run python -m tools.corpus_v2.extract_aozora_dialogue \
        --src datasets/aozora_clean.jsonl \
        --out datasets/v2/aozora_dialogue.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


# Lines that start with 「 and end with 」 (with optional trailing
# period) are unambiguously dialogue. We also accept 『 / 』 nested
# quotes. The reading follows the same bracket so we require both
# sides to match.
DIALOGUE_RE = re.compile(r"^[「『].*[」』][。！？!?]?\s*$")

# Minimum / maximum char lengths to keep; very short filler like
# 「……」 or "「あっ」" aren't useful training targets.
MIN_LEN = 3
MAX_LEN = 80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="datasets/aozora_clean.jsonl")
    parser.add_argument("--out", default="datasets/v2/aozora_dialogue.jsonl")
    parser.add_argument("--report-every", type=int, default=500_000)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with src.open("rb") as f, out.open("w", encoding="utf-8") as g:
        for line in f:
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            surface = d.get("surface", "")
            reading = d.get("reading", "")
            if not surface or not reading:
                continue
            if not DIALOGUE_RE.match(surface):
                continue
            if not DIALOGUE_RE.match(reading):
                continue
            if len(surface) < MIN_LEN or len(surface) > MAX_LEN:
                continue

            out_rec = {
                "reading": reading,
                "surface": surface,
                "context": d.get("context", ""),
                "source": "aozora_dialogue",
            }
            g.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            kept += 1
            if total % args.report_every == 0:
                print(f"  scanned {total:,} kept {kept:,}", flush=True)

    print(f"done: scanned {total:,} kept {kept:,} -> {out}")


if __name__ == "__main__":
    main()
