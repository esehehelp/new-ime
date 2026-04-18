"""Convert mozc's open-source dictionary files to fixed_dict.tsv format.

Input: dictionary00.txt .. dictionary09.txt from mozc/src/data/dictionary_oss/
       format: kana\\tleft_id\\tright_id\\tcost\\tsurface
Output: a fixed_dict TSV (kana\\tsurface, one line per homophone)

We keep an entry when:
  - surface != kana (skip identity rewrites — CTC handles them better)
  - cost <= --max-cost (lower mozc cost = more common; default 6000
    retains roughly the top ~80% of the dict while dropping long-tail
    proper nouns)
  - kana is all hiragana (skip katakana-reading loanword entries which
    would collide with our katakana passthrough policy)

Homophones are ordered by cost (most common first); ties by insertion
order.

Usage:
    uv run python -m scripts.import_mozc_dict \
        --src tools/mozc_import \
        --out models/fixed_dict_mozc.tsv \
        --max-cost 6000 \
        --max-per-reading 10

Licensing: mozc dict is Apache-2. Keep LICENSE notice when redistributing.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def is_all_hiragana(s: str) -> bool:
    """Accept ー (prolongation) too — common in katakana-readings we want
    to keep when the surface is a legit loanword transcription."""
    for c in s:
        if c in ("\u30FC",):  # ー
            continue
        if "\u3041" <= c <= "\u309F":  # hiragana
            continue
        return False
    return bool(s)


def has_kanji(s: str) -> bool:
    """True iff the surface contains at least one CJK ideograph. Pure
    hiragana / katakana surfaces aren't worth diverting from CTC — the
    model already gets those right and short dict matches just replace
    grammatical particles (の が は etc) with rare kanji."""
    for c in s:
        if "\u4E00" <= c <= "\u9FFF":  # CJK Unified Ideographs
            return True
        if "\u3400" <= c <= "\u4DBF":  # Extension A
            return True
    return False


# Readings that are almost always particles / functional morphemes;
# even with a cost filter the mozc dict emits rare kanji spellings for
# these that hurt more than they help. Kept as literals so it's obvious
# when someone adds / removes an entry.
PARTICLE_BLACKLIST = {
    "の", "が", "は", "を", "に", "と", "で", "へ", "や", "か",
    "から", "まで", "より", "ので", "のに", "けど", "けれど", "たら",
    "れば", "ても", "でも", "しか", "だけ", "ばかり", "ほど", "ながら",
    "って", "とか", "など", "なり", "やら", "でき", "いる", "ある",
    "する", "なる", "です", "ます", "だった", "です", "ました",
    "だろう", "でしょう", "かも", "ね", "よ", "な", "わ", "ぞ", "ぜ",
    "もの", "こと", "ため",  "とき", "ところ", "ほう", "うち",
    "た", "て", "で", "ば", "ば", "ろ", "よう", "そう",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="tools/mozc_import", help="dir with dictionary00.txt..09.txt")
    parser.add_argument("--out", default="models/fixed_dict_mozc.tsv")
    parser.add_argument(
        "--max-cost",
        type=int,
        default=6000,
        help="Drop entries with cost above this (mozc cost: lower = more common)",
    )
    parser.add_argument(
        "--max-per-reading",
        type=int,
        default=8,
        help="Cap homophone count per reading so common kana don't blow up the file",
    )
    parser.add_argument(
        "--min-kana-len",
        type=int,
        default=3,
        help="Skip very short readings — too ambiguous, and CTC handles grammar particles",
    )
    parser.add_argument(
        "--require-kanji",
        action="store_true",
        default=True,
        help="Drop entries whose surface is pure kana (CTC handles these) — default on",
    )
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # group[kana] -> list of (cost, surface) pairs, deduped by surface.
    group: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_pair: set[tuple[str, str]] = set()
    total_in = 0
    kept_in = 0
    for f in sorted(src_dir.glob("dictionary*.txt")):
        for raw in f.read_text(encoding="utf-8").splitlines():
            total_in += 1
            parts = raw.split("\t")
            if len(parts) < 5:
                continue
            kana, _, _, cost_s, surface = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                cost = int(cost_s)
            except ValueError:
                continue
            if cost > args.max_cost:
                continue
            if len(kana) < args.min_kana_len:
                continue
            if not is_all_hiragana(kana):
                continue
            if surface == kana:
                continue
            if kana in PARTICLE_BLACKLIST:
                continue
            if args.require_kanji and not has_kanji(surface):
                continue
            if (kana, surface) in seen_pair:
                continue
            seen_pair.add((kana, surface))
            group[kana].append((cost, surface))
            kept_in += 1

    # Sort homophones by ascending cost; cap per-reading.
    lines = []
    total_out = 0
    for kana in sorted(group.keys()):
        pairs = sorted(group[kana])[: args.max_per_reading]
        for _, surf in pairs:
            lines.append(f"{kana}\t{surf}")
            total_out += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"read {total_in} lines, kept {kept_in}, wrote {total_out} to {out_path}")
    print(f"unique readings: {len(group)}")


if __name__ == "__main__":
    main()
