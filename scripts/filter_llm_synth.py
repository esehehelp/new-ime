"""Drop low-quality rows from LLM-generated synth JSONL.

The single-row LLM synth scripts (A homophone, B names, C context-rich)
have non-trivial rates of:
  - reading containing non-kana characters (kanji left in the kana field)
  - reading that's far shorter than the surface kana would be (truncated)
  - reading that's far longer than surface (kana for a non-existent surface)

This filter keeps a row only if:
  1. reading is hiragana/katakana/digits/punctuation only (no kanji)
  2. left_context_reading (if present) likewise
  3. reading char count is within [0.6, 1.6] of surface char count
     (loose ratio because mixed surfaces have varying kana density)

Usage:
    python scripts/filter_llm_synth.py <in.jsonl> [<in2.jsonl> ...]
    → emits <in>.filtered.jsonl per input, plus a summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def has_no_kanji(s: str) -> bool:
    """True if string has no CJK ideographs (i.e., it's all kana/digits/punct)."""
    for ch in s:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:  # CJK Unified Ideographs
            return False
        if 0x3400 <= code <= 0x4DBF:  # CJK Extension A
            return False
        if 0x20000 <= code <= 0x2A6DF:  # CJK Extension B
            return False
    return True


def good_reading(reading: str, surface: str) -> tuple[bool, str]:
    if not reading or not surface:
        return False, "empty"
    if not has_no_kanji(reading):
        return False, "kanji_in_reading"
    rlen = len(reading)
    slen = len(surface)
    if slen == 0:
        return False, "empty_surface"
    ratio = rlen / slen
    if ratio < 0.6 or ratio > 2.5:
        return False, f"len_ratio={ratio:.2f}"
    return True, ""


def filter_file(path: Path) -> dict:
    out_path = path.with_suffix(".filtered.jsonl")
    counts = {"kept": 0, "kanji_in_reading": 0, "len_ratio": 0, "empty": 0, "parse": 0}
    with path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["parse"] += 1
                continue
            reading = row.get("reading") or ""
            surface = row.get("surface") or ""
            ok, why = good_reading(reading, surface)
            if not ok:
                if why.startswith("len_ratio"):
                    counts["len_ratio"] += 1
                else:
                    counts[why] = counts.get(why, 0) + 1
                continue
            # Also drop if left_context_reading has kanji
            lcr = row.get("left_context_reading") or ""
            if lcr and not has_no_kanji(lcr):
                counts["kanji_in_reading"] += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["kept"] += 1
    return {"path": str(path), "out": str(out_path), **counts}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    total = {"kept": 0, "dropped": 0}
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"skip: {path} not found", file=sys.stderr)
            continue
        stats = filter_file(path)
        kept = stats["kept"]
        dropped = sum(v for k, v in stats.items() if k not in {"path", "out", "kept"})
        total["kept"] += kept
        total["dropped"] += dropped
        print(
            f"{path.name}: kept={kept} dropped={dropped} "
            f"kanji_in_reading={stats.get('kanji_in_reading', 0)} "
            f"len_ratio={stats.get('len_ratio', 0)} "
            f"→ {stats['out']}",
            file=sys.stderr,
        )
    print(f"TOTAL kept={total['kept']} dropped={total['dropped']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
