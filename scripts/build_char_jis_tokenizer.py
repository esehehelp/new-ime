"""Build a wider char tokenizer for CTC: full CJK Basic + ASCII rescue.

設計方針:
- char 単位 (subword 化しない、CTC alignment ambiguity 回避)
- ASCII printable / 全角 ASCII / JP_SYMBOLS は frequency 関係なく一律入れる
  (char-5k で `%`, `[`, `]`, `@`, `^`, `~` が freq cutoff で落ちた罠を排除)
- CJK Unified Ideographs (U+4E00-U+9FFF) 全 20992 を入れる
  → 人名 / 固有名詞 / rare kanji も decode 出力可能 (char-5k では 1903 ids が
    train で出ない死に weight だったが、新 vocab は逆に "出力可能性を最大化")
- byte fallback 256 (絵文字 / Extension A 以降の rare chars 用 fail-safe)
- 互換性は無視 (shard 再 compile 必須)

CJK Extension A (U+3400-U+4DBF, 6592 chars) は v2 で追加検討。Basic だけで
JIS 第一+第二+補助は完全カバー。

vocab 約 22000 で output projection params: hidden 384 × 22000 = 8.4M。
tied embedding 経由で encoder embedding にも反映 → 30M model が ~38M 規模。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from new_ime.data.tokenizer import (
    HIRAGANA_END,
    HIRAGANA_START,
    JP_SYMBOLS,
    KANA_MARKS,
    KATAKANA_END,
    KATAKANA_START,
    SPECIAL_TOKENS,
)


def build() -> dict[str, int]:
    token_to_id: dict[str, int] = {}

    def add(token: str) -> None:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    # 1. Specials (PAD/UNK/SEP/CLS/BLANK/MASK) — IDs 0-5 (frozen by Rust side)
    for tok in SPECIAL_TOKENS:
        add(tok)

    # 2. Byte fallback 256 — IDs 6-261 (frozen)
    for b in range(256):
        add(f"<0x{b:02X}>")

    # 3. Hiragana (U+3041..U+3096)
    for cp in range(HIRAGANA_START, HIRAGANA_END + 1):
        add(chr(cp))

    # 4. Katakana (U+30A1..U+30FA)
    for cp in range(KATAKANA_START, KATAKANA_END + 1):
        add(chr(cp))

    # 5. Kana marks
    for ch in KANA_MARKS:
        add(ch)

    # 6. ASCII printable (U+0020..U+007E) — 95 chars
    for cp in range(0x20, 0x7F):
        add(chr(cp))

    # 7. Full-width ASCII (U+FF01..U+FF5E) — 94 chars
    for cp in range(0xFF01, 0xFF5F):
        add(chr(cp))

    # 8. JP symbols
    for ch in JP_SYMBOLS:
        add(ch)

    # 9. CJK Unified Ideographs Basic (U+4E00..U+9FFF) — 20992 chars
    #    JIS X 0208 第一+第二+補助 + 常用漢字すべて含む
    for cp in range(0x4E00, 0xA000):
        ch = chr(cp)
        add(ch)

    # 10. Symbol / punct blocks (corpus サンプルで OOV 上位を占めた範囲)
    #     '%', '×', '°', '〜', '―', '“', '”', '★', '♪', 'Ω', 'κ', '①', '㎡' 等
    symbol_ranges = [
        (0x00A0, 0x00FF),  # Latin-1 Supplement (×, °, ¥, »)
        (0x0300, 0x036F),  # Combining Diacritical Marks
        (0x0370, 0x03FF),  # Greek and Coptic (Ω, κ, π)
        (0x2000, 0x206F),  # General Punctuation (―, ", ", —, ※, ‼, NBSP系)
        (0x2100, 0x214F),  # Letterlike Symbols (℃, ™)
        (0x2150, 0x218F),  # Number Forms (Ⅰ-Ⅻ)
        (0x2190, 0x21FF),  # Arrows (→, ⇒)
        (0x2200, 0x22FF),  # Mathematical Operators (−, ≠, ≤)
        (0x2300, 0x23FF),  # Misc Technical
        (0x2460, 0x24FF),  # Enclosed Alphanumerics (①, ②, ③)
        (0x2500, 0x257F),  # Box Drawing
        (0x2580, 0x259F),  # Block Elements (█)
        (0x25A0, 0x25FF),  # Geometric Shapes (■, ★, ●, ◎)
        (0x2600, 0x26FF),  # Misc Symbols (♪, ♡, ☆)
        (0x2700, 0x27BF),  # Dingbats (✨)
        (0x3000, 0x303F),  # CJK Symbols and Punctuation (々, 〆, 〇 — JP_SYMBOLS と部分重複)
        (0x3099, 0x309F),  # Combining 濁点 / 半濁点 + Hiragana 拡張
        (0x3300, 0x33FF),  # CJK Compatibility (㎏, ㎡, ㎜)
        (0xFE30, 0xFE4F),  # CJK Compatibility Forms (︱, ︵)
        (0xFF00, 0xFFEF),  # Halfwidth/Fullwidth Forms (ｶ, ﾞ, ￥; 半角全部 + 全角ASCII 重複)
    ]
    for start, end in symbol_ranges:
        for cp in range(start, end + 1):
            ch = chr(cp)
            add(ch)

    return token_to_id


def main() -> int:
    token_to_id = build()
    out_path = Path("datasets/tokenizers/char-jis-24k.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"type": "shared", "max_kanji": 0, "token_to_id": token_to_id},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"vocab_size = {len(token_to_id)}")
    print(f"written: {out_path}")
    print(f"  specials:        6 (IDs 0-5)")
    print(f"  byte fallback:   256 (IDs 6-261)")
    n_kana = (HIRAGANA_END - HIRAGANA_START + 1) + (KATAKANA_END - KATAKANA_START + 1) + len(KANA_MARKS)
    print(f"  kana group:      {n_kana}")
    print(f"  ASCII + 全角 + JP symbols: {95 + 94 + len(JP_SYMBOLS)}")
    print(f"  CJK Basic:       {0xA000 - 0x4E00}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
