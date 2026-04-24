//! かな直接レベルの混同 rule で使う静的テーブル群。
//!
//! 濁点/半濁点 / 大小 / 長音 / 同音異表記 の 4 種。`rules` 側が位置ごとに
//! applicable かを判定、`rng` で uniform 抽選、1 位置だけ書き換える。

/// 清音 / 濁音 / 半濁音 の 3-way。半濁音が無い行 (か・さ・た 行) は空文字列。
pub const DAKUTEN_TRIPLES: &[(&str, &str, &str)] = &[
    ("か", "が", ""), ("き", "ぎ", ""), ("く", "ぐ", ""), ("け", "げ", ""), ("こ", "ご", ""),
    ("さ", "ざ", ""), ("し", "じ", ""), ("す", "ず", ""), ("せ", "ぜ", ""), ("そ", "ぞ", ""),
    ("た", "だ", ""), ("ち", "ぢ", ""), ("つ", "づ", ""), ("て", "で", ""), ("と", "ど", ""),
    ("は", "ば", "ぱ"), ("ひ", "び", "ぴ"), ("ふ", "ぶ", "ぷ"), ("へ", "べ", "ぺ"), ("ほ", "ぼ", "ぽ"),
];

/// 大きな仮名 → 小書き仮名。逆向きも rule 側で対称に扱う。
pub const SMALL_KANA_PAIRS: &[(&str, &str)] = &[
    ("や", "ゃ"), ("ゆ", "ゅ"), ("よ", "ょ"),
    ("つ", "っ"),
    ("あ", "ぁ"), ("い", "ぃ"), ("う", "ぅ"), ("え", "ぇ"), ("お", "ぉ"),
    ("わ", "ゎ"),
];

/// 長音 (ー) と、直前の母音に対応する純母音の相互変換候補。
/// 例: "こう" ↔ "こー", "せい" ↔ "せー"
pub const CHOUON_VOWEL_MAP: &[(char, char)] = &[
    ('あ', 'ー'), ('い', 'ー'), ('う', 'ー'), ('え', 'ー'), ('お', 'ー'),
];

/// 同音異表記 (ぢ/じ, づ/ず, を/お, は/わ, へ/え)。default weight は低め。
pub const HOMOPHONE_KANA_PAIRS: &[(&str, &str)] = &[
    ("じ", "ぢ"), ("ず", "づ"),
    ("を", "お"),
    ("は", "わ"), ("へ", "え"),
];

/// ひらがな 1 char を対応カタカナ 1 char にマップ (rule_hira_kata で使用)。
/// U+3041..U+3096 → U+30A1..U+30F6。BMP 内、unicode offset +0x60。
pub const HIRA_KATA_OFFSET: u32 = 0x60;
pub const HIRAGANA_START: u32 = 0x3041;
pub const HIRAGANA_END: u32 = 0x3096;
pub const KATAKANA_START: u32 = 0x30A1;
pub const KATAKANA_END: u32 = 0x30F6;

/// 与えた char がひらがな範囲か。
pub fn is_hiragana(c: char) -> bool {
    let u = c as u32;
    u >= HIRAGANA_START && u <= HIRAGANA_END
}

/// 与えた char がカタカナ範囲か (ー や ・ は除く)。
pub fn is_katakana(c: char) -> bool {
    let u = c as u32;
    u >= KATAKANA_START && u <= KATAKANA_END
}

pub fn hira_to_kata(c: char) -> Option<char> {
    if is_hiragana(c) {
        char::from_u32(c as u32 + HIRA_KATA_OFFSET)
    } else {
        None
    }
}

pub fn kata_to_hira(c: char) -> Option<char> {
    if is_katakana(c) {
        char::from_u32(c as u32 - HIRA_KATA_OFFSET)
    } else {
        None
    }
}
