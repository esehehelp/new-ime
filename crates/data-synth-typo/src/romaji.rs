//! hiragana ↔ romaji 変換 (Hepburn-ish)。longest-prefix match で拗音を先に。
//!
//! このモジュールは「ユーザが qwerty romaji 入力で犯しうるキー誤打」を
//! reading 側に注入するための橋渡し専用。完全な IME 変換ではないので、
//! 未対応かな (ゐ/ゑ/ヴァ/ヷ 等) を食わせたら None を返す → 呼び出し側で
//! rule skip する。

// longest-prefix-match 用に「拗音・特殊」を先頭に、清音/濁音/半濁音は後ろ。
// promote 効率のため要素順は長さ降順を保つこと。
pub(crate) const HIRA_TO_ROMAJI: &[(&str, &str)] = &[
    // ── 3 char (special) ──────────────────────────────────────
    // ── 2 char (拗音 + "っ"-cluster) ──────────────────────────
    ("きゃ", "kya"), ("きゅ", "kyu"), ("きょ", "kyo"),
    ("しゃ", "sha"), ("しゅ", "shu"), ("しょ", "sho"),
    ("ちゃ", "cha"), ("ちゅ", "chu"), ("ちょ", "cho"),
    ("にゃ", "nya"), ("にゅ", "nyu"), ("にょ", "nyo"),
    ("ひゃ", "hya"), ("ひゅ", "hyu"), ("ひょ", "hyo"),
    ("みゃ", "mya"), ("みゅ", "myu"), ("みょ", "myo"),
    ("りゃ", "rya"), ("りゅ", "ryu"), ("りょ", "ryo"),
    ("ぎゃ", "gya"), ("ぎゅ", "gyu"), ("ぎょ", "gyo"),
    ("じゃ", "ja"),  ("じゅ", "ju"),  ("じょ", "jo"),
    ("ぢゃ", "ja"),  ("ぢゅ", "ju"),  ("ぢょ", "jo"),
    ("びゃ", "bya"), ("びゅ", "byu"), ("びょ", "byo"),
    ("ぴゃ", "pya"), ("ぴゅ", "pyu"), ("ぴょ", "pyo"),
    ("ふぁ", "fa"),  ("ふぃ", "fi"),  ("ふぇ", "fe"),  ("ふぉ", "fo"),
    ("ゔぁ", "va"),  ("ゔぃ", "vi"),  ("ゔ",   "vu"),  ("ゔぇ", "ve"),  ("ゔぉ", "vo"),
    // ── 1 char (清音・濁音・半濁音) ────────────────────────────
    ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
    ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
    ("さ", "sa"), ("し", "shi"), ("す", "su"), ("せ", "se"), ("そ", "so"),
    ("た", "ta"), ("ち", "chi"), ("つ", "tsu"), ("て", "te"), ("と", "to"),
    ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
    ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
    ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
    ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
    ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
    ("わ", "wa"), ("を", "wo"), ("ん", "n"),
    ("が", "ga"), ("ぎ", "gi"), ("ぐ", "gu"), ("げ", "ge"), ("ご", "go"),
    ("ざ", "za"), ("じ", "ji"), ("ず", "zu"), ("ぜ", "ze"), ("ぞ", "zo"),
    ("だ", "da"), ("ぢ", "ji"), ("づ", "zu"), ("で", "de"), ("ど", "do"),
    ("ば", "ba"), ("び", "bi"), ("ぶ", "bu"), ("べ", "be"), ("ぼ", "bo"),
    ("ぱ", "pa"), ("ぴ", "pi"), ("ぷ", "pu"), ("ぺ", "pe"), ("ぽ", "po"),
    // 長音符・句読点はそのまま通す (romaji 上では `-` `、` `。` 等、往復不能は
    // romaji_to_kana 側で ASCII passthrough として扱う)。
    ("ー", "-"),
];

/// 単独では variants を持つ特殊かな (promote 管理する必要あり)。
/// 小書き単独の "ゃ" 等はそのまま attach されない限り round-trip できないので
/// 呼び出し側で rule skip。
const STANDALONE_FAIL: &[&str] = &[
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "ゃ", "ゅ", "ょ", "ゎ", "っ",
];

/// hiragana 文字列を romaji に変換。未対応かなを含む場合は None。
pub fn kana_to_romaji(kana: &str) -> Option<String> {
    let bytes = kana.as_bytes();
    let mut out = String::with_capacity(kana.len() * 2);
    let mut i = 0;
    while i < bytes.len() {
        // 次 codepoint の表示長 (バイト数) を求める
        let remaining = std::str::from_utf8(&bytes[i..]).ok()?;

        // "っ" は次の子音を重ねる扱い (longest-match table には載せない、
        // ここで専用処理)。末尾の "っ" は "tsu" として扱う。
        if remaining.starts_with('っ') {
            let cho = 'っ'.len_utf8();
            // 次 token を lookup
            let after = &remaining[cho..];
            if after.is_empty() {
                out.push_str("tsu");
                i += cho;
                continue;
            }
            if let Some((k, r)) = longest_prefix(after) {
                // 子音頭を 1 文字重ねて emit (例: "って" → "tte")
                if let Some(c) = r.chars().next() {
                    if c.is_ascii_alphabetic() {
                        out.push(c);
                    } else {
                        // 母音始まり (a/i/u/e/o) は重ねられない → "ltu" で明示
                        // ここは稀なので romaji 上 "xtu" (conservative)
                        out.push_str("xtu");
                    }
                }
                out.push_str(r);
                i += cho + k.len();
                continue;
            }
            // 次 token が未対応 → 全体 None
            return None;
        }

        // standalone 小書きは独立 romaji が無い (通常 preceding kana と合成済)
        for bad in STANDALONE_FAIL {
            if remaining.starts_with(bad) {
                return None;
            }
        }

        if let Some((k, r)) = longest_prefix(remaining) {
            out.push_str(r);
            i += k.len();
            continue;
        }
        // table にない文字 (ascii 記号、改行、その他) は 1 char だけ passthrough
        let c = remaining.chars().next()?;
        out.push(c);
        i += c.len_utf8();
    }
    Some(out)
}

fn longest_prefix(s: &str) -> Option<(&'static str, &'static str)> {
    for (k, r) in HIRA_TO_ROMAJI {
        if s.starts_with(k) {
            return Some((*k, *r));
        }
    }
    None
}

/// romaji → hiragana 逆変換。不正 romaji はそのまま ASCII で保持。
/// Greedy longest-match on ASCII lowercase; 促音は "tt" → "って" 相当に戻さず、
/// 単純に子音重複を "っ + (その子音の kana)" と解釈する小さな前処理を挟む。
pub fn romaji_to_kana(romaji: &str) -> String {
    let mut out = String::with_capacity(romaji.len() * 3);
    let mut rest: &str = romaji;
    while !rest.is_empty() {
        // 子音重複 → 促音
        let bytes = rest.as_bytes();
        if bytes.len() >= 2 {
            let a = bytes[0].to_ascii_lowercase();
            let b = bytes[1].to_ascii_lowercase();
            if a == b && is_doubleable_consonant(a) {
                out.push('っ');
                rest = &rest[1..];
                continue;
            }
        }
        // 撥音の特殊処理 "n" + (子音 | 終端 | "'") → "ん"
        // 先にテーブル match を試す。"na/ni/nu/ne/no/nya..." は "な..." に match する
        // ので、ここに来る "n" は単独撥音のみ。
        if let Some((k, r)) = longest_romaji_prefix(rest) {
            out.push_str(k);
            rest = &rest[r.len()..];
            continue;
        }
        // 単独 "n" は撥音として消費
        if rest.starts_with('n') || rest.starts_with('N') {
            out.push('ん');
            rest = &rest[1..];
            continue;
        }
        // 非 romaji (記号/数字/ひらがなそのまま) は 1 char ずつ passthrough
        let c = rest.chars().next().unwrap();
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

fn is_doubleable_consonant(b: u8) -> bool {
    // 母音/y/n/a..e 以外の子音のみ促音化を認める
    matches!(
        b,
        b'k' | b'g' | b's' | b'z' | b'j' | b't' | b'd' | b'c' | b'p' | b'b' | b'f' | b'h' | b'm' | b'r' | b'l' | b'w'
    )
}

/// romaji テーブルから longest prefix で `(kana, romaji_used)` を返す。
fn longest_romaji_prefix(s: &str) -> Option<(&'static str, &'static str)> {
    // HIRA_TO_ROMAJI は kana 長い順に並んでいるが、romaji 長で検索したいので
    // 線形検索 + 一致時に最長記録で打ち切り。
    let lower: String = s
        .chars()
        .take(4)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut best: Option<(&'static str, &'static str)> = None;
    for (k, r) in HIRA_TO_ROMAJI {
        if lower.starts_with(*r) {
            match best {
                Some((_, br)) if br.len() >= r.len() => {}
                _ => best = Some((*k, *r)),
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_basic() {
        for word in ["こんにちは", "ありがとう", "さようなら", "にっぽん", "きょう", "じしょ"] {
            let r = kana_to_romaji(word).expect(word);
            let back = romaji_to_kana(&r);
            assert_eq!(back, word, "round-trip failed: {} -> {} -> {}", word, r, back);
        }
    }

    #[test]
    fn rejects_standalone_small_kana() {
        assert!(kana_to_romaji("ゃ").is_none());
        assert!(kana_to_romaji("ぁい").is_none());
    }

    #[test]
    fn dakuten_and_handakuten() {
        assert_eq!(kana_to_romaji("がっこう").as_deref(), Some("gakkou"));
        assert_eq!(kana_to_romaji("ぱーてぃー"), None, "ぱ + ー = partii but ティ is not covered");
        // 簡易 assert: 濁音/半濁音の基本
        assert_eq!(kana_to_romaji("ぱん").as_deref(), Some("pan"));
    }
}
