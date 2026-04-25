//! Katakana → hiragana conversion, bit-for-bit compatible with `jaconv.kata2hira`.
//!
//! Behaviour is fixed by the existing training/eval data pipeline, so the
//! conversion rules MUST match jaconv exactly:
//!
//! - `U+30A1..=U+30F6` (most katakana) → shift down by `0x60` into hiragana.
//! - `U+30FD` (ヽ, iteration mark) → `U+309D` (ゝ).
//! - `U+30FE` (ヾ, voiced iteration mark) → `U+309E` (ゞ).
//! - `U+30FC` (ー, prolonged sound mark) → unchanged (same codepoint is used
//!   for hiragana prolongation).
//! - All other characters (ASCII, punctuation, already-hiragana, etc.) pass
//!   through untouched.

/// Convert katakana characters to hiragana using the jaconv-compatible rules
/// described at the module level.
pub fn kata_to_hira(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        let code = ch as u32;
        let mapped = match code {
            0x30A1..=0x30F6 => code - 0x60,
            0x30FD => 0x309D,
            0x30FE => 0x309E,
            _ => {
                out.push(ch);
                continue;
            }
        };
        if let Some(mapped_ch) = char::from_u32(mapped) {
            out.push(mapped_ch);
        } else {
            out.push(ch);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_jaconv_samples() {
        // Ground truth verified against Python jaconv.
        assert_eq!(kata_to_hira("フクザワユキチ"), "ふくざわゆきち");
        assert_eq!(
            kata_to_hira("イマ、ショウボウショクインニハ"),
            "いま、しょうぼうしょくいんには",
        );
        assert_eq!(kata_to_hira("モード"), "もーど");
        assert_eq!(kata_to_hira("ABC123"), "ABC123");
        assert_eq!(kata_to_hira("ヽヾ"), "ゝゞ");
    }

    #[test]
    fn empty_input() {
        assert_eq!(kata_to_hira(""), "");
    }

    #[test]
    fn prolonged_mark_preserved() {
        assert_eq!(kata_to_hira("コーヒー"), "こーひー");
    }

    #[test]
    fn half_width_kana_untouched() {
        // Half-width katakana (U+FF65..) is not in the main range; jaconv
        // would not convert these via kata2hira either.
        assert_eq!(kata_to_hira("ｱｲｳ"), "ｱｲｳ");
    }
}
