//! Goji-data の rule 実装。かな直接レベルで音形/字形の混同を注入する。

use crate::kana_tables::{
    hira_to_kata, is_hiragana, is_katakana, kata_to_hira, CHOUON_VOWEL_MAP, DAKUTEN_TRIPLES,
    HOMOPHONE_KANA_PAIRS, SMALL_KANA_PAIRS,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GojiKind {
    DakutenFlip,
    SmallKana,
    ChouonConfusion,
    HiraKataConfuse,
    HomophoneKana,
}

#[derive(Clone, Debug)]
pub struct GojiConfig {
    pub mean_edits: f64,
    pub weights: Vec<(GojiKind, f64)>,
    pub length_drift_max: f64,
}

impl Default for GojiConfig {
    fn default() -> Self {
        Self {
            mean_edits: 1.0,
            weights: vec![
                (GojiKind::DakutenFlip, 0.40),
                (GojiKind::SmallKana, 0.25),
                (GojiKind::ChouonConfusion, 0.20),
                (GojiKind::HiraKataConfuse, 0.10),
                (GojiKind::HomophoneKana, 0.05),
            ],
            length_drift_max: 0.5,
        }
    }
}

/// 1 input reading にノイズを適用。None なら emit しない (品質ガード / 無変化)。
pub fn apply(reading: &str, cfg: &GojiConfig, rng: &mut StdRng) -> Option<String> {
    if reading.is_empty() {
        return None;
    }
    let lambda = cfg.mean_edits.max(0.0);
    let edits = if lambda == 0.0 {
        1
    } else {
        Poisson::new(lambda).ok()?.sample(rng).max(1.0) as u32
    };

    let mut chars: Vec<String> = reading.chars().map(|c| c.to_string()).collect();
    let mut any_applied = false;

    for _ in 0..edits {
        let Some(kind) = pick_kind(&cfg.weights, rng) else {
            break;
        };
        let ok = match kind {
            GojiKind::DakutenFlip => rule_dakuten_flip(&mut chars, rng),
            GojiKind::SmallKana => rule_small_kana(&mut chars, rng),
            GojiKind::ChouonConfusion => rule_chouon(&mut chars, rng),
            GojiKind::HiraKataConfuse => rule_hira_kata(&mut chars, rng),
            GojiKind::HomophoneKana => rule_homophone_kana(&mut chars, rng),
        };
        any_applied |= ok;
    }

    if !any_applied {
        return None;
    }

    let out: String = chars.concat();
    if out.is_empty() || out == reading {
        return None;
    }
    let orig_chars = reading.chars().count() as isize;
    let new_chars = out.chars().count() as isize;
    let drift = (new_chars - orig_chars).abs() as f64 / orig_chars.max(1) as f64;
    if drift > cfg.length_drift_max {
        return None;
    }
    Some(out)
}

fn pick_kind(weights: &[(GojiKind, f64)], rng: &mut StdRng) -> Option<GojiKind> {
    let total: f64 = weights.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return None;
    }
    let r = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (k, w) in weights {
        acc += *w;
        if r <= acc {
            return Some(*k);
        }
    }
    weights.last().map(|(k, _)| *k)
}

// ── rules ──────────────────────────────────────────────────────────────

/// 清音 ↔ 濁音 ↔ 半濁音 (random within the 3-way for は行、2-way for others)。
pub fn rule_dakuten_flip(chars: &mut [String], rng: &mut StdRng) -> bool {
    let positions: Vec<(usize, [bool; 3])> = chars
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            for tpl in DAKUTEN_TRIPLES {
                let slots = [c == tpl.0, c == tpl.1, c == tpl.2 && !tpl.2.is_empty()];
                if slots.iter().any(|b| *b) {
                    return Some((i, slots));
                }
            }
            None
        })
        .collect();
    if positions.is_empty() {
        return false;
    }
    let (idx, slots) = positions[rng.gen_range(0..positions.len())].clone();
    // 対応する triple を引き直す
    let cur = &chars[idx];
    let tpl = DAKUTEN_TRIPLES
        .iter()
        .find(|t| cur == t.0 || cur == t.1 || cur == t.2)
        .unwrap();
    // 現在 slot 以外の有効な slot からランダム選択
    let candidates: Vec<&str> = [tpl.0, tpl.1, tpl.2]
        .iter()
        .enumerate()
        .filter(|(i, s)| !slots[*i] && !s.is_empty())
        .map(|(_, s)| *s)
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let pick = candidates[rng.gen_range(0..candidates.len())];
    chars[idx] = pick.to_string();
    true
}

/// 大 ↔ 小書き の入替。促音 (っ) の挿入/削除もここで扱う。
pub fn rule_small_kana(chars: &mut Vec<String>, rng: &mut StdRng) -> bool {
    // 2 operation: (A) 大小 flip in-place, (B) っ の挿入 / 削除
    let flip_positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            for (big, small) in SMALL_KANA_PAIRS {
                if c == big || c == small {
                    return Some(i);
                }
            }
            None
        })
        .collect();
    let op = rng.gen_range(0..2);
    if op == 0 && !flip_positions.is_empty() {
        let i = flip_positions[rng.gen_range(0..flip_positions.len())];
        let cur = &chars[i];
        for (big, small) in SMALL_KANA_PAIRS {
            if cur == big {
                chars[i] = small.to_string();
                return true;
            }
            if cur == small {
                chars[i] = big.to_string();
                return true;
            }
        }
        false
    } else {
        // っ 挿入 or 削除
        let tsu_positions: Vec<usize> = chars
            .iter()
            .enumerate()
            .filter(|(_, c)| c == &"っ")
            .map(|(i, _)| i)
            .collect();
        if !tsu_positions.is_empty() && rng.gen::<f64>() < 0.5 {
            // 削除
            let idx = tsu_positions[rng.gen_range(0..tsu_positions.len())];
            chars.remove(idx);
            return true;
        }
        // 挿入: 2..len-1 の適当な位置 (先頭末尾を避ける)
        if chars.len() < 3 {
            return false;
        }
        let pos = rng.gen_range(1..chars.len());
        chars.insert(pos, "っ".into());
        true
    }
}

/// 長音 ー と母音の相互変換。
///
/// Pattern A: `ー` → 母音。前の字の母音列で意図される vowel を推定するのは難しい
/// ので、「あ/い/う/え/お」をランダム pick する。
/// Pattern B: 任意の母音 (2 文字目以降) → `ー`。例: "こう" (i=1 'う') → "こー"。
pub fn rule_chouon(chars: &mut Vec<String>, rng: &mut StdRng) -> bool {
    // Suppress unused warning — table kept for documentation / future tuning.
    let _ = CHOUON_VOWEL_MAP;
    const VOWELS: &[&str] = &["あ", "い", "う", "え", "お"];
    let mut candidates: Vec<(usize, &'static str)> = Vec::new();
    for i in 1..chars.len() {
        let cur = chars[i].as_str();
        // Pattern A: ー → vowel (pick later when we commit)
        if cur == "ー" {
            // put a sentinel; actual vowel picked at commit time
            candidates.push((i, ""));
            continue;
        }
        // Pattern B: vowel at i>=1 → ー
        if VOWELS.iter().any(|v| *v == cur) {
            candidates.push((i, "ー"));
        }
    }
    if candidates.is_empty() {
        return false;
    }
    let (idx, new_char) = candidates[rng.gen_range(0..candidates.len())];
    let replacement = if new_char.is_empty() {
        // ー → random vowel
        VOWELS[rng.gen_range(0..VOWELS.len())]
    } else {
        new_char
    };
    if chars[idx] == replacement {
        return false;
    }
    chars[idx] = replacement.to_string();
    true
}

/// 1 char だけカタカナ <-> ひらがな を入れ替える。
pub fn rule_hira_kata(chars: &mut [String], rng: &mut StdRng) -> bool {
    let positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.chars()
                .next()
                .map(|ch| is_hiragana(ch) || is_katakana(ch))
                .unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect();
    if positions.is_empty() {
        return false;
    }
    let idx = positions[rng.gen_range(0..positions.len())];
    let cur = chars[idx].chars().next().unwrap();
    let new_c = if is_hiragana(cur) {
        hira_to_kata(cur)
    } else {
        kata_to_hira(cur)
    };
    if let Some(nc) = new_c {
        chars[idx] = nc.to_string();
        true
    } else {
        false
    }
}

/// ぢ ↔ じ, づ ↔ ず, を ↔ お, は ↔ わ, へ ↔ え。
pub fn rule_homophone_kana(chars: &mut [String], rng: &mut StdRng) -> bool {
    let positions: Vec<usize> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            HOMOPHONE_KANA_PAIRS
                .iter()
                .any(|(a, b)| c == a || c == b)
        })
        .map(|(i, _)| i)
        .collect();
    if positions.is_empty() {
        return false;
    }
    let idx = positions[rng.gen_range(0..positions.len())];
    let cur = &chars[idx];
    for (a, b) in HOMOPHONE_KANA_PAIRS {
        if cur == a {
            chars[idx] = (*b).to_string();
            return true;
        }
        if cur == b {
            chars[idx] = (*a).to_string();
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn dakuten_flip_changes_reading() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::DakutenFlip, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = apply("かさたなはま", &cfg, &mut rng).expect("should flip");
        assert_ne!(out, "かさたなはま");
        // some dakuten char must appear
        assert!(out.contains(|c: char| "がざだばぱ".contains(c)));
    }

    #[test]
    fn small_kana_flip() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::SmallKana, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let mut hit = 0;
        for _ in 0..30 {
            if let Some(out) = apply("きょうはようこそ", &cfg, &mut rng) {
                if out != "きょうはようこそ" {
                    hit += 1;
                }
            }
        }
        assert!(hit > 0, "small-kana rule produced no change in 30 trials");
    }

    #[test]
    fn chouon_confusion_swaps_vowel_and_bar() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::ChouonConfusion, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(2);
        let out = apply("こうしえん", &cfg, &mut rng).expect("should swap");
        assert_ne!(out, "こうしえん");
    }

    #[test]
    fn hira_kata_confuse_basic() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::HiraKataConfuse, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(3);
        let out = apply("ありがとう", &cfg, &mut rng).expect("should convert one char");
        assert_ne!(out, "ありがとう");
        // at least 1 kata in the output
        assert!(out.chars().any(is_katakana));
    }

    #[test]
    fn homophone_kana_swap() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::HomophoneKana, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(4);
        let out = apply("これはおすすめですね", &cfg, &mut rng).expect("should swap");
        assert_ne!(out, "これはおすすめですね");
    }

    #[test]
    fn empty_input_returns_none() {
        let cfg = GojiConfig::default();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(apply("", &cfg, &mut rng).is_none());
    }

    #[test]
    fn no_applicable_chars_returns_none() {
        let cfg = GojiConfig {
            mean_edits: 1.0,
            weights: vec![(GojiKind::DakutenFlip, 1.0)],
            ..GojiConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        // "まん" has no dakuten candidates, rule can't fire.
        assert!(apply("まん", &cfg, &mut rng).is_none());
    }
}
