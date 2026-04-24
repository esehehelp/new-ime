//! Type-data の rule 実装。reading を romaji 経由で変形して返す。
//!
//! Phase A はひとまず `AdjacentKey` のみ。Phase B で Deletion / Insertion /
//! Transposition / NoConvert を追加する。

use crate::{qwerty, romaji};
use rand::rngs::StdRng;
use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TypoKind {
    AdjacentKey,
}

#[derive(Clone, Debug)]
pub struct TypoConfig {
    pub mean_edits: f64,
    pub weights: Vec<(TypoKind, f64)>,
}

impl Default for TypoConfig {
    fn default() -> Self {
        Self {
            mean_edits: 1.0,
            weights: vec![(TypoKind::AdjacentKey, 1.0)],
        }
    }
}

/// 1 input reading にノイズを適用。品質ガードで通らなければ None。
pub fn apply(reading: &str, cfg: &TypoConfig, rng: &mut StdRng) -> Option<String> {
    if reading.is_empty() {
        return None;
    }
    let romaji = romaji::kana_to_romaji(reading)?;
    if romaji.is_empty() {
        return None;
    }
    // Phase A: 固定で 1 回適用 (Phase B で Poisson 化)。
    let mut buf: Vec<u8> = romaji.into_bytes();
    let kind = pick_kind(&cfg.weights, rng)?;
    match kind {
        TypoKind::AdjacentKey => {
            if !rule_adjacent_key(&mut buf, rng) {
                return None;
            }
        }
    }
    let noisy_romaji = String::from_utf8(buf).ok()?;
    if noisy_romaji.is_empty() {
        return None;
    }
    let back = romaji::romaji_to_kana(&noisy_romaji);
    if back.is_empty() {
        return None;
    }
    let orig_chars = reading.chars().count() as isize;
    let new_chars = back.chars().count() as isize;
    let drift = (new_chars - orig_chars).abs() as f64 / orig_chars.max(1) as f64;
    if drift > 0.5 {
        return None;
    }
    Some(back)
}

fn pick_kind(weights: &[(TypoKind, f64)], rng: &mut StdRng) -> Option<TypoKind> {
    if weights.is_empty() {
        return None;
    }
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

/// ascii 1 char を qwerty 隣接キーへ置換。swap 対象が見つからなければ false。
pub fn rule_adjacent_key(buf: &mut Vec<u8>, rng: &mut StdRng) -> bool {
    // applicable 位置 = ascii_alphabetic
    let positions: Vec<usize> = buf
        .iter()
        .enumerate()
        .filter(|(_, b)| (**b as char).is_ascii_alphabetic())
        .map(|(i, _)| i)
        .collect();
    if positions.is_empty() {
        return false;
    }
    let idx = positions[rng.gen_range(0..positions.len())];
    let orig = buf[idx] as char;
    let neighbors = qwerty::adjacent_keys(orig);
    if neighbors.is_empty() {
        return false;
    }
    let pick = neighbors[rng.gen_range(0..neighbors.len())];
    // 大文字 preservation (本 crate は lowercase 前提だが安全のため)
    let new_byte = if orig.is_ascii_uppercase() {
        pick.to_ascii_uppercase() as u8
    } else {
        pick as u8
    };
    buf[idx] = new_byte;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn adjacent_key_produces_kana_change() {
        let cfg = TypoConfig::default();
        let mut rng = StdRng::seed_from_u64(0);
        let out = apply("こんにちは", &cfg, &mut rng).expect("should produce noisy reading");
        assert_ne!(out, "こんにちは", "output should differ from input");
        assert!(!out.is_empty());
    }

    #[test]
    fn empty_input_returns_none() {
        let cfg = TypoConfig::default();
        let mut rng = StdRng::seed_from_u64(0);
        assert!(apply("", &cfg, &mut rng).is_none());
    }

    #[test]
    fn unsupported_kana_returns_none() {
        let cfg = TypoConfig::default();
        let mut rng = StdRng::seed_from_u64(0);
        // "ゑ" は romaji table に無い → None
        assert!(apply("ゑ", &cfg, &mut rng).is_none());
    }
}
