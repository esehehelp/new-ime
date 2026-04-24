//! Type-data の rule 実装。reading を romaji 経由で変形して返す。
//!
//! 5 rule: AdjacentKey / Deletion / Insertion / Transposition / NoConvert。
//! 1 row あたりの編集回数は Poisson(mean_edits) でサンプリング。

use crate::{qwerty, romaji};
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution, Poisson};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TypoKind {
    AdjacentKey,
    Deletion,
    Insertion,
    Transposition,
    /// 末尾をわざと romaji のまま残す (IME 変換忘れ)。1 row で最大 1 回。
    NoConvert,
}

#[derive(Clone, Debug)]
pub struct TypoConfig {
    pub mean_edits: f64,
    pub weights: Vec<(TypoKind, f64)>,
    /// ノイズ後 reading が元の ±`length_drift_max` 超えた場合 drop。
    pub length_drift_max: f64,
}

impl Default for TypoConfig {
    fn default() -> Self {
        Self {
            mean_edits: 1.0,
            weights: vec![
                (TypoKind::AdjacentKey, 0.45),
                (TypoKind::Deletion, 0.20),
                (TypoKind::Insertion, 0.15),
                (TypoKind::Transposition, 0.15),
                (TypoKind::NoConvert, 0.05),
            ],
            length_drift_max: 0.5,
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

    // Poisson(λ). 0 回だった場合は最低 1 回の適用を強制 (ノイズ 0 の出力は
    // 意味がない → emit しない設計なので、0 を弾く)。
    let lambda = cfg.mean_edits.max(0.0);
    let edits = if lambda == 0.0 {
        1
    } else {
        let sample = Poisson::new(lambda).ok()?.sample(rng) as u32;
        sample.max(1)
    };

    let mut buf: Vec<u8> = romaji.into_bytes();
    let mut no_convert_tail: Option<String> = None;
    let mut any_applied = false;

    for _ in 0..edits {
        let Some(kind) = pick_kind(&cfg.weights, rng) else {
            break;
        };
        let ok = match kind {
            TypoKind::AdjacentKey => rule_adjacent_key(&mut buf, rng),
            TypoKind::Deletion => rule_deletion(&mut buf, rng),
            TypoKind::Insertion => rule_insertion(&mut buf, rng),
            TypoKind::Transposition => rule_transposition(&mut buf, rng),
            TypoKind::NoConvert => {
                if no_convert_tail.is_some() {
                    false
                } else {
                    // buf 末尾の romaji 1..=3 byte を切り出して「kana 化しない」
                    // ことで、末尾に素の romaji が残る挙動を再現する。
                    rule_no_convert(&mut buf, &mut no_convert_tail, rng)
                }
            }
        };
        any_applied |= ok;
    }

    if !any_applied {
        return None;
    }

    let noisy_romaji = String::from_utf8(buf).ok()?;
    let mut back = romaji::romaji_to_kana(&noisy_romaji);
    if let Some(tail) = no_convert_tail {
        back.push_str(&tail);
    }
    if back.is_empty() {
        return None;
    }

    let orig_chars = reading.chars().count() as isize;
    let new_chars = back.chars().count() as isize;
    let drift = (new_chars - orig_chars).abs() as f64 / orig_chars.max(1) as f64;
    if drift > cfg.length_drift_max {
        return None;
    }
    if back == reading {
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

// ── rules ───────────────────────────────────────────────────────────────

/// ascii 1 char を qwerty 隣接キーへ置換。
pub fn rule_adjacent_key(buf: &mut Vec<u8>, rng: &mut StdRng) -> bool {
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
    let new_byte = if orig.is_ascii_uppercase() {
        pick.to_ascii_uppercase() as u8
    } else {
        pick as u8
    };
    if buf[idx] == new_byte {
        return false;
    }
    buf[idx] = new_byte;
    true
}

/// ascii 1 char を脱字。romaji が 2 byte 以下なら skip (読み壊滅防止)。
pub fn rule_deletion(buf: &mut Vec<u8>, rng: &mut StdRng) -> bool {
    let positions: Vec<usize> = buf
        .iter()
        .enumerate()
        .filter(|(_, b)| (**b as char).is_ascii_alphabetic())
        .map(|(i, _)| i)
        .collect();
    if positions.len() < 3 {
        return false;
    }
    let idx = positions[rng.gen_range(0..positions.len())];
    buf.remove(idx);
    true
}

/// 隣接キーを追加挿入 (fat finger)。
pub fn rule_insertion(buf: &mut Vec<u8>, rng: &mut StdRng) -> bool {
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
    let extra = neighbors[rng.gen_range(0..neighbors.len())] as u8;
    buf.insert(idx + 1, extra);
    true
}

/// 隣接する ascii 2 char を swap。
pub fn rule_transposition(buf: &mut Vec<u8>, rng: &mut StdRng) -> bool {
    let pairs: Vec<usize> = (0..buf.len().saturating_sub(1))
        .filter(|i| {
            (buf[*i] as char).is_ascii_alphabetic()
                && (buf[*i + 1] as char).is_ascii_alphabetic()
                && buf[*i] != buf[*i + 1]
        })
        .collect();
    if pairs.is_empty() {
        return false;
    }
    let i = pairs[rng.gen_range(0..pairs.len())];
    buf.swap(i, i + 1);
    true
}

/// 末尾の romaji 1..=3 文字を「kana 化せず」残すことで、IME 変換忘れを模擬。
/// 呼び出し側は `back = romaji_to_kana(...); back += tail` で末尾につなぐ。
pub fn rule_no_convert(
    buf: &mut Vec<u8>,
    tail: &mut Option<String>,
    rng: &mut StdRng,
) -> bool {
    let len = buf.len();
    // 末尾 ascii を探索
    let ascii_count = buf
        .iter()
        .rev()
        .take_while(|b| (**b as char).is_ascii_alphabetic())
        .count();
    if ascii_count < 2 {
        return false;
    }
    let take = rng.gen_range(1..=ascii_count.min(3));
    let start = len - take;
    let bytes = buf.split_off(start);
    let Ok(s) = String::from_utf8(bytes) else {
        return false;
    };
    *tail = Some(s);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn adjacent_key_produces_kana_change() {
        let cfg = TypoConfig {
            mean_edits: 1.0,
            weights: vec![(TypoKind::AdjacentKey, 1.0)],
            ..TypoConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = apply("こんにちは", &cfg, &mut rng).expect("noisy reading");
        assert_ne!(out, "こんにちは");
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
        assert!(apply("ゑ", &cfg, &mut rng).is_none());
    }

    #[test]
    fn very_short_resists_deletion() {
        // 1 char → romaji 1-2 byte → rule_deletion skip → apply() が None を返すか
        // no-op 回避ロジックにより emit されない。
        let cfg = TypoConfig {
            mean_edits: 1.0,
            weights: vec![(TypoKind::Deletion, 1.0)],
            ..TypoConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(7);
        // 1 char input は romaji=1-2 byte → positions.len()<3 で deletion 不可
        assert!(apply("あ", &cfg, &mut rng).is_none());
    }

    #[test]
    fn transposition_rule_basic() {
        let cfg = TypoConfig {
            mean_edits: 1.0,
            weights: vec![(TypoKind::Transposition, 1.0)],
            ..TypoConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let out = apply("ありがとう", &cfg, &mut rng).expect("should produce output");
        assert_ne!(out, "ありがとう");
    }

    #[test]
    fn poisson_nonzero_mean_emits_edits() {
        let cfg = TypoConfig {
            mean_edits: 2.0,
            ..TypoConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(42);
        let n_with_output = (0..100)
            .filter(|_| apply("こんにちはせかい", &cfg, &mut rng).is_some())
            .count();
        assert!(
            n_with_output >= 90,
            "expected ≥90/100 samples to produce output with λ=2.0, got {}",
            n_with_output
        );
    }

    #[test]
    fn no_convert_leaves_ascii_tail() {
        let cfg = TypoConfig {
            mean_edits: 1.0,
            weights: vec![(TypoKind::NoConvert, 1.0)],
            ..TypoConfig::default()
        };
        let mut rng = StdRng::seed_from_u64(2);
        let out = apply("ありがとう", &cfg, &mut rng).expect("should leak ascii");
        // 末尾は ascii のはず (romaji 化失敗で kana に戻っていない)
        let last = out.chars().last().unwrap();
        assert!(
            last.is_ascii() || !last.is_alphanumeric(),
            "expected ascii tail, got: {}",
            out
        );
    }
}
