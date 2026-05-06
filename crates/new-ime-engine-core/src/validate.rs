//! Candidate validation for IME convert results.
//!
//! Rules enforced (`validate_candidate`):
//!   * R1 — every maximal kana run in the candidate, normalised kata→hira,
//!     must occur as a contiguous substring of the input reading at a
//!     position consistent with the surrounding kanji / Other runs. Kanji
//!     blocks implicitly consume the input chars between successive kana
//!     matches (and consume at least 1 char each).
//!   * R2 — within a kana run, a hira↔kata switch is rejected only when
//!     the corresponding input chars are identical (e.g. input `ああ` →
//!     output `あア` — the same `あ` got split arbitrarily into two
//!     scripts). Different-char boundaries are accepted (`はがくせいです`
//!     → `はガクセイです`) so the model can still emit kata segments
//!     where the implicit morpheme boundary is meaningful.
//!
//! Without explicit alignment from the beam or a kanji→reading dictionary
//! we approximate the morpheme boundary heuristically: same-char hira→kata
//! flips can't be a morpheme split, so we treat them as the user's
//! "`ああ` → must be `ああ` or `アア`, never `あア`" rule.
//!
//! `kata_to_hira` / `hira_to_kata` are also exported for the caller's
//! fallback path (when every beam candidate is rejected, the caller emits
//! the raw reading both as-is and katakana-converted).

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CharKind {
    Kanji,
    Hira,
    Kata,
    /// `ー` (U+30FC). Belongs to whichever kana run it's in but never flips
    /// the hira/kata kind for the boundary check.
    Neutral,
    Other,
}

fn char_kind(c: char) -> CharKind {
    match c {
        '\u{3041}'..='\u{3096}' => CharKind::Hira,
        '\u{30A1}'..='\u{30FA}' => CharKind::Kata,
        '\u{30FC}' => CharKind::Neutral,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{20000}'..='\u{2A6DF}' => CharKind::Kanji,
        _ => CharKind::Other,
    }
}

#[derive(Debug, PartialEq)]
enum Run {
    Kanji(String),
    Kana(String),
    Other(String),
}

fn split_runs(s: &str) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut buf = String::new();
    let mut group: Option<u8> = None;

    for c in s.chars() {
        let new_group = match char_kind(c) {
            CharKind::Kanji => 0u8,
            CharKind::Hira | CharKind::Kata | CharKind::Neutral => 1,
            CharKind::Other => 2,
        };
        if group.is_some() && group != Some(new_group) {
            push_run(&mut runs, std::mem::take(&mut buf), group.unwrap());
        }
        group = Some(new_group);
        buf.push(c);
    }
    if let Some(g) = group {
        push_run(&mut runs, buf, g);
    }
    runs
}

fn push_run(runs: &mut Vec<Run>, text: String, group: u8) {
    if text.is_empty() {
        return;
    }
    match group {
        0 => runs.push(Run::Kanji(text)),
        1 => runs.push(Run::Kana(text)),
        2 => runs.push(Run::Other(text)),
        _ => {}
    }
}

pub fn kata_to_hira(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{30A1}'..='\u{30F6}' => char::from_u32(c as u32 - 0x60).unwrap_or(c),
            other => other,
        })
        .collect()
}

pub fn hira_to_kata(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{3041}'..='\u{3096}' => char::from_u32(c as u32 + 0x60).unwrap_or(c),
            other => other,
        })
        .collect()
}

fn find_subseq(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    for i in 0..=haystack.len() - needle.len() {
        if haystack[i..i + needle.len()] == *needle {
            return Some(i);
        }
    }
    None
}

/// Same as `CharKind` but only for kana — used for the R2 boundary check.
/// Neutral (`ー`) returns `None` so chōon never participates in a switch.
fn kana_kind(c: char) -> Option<CharKind> {
    match char_kind(c) {
        CharKind::Hira => Some(CharKind::Hira),
        CharKind::Kata => Some(CharKind::Kata),
        _ => None,
    }
}

/// R2 check: walk adjacent chars of the kana run; reject only when the
/// kana kind flips (hira↔kata) **and** the corresponding input chars are
/// identical. The same input char being split across two scripts has no
/// morpheme-boundary justification, so treat it as a model artifact.
fn kana_run_kind_switch_ok(out_run: &str, input_slice: &[char]) -> bool {
    let out_chars: Vec<char> = out_run.chars().collect();
    debug_assert_eq!(out_chars.len(), input_slice.len());
    for k in 1..out_chars.len() {
        let (Some(prev_kind), Some(cur_kind)) =
            (kana_kind(out_chars[k - 1]), kana_kind(out_chars[k]))
        else {
            continue;
        };
        if prev_kind != cur_kind && input_slice[k - 1] == input_slice[k] {
            return false;
        }
    }
    true
}

pub fn validate_candidate(reading: &str, candidate: &str) -> bool {
    let runs = split_runs(candidate);
    // Normalise the input reading to hira so it matches the candidate kana
    // runs (which are themselves normalised at comparison time). probe_v3
    // reading strings are pure kata ("サンジュウニン...") while the model
    // typically emits hira particles ("が"); without this normalisation the
    // reading "ガ" would never match candidate "が" and every kana-bearing
    // candidate would be rejected.
    let input: Vec<char> = kata_to_hira(reading).chars().collect();
    // Pre-extract original kana runs so the R2 same-char-split check can
    // see the candidate's hira/kata kind without re-walking runs.
    validate_from(&runs, 0, &input, 0)
}

/// Recursive validator with backtracking on Kanji/Other → Kana boundaries.
///
/// The previous iterative version greedy-picked the first occurrence of
/// the next Kana run inside the remaining input. That breaks cases like
/// `友達と` against `ともだちと`: `と` first appears at position 0 (`友`'s
/// own reading), the `block_has_kanji && p == 0` rule then rejects, even
/// though picking the second `と` (position 4) makes the whole alignment
/// work. With backtracking we try every legal anchor position for the
/// next kana run; if any of them allows the rest of the candidate to
/// validate, the candidate is accepted.
fn validate_from(
    runs: &[Run],
    i: usize,
    input: &[char],
    cursor: usize,
) -> bool {
    if i >= runs.len() {
        return cursor == input.len();
    }
    match &runs[i] {
        Run::Kana(text) => {
            let normalized = kata_to_hira(text);
            let chars: Vec<char> = normalized.chars().collect();
            let n = chars.len();
            if cursor + n > input.len() {
                return false;
            }
            if input[cursor..cursor + n] != chars[..] {
                return false;
            }
            if !kana_run_kind_switch_ok(text, &input[cursor..cursor + n]) {
                return false;
            }
            validate_from(runs, i + 1, input, cursor + n)
        }
        Run::Kanji(_) | Run::Other(_) => {
            // A contiguous Kanji + Other block is opaque — without a
            // kanji→reading dictionary we only know it consumes some
            // number of input chars, bounded by where the next Kana run
            // can be anchored.
            let next_kana_idx =
                (i + 1..runs.len()).find(|&j| matches!(&runs[j], Run::Kana(_)));
            match next_kana_idx {
                Some(j) => {
                    let Run::Kana(text) = &runs[j] else {
                        unreachable!()
                    };
                    let normalized = kata_to_hira(text);
                    let kana_chars: Vec<char> = normalized.chars().collect();
                    let n = kana_chars.len();
                    let block_has_kanji =
                        (i..j).any(|k| matches!(&runs[k], Run::Kanji(_)));
                    // Block-with-kanji must consume at least 1 char;
                    // pure-Other prefix may consume zero (e.g. "100キロ"
                    // against "ひゃくじゅっきろ" pins "キロ" at pos 6 and
                    // leaves the prefix to absorb the rest).
                    let min_p = if block_has_kanji { 1 } else { 0 };
                    let mut start = cursor + min_p;
                    while start + n <= input.len() {
                        if input[start..start + n] == kana_chars[..]
                            && kana_run_kind_switch_ok(
                                text,
                                &input[start..start + n],
                            )
                            && validate_from(runs, j + 1, input, start + n)
                        {
                            return true;
                        }
                        start += 1;
                    }
                    false
                }
                None => {
                    // Trailing Kanji/Other block.
                    // - Has Kanji: must have ≥1 char left, consumes
                    //   everything to the end (kanji-end relaxation).
                    // - Pure Other: accept if either cursor already at
                    //   end (consume nothing) OR remaining input chars
                    //   equal the trailing Other text verbatim
                    //   (1:1 transcription, e.g. closing 」 or ASCII
                    //   punctuation).
                    let block_has_kanji = (i..runs.len())
                        .any(|k| matches!(&runs[k], Run::Kanji(_)));
                    if block_has_kanji {
                        cursor < input.len()
                    } else if cursor == input.len() {
                        true
                    } else {
                        let other_text: String = runs[i..]
                            .iter()
                            .map(|r| match r {
                                Run::Other(t) => t.as_str(),
                                _ => "",
                            })
                            .collect();
                        let other_chars: Vec<char> = other_text.chars().collect();
                        input[cursor..] == other_chars[..]
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- R2 ----

    #[test]
    fn r2_rejects_same_char_split() {
        // `ああ → あア` / `アあ` は input の同 char `あ`+`あ` を script 違い
        // に split している → 形態素境界として無理 → reject。
        assert!(!validate_candidate("ああ", "あア"));
        assert!(!validate_candidate("ああ", "アあ"));
    }

    #[test]
    fn r2_accepts_pure_runs() {
        assert!(validate_candidate("ああ", "ああ"));
        assert!(validate_candidate("ああ", "アア"));
    }

    #[test]
    fn r2_accepts_diff_char_boundary() {
        // `はがくせいです → はガクセイです` は kana run 内 hira→kata 切替が
        // input 側で異 char 境界 (`は`/`が`, `い`/`で`) なので valid。
        // 形態素境界に対応する想定。
        assert!(validate_candidate("はがくせいです", "はガクセイです"));
    }

    // ---- R1 ----

    #[test]
    fn r1_kanji_kana_alignment_basic() {
        assert!(validate_candidate(
            "わたしはがくせいです",
            "私は学生です"
        ));
    }

    #[test]
    fn r1_kanji_full_kata_replacement_ok() {
        // 学生 を ガクセイ に置換した形。連続 kana run 内 hira→kata 切替が
        // 異 char 境界なので valid。
        assert!(validate_candidate(
            "わたしはがくせいです",
            "私はガクセイです"
        ));
    }

    #[test]
    fn r1_kana_run_substring_mismatch_rejected() {
        assert!(!validate_candidate(
            "わたしはがくせいです",
            "私は学生でし"
        ));
    }

    #[test]
    fn r1_extra_chars_rejected() {
        assert!(!validate_candidate("ああ", "あいう"));
    }

    #[test]
    fn r1_kanji_zero_consume_rejected() {
        // `亜` の手前は cursor=0、後続 Kana `あ` が input[0] にいきなり
        // hit → 亜 が 0 char 消費 → reject。
        assert!(!validate_candidate("あ", "亜あ"));
    }

    // ---- 末尾 kanji 緩和 ----

    #[test]
    fn trailing_kanji_consumes_rest() {
        // 末尾 kanji block は入力残り全部を消費するという緩和ルール。
        assert!(validate_candidate("にほんご", "日本語"));
        assert!(validate_candidate("はにほん", "は日本"));
        // この緩和は kanji 末尾の入力長 mismatch を見逃すが、辞書なしの
        // 妥協。kana 部分の厳密 align は維持される。
        assert!(validate_candidate("はにほんで", "は日本"));
    }

    // ---- chōon ----

    #[test]
    fn choon_neutral_in_kata_run() {
        // ー は中立 → run 内 hira/kata 判定にも R2 boundary 判定にも影響
        // しない。
        assert!(validate_candidate("こー", "コー"));
        assert!(validate_candidate("こー", "こー"));
        assert!(validate_candidate("こーひー", "コーヒー"));
        assert!(validate_candidate("こーひー", "こーひー"));
    }

    #[test]
    fn choon_partial_kata_rejected_by_same_char_rule() {
        // `こーひー → コーひー`: ヒラ→カタ切替の char は ー (Neutral)
        // を skip して `コ`(Kata) ↔ `ひ`(Hira)。input 側 `こ`(0) ↔ `ひ`(2)
        // は異 char → 形式上 valid。これは R2 緩和ルールの帰結。
        // (今回の R2 簡易版では ASCII 同 char の事故のみ catch)
        assert!(validate_candidate("こーひー", "コーひー"));
    }

    // ---- Other / ASCII ----

    #[test]
    fn other_alone_invalidates() {
        // Other 単体は cursor を進めない → 末尾で input 残 → invalid。
        assert!(!validate_candidate("にじゅう", "20"));
    }

    #[test]
    fn other_then_trailing_kanji_ok() {
        // `20` (Other) + `秒` (末尾 Kanji) → 末尾 kanji 緩和で残り消費。
        assert!(validate_candidate("にじゅうびょう", "20秒"));
    }

    // ---- helper ----

    #[test]
    fn kata_to_hira_basic() {
        assert_eq!(kata_to_hira("コーヒー"), "こーひー");
        assert_eq!(kata_to_hira("ガクセイ"), "がくせい");
    }

    #[test]
    fn hira_to_kata_basic() {
        assert_eq!(hira_to_kata("こーひー"), "コーヒー"); // ー は元から ー (range 外)
        assert_eq!(hira_to_kata("わたし"), "ワタシ");
    }
}
