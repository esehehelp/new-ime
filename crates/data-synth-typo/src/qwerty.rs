//! QWERTY (US) 物理配列の上下左右隣接 key map。打鍵ミス rule で使う。
//!
//! 角キーや `;` `,` `.` は romaji に出てこないため未定義。lookup 時は
//! `Option<_>` で扱い、非対応 key は rule skip。

/// key `c` に対し、物理的に隣接 (上下左右斜め) する key の集合。
/// 小文字 ascii のみ返す。
pub fn adjacent_keys(c: char) -> &'static [char] {
    // Staggered ANSI qwerty. Each key's neighbours are the ≤ 1-row, ≤ 1-
    // lateral-offset keys. Manually kept symmetric — `adjacency_is_symmetric`
    // test guards against drift.
    match c.to_ascii_lowercase() {
        // top row
        'q' => &['w', 'a'],
        'w' => &['q', 'e', 'a', 's'],
        'e' => &['w', 'r', 's', 'd'],
        'r' => &['e', 't', 'd', 'f'],
        't' => &['r', 'y', 'f', 'g'],
        'y' => &['t', 'u', 'g', 'h'],
        'u' => &['y', 'i', 'h', 'j'],
        'i' => &['u', 'o', 'j', 'k'],
        'o' => &['i', 'p', 'k', 'l'],
        'p' => &['o', 'l'],
        // home row
        'a' => &['q', 'w', 's', 'z'],
        's' => &['w', 'e', 'a', 'd', 'z', 'x'],
        'd' => &['e', 'r', 's', 'f', 'x', 'c'],
        'f' => &['r', 't', 'd', 'g', 'c', 'v'],
        'g' => &['t', 'y', 'f', 'h', 'v', 'b'],
        'h' => &['y', 'u', 'g', 'j', 'b', 'n'],
        'j' => &['u', 'i', 'h', 'k', 'n', 'm'],
        'k' => &['i', 'o', 'j', 'l', 'm'],
        'l' => &['o', 'p', 'k'],
        // bottom row
        'z' => &['a', 's', 'x'],
        'x' => &['s', 'd', 'z', 'c'],
        'c' => &['d', 'f', 'x', 'v'],
        'v' => &['f', 'g', 'c', 'b'],
        'b' => &['g', 'h', 'v', 'n'],
        'n' => &['h', 'j', 'b', 'm'],
        'm' => &['j', 'k', 'n'],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacency_is_symmetric() {
        for c in 'a'..='z' {
            for &n in adjacent_keys(c) {
                assert!(
                    adjacent_keys(n).contains(&c),
                    "adjacency not symmetric: {} lists {} but {} doesn't list {}",
                    c,
                    n,
                    n,
                    c
                );
            }
        }
    }

    #[test]
    fn non_letter_returns_empty() {
        assert!(adjacent_keys(' ').is_empty());
        assert!(adjacent_keys('1').is_empty());
    }
}
