//! Romaji → hiragana composing buffer (Rust port of engine/src/composing_text.cpp).
//!
//! Kept inline in the TSF crate so the DLL has no hiragana-specific runtime
//! dependency beyond `engine-core`. The logic intentionally matches the C++
//! behaviour: longest-match first, double-consonant → っ, n-before-consonant → ん.

const ROMAJI_TABLE: &[(&str, &str)] = &[
    // four-letter — explicit small-tsu spellings (`ltsu` / `xtsu`). Without
    // these the 3-letter `tsu` → つ entry would win and leave the `l`/`x`
    // prefix stranded in the buffer.
    ("ltsu", "っ"), ("xtsu", "っ"),
    // three-letter (yōon + small kana with y-row + extended th/dh/v)
    ("shi", "し"), ("chi", "ち"), ("tsu", "つ"), ("sha", "しゃ"),
    ("shu", "しゅ"), ("sho", "しょ"), ("cha", "ちゃ"), ("chu", "ちゅ"),
    ("cho", "ちょ"), ("tya", "ちゃ"), ("tyu", "ちゅ"), ("tyo", "ちょ"),
    ("sya", "しゃ"), ("syu", "しゅ"), ("syo", "しょ"), ("nya", "にゃ"),
    ("nyu", "にゅ"), ("nyo", "にょ"), ("hya", "ひゃ"), ("hyu", "ひゅ"),
    ("hyo", "ひょ"), ("mya", "みゃ"), ("myu", "みゅ"), ("myo", "みょ"),
    ("rya", "りゃ"), ("ryu", "りゅ"), ("ryo", "りょ"), ("gya", "ぎゃ"),
    ("gyu", "ぎゅ"), ("gyo", "ぎょ"), ("bya", "びゃ"), ("byu", "びゅ"),
    ("byo", "びょ"), ("pya", "ぴゃ"), ("pyu", "ぴゅ"), ("pyo", "ぴょ"),
    ("kya", "きゃ"), ("kyu", "きゅ"), ("kyo", "きょ"), ("jya", "じゃ"),
    ("jyu", "じゅ"), ("jyo", "じょ"),
    // small-kana — `x`- and `l`-prefixed aliases. MS-IME defaults to `l`
    // for small kana, so both families are accepted.
    ("xtu", "っ"), ("ltu", "っ"),
    ("xya", "ゃ"), ("xyu", "ゅ"), ("xyo", "ょ"), ("xwa", "ゎ"),
    ("lya", "ゃ"), ("lyu", "ゅ"), ("lyo", "ょ"), ("lwa", "ゎ"),
    // th/dh-row — small vowel on て / で base (katakana カ spellings).
    ("tha", "てぁ"), ("thi", "てぃ"), ("thu", "てゅ"), ("the", "てぇ"), ("tho", "てょ"),
    ("dha", "でぁ"), ("dhi", "でぃ"), ("dhu", "でゅ"), ("dhe", "でぇ"), ("dho", "でょ"),
    // v-row yōon
    ("vya", "ゔゃ"), ("vyu", "ゔゅ"), ("vyo", "ゔょ"),
    // two-letter
    ("ka", "か"), ("ki", "き"), ("ku", "く"), ("ke", "け"), ("ko", "こ"),
    ("sa", "さ"), ("si", "し"), ("su", "す"), ("se", "せ"), ("so", "そ"),
    ("ta", "た"), ("ti", "ち"), ("tu", "つ"), ("te", "て"), ("to", "と"),
    ("na", "な"), ("ni", "に"), ("nu", "ぬ"), ("ne", "ね"), ("no", "の"),
    ("ha", "は"), ("hi", "ひ"), ("hu", "ふ"), ("he", "へ"), ("ho", "ほ"),
    ("ma", "ま"), ("mi", "み"), ("mu", "む"), ("me", "め"), ("mo", "も"),
    ("ya", "や"), ("yi", "い"), ("yu", "ゆ"), ("ye", "いぇ"), ("yo", "よ"),
    ("ra", "ら"), ("ri", "り"), ("ru", "る"), ("re", "れ"), ("ro", "ろ"),
    ("wa", "わ"), ("wi", "ゐ"), ("wu", "う"), ("we", "ゑ"), ("wo", "を"),
    ("ga", "が"), ("gi", "ぎ"), ("gu", "ぐ"), ("ge", "げ"), ("go", "ご"),
    ("za", "ざ"), ("zi", "じ"), ("zu", "ず"), ("ze", "ぜ"), ("zo", "ぞ"),
    ("da", "だ"), ("di", "ぢ"), ("du", "づ"), ("de", "で"), ("do", "ど"),
    ("ba", "ば"), ("bi", "び"), ("bu", "ぶ"), ("be", "べ"), ("bo", "ぼ"),
    ("pa", "ぱ"), ("pi", "ぴ"), ("pu", "ぷ"), ("pe", "ぺ"), ("po", "ぽ"),
    ("fa", "ふぁ"), ("fi", "ふぃ"), ("fu", "ふ"), ("fe", "ふぇ"), ("fo", "ふぉ"),
    ("ja", "じゃ"), ("ji", "じ"), ("ju", "じゅ"), ("je", "じぇ"), ("jo", "じょ"),
    ("va", "ゔぁ"), ("vi", "ゔぃ"), ("vu", "ゔ"), ("ve", "ゔぇ"), ("vo", "ゔぉ"),
    ("xa", "ぁ"), ("xi", "ぃ"), ("xu", "ぅ"), ("xe", "ぇ"), ("xo", "ぉ"),
    ("la", "ぁ"), ("li", "ぃ"), ("lu", "ぅ"), ("le", "ぇ"), ("lo", "ぉ"),
    // Note: "nn" intentionally NOT in the table; handled specially in
    // `input_char` so `konnichiha` resolves as こんにちは (not こんいちは).
    // single vowels
    ("a", "あ"), ("i", "い"), ("u", "う"), ("e", "え"), ("o", "お"),
];

#[derive(Debug, Clone, Default)]
pub struct ComposingText {
    hiragana: String,
    romaji_buffer: String,
    /// Cursor measured in UTF-8 code points (chars) into `hiragana`.
    cursor: usize,
}

impl ComposingText {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.hiragana.is_empty() && self.romaji_buffer.is_empty()
    }

    /// Completed hiragana portion (used as the ONNX model's reading input
    /// during live conversion — pending romaji is excluded on purpose).
    pub fn hiragana(&self) -> &str {
        &self.hiragana
    }

    /// Un-converted romaji tail (e.g. `"k"` while the user is in the middle
    /// of typing `konnichiha`). Shown verbatim at the end of the preedit.
    pub fn pending_romaji(&self) -> &str {
        &self.romaji_buffer
    }

    pub fn input_char(&mut self, c: char) {
        let c = c.to_ascii_lowercase();
        if !c.is_ascii() {
            return;
        }

        // `-` is the chōon mark (ー, U+30FC), not a romaji letter. Insert it
        // verbatim, flushing any pending `n` as ん first so `san-` → `さんー`
        // (not `さんnー`). This also bypasses the doubled-consonant rule
        // below so `--` produces `ーー` (two long marks) instead of `っ`.
        if c == '-' {
            if self.romaji_buffer == "n" || self.romaji_buffer == "nn" {
                self.insert_at_cursor("ん");
            }
            self.romaji_buffer.clear();
            self.insert_at_cursor("ー");
            return;
        }

        // Special "nn" disambiguation, mirroring MS-IME behaviour:
        //   buffer == "nn" + vowel/y → first n = ん, second n + vowel forms にゃ/に/…
        //   buffer == "nn" + anything else (consonant incl. n, or commit) → "nn" == ん
        // This is why `konnichiha` must be こんにちは: the 'i' after "nn" says
        // "split", so first n → ん and second n + i → に.
        if self.romaji_buffer == "nn" {
            self.insert_at_cursor("ん");
            if matches!(c, 'a' | 'i' | 'u' | 'e' | 'o' | 'y') {
                self.romaji_buffer = "n".to_string();
            } else {
                self.romaji_buffer.clear();
            }
        }

        self.romaji_buffer.push(c);

        // `n` + consonant (not y/n/vowel) → flush as ん (single n case, e.g. "nk").
        if self.romaji_buffer.len() >= 2 {
            let bytes = self.romaji_buffer.as_bytes();
            if bytes[0] == b'n' {
                let s = bytes[1];
                let is_vowel = matches!(s, b'a' | b'i' | b'u' | b'e' | b'o');
                if !is_vowel && s != b'y' && s != b'n' {
                    self.insert_at_cursor("ん");
                    self.romaji_buffer.drain(..1);
                }
            }
        }

        // Doubled consonant → っ + restart with the second letter. `l` and
        // `x` count as consonants here, matching MS-IME: `lla` → っぁ,
        // `xxa` → っぁ. (`ll` alone still becomes `っl`; the subsequent
        // vowel finishes the small-kana form via the `la`/`xa` table.)
        if self.romaji_buffer.len() >= 2 {
            let bytes = self.romaji_buffer.as_bytes();
            let a = bytes[0];
            let b = bytes[1];
            if a == b && !matches!(a, b'a' | b'i' | b'u' | b'e' | b'o' | b'n') {
                self.insert_at_cursor("っ");
                self.romaji_buffer.drain(..1);
            }
        }

        self.try_convert_romaji();
    }

    fn try_convert_romaji(&mut self) -> bool {
        for (romaji, kana) in ROMAJI_TABLE {
            if self.romaji_buffer.starts_with(romaji) {
                let len = romaji.len();
                self.insert_at_cursor(kana);
                self.romaji_buffer.drain(..len);
                return true;
            }
        }
        false
    }

    fn insert_at_cursor(&mut self, kana: &str) {
        let byte_pos = self
            .hiragana
            .char_indices()
            .nth(self.cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.hiragana.len());
        self.hiragana.insert_str(byte_pos, kana);
        self.cursor += kana.chars().count();
    }

    pub fn delete_left(&mut self) {
        if !self.romaji_buffer.is_empty() {
            self.romaji_buffer.pop();
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let prev_char_start = self
            .hiragana
            .char_indices()
            .nth(self.cursor - 1)
            .map(|(b, _)| b)
            .unwrap_or(0);
        let next_char_start = self
            .hiragana
            .char_indices()
            .nth(self.cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.hiragana.len());
        self.hiragana.drain(prev_char_start..next_char_start);
        self.cursor -= 1;
    }

    pub fn move_cursor(&mut self, delta: i32) {
        let total = self.hiragana.chars().count() as i32;
        let new_cursor = (self.cursor as i32 + delta).clamp(0, total) as usize;
        self.cursor = new_cursor;
    }

    /// Preedit text = hiragana with the pending romaji buffer spliced at the cursor.
    pub fn preedit(&self) -> String {
        if self.romaji_buffer.is_empty() {
            return self.hiragana.clone();
        }
        let byte_pos = self
            .hiragana
            .char_indices()
            .nth(self.cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.hiragana.len());
        let mut out = String::with_capacity(self.hiragana.len() + self.romaji_buffer.len());
        out.push_str(&self.hiragana[..byte_pos]);
        out.push_str(&self.romaji_buffer);
        out.push_str(&self.hiragana[byte_pos..]);
        out
    }

    /// Commit flushes pending `n`/`nn` as ん then returns the hiragana and resets.
    pub fn commit(&mut self) -> String {
        if self.romaji_buffer == "n" || self.romaji_buffer == "nn" {
            self.insert_at_cursor("ん");
            self.romaji_buffer.clear();
        }
        let out = self.hiragana.clone();
        self.reset();
        out
    }

    /// Flush pending `n` as ん and return the full hiragana text, leaving the
    /// buffer otherwise in place for potential further editing (callers can
    /// `reset()` themselves). Used by the conversion path so the ONNX model
    /// sees a clean reading even if the user pressed Space before typing the
    /// final vowel.
    pub fn commit_reading(&mut self) -> String {
        if self.romaji_buffer == "n" || self.romaji_buffer == "nn" {
            self.insert_at_cursor("ん");
            self.romaji_buffer.clear();
        }
        self.hiragana.clone()
    }

    /// Drop any pending romaji (flushing `n` → ん first) and splice the given
    /// kana at the cursor. Used for punctuation (`,` → `、`, `.` → `。`) so
    /// the composition keeps going instead of auto-committing the half-width
    /// character.
    pub fn flush_romaji_and_insert(&mut self, kana: &str) {
        if self.romaji_buffer == "n" || self.romaji_buffer == "nn" {
            self.insert_at_cursor("ん");
        }
        self.romaji_buffer.clear();
        self.insert_at_cursor(kana);
    }

    pub fn reset(&mut self) {
        self.hiragana.clear();
        self.romaji_buffer.clear();
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_all(s: &str) -> ComposingText {
        let mut c = ComposingText::new();
        for ch in s.chars() {
            c.input_char(ch);
        }
        c
    }

    fn commit_all(s: &str) -> String {
        type_all(s).commit()
    }

    #[test]
    fn basic_vowels() {
        assert_eq!(commit_all("aiueo"), "あいうえお");
    }

    #[test]
    fn konnichiha() {
        assert_eq!(commit_all("konnichiha"), "こんにちは");
    }

    #[test]
    fn sokuon() {
        assert_eq!(commit_all("kitte"), "きって");
    }

    #[test]
    fn n_flush_on_commit() {
        assert_eq!(commit_all("shin"), "しん");
    }

    #[test]
    fn double_n_commit_is_single_n() {
        assert_eq!(commit_all("nn"), "ん");
    }

    #[test]
    fn shinbun() {
        assert_eq!(commit_all("shinbun"), "しんぶん");
    }

    #[test]
    fn preedit_includes_pending() {
        let mut c = ComposingText::new();
        c.input_char('k');
        assert_eq!(c.preedit(), "k");
        c.input_char('o');
        assert_eq!(c.preedit(), "こ");
    }

    #[test]
    fn choon_single_hyphen() {
        assert_eq!(commit_all("ra-men"), "らーめん");
    }

    #[test]
    fn choon_double_hyphen_is_two_marks_not_sokuon() {
        assert_eq!(commit_all("a--"), "あーー");
    }

    #[test]
    fn n_before_choon_flushes() {
        assert_eq!(commit_all("san-"), "さんー");
    }

    // ---- small-kana family ----

    #[test]
    fn l_prefix_small_vowels() {
        assert_eq!(commit_all("la"), "ぁ");
        assert_eq!(commit_all("li"), "ぃ");
        assert_eq!(commit_all("lu"), "ぅ");
        assert_eq!(commit_all("le"), "ぇ");
        assert_eq!(commit_all("lo"), "ぉ");
    }

    #[test]
    fn x_prefix_small_vowels_still_work() {
        assert_eq!(commit_all("xa"), "ぁ");
        assert_eq!(commit_all("xi"), "ぃ");
        assert_eq!(commit_all("xu"), "ぅ");
        assert_eq!(commit_all("xe"), "ぇ");
        assert_eq!(commit_all("xo"), "ぉ");
    }

    #[test]
    fn l_prefix_small_y_row() {
        assert_eq!(commit_all("lya"), "ゃ");
        assert_eq!(commit_all("lyu"), "ゅ");
        assert_eq!(commit_all("lyo"), "ょ");
    }

    #[test]
    fn small_tsu_spellings() {
        assert_eq!(commit_all("ltu"), "っ");
        assert_eq!(commit_all("xtu"), "っ");
        assert_eq!(commit_all("ltsu"), "っ");
        assert_eq!(commit_all("xtsu"), "っ");
    }

    #[test]
    fn small_wa() {
        assert_eq!(commit_all("lwa"), "ゎ");
        assert_eq!(commit_all("xwa"), "ゎ");
    }

    #[test]
    fn ll_triggers_sokuon_then_small_vowel() {
        // Matches MS-IME: first `l` becomes っ, then `la` → ぁ.
        assert_eq!(commit_all("lla"), "っぁ");
        assert_eq!(commit_all("xxi"), "っぃ");
    }

    // ---- extended consonants ----

    #[test]
    fn th_row() {
        assert_eq!(commit_all("thi"), "てぃ");
        assert_eq!(commit_all("tha"), "てぁ");
        assert_eq!(commit_all("thu"), "てゅ");
    }

    #[test]
    fn dh_row() {
        assert_eq!(commit_all("dhi"), "でぃ");
        assert_eq!(commit_all("dha"), "でぁ");
    }

    #[test]
    fn v_row_full() {
        // All five vowels + yōon, hiragana form of the MS-IME ヴ mapping.
        assert_eq!(commit_all("va"), "ゔぁ");
        assert_eq!(commit_all("vi"), "ゔぃ");
        assert_eq!(commit_all("vu"), "ゔ");
        assert_eq!(commit_all("ve"), "ゔぇ");
        assert_eq!(commit_all("vo"), "ゔぉ");
        assert_eq!(commit_all("vya"), "ゔゃ");
        assert_eq!(commit_all("vyu"), "ゔゅ");
        assert_eq!(commit_all("vyo"), "ゔょ");
    }

    #[test]
    fn vy_alone_stays_pending_until_vowel() {
        // `vy` without a trailing vowel must not auto-collapse to ゔ — a
        // buffered `vy` waits for the next vowel to disambiguate ゃ/ゅ/ょ.
        // (Commit without a vowel simply drops the incomplete romaji.)
        let mut c = ComposingText::new();
        c.input_char('v');
        c.input_char('y');
        assert_eq!(c.preedit(), "vy");
        c.input_char('u');
        assert_eq!(c.commit(), "ゔゅ");
    }

    #[test]
    fn fa_fi_fe_fo() {
        assert_eq!(commit_all("fa"), "ふぁ");
        assert_eq!(commit_all("fi"), "ふぃ");
        assert_eq!(commit_all("fe"), "ふぇ");
        assert_eq!(commit_all("fo"), "ふぉ");
    }

    #[test]
    fn ja_family() {
        assert_eq!(commit_all("ja"), "じゃ");
        assert_eq!(commit_all("ji"), "じ");
        assert_eq!(commit_all("ju"), "じゅ");
        assert_eq!(commit_all("jo"), "じょ");
        assert_eq!(commit_all("jya"), "じゃ");
        assert_eq!(commit_all("jyu"), "じゅ");
    }

    // ---- regression cases ----

    #[test]
    fn tsu_still_works_with_four_letter_prefix_in_table() {
        // Adding `ltsu`/`xtsu` must not break plain `tsu`.
        assert_eq!(commit_all("tsu"), "つ");
        assert_eq!(commit_all("matsuri"), "まつり");
    }

    #[test]
    fn konpyu_ta_choon_chain() {
        assert_eq!(commit_all("konpyu-ta-"), "こんぴゅーたー");
    }
}
