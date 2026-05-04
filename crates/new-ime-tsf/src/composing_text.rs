//! Romaji → hiragana composing buffer (Rust port of engine/src/composing_text.cpp).
//!
//! Kept inline in the TSF crate so the DLL has no hiragana-specific runtime
//! dependency beyond `engine-core`. The logic intentionally matches the C++
//! behaviour: longest-match first, double-consonant → っ, n-before-consonant → ん.

const ROMAJI_TABLE: &[(&str, &str)] = &[
    // four-letter — explicit small-tsu spellings (`ltsu` / `xtsu`). Without
    // these the 3-letter `tsu` → つ entry would win and leave the `l`/`x`
    // prefix stranded in the buffer.
    ("ltsu", "っ"),
    ("xtsu", "っ"),
    // three-letter (yōon + small kana with y-row + extended th/dh/v)
    ("shi", "し"),
    ("chi", "ち"),
    ("tsu", "つ"),
    ("sha", "しゃ"),
    ("shu", "しゅ"),
    ("sho", "しょ"),
    ("cha", "ちゃ"),
    ("chu", "ちゅ"),
    ("cho", "ちょ"),
    ("tya", "ちゃ"),
    ("tyu", "ちゅ"),
    ("tyo", "ちょ"),
    ("sya", "しゃ"),
    ("syu", "しゅ"),
    ("syo", "しょ"),
    ("nya", "にゃ"),
    ("nyu", "にゅ"),
    ("nyo", "にょ"),
    ("hya", "ひゃ"),
    ("hyu", "ひゅ"),
    ("hyo", "ひょ"),
    ("mya", "みゃ"),
    ("myu", "みゅ"),
    ("myo", "みょ"),
    ("rya", "りゃ"),
    ("ryu", "りゅ"),
    ("ryo", "りょ"),
    ("gya", "ぎゃ"),
    ("gyu", "ぎゅ"),
    ("gyo", "ぎょ"),
    ("bya", "びゃ"),
    ("byu", "びゅ"),
    ("byo", "びょ"),
    ("pya", "ぴゃ"),
    ("pyu", "ぴゅ"),
    ("pyo", "ぴょ"),
    ("kya", "きゃ"),
    ("kyu", "きゅ"),
    ("kyo", "きょ"),
    ("jya", "じゃ"),
    ("jyu", "じゅ"),
    ("jyo", "じょ"),
    // small-kana — `x`- and `l`-prefixed aliases. MS-IME defaults to `l`
    // for small kana, so both families are accepted.
    ("xtu", "っ"),
    ("ltu", "っ"),
    ("xya", "ゃ"),
    ("xyu", "ゅ"),
    ("xyo", "ょ"),
    ("xwa", "ゎ"),
    ("lya", "ゃ"),
    ("lyu", "ゅ"),
    ("lyo", "ょ"),
    ("lwa", "ゎ"),
    // th/dh-row — small vowel on て / で base (katakana カ spellings).
    ("tha", "てぁ"),
    ("thi", "てぃ"),
    ("thu", "てゅ"),
    ("the", "てぇ"),
    ("tho", "てょ"),
    ("dha", "でぁ"),
    ("dhi", "でぃ"),
    ("dhu", "でゅ"),
    ("dhe", "でぇ"),
    ("dho", "でょ"),
    // v-row yōon
    ("vya", "ゔゃ"),
    ("vyu", "ゔゅ"),
    ("vyo", "ゔょ"),
    // two-letter
    ("ka", "か"),
    ("ki", "き"),
    ("ku", "く"),
    ("ke", "け"),
    ("ko", "こ"),
    ("sa", "さ"),
    ("si", "し"),
    ("su", "す"),
    ("se", "せ"),
    ("so", "そ"),
    ("ta", "た"),
    ("ti", "ち"),
    ("tu", "つ"),
    ("te", "て"),
    ("to", "と"),
    ("na", "な"),
    ("ni", "に"),
    ("nu", "ぬ"),
    ("ne", "ね"),
    ("no", "の"),
    ("ha", "は"),
    ("hi", "ひ"),
    ("hu", "ふ"),
    ("he", "へ"),
    ("ho", "ほ"),
    ("ma", "ま"),
    ("mi", "み"),
    ("mu", "む"),
    ("me", "め"),
    ("mo", "も"),
    ("ya", "や"),
    ("yi", "い"),
    ("yu", "ゆ"),
    ("ye", "いぇ"),
    ("yo", "よ"),
    ("ra", "ら"),
    ("ri", "り"),
    ("ru", "る"),
    ("re", "れ"),
    ("ro", "ろ"),
    ("wa", "わ"),
    ("wi", "ゐ"),
    ("wu", "う"),
    ("we", "ゑ"),
    ("wo", "を"),
    ("ga", "が"),
    ("gi", "ぎ"),
    ("gu", "ぐ"),
    ("ge", "げ"),
    ("go", "ご"),
    ("za", "ざ"),
    ("zi", "じ"),
    ("zu", "ず"),
    ("ze", "ぜ"),
    ("zo", "ぞ"),
    ("da", "だ"),
    ("di", "ぢ"),
    ("du", "づ"),
    ("de", "で"),
    ("do", "ど"),
    ("ba", "ば"),
    ("bi", "び"),
    ("bu", "ぶ"),
    ("be", "べ"),
    ("bo", "ぼ"),
    ("pa", "ぱ"),
    ("pi", "ぴ"),
    ("pu", "ぷ"),
    ("pe", "ぺ"),
    ("po", "ぽ"),
    ("fa", "ふぁ"),
    ("fi", "ふぃ"),
    ("fu", "ふ"),
    ("fe", "ふぇ"),
    ("fo", "ふぉ"),
    ("ja", "じゃ"),
    ("ji", "じ"),
    ("ju", "じゅ"),
    ("je", "じぇ"),
    ("jo", "じょ"),
    ("va", "ゔぁ"),
    ("vi", "ゔぃ"),
    ("vu", "ゔ"),
    ("ve", "ゔぇ"),
    ("vo", "ゔぉ"),
    ("xa", "ぁ"),
    ("xi", "ぃ"),
    ("xu", "ぅ"),
    ("xe", "ぇ"),
    ("xo", "ぉ"),
    ("la", "ぁ"),
    ("li", "ぃ"),
    ("lu", "ぅ"),
    ("le", "ぇ"),
    ("lo", "ぉ"),
    // Note: "nn" intentionally NOT in the table; handled specially in
    // `input_char` so `konnichiha` resolves as こんにちは (not こんいちは).
    // single vowels
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
];

#[derive(Debug, Clone, Default)]
pub struct ComposingText {
    hiragana: String,
    romaji_buffer: String,
    /// Cursor measured in UTF-8 code points (chars) into `hiragana`.
    cursor: usize,
    /// True when the current `romaji_buffer == "n"` is the second n of an
    /// "nn" sequence that was already eagerly flushed as ん into
    /// `hiragana`. A subsequent vowel/y can still combine with this n
    /// (so `nni` → んに, `nnyo` → んにょ), but a consonant / `n` / end
    /// of input must NOT re-flush ん because it's already there. See
    /// `input_char` for the full state machine.
    n_promoted: bool,
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
    /// The eagerly-flushed-`nn` sentinel `n` is hidden so the user sees
    /// just `ん` after typing two n's, not `んn`.
    pub fn pending_romaji(&self) -> &str {
        if self.n_promoted && self.romaji_buffer == "n" {
            return "";
        }
        &self.romaji_buffer
    }

    pub fn input_char(&mut self, c: char) {
        let c = c.to_ascii_lowercase();
        if !c.is_ascii() {
            return;
        }

        // `-` is the chōon mark (ー, U+30FC), not a romaji letter. Insert it
        // verbatim, flushing any pending `n` as ん first so `san-` → `さんー`
        // (not `さんnー`). When `n_promoted` is true the ん has already been
        // emitted (eager nn flush), so we just drop the sentinel without
        // double-flushing. The chōon path also bypasses the doubled-
        // consonant rule below so `--` produces `ーー`, not `っ`.
        if c == '-' {
            if self.romaji_buffer == "n" && !self.n_promoted {
                self.insert_at_cursor("ん");
            }
            self.romaji_buffer.clear();
            self.n_promoted = false;
            self.insert_at_cursor("ー");
            return;
        }

        // Eager `nn` → ん: as soon as the user types the 2nd `n` on a fresh
        // single-n buffer, commit the first n as ん so the preedit reflects
        // it without waiting for a third keystroke. The 2nd n stays in the
        // buffer as a sentinel + `n_promoted=true` flag so a vowel/y next
        // can still combine with it (`nni` → んに, `nnyo` → んにょ); a
        // consonant or end-of-input must NOT re-flush, since the ん was
        // already emitted on this 2nd-n press.
        if self.romaji_buffer == "n" && c == 'n' && !self.n_promoted {
            self.insert_at_cursor("ん");
            self.n_promoted = true;
            // `romaji_buffer` stays as "n"; that's the sentinel.
            return;
        }

        // In-play 2nd-n disambiguation. The buffer holds "n" but its ん
        // was already painted; the next keystroke decides whether the n
        // joins a vowel/y or gets discarded.
        if self.n_promoted {
            debug_assert_eq!(self.romaji_buffer, "n");
            self.n_promoted = false;
            match c {
                'a' | 'i' | 'u' | 'e' | 'o' | 'y' => {
                    // Vowel/y: keep the buffered "n" so the existing combine
                    // logic below produces na/ni/nya/etc.
                }
                'n' => {
                    // 3rd consecutive n: the in-play sentinel n is dropped
                    // (it was already counted as ん); the new n becomes a
                    // fresh single-n buffer waiting for its own pair or
                    // vowel. Net effect: buffer stays "n", flag now false.
                    return;
                }
                _ => {
                    // Anything else (other consonants): discard the
                    // sentinel and start over with the new char as if the
                    // user just began typing.
                    self.romaji_buffer.clear();
                }
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
            // Sentinel was popped: the in-play state is no longer valid,
            // and the eagerly-emitted ん is still in `hiragana` (the next
            // BS will pop it).
            self.n_promoted = false;
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
        let pending = self.pending_romaji();
        if pending.is_empty() {
            return self.hiragana.clone();
        }
        let byte_pos = self
            .hiragana
            .char_indices()
            .nth(self.cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.hiragana.len());
        let mut out = String::with_capacity(self.hiragana.len() + pending.len());
        out.push_str(&self.hiragana[..byte_pos]);
        out.push_str(pending);
        out.push_str(&self.hiragana[byte_pos..]);
        out
    }

    /// Commit flushes a pending raw `n` as ん, then returns the hiragana
    /// and resets. When `n_promoted` is set, the ん was already painted on
    /// the eager flush, so the sentinel `n` is dropped without re-emitting.
    pub fn commit(&mut self) -> String {
        self.flush_pending_n();
        let out = self.hiragana.clone();
        self.reset();
        out
    }

    /// Flush pending `n` as ん and return the full hiragana text, leaving
    /// the buffer otherwise in place for potential further editing (callers
    /// can `reset()` themselves). Used by the conversion path so the ONNX
    /// model sees a clean reading even if the user pressed Space before
    /// typing the final vowel. Eager-promoted n is just dropped.
    pub fn commit_reading(&mut self) -> String {
        self.flush_pending_n();
        self.hiragana.clone()
    }

    /// Drop any pending romaji (flushing `n` → ん first, unless already
    /// promoted) and splice the given kana at the cursor. Used for
    /// punctuation (`,` → `、`, `.` → `。`) so the composition keeps going
    /// instead of auto-committing the half-width character.
    pub fn flush_romaji_and_insert(&mut self, kana: &str) {
        self.flush_pending_n();
        self.romaji_buffer.clear();
        self.insert_at_cursor(kana);
    }

    fn flush_pending_n(&mut self) {
        if self.romaji_buffer == "n" {
            if !self.n_promoted {
                self.insert_at_cursor("ん");
            }
            self.romaji_buffer.clear();
        }
        self.n_promoted = false;
    }

    pub fn reset(&mut self) {
        self.hiragana.clear();
        self.romaji_buffer.clear();
        self.cursor = 0;
        self.n_promoted = false;
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
    fn nn_eagerly_visible_in_preedit() {
        // The whole point of this state machine: typing two n's should put
        // ん on screen *now*, not wait for a third keystroke.
        let mut c = ComposingText::new();
        c.input_char('n');
        assert_eq!(c.preedit(), "n");
        c.input_char('n');
        assert_eq!(c.preedit(), "ん");
    }

    #[test]
    fn nn_then_vowel_combines_via_in_play_n() {
        // Sentinel `n` after eager flush merges with the next vowel.
        assert_eq!(commit_all("nna"), "んな");
        assert_eq!(commit_all("nni"), "んに");
        assert_eq!(commit_all("nne"), "んね");
    }

    #[test]
    fn nn_then_yoon() {
        // 3-letter yōon spelling still works after eager flush.
        assert_eq!(commit_all("nnyo"), "んにょ");
        assert_eq!(commit_all("nnya"), "んにゃ");
    }

    #[test]
    fn nnn_is_one_kana_plus_pending() {
        // 1st n: pending. 2nd n: eager ん, sentinel "n", flag on.
        // 3rd n: drop sentinel, fresh single-n pending. Commit flushes it as ん.
        assert_eq!(commit_all("nnn"), "んん");
    }

    #[test]
    fn nnnn_is_two_kana() {
        // Two complete `nn` pairs.
        assert_eq!(commit_all("nnnn"), "んん");
    }

    #[test]
    fn nn_then_consonant_does_not_double_flush() {
        // After eager nn → ん, a consonant must NOT trigger another ん via
        // the n+consonant rule (that ん is already on screen).
        assert_eq!(commit_all("nnka"), "んか");
        assert_eq!(commit_all("nnta"), "んた");
    }

    #[test]
    fn nn_choon_is_n_choon() {
        // `n-` → んー. `nn-` should also be んー (single ん, single ー).
        assert_eq!(commit_all("nn-"), "んー");
    }

    #[test]
    fn nnk_a_combined_sequence() {
        // ん + か (sentinel-n is discarded on the consonant, k waits for a).
        assert_eq!(commit_all("nnka"), "んか");
    }

    #[test]
    fn konnichiha_still_works() {
        // The original regression case the prior nn-disambig was written
        // for. Eager flush must not break it.
        assert_eq!(commit_all("konnichiha"), "こんにちは");
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
