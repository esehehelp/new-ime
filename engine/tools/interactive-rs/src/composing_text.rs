#[derive(Debug, Default, Clone)]
pub struct ComposingText {
    hiragana: String,
    romaji_buffer: String,
    cursor: usize,
}

impl ComposingText {
    pub fn input_char(&mut self, mut c: char) {
        if c.is_ascii_uppercase() {
            c = c.to_ascii_lowercase();
        }

        self.romaji_buffer.push(c);

        if self.romaji_buffer.len() >= 2 && self.romaji_buffer.as_bytes()[0] == b'n' {
            let second = self.romaji_buffer.as_bytes()[1] as char;
            if !matches!(second, 'a' | 'i' | 'u' | 'e' | 'o' | 'y' | 'n') {
                self.insert_at_cursor("ん");
                self.romaji_buffer.remove(0);
            }
        }

        if self.romaji_buffer.len() >= 2 {
            let first = self.romaji_buffer.as_bytes()[0] as char;
            let second = self.romaji_buffer.as_bytes()[1] as char;
            if first == second && !matches!(first, 'a' | 'i' | 'u' | 'e' | 'o' | 'n') {
                self.insert_at_cursor("っ");
                self.romaji_buffer.remove(0);
            }
        }

        self.try_convert_romaji();
    }

    pub fn delete_left(&mut self) {
        if self.cursor == 0 && self.romaji_buffer.is_empty() {
            return;
        }

        if !self.romaji_buffer.is_empty() {
            self.romaji_buffer.pop();
            return;
        }

        let chars: Vec<(usize, char)> = self.hiragana.char_indices().collect();
        if self.cursor == 0 || self.cursor > chars.len() {
            return;
        }

        let start = chars[self.cursor - 1].0;
        let end = if self.cursor < chars.len() {
            chars[self.cursor].0
        } else {
            self.hiragana.len()
        };
        self.hiragana.replace_range(start..end, "");
        self.cursor -= 1;
    }

    pub fn delete_right(&mut self) {
        let chars: Vec<(usize, char)> = self.hiragana.char_indices().collect();
        if self.cursor >= chars.len() {
            return;
        }

        let start = chars[self.cursor].0;
        let end = if self.cursor + 1 < chars.len() {
            chars[self.cursor + 1].0
        } else {
            self.hiragana.len()
        };
        self.hiragana.replace_range(start..end, "");
    }

    pub fn move_cursor(&mut self, offset: isize) {
        let total_chars = self.hiragana.chars().count();
        let next = self.cursor as isize + offset;
        self.cursor = next.clamp(0, total_chars as isize) as usize;
    }

    pub fn hiragana(&self) -> &str {
        &self.hiragana
    }

    pub fn display(&self) -> String {
        if self.romaji_buffer.is_empty() {
            return self.hiragana.clone();
        }

        let byte_pos = char_to_byte_pos(&self.hiragana, self.cursor);
        let mut out = self.hiragana.clone();
        out.insert_str(byte_pos, &self.romaji_buffer);
        out
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn empty(&self) -> bool {
        self.hiragana.is_empty() && self.romaji_buffer.is_empty()
    }

    pub fn reset(&mut self) {
        self.hiragana.clear();
        self.romaji_buffer.clear();
        self.cursor = 0;
    }

    fn try_convert_romaji(&mut self) -> bool {
        for (romaji, hiragana) in ROMAJI_TABLE {
            if self.romaji_buffer.starts_with(romaji) {
                self.insert_at_cursor(hiragana);
                self.romaji_buffer.drain(..romaji.len());
                return true;
            }
        }
        false
    }

    fn insert_at_cursor(&mut self, kana: &str) {
        let byte_pos = char_to_byte_pos(&self.hiragana, self.cursor);
        self.hiragana.insert_str(byte_pos, kana);
        self.cursor += kana.chars().count();
    }
}

fn char_to_byte_pos(text: &str, cursor: usize) -> usize {
    text.char_indices()
        .nth(cursor)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

const ROMAJI_TABLE: &[(&str, &str)] = &[
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
    ("xtu", "っ"),
    ("xya", "ゃ"),
    ("xyu", "ゅ"),
    ("xyo", "ょ"),
    ("xwa", "ゎ"),
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
    ("nn", "ん"),
    ("xa", "ぁ"),
    ("xi", "ぃ"),
    ("xu", "ぅ"),
    ("xe", "ぇ"),
    ("xo", "ぉ"),
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
];

#[cfg(test)]
mod tests {
    use super::ComposingText;

    #[test]
    fn basic_romaji() {
        let mut ct = ComposingText::default();
        ct.input_char('k');
        assert_eq!(ct.display(), "k");
        ct.input_char('a');
        assert_eq!(ct.hiragana(), "か");
        assert_eq!(ct.display(), "か");
    }

    #[test]
    fn word() {
        let mut ct = ComposingText::default();
        for c in "kanji".chars() {
            ct.input_char(c);
        }
        assert_eq!(ct.hiragana(), "かんじ");
    }

    #[test]
    fn double_consonant() {
        let mut ct = ComposingText::default();
        for c in "kitte".chars() {
            ct.input_char(c);
        }
        assert_eq!(ct.hiragana(), "きって");
    }

    #[test]
    fn nn() {
        let mut ct = ComposingText::default();
        ct.input_char('n');
        ct.input_char('n');
        assert_eq!(ct.hiragana(), "ん");
    }

    #[test]
    fn n_before_consonant() {
        let mut ct = ComposingText::default();
        for c in "kanka".chars() {
            ct.input_char(c);
        }
        assert_eq!(ct.hiragana(), "かんか");
    }

    #[test]
    fn delete_left() {
        let mut ct = ComposingText::default();
        for c in "ka".chars() {
            ct.input_char(c);
        }
        ct.delete_left();
        assert!(ct.hiragana().is_empty());
    }

    #[test]
    fn delete_romaji_buffer() {
        let mut ct = ComposingText::default();
        ct.input_char('k');
        ct.delete_left();
        assert!(ct.empty());
    }

    #[test]
    fn reset() {
        let mut ct = ComposingText::default();
        for c in "tesuto".chars() {
            ct.input_char(c);
        }
        ct.reset();
        assert!(ct.empty());
        assert_eq!(ct.cursor(), 0);
    }

    #[test]
    fn uppercase() {
        let mut ct = ComposingText::default();
        for c in "KA".chars() {
            ct.input_char(c);
        }
        assert_eq!(ct.hiragana(), "か");
    }
}
