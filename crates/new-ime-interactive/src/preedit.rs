#[derive(Debug, Default, Clone)]
pub struct Preedit {
    text: String,
    cursor_byte_pos: usize,
    active_segment: isize,
    highlighted: bool,
    segment_ranges: Vec<(usize, usize)>,
}

impl Preedit {
    pub fn set_simple(&mut self, text: impl Into<String>, cursor_pos: usize) {
        self.text = text.into();
        self.highlighted = false;
        self.active_segment = -1;
        self.segment_ranges.clear();
        self.cursor_byte_pos = self
            .text
            .char_indices()
            .nth(cursor_pos)
            .map(|(idx, _)| idx)
            .unwrap_or(self.text.len());
    }

    pub fn set_highlighted(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.highlighted = true;
        self.active_segment = -1;
        self.cursor_byte_pos = self.text.len();
        self.segment_ranges.clear();
    }

    pub fn set_segments(&mut self, segments: &[String], active_segment: isize) {
        self.text.clear();
        self.segment_ranges.clear();
        self.highlighted = false;
        self.active_segment = active_segment;

        for segment in segments {
            let start = self.text.len();
            self.text.push_str(segment);
            let end = self.text.len();
            self.segment_ranges.push((start, end));
        }

        self.cursor_byte_pos = self.text.len();
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn cursor_byte_pos(&self) -> usize {
        self.cursor_byte_pos
    }

    pub fn active_segment(&self) -> isize {
        self.active_segment
    }

    pub fn segment_ranges(&self) -> &[(usize, usize)] {
        &self.segment_ranges
    }

    pub fn is_highlighted(&self) -> bool {
        self.highlighted
    }

    pub fn reset(&mut self) {
        self.text.clear();
        self.cursor_byte_pos = 0;
        self.active_segment = -1;
        self.highlighted = false;
        self.segment_ranges.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::Preedit;

    #[test]
    fn simple_cursor_tracks_utf8_bytes() {
        let mut preedit = Preedit::default();
        preedit.set_simple("かなabc", 2);
        assert_eq!(preedit.cursor_byte_pos(), "かな".len());
        assert_eq!(preedit.text(), "かなabc");
        assert!(!preedit.is_highlighted());
    }

    #[test]
    fn highlighted_selects_whole_text() {
        let mut preedit = Preedit::default();
        preedit.set_highlighted("変換候補");
        assert_eq!(preedit.cursor_byte_pos(), "変換候補".len());
        assert!(preedit.is_highlighted());
    }

    #[test]
    fn segments_build_ranges() {
        let mut preedit = Preedit::default();
        preedit.set_segments(&["東京".to_string(), "駅".to_string()], 0);
        assert_eq!(preedit.text(), "東京駅");
        assert_eq!(
            preedit.segment_ranges(),
            &[("".len(), "東京".len()), ("東京".len(), "東京駅".len())]
        );
        assert_eq!(preedit.active_segment(), 0);
    }
}
