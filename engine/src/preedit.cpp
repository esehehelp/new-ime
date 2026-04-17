#include "preedit.h"

namespace newime {

void Preedit::set_simple(const std::string& text, int cursor_pos) {
    text_ = text;
    highlighted_ = false;
    active_segment_ = -1;
    segment_ranges_.clear();

    // Convert character position to byte position
    cursor_byte_pos_ = 0;
    int chars = 0;
    for (size_t i = 0; i < text_.size() && chars < cursor_pos;) {
        unsigned char ch = text_[i];
        int byte_len = 1;
        if (ch >= 0xF0) byte_len = 4;
        else if (ch >= 0xE0) byte_len = 3;
        else if (ch >= 0xC0) byte_len = 2;
        i += byte_len;
        cursor_byte_pos_ = static_cast<int>(i);
        chars++;
    }
}

void Preedit::set_highlighted(const std::string& text) {
    text_ = text;
    highlighted_ = true;
    active_segment_ = -1;
    cursor_byte_pos_ = static_cast<int>(text_.size());
    segment_ranges_.clear();
}

void Preedit::set_segments(const std::vector<std::string>& segments, int active_segment) {
    text_.clear();
    segment_ranges_.clear();
    highlighted_ = false;
    active_segment_ = active_segment;

    for (const auto& seg : segments) {
        int start = static_cast<int>(text_.size());
        text_ += seg;
        int end = static_cast<int>(text_.size());
        segment_ranges_.emplace_back(start, end);
    }

    cursor_byte_pos_ = static_cast<int>(text_.size());
}

void Preedit::reset() {
    text_.clear();
    cursor_byte_pos_ = 0;
    active_segment_ = -1;
    highlighted_ = false;
    segment_ranges_.clear();
}

} // namespace newime
