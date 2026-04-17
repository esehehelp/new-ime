#pragma once

#include <string>
#include <vector>

namespace newime {

/// Manages preedit display for fcitx5.
/// Handles both simple (single segment) and multi-segment display.
class Preedit {
public:
    Preedit() = default;

    /// Set simple preedit text (composing mode, no segments)
    void set_simple(const std::string& text, int cursor_pos);

    /// Set preedit with highlight (prediction shown inline)
    void set_highlighted(const std::string& text);

    /// Set multi-segment preedit (candidate selection mode)
    void set_segments(const std::vector<std::string>& segments, int active_segment);

    /// Get display text
    const std::string& text() const { return text_; }

    /// Get cursor position (byte offset in text)
    int cursor_byte_pos() const { return cursor_byte_pos_; }

    /// Get active segment index (-1 if no segments)
    int active_segment() const { return active_segment_; }

    /// Get segment boundaries (byte offsets)
    const std::vector<std::pair<int, int>>& segment_ranges() const { return segment_ranges_; }

    /// Whether the entire text should be highlighted
    bool is_highlighted() const { return highlighted_; }

    /// Clear
    void reset();

private:
    std::string text_;
    int cursor_byte_pos_ = 0;
    int active_segment_ = -1;
    bool highlighted_ = false;
    std::vector<std::pair<int, int>> segment_ranges_;
};

} // namespace newime
