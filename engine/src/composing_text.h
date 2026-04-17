#pragma once

#include <string>
#include <vector>

namespace newime {

/// Romaji to hiragana conversion and composing text management.
/// Handles the input state before conversion is requested.
class ComposingText {
public:
    ComposingText() = default;

    /// Add a character (romaji or direct kana)
    void input_char(char c);

    /// Delete one character to the left of cursor
    void delete_left();

    /// Delete one character to the right of cursor
    void delete_right();

    /// Move cursor by offset (negative = left, positive = right)
    void move_cursor(int offset);

    /// Get confirmed hiragana string
    const std::string& hiragana() const { return hiragana_; }

    /// Get display string (hiragana + pending romaji)
    std::string display() const;

    /// Get cursor position in display string (character count)
    int cursor() const { return cursor_; }

    /// Check if there is any composing text
    bool empty() const { return hiragana_.empty() && romaji_buffer_.empty(); }

    /// Clear all state
    void reset();

private:
    /// Try to convert romaji buffer to hiragana. Returns true if conversion happened.
    bool try_convert_romaji();

    /// Insert hiragana string at cursor position
    void insert_at_cursor(const std::string& kana);

    std::string hiragana_;       // Confirmed hiragana
    std::string romaji_buffer_;  // Pending romaji input
    int cursor_ = 0;             // Cursor position (in UTF-8 character count)

    // Romaji table (initialized statically)
    struct RomajiEntry {
        const char* romaji;
        const char* hiragana;
    };
    static const std::vector<RomajiEntry>& romaji_table();
};

} // namespace newime
