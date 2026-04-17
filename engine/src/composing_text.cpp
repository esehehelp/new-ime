#include "composing_text.h"
#include <algorithm>
#include <cstring>

namespace newime {

const std::vector<ComposingText::RomajiEntry>& ComposingText::romaji_table() {
    // Sorted by length descending so longer matches are tried first
    static const std::vector<RomajiEntry> table = {
        // Three-character combinations
        {"shi", "し"}, {"chi", "ち"}, {"tsu", "つ"}, {"sha", "しゃ"},
        {"shu", "しゅ"}, {"sho", "しょ"}, {"cha", "ちゃ"}, {"chu", "ちゅ"},
        {"cho", "ちょ"}, {"tya", "ちゃ"}, {"tyu", "ちゅ"}, {"tyo", "ちょ"},
        {"sya", "しゃ"}, {"syu", "しゅ"}, {"syo", "しょ"}, {"nya", "にゃ"},
        {"nyu", "にゅ"}, {"nyo", "にょ"}, {"hya", "ひゃ"}, {"hyu", "ひゅ"},
        {"hyo", "ひょ"}, {"mya", "みゃ"}, {"myu", "みゅ"}, {"myo", "みょ"},
        {"rya", "りゃ"}, {"ryu", "りゅ"}, {"ryo", "りょ"}, {"gya", "ぎゃ"},
        {"gyu", "ぎゅ"}, {"gyo", "ぎょ"}, {"bya", "びゃ"}, {"byu", "びゅ"},
        {"byo", "びょ"}, {"pya", "ぴゃ"}, {"pyu", "ぴゅ"}, {"pyo", "ぴょ"},
        {"kya", "きゃ"}, {"kyu", "きゅ"}, {"kyo", "きょ"}, {"jya", "じゃ"},
        {"jyu", "じゅ"}, {"jyo", "じょ"}, {"xtu", "っ"},  {"xya", "ゃ"},
        {"xyu", "ゅ"},  {"xyo", "ょ"},  {"xwa", "ゎ"},

        // Two-character combinations
        {"ka", "か"}, {"ki", "き"}, {"ku", "く"}, {"ke", "け"}, {"ko", "こ"},
        {"sa", "さ"}, {"si", "し"}, {"su", "す"}, {"se", "せ"}, {"so", "そ"},
        {"ta", "た"}, {"ti", "ち"}, {"tu", "つ"}, {"te", "て"}, {"to", "と"},
        {"na", "な"}, {"ni", "に"}, {"nu", "ぬ"}, {"ne", "ね"}, {"no", "の"},
        {"ha", "は"}, {"hi", "ひ"}, {"hu", "ふ"}, {"he", "へ"}, {"ho", "ほ"},
        {"ma", "ま"}, {"mi", "み"}, {"mu", "む"}, {"me", "め"}, {"mo", "も"},
        {"ya", "や"}, {"yi", "い"}, {"yu", "ゆ"}, {"ye", "いぇ"}, {"yo", "よ"},
        {"ra", "ら"}, {"ri", "り"}, {"ru", "る"}, {"re", "れ"}, {"ro", "ろ"},
        {"wa", "わ"}, {"wi", "ゐ"}, {"wu", "う"}, {"we", "ゑ"}, {"wo", "を"},
        {"ga", "が"}, {"gi", "ぎ"}, {"gu", "ぐ"}, {"ge", "げ"}, {"go", "ご"},
        {"za", "ざ"}, {"zi", "じ"}, {"zu", "ず"}, {"ze", "ぜ"}, {"zo", "ぞ"},
        {"da", "だ"}, {"di", "ぢ"}, {"du", "づ"}, {"de", "で"}, {"do", "ど"},
        {"ba", "ば"}, {"bi", "び"}, {"bu", "ぶ"}, {"be", "べ"}, {"bo", "ぼ"},
        {"pa", "ぱ"}, {"pi", "ぴ"}, {"pu", "ぷ"}, {"pe", "ぺ"}, {"po", "ぽ"},
        {"fa", "ふぁ"}, {"fi", "ふぃ"}, {"fu", "ふ"}, {"fe", "ふぇ"}, {"fo", "ふぉ"},
        {"ja", "じゃ"}, {"ji", "じ"}, {"ju", "じゅ"}, {"je", "じぇ"}, {"jo", "じょ"},
        {"nn", "ん"}, {"xa", "ぁ"}, {"xi", "ぃ"}, {"xu", "ぅ"}, {"xe", "ぇ"},
        {"xo", "ぉ"},

        // Single vowels
        {"a", "あ"}, {"i", "い"}, {"u", "う"}, {"e", "え"}, {"o", "お"},
    };
    return table;
}

void ComposingText::input_char(char c) {
    // Convert to lowercase
    if (c >= 'A' && c <= 'Z') {
        c = c - 'A' + 'a';
    }

    romaji_buffer_ += c;

    // Handle 'n' before consonant → 'ん'
    if (romaji_buffer_.size() >= 2 && romaji_buffer_[0] == 'n') {
        char second = romaji_buffer_[1];
        // n + consonant (not 'a','i','u','e','o','y','n') → ん + restart
        if (second != 'a' && second != 'i' && second != 'u' &&
            second != 'e' && second != 'o' && second != 'y' &&
            second != 'n') {
            insert_at_cursor("ん");
            romaji_buffer_ = romaji_buffer_.substr(1);
        }
    }

    // Handle double consonant → っ
    if (romaji_buffer_.size() >= 2) {
        char first = romaji_buffer_[0];
        char second = romaji_buffer_[1];
        if (first == second && first != 'a' && first != 'i' &&
            first != 'u' && first != 'e' && first != 'o' && first != 'n') {
            insert_at_cursor("っ");
            romaji_buffer_ = romaji_buffer_.substr(1);
        }
    }

    // Try romaji conversion
    try_convert_romaji();
}

bool ComposingText::try_convert_romaji() {
    for (const auto& entry : romaji_table()) {
        size_t len = std::strlen(entry.romaji);
        if (romaji_buffer_.size() >= len &&
            romaji_buffer_.substr(0, len) == entry.romaji) {
            insert_at_cursor(entry.hiragana);
            romaji_buffer_ = romaji_buffer_.substr(len);
            return true;
        }
    }
    return false;
}

void ComposingText::insert_at_cursor(const std::string& kana) {
    // Count UTF-8 characters up to cursor position to find byte offset
    int char_count = 0;
    size_t byte_pos = 0;
    while (char_count < cursor_ && byte_pos < hiragana_.size()) {
        unsigned char ch = hiragana_[byte_pos];
        if (ch < 0x80) byte_pos += 1;
        else if (ch < 0xE0) byte_pos += 2;
        else if (ch < 0xF0) byte_pos += 3;
        else byte_pos += 4;
        char_count++;
    }

    hiragana_.insert(byte_pos, kana);

    // Count characters in inserted kana
    int inserted_chars = 0;
    for (size_t i = 0; i < kana.size();) {
        unsigned char ch = kana[i];
        if (ch < 0x80) i += 1;
        else if (ch < 0xE0) i += 2;
        else if (ch < 0xF0) i += 3;
        else i += 4;
        inserted_chars++;
    }
    cursor_ += inserted_chars;
}

void ComposingText::delete_left() {
    if (cursor_ == 0 && romaji_buffer_.empty()) return;

    if (!romaji_buffer_.empty()) {
        romaji_buffer_.pop_back();
        return;
    }

    // Delete one UTF-8 character before cursor
    int char_count = 0;
    size_t prev_pos = 0;
    size_t byte_pos = 0;
    while (char_count < cursor_ && byte_pos < hiragana_.size()) {
        prev_pos = byte_pos;
        unsigned char ch = hiragana_[byte_pos];
        if (ch < 0x80) byte_pos += 1;
        else if (ch < 0xE0) byte_pos += 2;
        else if (ch < 0xF0) byte_pos += 3;
        else byte_pos += 4;
        char_count++;
    }

    hiragana_.erase(prev_pos, byte_pos - prev_pos);
    cursor_--;
}

void ComposingText::delete_right() {
    // Find byte position at cursor
    int char_count = 0;
    size_t byte_pos = 0;
    while (char_count < cursor_ && byte_pos < hiragana_.size()) {
        unsigned char ch = hiragana_[byte_pos];
        if (ch < 0x80) byte_pos += 1;
        else if (ch < 0xE0) byte_pos += 2;
        else if (ch < 0xF0) byte_pos += 3;
        else byte_pos += 4;
        char_count++;
    }

    if (byte_pos >= hiragana_.size()) return;

    // Find end of character at cursor
    size_t end_pos = byte_pos;
    unsigned char ch = hiragana_[end_pos];
    if (ch < 0x80) end_pos += 1;
    else if (ch < 0xE0) end_pos += 2;
    else if (ch < 0xF0) end_pos += 3;
    else end_pos += 4;

    hiragana_.erase(byte_pos, end_pos - byte_pos);
}

void ComposingText::move_cursor(int offset) {
    // Count total characters
    int total_chars = 0;
    for (size_t i = 0; i < hiragana_.size();) {
        unsigned char ch = hiragana_[i];
        if (ch < 0x80) i += 1;
        else if (ch < 0xE0) i += 2;
        else if (ch < 0xF0) i += 3;
        else i += 4;
        total_chars++;
    }

    cursor_ = std::clamp(cursor_ + offset, 0, total_chars);
}

std::string ComposingText::display() const {
    // Insert romaji buffer at cursor position
    if (romaji_buffer_.empty()) return hiragana_;

    int char_count = 0;
    size_t byte_pos = 0;
    while (char_count < cursor_ && byte_pos < hiragana_.size()) {
        unsigned char ch = hiragana_[byte_pos];
        if (ch < 0x80) byte_pos += 1;
        else if (ch < 0xE0) byte_pos += 2;
        else if (ch < 0xF0) byte_pos += 3;
        else byte_pos += 4;
        char_count++;
    }

    std::string result = hiragana_;
    result.insert(byte_pos, romaji_buffer_);
    return result;
}

void ComposingText::reset() {
    hiragana_.clear();
    romaji_buffer_.clear();
    cursor_ = 0;
}

} // namespace newime
