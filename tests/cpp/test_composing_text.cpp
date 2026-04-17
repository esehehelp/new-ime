#include "composing_text.h"
#include <cassert>
#include <iostream>

using newime::ComposingText;

void test_basic_romaji() {
    ComposingText ct;
    // "ka" → "か"
    ct.input_char('k');
    assert(ct.display() == "k");
    ct.input_char('a');
    assert(ct.hiragana() == "か");
    assert(ct.display() == "か");
    std::cout << "PASS: test_basic_romaji\n";
}

void test_word() {
    ComposingText ct;
    // "kanji" → "かんじ"
    for (char c : std::string("kanji")) ct.input_char(c);
    assert(ct.hiragana() == "かんじ");
    std::cout << "PASS: test_word\n";
}

void test_double_consonant() {
    ComposingText ct;
    // "kitte" → "きって"
    for (char c : std::string("kitte")) ct.input_char(c);
    assert(ct.hiragana() == "きって");
    std::cout << "PASS: test_double_consonant\n";
}

void test_nn() {
    ComposingText ct;
    // "nn" → "ん"
    ct.input_char('n');
    ct.input_char('n');
    assert(ct.hiragana() == "ん");
    std::cout << "PASS: test_nn\n";
}

void test_n_before_consonant() {
    ComposingText ct;
    // "kanka" → "かんか"
    for (char c : std::string("kanka")) ct.input_char(c);
    assert(ct.hiragana() == "かんか");
    std::cout << "PASS: test_n_before_consonant\n";
}

void test_shi() {
    ComposingText ct;
    for (char c : std::string("shi")) ct.input_char(c);
    assert(ct.hiragana() == "し");
    std::cout << "PASS: test_shi\n";
}

void test_chi() {
    ComposingText ct;
    for (char c : std::string("chi")) ct.input_char(c);
    assert(ct.hiragana() == "ち");
    std::cout << "PASS: test_chi\n";
}

void test_tsu() {
    ComposingText ct;
    for (char c : std::string("tsu")) ct.input_char(c);
    assert(ct.hiragana() == "つ");
    std::cout << "PASS: test_tsu\n";
}

void test_delete_left() {
    ComposingText ct;
    for (char c : std::string("ka")) ct.input_char(c);
    assert(ct.hiragana() == "か");
    ct.delete_left();
    assert(ct.hiragana().empty());
    std::cout << "PASS: test_delete_left\n";
}

void test_delete_romaji_buffer() {
    ComposingText ct;
    ct.input_char('k');
    assert(ct.display() == "k");
    ct.delete_left();
    assert(ct.display().empty());
    assert(ct.empty());
    std::cout << "PASS: test_delete_romaji_buffer\n";
}

void test_reset() {
    ComposingText ct;
    for (char c : std::string("tesuto")) ct.input_char(c);
    assert(!ct.empty());
    ct.reset();
    assert(ct.empty());
    assert(ct.hiragana().empty());
    assert(ct.cursor() == 0);
    std::cout << "PASS: test_reset\n";
}

void test_uppercase() {
    ComposingText ct;
    for (char c : std::string("KA")) ct.input_char(c);
    assert(ct.hiragana() == "か");
    std::cout << "PASS: test_uppercase\n";
}

int main() {
    test_basic_romaji();
    test_word();
    test_double_consonant();
    test_nn();
    test_n_before_consonant();
    test_shi();
    test_chi();
    test_tsu();
    test_delete_left();
    test_delete_romaji_buffer();
    test_reset();
    test_uppercase();

    std::cout << "\nAll composing text tests passed!\n";
    return 0;
}
