#include "ctc_decoder.h"
#include <cassert>
#include <iostream>
#include <cmath>

using newime::CTCDecoder;

void test_greedy_simple() {
    // Vocab: 0=blank, 1="あ", 2="い", 3="う"
    std::vector<std::string> vocab = {"", "あ", "い", "う"};
    CTCDecoder decoder(vocab, 0);

    // 4 timesteps, 4 vocab tokens
    // Logits: [blank, あ, い, う] per timestep
    std::vector<float> logits = {
        -10, 5, -10, -10,  // t0: あ
        -10, 5, -10, -10,  // t1: あ (repeat, should be collapsed)
        -1,  -10, 5, -10,  // t2: い
        -10, -10, -10, 5,  // t3: う
    };

    auto result = decoder.greedy_decode(logits, 4, 4);
    assert(result.text == "あいう");  // Repeated あ collapsed
    std::cout << "PASS: test_greedy_simple (got: " << result.text << ")\n";
}

void test_greedy_with_blanks() {
    std::vector<std::string> vocab = {"", "か", "ん", "じ"};
    CTCDecoder decoder(vocab, 0);

    std::vector<float> logits = {
        5, -10, -10, -10,  // t0: blank
        -10, 5, -10, -10,  // t1: か
        5, -10, -10, -10,  // t2: blank
        -10, -10, 5, -10,  // t3: ん
        -10, -10, -10, 5,  // t4: じ
    };

    auto result = decoder.greedy_decode(logits, 5, 4);
    assert(result.text == "かんじ");
    std::cout << "PASS: test_greedy_with_blanks (got: " << result.text << ")\n";
}

void test_greedy_all_blank() {
    std::vector<std::string> vocab = {"", "あ"};
    CTCDecoder decoder(vocab, 0);

    std::vector<float> logits = {
        5, -10,  // blank
        5, -10,  // blank
    };

    auto result = decoder.greedy_decode(logits, 2, 2);
    assert(result.text.empty());
    std::cout << "PASS: test_greedy_all_blank\n";
}

void test_beam_search_basic() {
    std::vector<std::string> vocab = {"", "あ", "い"};
    CTCDecoder decoder(vocab, 0);

    // Clear preference: あい
    std::vector<float> logits = {
        -10, 5, -10,   // t0: あ
        -1, -10,  5,   // t1: い
    };

    auto results = decoder.beam_search(logits, 2, 3, 3);
    assert(!results.empty());
    assert(results[0].text == "あい");
    std::cout << "PASS: test_beam_search_basic (top: " << results[0].text << ")\n";
}

void test_beam_search_multiple_candidates() {
    std::vector<std::string> vocab = {"", "か", "が"};
    CTCDecoder decoder(vocab, 0);

    // Ambiguous: か and が have similar scores
    std::vector<float> logits = {
        -5, 3, 2.5,  // t0: か slightly preferred over が
    };

    auto results = decoder.beam_search(logits, 1, 3, 3);
    assert(results.size() >= 2);
    // Both か and が should appear as candidates
    bool has_ka = false, has_ga = false;
    for (const auto& r : results) {
        if (r.text == "か") has_ka = true;
        if (r.text == "が") has_ga = true;
    }
    assert(has_ka);
    assert(has_ga);
    std::cout << "PASS: test_beam_search_multiple_candidates\n";
}

int main() {
    test_greedy_simple();
    test_greedy_with_blanks();
    test_greedy_all_blank();
    test_beam_search_basic();
    test_beam_search_multiple_candidates();

    std::cout << "\nAll CTC decoder tests passed!\n";
    return 0;
}
