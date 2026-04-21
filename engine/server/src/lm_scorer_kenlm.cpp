#ifdef NEWIME_ENABLE_KENLM

#include "lm_scorer_kenlm.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lm/model.hh"

namespace newime {

namespace {

// Natural log of 10, used to convert KenLM's log10 scores to natural log.
constexpr float LOG_10 = 2.30258509299f;

// Split a UTF-8 string into its codepoint substrings (each glyph as a
// standalone UTF-8 slice). Assumes well-formed UTF-8 input.
std::vector<std::string> split_utf8_chars(std::string_view s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n;
        if ((c & 0x80) == 0) n = 1;
        else if ((c & 0xE0) == 0xC0) n = 2;
        else if ((c & 0xF0) == 0xE0) n = 3;
        else if ((c & 0xF8) == 0xF0) n = 4;
        else { n = 1; }  // malformed — advance by one byte
        if (i + n > s.size()) n = s.size() - i;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

} // namespace

struct KenLMCharScorer::Impl {
    std::unique_ptr<lm::base::Model> model;
    explicit Impl(const std::string& path)
        : model(lm::ngram::LoadVirtual(path.c_str())) {}
};

KenLMCharScorer::KenLMCharScorer(const std::string& lm_path)
    : impl_(std::make_unique<Impl>(lm_path)) {}

KenLMCharScorer::~KenLMCharScorer() = default;

float KenLMCharScorer::score(std::string_view prefix) {
    if (prefix.empty()) return 0.0f;

    std::string key(prefix);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    auto chars = split_utf8_chars(prefix);
    const auto& vocab = impl_->model->BaseVocabulary();
    std::vector<uint8_t> state(impl_->model->StateSize());
    std::vector<uint8_t> next(impl_->model->StateSize());
    impl_->model->BeginSentenceWrite(state.data());
    float total_log10 = 0.0f;
    for (const auto& ch : chars) {
        lm::WordIndex w = vocab.Index(ch);
        total_log10 += impl_->model->BaseScore(state.data(), w, next.data());
        state.swap(next);
    }
    float natlog = total_log10 * LOG_10;
    cache_.emplace(std::move(key), natlog);
    return natlog;
}

} // namespace newime

#endif // NEWIME_ENABLE_KENLM
