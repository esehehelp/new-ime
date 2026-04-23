// C shim around KenLM's lm::ngram::LoadVirtual interface.
// Exposes a minimal C API so Rust can FFI in without knowing about the
// virtual base class or the state-size protocol.
//
// Character-level scoring: the input UTF-8 string is split into individual
// codepoints, each treated as one KenLM "word". This matches how the model
// was trained (text is pre-split with spaces between every glyph) and the
// historical runtime scorer implementation retained only for score parity.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "lm/model.hh"

namespace {

constexpr float LOG_10 = 2.30258509299f;

struct Handle {
    std::unique_ptr<lm::base::Model> model;
    std::vector<uint8_t> state;
    std::vector<uint8_t> next;
};

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
        else { n = 1; }
        if (i + n > s.size()) n = s.size() - i;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

} // namespace

extern "C" {

void* kenlm_shim_load(const char* path) {
    try {
        lm::base::Model* m = lm::ngram::LoadVirtual(path);
        if (!m) return nullptr;
        auto* h = new Handle();
        h->model.reset(m);
        h->state.resize(m->StateSize());
        h->next.resize(m->StateSize());
        return h;
    } catch (...) {
        return nullptr;
    }
}

void kenlm_shim_free(void* handle) {
    if (!handle) return;
    delete static_cast<Handle*>(handle);
}

// Score a UTF-8 prefix as a full sentence (with BOS). Returns the natural-log
// probability of the whole prefix (sum of per-char BaseScore * ln(10)).
// Returns 0.0 for an empty prefix. Never throws.
float kenlm_shim_score(void* handle, const char* utf8, size_t len) {
    if (!handle || !utf8 || len == 0) return 0.0f;
    auto* h = static_cast<Handle*>(handle);
    try {
        auto chars = split_utf8_chars(std::string_view(utf8, len));
        const auto& vocab = h->model->BaseVocabulary();
        h->model->BeginSentenceWrite(h->state.data());
        float total_log10 = 0.0f;
        for (const auto& ch : chars) {
            lm::WordIndex w = vocab.Index(ch);
            total_log10 += h->model->BaseScore(h->state.data(), w, h->next.data());
            h->state.swap(h->next);
        }
        return total_log10 * LOG_10;
    } catch (...) {
        return 0.0f;
    }
}

} // extern "C"
