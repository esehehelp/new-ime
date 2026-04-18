#pragma once

#ifdef NEWIME_ENABLE_KENLM

#include "lm_scorer.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace newime {

/// KenLM-backed scorer. LM model is expected to be a character n-gram
/// trained on char-spaced corpus (each glyph is one KenLM word).
///
/// The scorer splits the prefix into UTF-8 codepoints and joins them
/// with spaces to match the LM tokenization, then queries KenLM.
/// Per-prefix scores are cached since beams frequently re-evaluate
/// the same prefix text across timesteps.
class KenLMCharScorer : public LMScorer {
public:
    /// @param lm_path path to KenLM .arpa or .bin file.
    explicit KenLMCharScorer(const std::string& lm_path);
    ~KenLMCharScorer() override;

    float score(std::string_view prefix) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::unordered_map<std::string, float> cache_;
};

} // namespace newime

#endif // NEWIME_ENABLE_KENLM
