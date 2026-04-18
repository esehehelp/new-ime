#pragma once

#include <string>
#include <vector>
#include <memory>

namespace newime {

struct ConversionCandidate {
    std::string text;
    std::string reading;
    float score;
};

/// Abstract inference backend.
/// Implementations: OnnxInferenceEngine (production), MockInferenceEngine (testing).
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    /// Run encoder + decoder, return raw CTC logits.
    /// @param input_ids Tokenized encoder input.
    /// @return Logits as flat array (seq_len * vocab_size), plus output seq_len.
    struct InferenceResult {
        std::vector<float> logits;
        int seq_len;
        int vocab_size;
    };

    virtual InferenceResult infer(const std::vector<int>& input_ids) = 0;
};

/// Mock inference backend for testing.
/// Returns dummy logits that decode to a fixed transformation of input.
class MockInferenceEngine : public InferenceBackend {
public:
    explicit MockInferenceEngine(int vocab_size = 6500);
    InferenceResult infer(const std::vector<int>& input_ids) override;

private:
    int vocab_size_;
};

/// Main conversion engine. Orchestrates tokenization, inference, and CTC decoding.
class ConversionEngine {
public:
    /// Create with a specific inference backend.
    explicit ConversionEngine(std::unique_ptr<InferenceBackend> backend);

    /// Convert kana input to kanji-kana mixed candidates.
    std::vector<ConversionCandidate> convert(
        const std::string& kana_input,
        const std::string& left_context,
        int num_candidates = 5,
        bool use_refinement = false);

    /// Set surrounding text context.
    void set_context(const std::string& surrounding_text, int cursor_position);

private:
    std::unique_ptr<InferenceBackend> backend_;
    std::string context_;
};

} // namespace newime
