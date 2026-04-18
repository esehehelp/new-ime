#include "conversion_engine.h"
#include "ctc_decoder.h"

#include <cmath>

namespace newime {

// ---- MockInferenceEngine ----

MockInferenceEngine::MockInferenceEngine(int vocab_size)
    : vocab_size_(vocab_size) {}

InferenceBackend::InferenceResult MockInferenceEngine::infer(
    const std::vector<int>& input_ids)
{
    // Mock: produce logits that are identity-ish (each input token maps
    // to the same output token ID, with some blank tokens in between).
    // This makes the mock CTC output approximately echo the input.
    int seq_len = static_cast<int>(input_ids.size());

    std::vector<float> logits(seq_len * vocab_size_, -10.0f);

    for (int t = 0; t < seq_len; t++) {
        int token = input_ids[t];
        if (token >= 0 && token < vocab_size_) {
            // Make the input token the most likely output
            logits[t * vocab_size_ + token] = 5.0f;
        }
        // Also give blank a moderate score so CTC can use it
        logits[t * vocab_size_ + 0] = -1.0f;  // blank_id = 0
    }

    return {logits, seq_len, vocab_size_};
}

// ---- ConversionEngine ----

ConversionEngine::ConversionEngine(std::unique_ptr<InferenceBackend> backend)
    : backend_(std::move(backend)) {}

std::vector<ConversionCandidate> ConversionEngine::convert(
    const std::string& kana_input,
    const std::string& /*left_context*/,
    int num_candidates,
    bool /*use_refinement*/)
{
    // Simple tokenization: each UTF-8 character → its index
    // In production, this uses the actual tokenizer vocabulary
    std::vector<int> input_ids;
    for (size_t i = 0; i < kana_input.size();) {
        unsigned char ch = kana_input[i];
        int byte_len = 1;
        if (ch >= 0xF0) byte_len = 4;
        else if (ch >= 0xE0) byte_len = 3;
        else if (ch >= 0xC0) byte_len = 2;

        // Use a simple hash as token ID for mock purposes
        int token_id = 0;
        for (int b = 0; b < byte_len && i + b < kana_input.size(); b++) {
            token_id = (token_id * 256 + static_cast<unsigned char>(kana_input[i + b])) % 6000;
        }
        token_id = std::max(token_id, 1);  // Avoid blank token (0)
        input_ids.push_back(token_id);
        i += byte_len;
    }

    if (input_ids.empty()) return {};

    // Run inference
    auto result = backend_->infer(input_ids);

    // Build vocab for CTC decoder (mock: token IDs map back to kana characters)
    // In production, this is the output tokenizer's vocabulary
    std::vector<std::string> vocab(result.vocab_size);
    vocab[0] = "";  // blank

    // For mock: decode input characters back
    size_t char_idx = 0;
    for (size_t i = 0; i < kana_input.size() && char_idx < input_ids.size();) {
        unsigned char ch = kana_input[i];
        int byte_len = 1;
        if (ch >= 0xF0) byte_len = 4;
        else if (ch >= 0xE0) byte_len = 3;
        else if (ch >= 0xC0) byte_len = 2;

        int tid = input_ids[char_idx];
        if (tid > 0 && tid < result.vocab_size) {
            vocab[tid] = kana_input.substr(i, byte_len);
        }
        i += byte_len;
        char_idx++;
    }

    CTCDecoder decoder(vocab, /*blank_id=*/0);

    // Use beam search for multiple candidates
    auto decoded = decoder.beam_search(
        result.logits, result.seq_len, result.vocab_size,
        num_candidates);

    std::vector<ConversionCandidate> candidates;
    for (auto& d : decoded) {
        candidates.push_back({d.text, kana_input, d.score});
    }

    return candidates;
}

void ConversionEngine::set_context(const std::string& surrounding_text,
                                    int /*cursor_position*/) {
    context_ = surrounding_text;
}

} // namespace newime
