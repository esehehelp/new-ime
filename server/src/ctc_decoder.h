#pragma once

#include <string>
#include <vector>

namespace newime {

/// Result from CTC decoding.
struct DecodedCandidate {
    std::string text;
    float score;
};

/// CTC greedy and beam search decoder.
///
/// Takes CTC logits (seq_len x vocab_size) and produces decoded strings.
/// The BLANK token at index 0 is used for CTC collapse.
class CTCDecoder {
public:
    /// @param vocab Token ID to character mapping.
    /// @param blank_id Token ID for CTC blank (default 0).
    explicit CTCDecoder(const std::vector<std::string>& vocab, int blank_id = 0);

    /// Greedy decode: argmax at each position, collapse blanks and repeats.
    DecodedCandidate greedy_decode(const std::vector<float>& logits,
                                   int seq_len, int vocab_size) const;

    /// Beam search: returns top-k candidates.
    std::vector<DecodedCandidate> beam_search(const std::vector<float>& logits,
                                               int seq_len, int vocab_size,
                                               int beam_width = 5) const;

private:
    std::vector<std::string> vocab_;
    int blank_id_;
};

} // namespace newime
