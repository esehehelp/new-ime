#include "ctc_decoder.h"
#include "lm_scorer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>

namespace {

// Count UTF-8 codepoints (glyphs), not bytes. Used for the length-penalty
// term so alpha/beta tuning behaves the same on ASCII and CJK.
int utf8_codepoints(const std::string& s) {
    int n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

} // namespace

namespace newime {

CTCDecoder::CTCDecoder(const std::vector<std::string>& vocab, int blank_id)
    : vocab_(vocab), blank_id_(blank_id) {}

DecodedCandidate CTCDecoder::greedy_decode(const std::vector<float>& logits,
                                            int seq_len, int vocab_size) const {
    std::string result;
    float total_score = 0.0f;
    int prev_token = -1;

    for (int t = 0; t < seq_len; t++) {
        // Find argmax
        int best_id = 0;
        float best_val = logits[t * vocab_size];
        for (int v = 1; v < vocab_size; v++) {
            float val = logits[t * vocab_size + v];
            if (val > best_val) {
                best_val = val;
                best_id = v;
            }
        }

        total_score += best_val;

        // CTC collapse: skip blank and repeated tokens
        if (best_id != blank_id_ && best_id != prev_token) {
            if (best_id >= 0 && best_id < static_cast<int>(vocab_.size())) {
                result += vocab_[best_id];
            }
        }
        prev_token = best_id;
    }

    return {result, total_score / static_cast<float>(seq_len)};
}

std::vector<DecodedCandidate> CTCDecoder::beam_search(
    const std::vector<float>& logits,
    int seq_len, int vocab_size,
    int beam_width,
    LMScorer* lm_scorer,
    float lm_alpha,
    float lm_beta) const
{
    const bool use_lm = (lm_scorer != nullptr) && (lm_alpha != 0.0f);
    const bool use_beta = (lm_beta != 0.0f);

    auto fused_score = [&](const std::string& prefix, float ctc_log) -> float {
        float out = ctc_log;
        if (use_lm) out += lm_alpha * lm_scorer->score(prefix);
        if (use_beta) out += lm_beta * static_cast<float>(utf8_codepoints(prefix));
        return out;
    };
    // Prefix beam search (Hannun et al. 2014)
    //
    // Each beam tracks: prefix string, probability of ending in blank,
    // probability of ending in non-blank.

    struct Beam {
        std::string prefix;
        int last_token = -1;
        float score_blank = 0.0f;      // log prob ending in blank
        float score_non_blank = 0.0f;  // log prob ending in non-blank

        float total_score() const {
            // log-sum-exp of blank and non-blank scores
            float max_s = std::max(score_blank, score_non_blank);
            return max_s + std::log(
                std::exp(score_blank - max_s) + std::exp(score_non_blank - max_s));
        }
    };

    constexpr float NEG_INF = -1e30f;

    // Initialize with empty prefix
    std::map<std::string, Beam> beams;
    Beam initial;
    initial.score_blank = 0.0f;  // log(1) = 0
    initial.score_non_blank = NEG_INF;
    initial.last_token = -1;
    beams[""] = initial;

    for (int t = 0; t < seq_len; t++) {
        std::map<std::string, Beam> new_beams;

        // Compute log-softmax for this timestep
        std::vector<float> log_probs(vocab_size);
        float max_logit = *std::max_element(
            logits.begin() + t * vocab_size,
            logits.begin() + (t + 1) * vocab_size);
        float sum_exp = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            sum_exp += std::exp(logits[t * vocab_size + v] - max_logit);
        }
        float log_sum = max_logit + std::log(sum_exp);
        for (int v = 0; v < vocab_size; v++) {
            log_probs[v] = logits[t * vocab_size + v] - log_sum;
        }

        for (auto& [prefix, beam] : beams) {
            float prev_total = beam.total_score();

            // Blank extension
            {
                auto& nb = new_beams[prefix];
                if (nb.prefix.empty()) {
                    nb.prefix = prefix;
                    nb.last_token = beam.last_token;
                    nb.score_blank = NEG_INF;
                    nb.score_non_blank = NEG_INF;
                }
                float new_score = prev_total + log_probs[blank_id_];
                float max_s = std::max(nb.score_blank, new_score);
                nb.score_blank = max_s + std::log(
                    std::exp(nb.score_blank - max_s) + std::exp(new_score - max_s));
            }

            // Non-blank extensions (top-k tokens for efficiency)
            std::vector<int> top_indices(vocab_size);
            std::iota(top_indices.begin(), top_indices.end(), 0);
            std::partial_sort(top_indices.begin(),
                              top_indices.begin() + std::min(beam_width * 2, vocab_size),
                              top_indices.end(),
                              [&log_probs](int a, int b) {
                                  return log_probs[a] > log_probs[b];
                              });

            int num_extend = std::min(beam_width * 2, vocab_size);
            for (int k = 0; k < num_extend; k++) {
                int v = top_indices[k];
                if (v == blank_id_) continue;

                std::string new_prefix = prefix;
                float score;

                if (v == beam.last_token) {
                    // Same token: only extend from blank state (to allow repeat)
                    score = beam.score_blank + log_probs[v];
                } else {
                    score = prev_total + log_probs[v];
                }

                if (v >= 0 && v < static_cast<int>(vocab_.size())) {
                    new_prefix += vocab_[v];
                }

                auto& nb = new_beams[new_prefix];
                if (nb.prefix.empty()) {
                    nb.prefix = new_prefix;
                    nb.score_blank = NEG_INF;
                    nb.score_non_blank = NEG_INF;
                }
                nb.last_token = v;
                float max_s = std::max(nb.score_non_blank, score);
                nb.score_non_blank = max_s + std::log(
                    std::exp(nb.score_non_blank - max_s) + std::exp(score - max_s));
            }
        }

        // Prune to beam_width by the fused score (CTC + alpha * LM + beta * len).
        std::vector<std::pair<std::string, Beam>> sorted_beams(
            new_beams.begin(), new_beams.end());
        std::partial_sort(sorted_beams.begin(),
                          sorted_beams.begin() + std::min(beam_width, static_cast<int>(sorted_beams.size())),
                          sorted_beams.end(),
                          [&](const auto& a, const auto& b) {
                              return fused_score(a.first, a.second.total_score())
                                   > fused_score(b.first, b.second.total_score());
                          });

        beams.clear();
        for (int i = 0; i < std::min(beam_width, static_cast<int>(sorted_beams.size())); i++) {
            beams[sorted_beams[i].first] = sorted_beams[i].second;
        }
    }

    // Collect results. Final ranking uses the same fused score as pruning.
    std::vector<DecodedCandidate> results;
    for (auto& [prefix, beam] : beams) {
        results.push_back({prefix, fused_score(prefix, beam.total_score())});
    }
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    return results;
}

} // namespace newime
