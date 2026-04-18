#pragma once

#include <string>
#include <string_view>

namespace newime {

/// Abstract LM scorer for CTC prefix beam shallow fusion.
///
/// score() returns the natural-log probability of the full prefix so far.
/// The CTC beam uses the delta (new minus previous) when a prefix is
/// extended, so implementations are free to cache.
class LMScorer {
public:
    virtual ~LMScorer() = default;

    /// Full-prefix log-probability (natural log). Prefix is the decoded
    /// surface string (Unicode codepoints as UTF-8 bytes).
    virtual float score(std::string_view prefix) = 0;
};

} // namespace newime
