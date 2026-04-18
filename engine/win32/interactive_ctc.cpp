// Interactive CLI for CTC-NAT + optional KenLM shallow fusion.
//
// Standalone counterpart to interactive.cpp (which drives the AR path via
// new-ime-engine.dll). Links directly against onnxruntime.dll plus the
// server/src CTC decoder and the vendored kenlm static libs — no DLL
// boundary on the model path, no FFI layer.
//
// Build: see build_ctc.bat in this folder.
//
// CLI:
//   interactive_ctc.exe --onnx path.onnx [--tokenizer path.tokenizer.json]
//                        [--lm model.bin --alpha 0.8 --beta 1.0]
//                        [--beam 4]

#include "ctc_decoder.h"
#include "lm_scorer_kenlm.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <windows.h>

namespace {

// Very small tokenizer sidecar reader. The training-side SharedCharTokenizer
// dumps JSON with fields `vocab` (id -> token) and special token IDs. We
// don't pull in a JSON dep — we parse the two fields we need by regex-ish
// scanning. The sidecar format is stable across checkpoints.
struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int blank_id = 4;
    int cls_id = 2;
    int sep_id = 3;
    int pad_id = 0;
    int unk_id = 1;

    // Parse one JSON-encoded string starting at j[i] (must point at opening
    // quote). Handles \" \\ \n \t \uXXXX. Leaves i just past the closing
    // quote. Returns the decoded UTF-8 string.
    static std::string parse_json_string(const std::string& j, size_t& i) {
        ++i;  // opening "
        std::string out;
        while (i < j.size() && j[i] != '"') {
            if (j[i] == '\\' && i + 1 < j.size()) {
                char n = j[i + 1];
                if (n == '"') { out += '"'; i += 2; }
                else if (n == '\\') { out += '\\'; i += 2; }
                else if (n == '/') { out += '/'; i += 2; }
                else if (n == 'n') { out += '\n'; i += 2; }
                else if (n == 't') { out += '\t'; i += 2; }
                else if (n == 'r') { out += '\r'; i += 2; }
                else if (n == 'b') { out += '\b'; i += 2; }
                else if (n == 'f') { out += '\f'; i += 2; }
                else if (n == 'u' && i + 5 < j.size()) {
                    unsigned cp = 0;
                    for (int k = 0; k < 4; ++k) {
                        char c = j[i + 2 + k];
                        cp <<= 4;
                        if (c >= '0' && c <= '9') cp |= (c - '0');
                        else if (c >= 'a' && c <= 'f') cp |= (c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') cp |= (c - 'A' + 10);
                    }
                    i += 6;
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | (cp >> 6));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (cp >> 12));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                } else {
                    i += 2;
                }
            } else {
                out += j[i++];
            }
        }
        if (i < j.size()) ++i;  // past closing "
        return out;
    }

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream buf;
        buf << f.rdbuf();
        std::string j = buf.str();

        // The SharedCharTokenizer sidecar is
        //   {"type": "shared", "max_kanji": N, "token_to_id": {token: id, ...}}
        // We parse token_to_id, then derive id_to_token and look up the
        // special token IDs by name.
        const std::string needle = "\"token_to_id\":";
        auto p = j.find(needle);
        if (p == std::string::npos) return false;
        p += needle.size();
        while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n')) ++p;
        if (p >= j.size() || j[p] != '{') return false;
        ++p;  // past '{'

        int max_id = -1;
        while (p < j.size()) {
            while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n'
                                    || j[p] == ',')) ++p;
            if (p >= j.size() || j[p] == '}') break;
            if (j[p] != '"') return false;
            std::string tok = parse_json_string(j, p);
            while (p < j.size() && (j[p] == ' ' || j[p] == ':')) ++p;
            // Parse integer id
            int id = 0;
            while (p < j.size() && j[p] >= '0' && j[p] <= '9') {
                id = id * 10 + (j[p] - '0'); ++p;
            }
            token_to_id[tok] = id;
            if (id > max_id) max_id = id;
        }
        if (max_id < 0) return false;

        id_to_token.assign(static_cast<size_t>(max_id) + 1, std::string{});
        for (auto& [tok, id] : token_to_id) {
            id_to_token[static_cast<size_t>(id)] = tok;
        }

        auto look = [&](const char* name, int& out) {
            auto it = token_to_id.find(name);
            if (it != token_to_id.end()) out = it->second;
        };
        look("[PAD]", pad_id);
        look("[UNK]", unk_id);
        look("[SEP]", sep_id);
        look("[CLS]", cls_id);
        look("[BLANK]", blank_id);

        return !id_to_token.empty();
    }

    int encode_char(const std::string& utf8_char) const {
        auto it = token_to_id.find(utf8_char);
        return it == token_to_id.end() ? unk_id : it->second;
    }
};

// Replace ASCII/fullwidth-Latin punctuation with the Japanese forms the
// training corpus actually uses. Without this, inputs like "，" (U+FF0C) or
// ASCII "?" reach the model as rare tokens and the KenLM prior drives the
// beam to drop them. Only rewrites input characters; the model can still
// emit either form on its own if it learned them.
std::string normalize_input_punct(const std::string& s) {
    static const std::pair<std::string, std::string> subs[] = {
        {"\xEF\xBC\x8C", "\xE3\x80\x81"},  // ，  -> 、
        {"\xEF\xBC\x8E", "\xE3\x80\x82"},  // ．  -> 。
        {",", "\xE3\x80\x81"},              // ASCII ,  -> 、
        {"?", "\xEF\xBC\x9F"},              // ASCII ?  -> ？
        {"!", "\xEF\xBC\x81"},              // ASCII !  -> ！
    };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ) {
        bool matched = false;
        for (const auto& [from, to] : subs) {
            if (s.compare(i, from.size(), from) == 0) {
                out += to;
                i += from.size();
                matched = true;
                break;
            }
        }
        if (!matched) out += s[i++];
    }
    return out;
}

// Decode a UTF-8 sequence starting at s[i], return codepoint and byte length.
// On malformed input, returns (-1, 1).
std::pair<int32_t, size_t> utf8_decode_one(const std::string& s, size_t i) {
    if (i >= s.size()) return {-1, 0};
    unsigned char c0 = static_cast<unsigned char>(s[i]);
    if ((c0 & 0x80) == 0) return {static_cast<int32_t>(c0), 1};
    auto need = [&](size_t n) { return i + n <= s.size(); };
    if ((c0 & 0xE0) == 0xC0 && need(2)) {
        int32_t cp = (c0 & 0x1F) << 6;
        cp |= (s[i + 1] & 0x3F);
        return {cp, 2};
    }
    if ((c0 & 0xF0) == 0xE0 && need(3)) {
        int32_t cp = (c0 & 0x0F) << 12;
        cp |= (s[i + 1] & 0x3F) << 6;
        cp |= (s[i + 2] & 0x3F);
        return {cp, 3};
    }
    if ((c0 & 0xF8) == 0xF0 && need(4)) {
        int32_t cp = (c0 & 0x07) << 18;
        cp |= (s[i + 1] & 0x3F) << 12;
        cp |= (s[i + 2] & 0x3F) << 6;
        cp |= (s[i + 3] & 0x3F);
        return {cp, 4};
    }
    return {-1, 1};
}

// True for hiragana, katakana, prolongation mark ー. We treat these as the
// "kana run" that gets fed to CTC; everything else (digits, latin,
// symbols, CJK already-converted kanji, etc.) is passthrough.
bool is_kana_codepoint(int32_t cp) {
    if (cp >= 0x3041 && cp <= 0x3096) return true;  // Hiragana
    if (cp >= 0x30A1 && cp <= 0x30FA) return true;  // Katakana
    if (cp == 0x30FC) return true;                   // ー prolong mark
    if (cp == 0x3093 || cp == 0x30F3) return true;   // ん / ン (inside range but explicit)
    return false;
}

// Split UTF-8 string into codepoint substrings.
std::vector<std::string> split_utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n = (c < 0x80) ? 1
                 : ((c & 0xE0) == 0xC0) ? 2
                 : ((c & 0xF0) == 0xE0) ? 3
                 : ((c & 0xF8) == 0xF0) ? 4 : 1;
        if (i + n > s.size()) n = s.size() - i;
        out.emplace_back(s.substr(i, n));
        i += n;
    }
    return out;
}

// Build the encoder input: [CLS] ctx... [SEP] reading...
// padded with PAD to max_seq_len. Returns input_ids and attention_mask.
struct EncodedInput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    int seq_len;
};

EncodedInput encode(const Tokenizer& tok, const std::string& context,
                    const std::string& reading, int max_seq_len,
                    int max_context) {
    auto ctx_chars = split_utf8_chars(context);
    if (static_cast<int>(ctx_chars.size()) > max_context) {
        ctx_chars.erase(
            ctx_chars.begin(),
            ctx_chars.begin() + (ctx_chars.size() - max_context));
    }
    auto read_chars = split_utf8_chars(reading);

    std::vector<int64_t> ids;
    ids.reserve(max_seq_len);
    ids.push_back(tok.cls_id);
    for (auto& c : ctx_chars) ids.push_back(tok.encode_char(c));
    ids.push_back(tok.sep_id);
    for (auto& c : read_chars) ids.push_back(tok.encode_char(c));
    if (static_cast<int>(ids.size()) > max_seq_len) {
        ids.resize(max_seq_len);
    }
    int actual = static_cast<int>(ids.size());
    std::vector<int64_t> mask(max_seq_len, 0);
    for (int i = 0; i < actual; ++i) mask[i] = 1;
    while (static_cast<int>(ids.size()) < max_seq_len) ids.push_back(tok.pad_id);

    return {std::move(ids), std::move(mask), actual};
}

struct Args {
    std::string onnx_path;
    std::string tokenizer_path;
    std::string lm_path;
    std::string dict_path;
    float alpha = 0.8f;
    float beta = 1.0f;
    int beam = 4;
    int seq_len = 128;
    int max_context = 32;
    bool help = false;
};

// Longest-prefix kana dictionary, multi-surface. A single reading can map
// to any number of surfaces (homophones). Loaded from a TSV where every
// line is ``kana<TAB>surface`` — to register multiple homophones, just
// repeat the kana on separate lines. That format is what mozc's open
// dictionary already produces, so an import script can dump it directly.
//
// With mozc-ut loaded the entry count approaches 1.5M, so the lookup
// path uses a hash-map keyed by the exact kana bytes plus a per-byte-
// length bucket, turning match_prefix into O(max_len_bucket) hash
// lookups (~20 probes) regardless of total entry count.
struct FixedDict {
    struct Entry {
        std::string kana;
        std::vector<std::string> surfaces;
    };

    std::vector<Entry> entries;
    // kana (byte string) -> index into entries. Used at load time to
    // merge homophones and at lookup time to check whether a given
    // prefix is a key.
    std::unordered_map<std::string, size_t> by_kana;
    // Sorted distinct byte-lengths of all keys, descending — used by
    // match_prefix to probe lengths from longest to shortest.
    std::vector<size_t> lengths_desc;

    size_t total_surfaces() const {
        size_t n = 0;
        for (const auto& e : entries) n += e.surfaces.size();
        return n;
    }

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;

        std::string line;
        line.reserve(256);
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            if (tab == 0 || tab + 1 >= line.size()) continue;

            std::string kana = line.substr(0, tab);
            std::string surface = line.substr(tab + 1);

            auto it = by_kana.find(kana);
            if (it == by_kana.end()) {
                Entry e;
                e.kana = kana;
                e.surfaces.push_back(std::move(surface));
                by_kana.emplace(std::move(kana), entries.size());
                entries.push_back(std::move(e));
            } else {
                auto& existing = entries[it->second].surfaces;
                bool dup = false;
                for (const auto& s : existing) if (s == surface) { dup = true; break; }
                if (!dup) existing.push_back(std::move(surface));
            }
        }

        // Rebuild descending-length bucket index.
        std::vector<size_t> uniq_lengths;
        uniq_lengths.reserve(entries.size());
        for (const auto& e : entries) uniq_lengths.push_back(e.kana.size());
        std::sort(uniq_lengths.begin(), uniq_lengths.end(), std::greater<size_t>());
        uniq_lengths.erase(std::unique(uniq_lengths.begin(), uniq_lengths.end()),
                           uniq_lengths.end());
        lengths_desc = std::move(uniq_lengths);
        return true;
    }

    std::pair<const Entry*, size_t> match_prefix(const std::string& kana, size_t at) const {
        const size_t remain = kana.size() - at;
        for (size_t L : lengths_desc) {
            if (L == 0 || L > remain) continue;
            auto it = by_kana.find(kana.substr(at, L));
            if (it != by_kana.end()) {
                return {&entries[it->second], L};
            }
        }
        return {nullptr, 0};
    }
};

// Split a normalized input line into alternating segments of kana and
// non-kana. Each segment carries an `is_kana` flag; non-kana segments
// pass through to the output unchanged. Kana segments go through the
// dict+CTC pipeline.
struct Segment {
    std::string text;
    bool is_kana;
};

std::vector<Segment> segment_by_kana(const std::string& s) {
    std::vector<Segment> out;
    size_t i = 0;
    while (i < s.size()) {
        auto [cp, n] = utf8_decode_one(s, i);
        if (cp < 0 || n == 0) { ++i; continue; }
        bool is_k = is_kana_codepoint(cp);
        if (out.empty() || out.back().is_kana != is_k) {
            out.push_back({std::string{}, is_k});
        }
        out.back().text.append(s, i, n);
        i += n;
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) return {};
            return argv[++i];
        };
        if (s == "--onnx") a.onnx_path = next();
        else if (s == "--tokenizer") a.tokenizer_path = next();
        else if (s == "--lm") a.lm_path = next();
        else if (s == "--dict") a.dict_path = next();
        else if (s == "--alpha") a.alpha = std::stof(next());
        else if (s == "--beta") a.beta = std::stof(next());
        else if (s == "--beam") a.beam = std::stoi(next());
        else if (s == "--seq-len") a.seq_len = std::stoi(next());
        else if (s == "--help" || s == "-h") a.help = true;
    }
    return a;
}

void usage() {
    std::cout <<
        "Usage: interactive_ctc.exe --onnx MODEL.onnx [options]\n"
        "  --tokenizer FILE    tokenizer sidecar .json (default: inferred from --onnx)\n"
        "  --lm FILE           KenLM .arpa or .bin (optional)\n"
        "  --dict FILE         fixed kana->surface TSV (optional)\n"
        "  --alpha F           LM weight (default 0.8)\n"
        "  --beta F            length penalty (default 1.0)\n"
        "  --beam N            beam width (default 4)\n"
        "  --seq-len N         model seq len (default 128, must match export)\n";
}

std::string infer_tokenizer_path(const std::string& onnx) {
    auto base = onnx;
    auto p = base.find_last_of('.');
    if (p != std::string::npos) base = base.substr(0, p);
    return base + ".tokenizer.json";
}

} // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    Args args = parse_args(argc, argv);
    if (args.help || args.onnx_path.empty()) { usage(); return args.help ? 0 : 1; }
    if (args.tokenizer_path.empty())
        args.tokenizer_path = infer_tokenizer_path(args.onnx_path);

    Tokenizer tokenizer;
    if (!tokenizer.load(args.tokenizer_path)) {
        std::cerr << "failed to load tokenizer from " << args.tokenizer_path << "\n";
        return 1;
    }
    std::cout << "tokenizer: " << tokenizer.id_to_token.size()
              << " tokens, blank=" << tokenizer.blank_id << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "interactive_ctc");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::wstring wpath(args.onnx_path.begin(), args.onnx_path.end());
    Ort::Session session(env, wpath.c_str(), opts);
    std::cout << "ONNX: " << args.onnx_path << "\n";

    std::unique_ptr<newime::LMScorer> lm_scorer;
#ifdef NEWIME_ENABLE_KENLM
    if (!args.lm_path.empty()) {
        try {
            lm_scorer = std::make_unique<newime::KenLMCharScorer>(args.lm_path);
            std::cout << "KenLM: " << args.lm_path
                      << " (alpha=" << args.alpha << ", beta=" << args.beta << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "KenLM load failed: " << e.what() << "\n";
            return 1;
        }
    }
#else
    if (!args.lm_path.empty()) {
        std::cerr << "NEWIME_ENABLE_KENLM not set at build time; --lm ignored\n";
    }
#endif

    newime::CTCDecoder decoder(tokenizer.id_to_token, tokenizer.blank_id);

    FixedDict dict;
    if (!args.dict_path.empty()) {
        // Accept a comma-separated list. Earlier paths win on ordering
        // when homophones overlap (user dict before mozc before mozc-ut),
        // which determines the tie-break if LM scores are equal.
        std::vector<std::string> paths;
        {
            std::string cur;
            for (char c : args.dict_path) {
                if (c == ',') { if (!cur.empty()) paths.push_back(std::move(cur)); cur.clear(); }
                else cur += c;
            }
            if (!cur.empty()) paths.push_back(std::move(cur));
        }
        for (const auto& p : paths) {
            if (!dict.load(p)) {
                std::cerr << "dict load failed: " << p << "\n";
            } else {
                std::cout << "dict: " << p << "\n";
            }
        }
        std::cout << "  total readings: " << dict.entries.size()
                  << " / surfaces: " << dict.total_surfaces() << "\n";
    }

    constexpr int TOP_K = 3;

    // A "chunk" is a contiguous piece of a kana run that becomes a single
    // decision point in the final top-N composition: either a fixed dict
    // match (one alternative) or a CTC decode (up to TOP_K alternatives).
    struct Chunk {
        // Parallel arrays: alt_texts[i] has score alt_scores[i]. For dict
        // hits both vectors hold one entry with score 0. For CTC chunks the
        // scores come from the fused CTC+LM+length ranking.
        std::vector<std::string> alt_texts;
        std::vector<float> alt_scores;
    };

    // One kana run -> list of Chunks. Each chunk is either a dict hit
    // (single-alt) or a CTC decode (TOP_K alts). The caller composes final
    // top-N strings by cross-product over chunks.
    // Contextual LM score = natlog P(left_ctx + surface) - natlog P(left_ctx).
    // The subtraction leaves only the marginal cost of the new surface
    // under KenLM, which is what we want to compare across candidates at
    // the same chunk position. Cached per (ctx, surface).
    auto ctx_lm_score = [&](const std::string& left_ctx, const std::string& surface) -> float {
        if (!lm_scorer) return 0.0f;
        float a = lm_scorer->score(std::string_view(left_ctx + surface));
        float b = lm_scorer->score(std::string_view(left_ctx));
        return a - b;
    };

    // Count UTF-8 codepoints for the length bonus — matches what
    // CTCDecoder does internally.
    auto glyph_count = [](const std::string& s) -> int {
        int n = 0;
        for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n;
        return n;
    };

    // A CTC chunk now unions CTC beam top-K with dict homophones. The
    // dict alts get CTC score 0 (CTC neither prefers nor rejects them —
    // it simply didn't enumerate them). Ranking is a fused score:
    //     CTC_score + lm_alpha_ctx * logp_lm(ctx + alt) + lm_beta * glyphs
    // For the non-CTC (dict) alts, CTC_score=0 means ranking is pure LM.
    // lm_alpha_ctx is read from the existing --alpha so tuning propagates.
    auto chunk_kana_run = [&](const std::string& kana_in,
                              const std::string& left_ctx_in,
                              double& ms_onnx, double& ms_decode) -> std::vector<Chunk> {
        std::string kana = kana_in;
        std::string left_ctx = left_ctx_in;
        std::vector<Chunk> chunks;
        while (!kana.empty()) {
            // Dict matches at the current position drive *two* things: a
            // pinned dict chunk (whole key) if no CTC-worthy tail follows,
            // OR a set of dict alternatives added to the CTC chunk that
            // covers the same kana span.
            const FixedDict::Entry* dict_hit = nullptr;
            size_t dict_hit_len = 0;
            if (!dict.entries.empty()) {
                auto [e, n] = dict.match_prefix(kana, 0);
                if (n > 0) { dict_hit = e; dict_hit_len = n; }
            }

            if (dict_hit && dict_hit->surfaces.size() >= 1) {
                Chunk c;
                // Every homophone becomes an alt. LM scored in context.
                float best_score = -1e30f;
                for (const auto& surf : dict_hit->surfaces) {
                    float s = args.alpha * ctx_lm_score(left_ctx, surf)
                            + args.beta  * glyph_count(surf);
                    c.alt_texts.push_back(surf);
                    c.alt_scores.push_back(s);
                    if (s > best_score) best_score = s;
                }
                // Pick best homophone for left context advancement.
                size_t best_idx = 0;
                for (size_t i = 0; i < c.alt_scores.size(); ++i)
                    if (c.alt_scores[i] > c.alt_scores[best_idx]) best_idx = i;
                left_ctx += c.alt_texts[best_idx];
                chunks.push_back(std::move(c));
                kana.erase(0, dict_hit_len);
                continue;
            }

            auto [cp, cp_n] = utf8_decode_one(kana, 0);
            if (cp_n == 0) break;
            size_t take = cp_n;
            while (take < kana.size()) {
                auto [e2, n2] = dict.entries.empty()
                    ? std::make_pair(static_cast<const FixedDict::Entry*>(nullptr), size_t{0})
                    : dict.match_prefix(kana, take);
                if (n2 > 0) break;
                auto [cp2, cp2_n] = utf8_decode_one(kana, take);
                if (cp2_n == 0) break;
                take += cp2_n;
            }

            std::string chunk_kana = kana.substr(0, take);
            auto enc = encode(tokenizer, left_ctx, chunk_kana, args.seq_len, args.max_context);
            std::array<int64_t, 2> shape{1, args.seq_len};
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, enc.input_ids.data(), enc.input_ids.size(),
                shape.data(), shape.size());
            Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, enc.attention_mask.data(), enc.attention_mask.size(),
                shape.data(), shape.size());
            const char* input_names[] = {"input_ids", "attention_mask"};
            const char* output_names[] = {"logits"};
            std::array<Ort::Value, 2> inputs{std::move(ids_tensor), std::move(mask_tensor)};
            auto t_a = std::chrono::steady_clock::now();
            auto outs = session.Run(Ort::RunOptions{nullptr}, input_names,
                                    inputs.data(), inputs.size(),
                                    output_names, 1);
            auto t_b = std::chrono::steady_clock::now();
            float* logits = outs[0].GetTensorMutableData<float>();
            auto info = outs[0].GetTensorTypeAndShapeInfo();
            int64_t V = info.GetShape()[2];
            std::vector<float> trimmed(logits, logits + enc.seq_len * V);
            auto candidates = decoder.beam_search(
                trimmed, enc.seq_len, static_cast<int>(V),
                args.beam, lm_scorer.get(), args.alpha, args.beta);
            auto t_c = std::chrono::steady_clock::now();
            ms_onnx += std::chrono::duration<double, std::milli>(t_b - t_a).count();
            ms_decode += std::chrono::duration<double, std::milli>(t_c - t_b).count();

            Chunk c;
            int take_k = std::min<int>(TOP_K, static_cast<int>(candidates.size()));
            for (int i = 0; i < take_k; ++i) {
                c.alt_texts.push_back(candidates[i].text);
                c.alt_scores.push_back(candidates[i].score);
            }
            if (c.alt_texts.empty()) {
                c.alt_texts.push_back("");
                c.alt_scores.push_back(0.0f);
            }
            left_ctx += c.alt_texts[0];
            chunks.push_back(std::move(c));
            kana.erase(0, take);
        }
        return chunks;
    };

    std::cout << "\nType kana to convert (empty line to quit).\n"
              << "Use 'ctx <text>' to set left context.\n\n";

    std::string context;
    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line.empty()) break;
        if (line.rfind("ctx ", 0) == 0) {
            context = normalize_input_punct(line.substr(4));
            std::cout << "context set (" << context.size() << " bytes)\n";
            continue;
        }

        auto t_start = std::chrono::steady_clock::now();
        line = normalize_input_punct(line);

        // Segment into alternating kana / non-kana runs. Non-kana (digits,
        // punctuation, latin, already-mixed text) passes through verbatim;
        // kana runs produce a list of Chunks each carrying up to TOP_K
        // alternatives. A final top-N is assembled by cross-product.
        auto segs = segment_by_kana(line);
        std::vector<Chunk> all_chunks;
        std::string left_ctx = context;
        double ms_onnx = 0.0, ms_decode = 0.0;
        for (auto& seg : segs) {
            if (!seg.is_kana) {
                Chunk c;
                c.alt_texts.push_back(seg.text);
                c.alt_scores.push_back(0.0f);
                all_chunks.push_back(std::move(c));
                left_ctx += seg.text;
                continue;
            }
            auto run_chunks = chunk_kana_run(seg.text, left_ctx, ms_onnx, ms_decode);
            for (auto& c : run_chunks) {
                left_ctx += c.alt_texts[0];
                all_chunks.push_back(std::move(c));
            }
        }
        auto t_end = std::chrono::steady_clock::now();

        // Cross-product over chunk alternatives, scored by sum of per-chunk
        // scores. Beam-prune during assembly so we never materialize more
        // than BEAM_PRUNE partial strings. For up to ~6 chunks with TOP_K=3
        // the full product is manageable, but the prune keeps it cheap
        // when the user pastes a long mixed input.
        constexpr int BEAM_PRUNE = 16;
        struct PartialCand { std::string text; float score; };
        std::vector<PartialCand> partials{{"", 0.0f}};
        for (const auto& ch : all_chunks) {
            std::vector<PartialCand> next;
            next.reserve(partials.size() * ch.alt_texts.size());
            for (const auto& p : partials) {
                for (size_t i = 0; i < ch.alt_texts.size(); ++i) {
                    next.push_back({p.text + ch.alt_texts[i], p.score + ch.alt_scores[i]});
                }
            }
            std::sort(next.begin(), next.end(),
                      [](const PartialCand& a, const PartialCand& b) {
                          return a.score > b.score;
                      });
            if (static_cast<int>(next.size()) > BEAM_PRUNE) next.resize(BEAM_PRUNE);
            partials = std::move(next);
        }

        std::vector<newime::DecodedCandidate> candidates;
        for (const auto& p : partials) {
            bool dup = false;
            for (const auto& existing : candidates) {
                if (existing.text == p.text) { dup = true; break; }
            }
            if (!dup) candidates.push_back({p.text, p.score});
        }

        // Training corpus underrepresents ？ ！ at sentence endings — Wiki
        // uses 。 overwhelmingly. The CTC path + LM prior converge onto
        // blank-at-？ for those inputs. If the user typed a trailing
        // sentence-final marker, echo it back onto the top candidates so
        // the raw input intent is preserved.
        auto ends_with = [](const std::string& s, const std::string& suf) {
            return s.size() >= suf.size()
                && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
        };
        for (const auto& mark : {std::string("\xEF\xBC\x9F"),   // ？
                                  std::string("\xEF\xBC\x81"),  // ！
                                  std::string("\xE3\x80\x82"),  // 。
                                  std::string("\xE3\x80\x81")}) // 、
        {
            if (ends_with(line, mark)) {
                for (auto& c : candidates) {
                    if (!ends_with(c.text, mark) && !c.text.empty()) {
                        c.text += mark;
                    }
                }
                break;
            }
        }

        double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        int shown = std::min<size_t>(3, candidates.size());
        for (int k = 0; k < shown; ++k) {
            std::cout << "  [" << k + 1 << "] "
                      << candidates[k].text
                      << "  (" << candidates[k].score << ")\n";
        }
        if (candidates.empty()) std::cout << "  (no candidates)\n";
        std::cout << "  [time] total=" << ms_total
                  << "ms (onnx=" << ms_onnx
                  << "ms, decode=" << ms_decode << "ms)\n";
    }

    return 0;
}
