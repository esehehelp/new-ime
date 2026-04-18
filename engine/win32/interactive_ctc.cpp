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
    float alpha = 0.8f;
    float beta = 1.0f;
    int beam = 4;
    int seq_len = 128;
    int max_context = 32;
    bool help = false;
};

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

    std::cout << "\nType kana to convert (empty line to quit).\n"
              << "Use 'ctx <text>' to set left context.\n\n";

    std::string context;
    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line.empty()) break;
        if (line.rfind("ctx ", 0) == 0) {
            context = line.substr(4);
            std::cout << "context set (" << context.size() << " bytes)\n";
            continue;
        }

        auto enc = encode(tokenizer, context, line, args.seq_len, args.max_context);
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
        auto outs = session.Run(Ort::RunOptions{nullptr}, input_names,
                                inputs.data(), inputs.size(),
                                output_names, 1);

        float* logits = outs[0].GetTensorMutableData<float>();
        auto info = outs[0].GetTensorTypeAndShapeInfo();
        auto out_shape = info.GetShape();
        int64_t T = out_shape[1];
        int64_t V = out_shape[2];

        // Trim logits to the actual (non-padded) prefix length.
        std::vector<float> trimmed(logits, logits + enc.seq_len * V);

        auto candidates = decoder.beam_search(
            trimmed, enc.seq_len, static_cast<int>(V),
            args.beam, lm_scorer.get(), args.alpha, args.beta);

        int shown = std::min<size_t>(3, candidates.size());
        for (int k = 0; k < shown; ++k) {
            std::cout << "  [" << k + 1 << "] "
                      << candidates[k].text
                      << "  (" << candidates[k].score << ")\n";
        }
        if (candidates.empty()) std::cout << "  (no candidates)\n";
    }

    return 0;
}
