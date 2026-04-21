/**
 * new-ime Windows engine DLL — TSF-oriented stateful CTC-NAT backend
 *
 * Exposes a mutable IME session API over a single global instance so TSF glue
 * can drive composition, candidate selection, and commit decisions.
 *
 * Current backend:
 *   - ONNX Runtime CTC-NAT inference
 *   - C++ prefix beam search via ctc_decoder.cpp
 *   - optional KenLM shallow fusion
 */

#include "ctc_decoder.h"
#include "lm_scorer.h"
#ifdef NEWIME_ENABLE_KENLM
#include "lm_scorer_kenlm.h"
#endif

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <windows.h>

namespace {

static std::string join_path(const std::string& a, const std::string& b);
static std::string module_dir();
static void debug_log(const std::string& line);
static bool ensure_runtime_loaded_impl_locked();
static bool ensure_runtime_loaded_locked();

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    int blank_id = 4;
    int cls_id = 2;
    int sep_id = 3;
    int pad_id = 0;
    int unk_id = 1;

    static bool hex_value(char c, unsigned char& out) {
        if (c >= '0' && c <= '9') {
            out = static_cast<unsigned char>(c - '0');
            return true;
        }
        if (c >= 'a' && c <= 'f') {
            out = static_cast<unsigned char>(c - 'a' + 10);
            return true;
        }
        if (c >= 'A' && c <= 'F') {
            out = static_cast<unsigned char>(c - 'A' + 10);
            return true;
        }
        return false;
    }

    static std::string decode_hex_bytes(const std::string& hex) {
        if ((hex.size() % 2) != 0) return {};
        std::string out;
        out.reserve(hex.size() / 2);
        for (size_t i = 0; i < hex.size(); i += 2) {
            unsigned char hi = 0;
            unsigned char lo = 0;
            if (!hex_value(hex[i], hi) || !hex_value(hex[i + 1], lo)) return {};
            out.push_back(static_cast<char>((hi << 4) | lo));
        }
        return out;
    }

    void finalize_special_ids() {
        auto look = [&](const char* name, int& out) {
            auto it = token_to_id.find(name);
            if (it != token_to_id.end()) out = it->second;
        };
        look("[PAD]", pad_id);
        look("[UNK]", unk_id);
        look("[SEP]", sep_id);
        look("[CLS]", cls_id);
        look("[BLANK]", blank_id);
    }

    bool load_vocab_hex_tsv(const std::string& path) {
        debug_log("Tokenizer::load_vocab_hex_tsv begin path=" + path);
        std::ifstream f(path);
        if (!f) {
            debug_log("Tokenizer::load_vocab_hex_tsv open failed");
            return false;
        }

        token_to_id.clear();
        id_to_token.clear();
        std::string line;
        int max_id = -1;
        size_t parsed = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            int id = std::atoi(line.substr(0, tab).c_str());
            std::string tok = decode_hex_bytes(line.substr(tab + 1));
            token_to_id[tok] = id;
            if (id > max_id) max_id = id;
            ++parsed;
        }
        if (max_id < 0) {
            debug_log("Tokenizer::load_vocab_hex_tsv no tokens parsed");
            return false;
        }
        id_to_token.assign(static_cast<size_t>(max_id) + 1, std::string{});
        for (const auto& [tok, id] : token_to_id) {
            if (id >= 0) id_to_token[static_cast<size_t>(id)] = tok;
        }
        finalize_special_ids();
        debug_log("Tokenizer::load_vocab_hex_tsv success parsed=" + std::to_string(parsed));
        return !id_to_token.empty();
    }

    static std::string parse_json_string(const std::string& j, size_t& i) {
        ++i;
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
        if (i < j.size()) ++i;
        return out;
    }

    bool load(const std::string& path) {
        const std::string tsv_path = path + ".vocab.hex.tsv";
        if (load_vocab_hex_tsv(tsv_path)) {
            return true;
        }
        debug_log("Tokenizer::load sidecar missing path=" + tsv_path);
        return false;
    }

    int encode_char(const std::string& utf8_char) const {
        auto it = token_to_id.find(utf8_char);
        return it == token_to_id.end() ? unk_id : it->second;
    }
};

struct EncodedInput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    int seq_len = 0;
};

struct CandidateResult {
    std::string text;
    int correspondingCount = 0;
    float score = 0.0f;
};

static constexpr int kDefaultSeqLen = 128;
static constexpr int kDefaultMaxContext = 32;
static constexpr int kDefaultBeamWidth = 5;
static constexpr float kDefaultLmAlpha = 0.4f;
static constexpr float kDefaultLmBeta = 0.6f;
static constexpr int kDefaultNumCandidates = 5;

static Ort::Env* g_env = nullptr;
static Ort::Session* g_session = nullptr;
static Tokenizer g_tokenizer;
static std::unique_ptr<newime::CTCDecoder> g_decoder;
static std::unique_ptr<newime::LMScorer> g_lm_scorer;
static std::string g_onnxPath;
static std::string g_tokenizerPath;
static std::string g_lmPath;
static std::string g_modelRoot;
static int g_seqLen = kDefaultSeqLen;
static int g_maxContext = kDefaultMaxContext;
static int g_beamWidth = kDefaultBeamWidth;
static float g_lmAlpha = kDefaultLmAlpha;
static float g_lmBeta = kDefaultLmBeta;
static bool g_initialized = false;
static bool g_runtimeReady = false;

static std::string g_composingText;
static std::string g_contextText;
static std::string g_lastCommittedText;
static size_t g_cursorBytePos = 0;
static int g_selectedCandidateIndex = 0;
static bool g_candidatesDirty = true;
static std::vector<CandidateResult> g_candidates;
static std::mutex g_mutex;

static bool is_packaged_process() {
    using FnGetCurrentPackageFullName = LONG (WINAPI*)(UINT32*, PWSTR);
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    if (!kernel) return false;
    auto fn = reinterpret_cast<FnGetCurrentPackageFullName>(
        GetProcAddress(kernel, "GetCurrentPackageFullName"));
    if (!fn) return false;
    UINT32 length = 0;
    LONG rc = fn(&length, nullptr);
    return rc == ERROR_INSUFFICIENT_BUFFER;
}

static void debug_log(const std::string& line) {
    std::string path = join_path(module_dir(), "new-ime-engine.log");
    HANDLE file = CreateFileA(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    std::string out = line + "\r\n";
    DWORD written = 0;
    WriteFile(file, out.data(), static_cast<DWORD>(out.size()), &written, nullptr);
    CloseHandle(file);
}

static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static int count_utf8_chars(const std::string& s) {
    int count = 0;
    for (size_t i = 0; i < s.size();) {
        i += utf8_char_len(static_cast<unsigned char>(s[i]));
        ++count;
    }
    return count;
}

static std::vector<std::string> split_utf8_chars(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        int n = utf8_char_len(static_cast<unsigned char>(s[i]));
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}

static size_t clamp_utf8_boundary_left(const std::string& s, size_t pos) {
    if (pos > s.size()) pos = s.size();
    while (pos > 0 && pos < s.size() &&
           (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return pos;
}

static size_t utf8_advance_chars(const std::string& s, size_t byte_pos, int char_delta) {
    size_t pos = clamp_utf8_boundary_left(s, byte_pos);
    if (char_delta > 0) {
        for (int i = 0; i < char_delta && pos < s.size(); ++i) {
            pos += utf8_char_len(static_cast<unsigned char>(s[pos]));
        }
    } else if (char_delta < 0) {
        for (int i = 0; i < -char_delta && pos > 0; ++i) {
            --pos;
            while (pos > 0 && (static_cast<unsigned char>(s[pos]) & 0xC0) == 0x80) {
                --pos;
            }
        }
    }
    return pos;
}

static int utf8_char_pos(const std::string& s, size_t byte_pos) {
    byte_pos = clamp_utf8_boundary_left(s, byte_pos);
    int chars = 0;
    for (size_t i = 0; i < byte_pos && i < s.size();) {
        i += utf8_char_len(static_cast<unsigned char>(s[i]));
        ++chars;
    }
    return chars;
}

static std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

static std::string dirname_of(const std::string& path) {
    size_t slash = path.find_last_of("\\/");
    return slash == std::string::npos ? std::string{} : path.substr(0, slash);
}

static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '\\' || last == '/') return a + b;
    return a + "\\" + b;
}

static std::string module_dir() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&module_dir), &module)) {
        return {};
    }
    char buf[MAX_PATH];
    DWORD len = GetModuleFileNameA(module, buf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return {};
    return dirname_of(std::string(buf, len));
}

static std::string current_dir() {
    char buf[MAX_PATH];
    DWORD len = GetCurrentDirectoryA(MAX_PATH, buf);
    if (len == 0 || len >= MAX_PATH) return {};
    return std::string(buf, len);
}

static bool path_exists(const std::string& path) {
    if (path.empty()) return false;
    DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES;
}

static std::string full_path(const std::string& path) {
    if (path.empty()) return {};
    DWORD needed = GetFullPathNameA(path.c_str(), 0, nullptr, nullptr);
    if (needed == 0) return path;
    std::string out(static_cast<size_t>(needed), '\0');
    DWORD written = GetFullPathNameA(path.c_str(), needed, out.data(), nullptr);
    if (written == 0 || written >= needed) return path;
    out.resize(static_cast<size_t>(written));
    return out;
}

static bool is_absolute_path(const std::string& path) {
    if (path.size() >= 2 && path[1] == ':') return true;
    if (path.size() >= 2 &&
        ((path[0] == '\\' && path[1] == '\\') ||
         (path[0] == '/' && path[1] == '/'))) {
        return true;
    }
    return false;
}

static std::string normalize_path(const std::string& input) {
    if (input.empty()) return input;
    if (is_absolute_path(input)) return input;

    std::vector<std::string> bases;
    auto cwd = current_dir();
    if (!cwd.empty()) bases.push_back(cwd);
    auto mod = module_dir();
    if (!mod.empty()) bases.push_back(mod);

    for (const auto& base : bases) {
        std::string candidate = full_path(join_path(base, input));
        if (path_exists(candidate)) {
            return candidate;
        }
    }

    if (!bases.empty()) {
        return full_path(join_path(bases.front(), input));
    }
    return full_path(input);
}

static std::string infer_tokenizer_path(const std::string& onnx_path) {
    auto pos = onnx_path.find_last_of('.');
    if (pos == std::string::npos) return onnx_path + ".tokenizer.json";
    return onnx_path.substr(0, pos) + ".tokenizer.json";
}

static std::string infer_lm_path(const std::string& model_root) {
    return join_path(join_path(model_root, "kenlm"), "kenlm_general_train_4gram.bin");
}

static std::string infer_onnx_path(const std::string& model_root) {
    return join_path(join_path(model_root, "onnx"), "ctc-nat-30m-student-step160000.int8.onnx");
}

static std::string infer_tokenizer_default(const std::string& model_root) {
    return join_path(join_path(model_root, "onnx"), "ctc-nat-30m-student-step160000.fp32.tokenizer.json");
}

static EncodedInput encode(const Tokenizer& tok,
                           const std::string& context,
                           const std::string& reading,
                           int max_seq_len,
                           int max_context) {
    auto ctx_chars = split_utf8_chars(context);
    if (static_cast<int>(ctx_chars.size()) > max_context) {
        ctx_chars.erase(ctx_chars.begin(),
                        ctx_chars.begin() + (ctx_chars.size() - max_context));
    }
    auto read_chars = split_utf8_chars(reading);

    std::vector<int64_t> ids;
    ids.reserve(static_cast<size_t>(max_seq_len));
    ids.push_back(tok.cls_id);
    for (const auto& c : ctx_chars) ids.push_back(tok.encode_char(c));
    ids.push_back(tok.sep_id);
    for (const auto& c : read_chars) ids.push_back(tok.encode_char(c));
    if (static_cast<int>(ids.size()) > max_seq_len) ids.resize(static_cast<size_t>(max_seq_len));
    int actual = static_cast<int>(ids.size());
    std::vector<int64_t> mask(static_cast<size_t>(max_seq_len), 0);
    for (int i = 0; i < actual; ++i) mask[static_cast<size_t>(i)] = 1;
    while (static_cast<int>(ids.size()) < max_seq_len) ids.push_back(tok.pad_id);

    return {std::move(ids), std::move(mask), actual};
}

static bool run_ctc_model(const std::string& reading,
                          const std::string& context,
                          std::vector<CandidateResult>& out_candidates) {
    out_candidates.clear();
    if (reading.empty()) return true;
    if ((!g_session || !g_decoder) && !ensure_runtime_loaded_locked()) {
        out_candidates.push_back({reading, count_utf8_chars(reading), 0.0f});
        return false;
    }

    auto enc = encode(g_tokenizer, context, reading, g_seqLen, g_maxContext);
    std::array<int64_t, 2> shape{1, g_seqLen};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, enc.input_ids.data(), enc.input_ids.size(), shape.data(), shape.size());
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        mem_info, enc.attention_mask.data(), enc.attention_mask.size(), shape.data(), shape.size());
    const char* input_names[] = {"input_ids", "attention_mask"};
    const char* output_names[] = {"logits"};
    std::array<Ort::Value, 2> inputs{std::move(ids_tensor), std::move(mask_tensor)};

    auto outs = g_session->Run(Ort::RunOptions{nullptr}, input_names,
                               inputs.data(), inputs.size(), output_names, 1);
    const float* logits = outs[0].GetTensorData<float>();
    auto info = outs[0].GetTensorTypeAndShapeInfo();
    int64_t vocab_size = info.GetShape()[2];
    std::vector<float> trimmed(logits, logits + static_cast<size_t>(enc.seq_len) * static_cast<size_t>(vocab_size));

    auto decoded = g_decoder->beam_search(
        trimmed,
        enc.seq_len,
        static_cast<int>(vocab_size),
        g_beamWidth,
        g_lm_scorer.get(),
        g_lmAlpha,
        g_lmBeta);

    int take = std::min<int>(kDefaultNumCandidates, static_cast<int>(decoded.size()));
    for (int i = 0; i < take; ++i) {
        out_candidates.push_back({decoded[static_cast<size_t>(i)].text,
                                  count_utf8_chars(reading),
                                  decoded[static_cast<size_t>(i)].score});
    }
    if (out_candidates.empty()) {
        out_candidates.push_back({reading, count_utf8_chars(reading), 0.0f});
    }
    return true;
}

static void refresh_candidates_locked() {
    if (!g_candidatesDirty) return;
    run_ctc_model(g_composingText, g_contextText, g_candidates);
    if (g_candidates.empty()) {
        g_selectedCandidateIndex = 0;
    } else if (g_selectedCandidateIndex < 0 ||
               g_selectedCandidateIndex >= static_cast<int>(g_candidates.size())) {
        g_selectedCandidateIndex = 0;
    }
    g_candidatesDirty = false;
}

static void replace_composition_locked(std::string text, size_t cursor_byte_pos) {
    g_composingText = std::move(text);
    g_cursorBytePos = clamp_utf8_boundary_left(g_composingText, cursor_byte_pos);
    g_candidatesDirty = true;
    g_candidates.clear();
    g_selectedCandidateIndex = 0;
}

static void clear_composition_locked() {
    g_composingText.clear();
    g_cursorBytePos = 0;
    g_candidates.clear();
    g_selectedCandidateIndex = 0;
    g_candidatesDirty = true;
}

static std::string selected_candidate_text_locked() {
    refresh_candidates_locked();
    if (g_candidates.empty()) return g_composingText;
    return g_candidates[static_cast<size_t>(g_selectedCandidateIndex)].text;
}

static bool ensure_runtime_loaded_impl_locked() {
    if (g_runtimeReady && g_session && g_decoder) return true;
    if (is_packaged_process()) {
        debug_log("Runtime load skipped in packaged process");
        return false;
    }
    debug_log("Runtime load begin");
    if (!g_env) {
        debug_log("Runtime create env");
        g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "new-ime-ctc");
    }
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::wstring wonnx(g_onnxPath.begin(), g_onnxPath.end());
    debug_log("Runtime create session");
    delete g_session;
    g_session = new Ort::Session(*g_env, wonnx.c_str(), opts);
    debug_log("Runtime create decoder");
    g_decoder = std::make_unique<newime::CTCDecoder>(
        g_tokenizer.id_to_token, g_tokenizer.blank_id);
#ifdef NEWIME_ENABLE_KENLM
    if (!g_lmPath.empty() && std::ifstream(g_lmPath)) {
        debug_log("Runtime create kenlm");
        g_lm_scorer = std::make_unique<newime::KenLMCharScorer>(g_lmPath);
    } else {
        g_lm_scorer.reset();
    }
#else
    g_lm_scorer.reset();
#endif
    g_runtimeReady = true;
    debug_log("Runtime load success");
    return true;
}

static bool ensure_runtime_loaded_locked() {
    return ensure_runtime_loaded_impl_locked();
}

static bool initialize_backend_locked(const std::string& dictionaryPath,
                                      const char* memoryPath) {
    debug_log("Initialize begin dictionaryPath=" + dictionaryPath +
              " memoryPath=" + std::string(memoryPath ? memoryPath : ""));
    std::string normalized_input = normalize_path(dictionaryPath);
    if (!normalized_input.empty()) {
        if (normalized_input.size() >= 5 &&
            normalized_input.substr(normalized_input.size() - 5) == ".onnx") {
            g_onnxPath = normalized_input;
            g_modelRoot = dirname_of(dirname_of(normalized_input));
        } else {
            g_modelRoot = normalized_input;
            g_onnxPath = infer_onnx_path(g_modelRoot);
        }
    } else if (g_modelRoot.empty()) {
        g_modelRoot = normalize_path("models");
        g_onnxPath = infer_onnx_path(g_modelRoot);
    } else if (g_onnxPath.empty()) {
        g_onnxPath = infer_onnx_path(g_modelRoot);
    }

    if (g_modelRoot.empty()) {
        g_modelRoot = dirname_of(dirname_of(g_onnxPath));
    }

    g_tokenizerPath = normalize_path(infer_tokenizer_path(g_onnxPath));
    if (!std::ifstream(g_tokenizerPath)) {
        g_tokenizerPath = normalize_path(infer_tokenizer_default(g_modelRoot));
    }

    g_lmPath = memoryPath && std::strlen(memoryPath) > 0
        ? normalize_path(memoryPath)
        : normalize_path(infer_lm_path(g_modelRoot));

    debug_log("Initialize resolved onnx=" + g_onnxPath);
    debug_log("Initialize resolved tokenizer=" + g_tokenizerPath);
    debug_log("Initialize resolved lm=" + g_lmPath);

    if (!g_tokenizer.load(g_tokenizerPath)) {
        debug_log("Initialize tokenizer load failed");
        return false;
    }

    (void)memoryPath;
    g_runtimeReady = false;
    g_decoder.reset();
    g_lm_scorer.reset();
    delete g_session;
    g_session = nullptr;
    g_initialized = true;
    debug_log("Initialize success");
    return true;
}

} // namespace

extern "C" {
#define EXPORT __declspec(dllexport)

EXPORT void LoadConfig(const char* configPath) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (configPath && std::strlen(configPath) > 0) {
            g_modelRoot = configPath;
            g_onnxPath.clear();
        }
    } catch (...) {}
}

EXPORT void Initialize(const char* dictionaryPath, const char* memoryPath) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        const std::string path = dictionaryPath ? dictionaryPath : "";
        debug_log("Export Initialize");
        initialize_backend_locked(path, memoryPath);
        g_composingText.clear();
        g_contextText.clear();
        g_lastCommittedText.clear();
        g_cursorBytePos = 0;
        g_candidates.clear();
        g_selectedCandidateIndex = 0;
        g_candidatesDirty = true;
    } catch (...) {
        g_initialized = false;
    }
}

EXPORT int IsInitialized(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_initialized ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

EXPORT void Shutdown(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        delete g_session; g_session = nullptr;
        delete g_env; g_env = nullptr;
        g_decoder.reset();
        g_lm_scorer.reset();
        g_initialized = false;
        g_runtimeReady = false;
    } catch (...) {}
}

EXPORT void AppendText(const char* input) {
    if (!input) return;
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        g_composingText.insert(g_cursorBytePos, input);
        g_cursorBytePos += std::strlen(input);
        g_candidatesDirty = true;
        g_candidates.clear();
        g_selectedCandidateIndex = 0;
    } catch (...) {}
}

EXPORT void RemoveText(int count) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        for (int i = 0; i < count && g_cursorBytePos > 0; ++i) {
            size_t prev = utf8_advance_chars(g_composingText, g_cursorBytePos, -1);
            g_composingText.erase(prev, g_cursorBytePos - prev);
            g_cursorBytePos = prev;
        }
        g_candidatesDirty = true;
        g_candidates.clear();
        g_selectedCandidateIndex = 0;
    } catch (...) {}
}

EXPORT void RemoveTextRight(int count) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        for (int i = 0; i < count && g_cursorBytePos < g_composingText.size(); ++i) {
            size_t next = utf8_advance_chars(g_composingText, g_cursorBytePos, 1);
            g_composingText.erase(g_cursorBytePos, next - g_cursorBytePos);
        }
        g_candidatesDirty = true;
        g_candidates.clear();
        g_selectedCandidateIndex = 0;
    } catch (...) {}
}

EXPORT void MoveCursor(int offset) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        g_cursorBytePos = utf8_advance_chars(g_composingText, g_cursorBytePos, offset);
    } catch (...) {}
}

EXPORT void ClearText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        clear_composition_locked();
    } catch (...) {}
}

EXPORT const char* GetComposedText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return _strdup("");
        std::string text = selected_candidate_text_locked();
        return _strdup(text.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT const char* GetCandidates(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return _strdup("[]");
        refresh_candidates_locked();
        std::string json = "[";
        for (size_t i = 0; i < g_candidates.size(); ++i) {
            if (i > 0) json += ",";
            const auto& cand = g_candidates[i];
            json += "{\"text\":\"" + json_escape(cand.text) + "\"";
            json += ",\"correspondingCount\":" + std::to_string(cand.correspondingCount);
            json += ",\"score\":" + std::to_string(cand.score);
            json += ",\"selected\":" + std::string(static_cast<int>(i) == g_selectedCandidateIndex ? "true" : "false");
            json += "}";
        }
        json += "]";
        return _strdup(json.c_str());
    } catch (...) {
        return _strdup("[]");
    }
}

EXPORT int GetCandidateCount(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return 0;
        refresh_candidates_locked();
        return static_cast<int>(g_candidates.size());
    } catch (...) {
        return 0;
    }
}

EXPORT int GetSelectedCandidateIndex(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return -1;
        refresh_candidates_locked();
        return g_candidates.empty() ? -1 : g_selectedCandidateIndex;
    } catch (...) {
        return -1;
    }
}

EXPORT void SetSelectedCandidateIndex(int index) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        refresh_candidates_locked();
        if (index >= 0 && index < static_cast<int>(g_candidates.size())) {
            g_selectedCandidateIndex = index;
        }
    } catch (...) {}
}

EXPORT void SelectCandidate(int index) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        refresh_candidates_locked();
        if (index >= 0 && index < static_cast<int>(g_candidates.size())) {
            g_selectedCandidateIndex = index;
            g_lastCommittedText = g_candidates[static_cast<size_t>(index)].text;
            if (!g_lastCommittedText.empty()) g_contextText = g_lastCommittedText;
            clear_composition_locked();
        }
    } catch (...) {}
}

EXPORT void ShrinkText(void) {
    RemoveText(1);
}
EXPORT void ExpandText(void) {
}

EXPORT void SetContext(const char* t) {
    if (!t) return;
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_contextText = t;
        g_candidatesDirty = true;
    } catch (...) {}
}

EXPORT void SetZenzaiEnabled(bool) {
}
EXPORT void SetZenzaiInferenceLimit(int) {
}
EXPORT void FreeString(const char* s) {
    try {
        free(const_cast<char*>(s));
    } catch (...) {}
}

EXPORT const char* GetCompositionText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        return _strdup(g_composingText.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT int GetCompositionCursor(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        return utf8_char_pos(g_composingText, g_cursorBytePos);
    } catch (...) {
        return 0;
    }
}

EXPORT void SetCompositionText(const char* text, int cursorChars) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return;
        std::string next = text ? text : "";
        size_t cursor = utf8_advance_chars(next, 0, std::max(0, cursorChars));
        replace_composition_locked(std::move(next), cursor);
    } catch (...) {}
}

EXPORT const char* GetPreedit(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return _strdup("");
        debug_log("Export GetPreedit");
        std::string text = selected_candidate_text_locked();
        return _strdup(text.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT const char* CommitSelectedCandidate(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return _strdup("");
        debug_log("Export CommitSelectedCandidate");
        g_lastCommittedText = selected_candidate_text_locked();
        if (!g_lastCommittedText.empty()) g_contextText = g_lastCommittedText;
        clear_composition_locked();
        return _strdup(g_lastCommittedText.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT const char* CommitRawText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_initialized) return _strdup("");
        g_lastCommittedText = g_composingText;
        if (!g_lastCommittedText.empty()) g_contextText = g_lastCommittedText;
        clear_composition_locked();
        return _strdup(g_lastCommittedText.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT const char* GetLastCommittedText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        return _strdup(g_lastCommittedText.c_str());
    } catch (...) {
        return _strdup("");
    }
}

EXPORT void ClearLastCommittedText(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_lastCommittedText.clear();
    } catch (...) {}
}

EXPORT const char* GetStateJson(void) {
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_initialized) refresh_candidates_locked();
        std::string selected = g_candidates.empty()
            ? g_composingText
            : g_candidates[static_cast<size_t>(g_selectedCandidateIndex)].text;
        std::string json = "{";
        json += "\"initialized\":" + std::string(g_initialized ? "true" : "false") + ",";
        json += "\"onnxPath\":\"" + json_escape(g_onnxPath) + "\",";
        json += "\"tokenizerPath\":\"" + json_escape(g_tokenizerPath) + "\",";
        json += "\"lmPath\":\"" + json_escape(g_lmPath) + "\",";
        json += "\"composition\":\"" + json_escape(g_composingText) + "\",";
        json += "\"cursor\":" + std::to_string(utf8_char_pos(g_composingText, g_cursorBytePos)) + ",";
        json += "\"context\":\"" + json_escape(g_contextText) + "\",";
        json += "\"preedit\":\"" + json_escape(selected) + "\",";
        json += "\"selectedCandidateIndex\":" + std::to_string(g_candidates.empty() ? -1 : g_selectedCandidateIndex) + ",";
        json += "\"lastCommitted\":\"" + json_escape(g_lastCommittedText) + "\",";
        json += "\"candidates\":[";
        for (size_t i = 0; i < g_candidates.size(); ++i) {
            if (i > 0) json += ",";
            const auto& cand = g_candidates[i];
            json += "{\"text\":\"" + json_escape(cand.text) + "\"";
            json += ",\"correspondingCount\":" + std::to_string(cand.correspondingCount);
            json += ",\"score\":" + std::to_string(cand.score);
            json += ",\"selected\":" + std::string(static_cast<int>(i) == g_selectedCandidateIndex ? "true" : "false");
            json += "}";
        }
        json += "]}";
        return _strdup(json.c_str());
    } catch (...) {
        return _strdup("{}");
    }
}

} // extern "C"
