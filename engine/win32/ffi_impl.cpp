/**
 * new-ime Windows engine DLL — ffi.h compatible with myime/mozc
 * ONNX Runtime AR greedy inference for kana-kanji conversion
 */

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <algorithm>
#include <sstream>

static constexpr int PAD_ID = 0;
static constexpr int SEP_ID = 1;
static constexpr int OUT_ID = 2;
static constexpr int EOS_ID = 3;
static constexpr int UNK_ID = 4;
static constexpr int MAX_POS = 256;
static constexpr int MAX_GEN = 128;

static Ort::Env* g_env = nullptr;
static Ort::Session* g_session = nullptr;
static std::unordered_map<std::string, int> g_char_to_id;
static std::unordered_map<int, std::string> g_id_to_char;
static int g_vocab_size = 0;

static std::string g_composingText;
static std::string g_contextText;
static std::string g_modelDir;

struct CandidateResult {
    std::string text;
    int correspondingCount;
};
static std::vector<CandidateResult> g_candidates;
static std::mutex g_mutex;
static bool g_initialized = false;

// UTF-8 helpers
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
        count++;
    }
    return count;
}

static std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size();) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// Vocab loader (simple JSON parser)
static bool load_vocab(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    g_char_to_id.clear();
    g_id_to_char.clear();

    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        pos++;
        size_t end = content.find('"', pos);
        if (end == std::string::npos) break;

        // Handle escaped quotes
        while (end != std::string::npos && end > 0 && content[end - 1] == '\\') {
            end = content.find('"', end + 1);
        }
        if (end == std::string::npos) break;

        std::string key = content.substr(pos, end - pos);
        pos = end + 1;

        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) break;
        pos = colon + 1;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\n' || content[pos] == '\r')) pos++;

        size_t numstart = pos;
        while (pos < content.size() && content[pos] >= '0' && content[pos] <= '9') pos++;
        if (numstart == pos) continue;

        int id = std::stoi(content.substr(numstart, pos - numstart));

        // Decode JSON escapes
        std::string decoded;
        for (size_t i = 0; i < key.size(); i++) {
            if (key[i] == '\\' && i + 1 < key.size()) {
                if (key[i + 1] == 'u' && i + 5 < key.size()) {
                    unsigned int cp = std::stoul(key.substr(i + 2, 4), nullptr, 16);
                    if (cp < 0x80) {
                        decoded += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        decoded += static_cast<char>(0xC0 | (cp >> 6));
                        decoded += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        decoded += static_cast<char>(0xE0 | (cp >> 12));
                        decoded += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        decoded += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    i += 5;
                } else if (key[i + 1] == '"') { decoded += '"'; i++; }
                else if (key[i + 1] == '\\') { decoded += '\\'; i++; }
                else decoded += key[i];
            } else {
                decoded += key[i];
            }
        }

        g_char_to_id[decoded] = id;
        g_id_to_char[id] = decoded;
    }

    g_vocab_size = static_cast<int>(g_char_to_id.size());
    return g_vocab_size > 0;
}

// Tokenizer
static std::vector<int64_t> encode(const std::string& text) {
    auto chars = split_utf8(text);
    std::vector<int64_t> ids;
    for (const auto& c : chars) {
        auto it = g_char_to_id.find(c);
        ids.push_back(it != g_char_to_id.end() ? it->second : UNK_ID);
    }
    return ids;
}

static std::string decode(const std::vector<int64_t>& ids) {
    std::string result;
    for (int64_t id : ids) {
        if (id <= UNK_ID) continue;
        auto it = g_id_to_char.find(static_cast<int>(id));
        if (it != g_id_to_char.end()) result += it->second;
    }
    return result;
}

// ONNX AR greedy inference
static std::vector<CandidateResult> run_inference(const std::string& hiragana, const std::string& context) {
    if (hiragana.empty() || !g_session) {
        return {{hiragana, count_utf8_chars(hiragana)}};
    }

    // Build: [context] SEP [hiragana] OUT
    std::vector<int64_t> input_ids;
    if (!context.empty()) {
        auto ctx_chars = split_utf8(context);
        int start = std::max(0, static_cast<int>(ctx_chars.size()) - 40);
        for (int i = start; i < static_cast<int>(ctx_chars.size()); i++) {
            auto it = g_char_to_id.find(ctx_chars[i]);
            input_ids.push_back(it != g_char_to_id.end() ? it->second : UNK_ID);
        }
    }
    input_ids.push_back(SEP_ID);
    auto hira_ids = encode(hiragana);
    input_ids.insert(input_ids.end(), hira_ids.begin(), hira_ids.end());
    input_ids.push_back(OUT_ID);

    int hira_count = count_utf8_chars(hiragana);
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Greedy decode with fixed-length padding (ONNX exported at MAX_POS=256)
    std::vector<int64_t> generated;
    for (int step = 0; step < MAX_GEN && static_cast<int>(input_ids.size()) < MAX_POS; step++) {
        int64_t real_len = static_cast<int64_t>(input_ids.size());

        // Pad to MAX_POS
        std::vector<int64_t> padded_ids(MAX_POS, PAD_ID);
        std::vector<int64_t> padded_mask(MAX_POS, 0);
        for (int64_t i = 0; i < real_len; i++) {
            padded_ids[i] = input_ids[i];
            padded_mask[i] = 1;
        }

        std::array<int64_t, 2> shape = {1, static_cast<int64_t>(MAX_POS)};
        auto in_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, padded_ids.data(), MAX_POS, shape.data(), 2);
        auto mask_tensor = Ort::Value::CreateTensor<int64_t>(mem_info, padded_mask.data(), MAX_POS, shape.data(), 2);

        const char* in_names[] = {"input_ids", "attention_mask"};
        const char* out_names[] = {"logits"};
        Ort::Value inputs[] = {std::move(in_tensor), std::move(mask_tensor)};

        auto outputs = g_session->Run(Ort::RunOptions{nullptr}, in_names, inputs, 2, out_names, 1);

        auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int vocab = static_cast<int>(logits_shape[2]);
        const float* logits = outputs[0].GetTensorData<float>();
        const float* last = logits + (real_len - 1) * vocab;

        int best = 0;
        float best_val = last[0];
        for (int v = 1; v < vocab; v++) {
            if (last[v] > best_val) { best_val = last[v]; best = v; }
        }

        if (best == EOS_ID || best == PAD_ID) break;
        generated.push_back(best);
        input_ids.push_back(best);
    }

    std::string result = decode(generated);
    if (result.empty()) result = hiragana;

    return {{result, hira_count}};
}

// --- FFI exports ---
extern "C" {
#define EXPORT __declspec(dllexport)

EXPORT void LoadConfig(const char* configPath) {
    if (configPath) g_modelDir = configPath;
}

EXPORT void Initialize(const char* dictionaryPath, const char* memoryPath) {
    (void)memoryPath;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (dictionaryPath && strlen(dictionaryPath) > 0) g_modelDir = dictionaryPath;
    if (g_modelDir.empty()) g_modelDir = "models";

    load_vocab(g_modelDir + "\\ar-31m-scratch.vocab.json");
    try {
        if (!g_env) g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "new-ime");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::wstring wpath = std::wstring(g_modelDir.begin(), g_modelDir.end()) + L"\\ar-31m-scratch_fixed.onnx";
        if (g_session) delete g_session;
        g_session = new Ort::Session(*g_env, wpath.c_str(), opts);
        g_initialized = true;
    } catch (...) {
        g_initialized = false;
    }
    g_composingText.clear();
    g_contextText.clear();
    g_candidates.clear();
}

EXPORT void Shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    delete g_session; g_session = nullptr;
    delete g_env; g_env = nullptr;
    g_initialized = false;
}

EXPORT void AppendText(const char* input) {
    if (!input) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_composingText += input;
}

EXPORT void RemoveText(int count) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int i = 0; i < count && !g_composingText.empty(); i++) {
        auto it = g_composingText.end();
        while (it != g_composingText.begin()) { --it; if ((*it & 0xC0) != 0x80) break; }
        g_composingText.erase(it, g_composingText.end());
    }
}

EXPORT void MoveCursor(int) {}
EXPORT void ClearText(void) { std::lock_guard<std::mutex> lock(g_mutex); g_composingText.clear(); g_candidates.clear(); }

EXPORT const char* GetComposedText(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_candidates = run_inference(g_composingText, g_contextText);
    return _strdup(g_candidates.empty() ? g_composingText.c_str() : g_candidates[0].text.c_str());
}

EXPORT const char* GetCandidates(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_candidates = run_inference(g_composingText, g_contextText);
    std::string json = "[";
    for (size_t i = 0; i < g_candidates.size(); i++) {
        if (i > 0) json += ",";
        json += "{\"text\":\"";
        for (char c : g_candidates[i].text) {
            if (c == '"') json += "\\\""; else if (c == '\\') json += "\\\\"; else json += c;
        }
        json += "\",\"correspondingCount\":" + std::to_string(g_candidates[i].correspondingCount) + "}";
    }
    json += "]";
    return _strdup(json.c_str());
}

EXPORT void SelectCandidate(int index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (index >= 0 && index < static_cast<int>(g_candidates.size())) g_contextText = g_candidates[index].text;
    g_composingText.clear();
    g_candidates.clear();
}

EXPORT void ShrinkText(void) { RemoveText(1); }
EXPORT void ExpandText(void) {}
EXPORT void SetContext(const char* t) { if (t) { std::lock_guard<std::mutex> lock(g_mutex); g_contextText = t; } }
EXPORT void SetZenzaiEnabled(bool) {}
EXPORT void SetZenzaiInferenceLimit(int) {}
EXPORT void FreeString(const char* s) { free(const_cast<char*>(s)); }

} // extern "C"
