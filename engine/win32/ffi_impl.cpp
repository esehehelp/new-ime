/**
 * new-ime Windows engine DLL — implements ffi.h interface
 * Drop-in replacement for azookey-engine.dll in myime/mozc
 *
 * Build: cl /LD ffi_impl.cpp /I<onnxruntime> /link onnxruntime.lib /out:new-ime-engine.dll
 */

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <mutex>

// TODO: Replace with actual ONNX Runtime inference
// For now, stub implementation that compiles

struct Candidate {
    std::string text;
    int correspondingCount;  // number of hiragana chars covered
};

static std::string g_composingText;
static std::string g_contextText;
static std::vector<Candidate> g_candidates;
static std::mutex g_mutex;
static bool g_initialized = false;

// Forward declarations for inference (to be implemented)
static std::vector<Candidate> run_inference(const std::string& hiragana, const std::string& context);

extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void LoadConfig(const char* configPath) {
    // TODO: Load model path, settings from JSON config
    (void)configPath;
}

EXPORT void Initialize(const char* dictionaryPath, const char* memoryPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // TODO: Load ONNX model, initialize tokenizer
    // dictionaryPath -> model directory
    // memoryPath -> user learning data directory
    (void)dictionaryPath;
    (void)memoryPath;
    g_initialized = true;
    g_composingText.clear();
    g_contextText.clear();
    g_candidates.clear();
}

EXPORT void Shutdown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_initialized = false;
    g_composingText.clear();
    g_candidates.clear();
}

EXPORT void AppendText(const char* input) {
    if (!input) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_composingText += input;
}

EXPORT void RemoveText(int count) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // Remove 'count' UTF-8 characters from end
    for (int i = 0; i < count && !g_composingText.empty(); i++) {
        // Walk back one UTF-8 character
        auto it = g_composingText.end();
        while (it != g_composingText.begin()) {
            --it;
            if ((*it & 0xC0) != 0x80) break;
        }
        g_composingText.erase(it, g_composingText.end());
    }
}

EXPORT void MoveCursor(int offset) {
    // TODO: Support cursor movement within composing text
    (void)offset;
}

EXPORT void ClearText(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_composingText.clear();
    g_candidates.clear();
}

EXPORT const char* GetComposedText(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_candidates = run_inference(g_composingText, g_contextText);
    if (g_candidates.empty()) {
        return _strdup("");
    }
    return _strdup(g_candidates[0].text.c_str());
}

EXPORT const char* GetCandidates(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_candidates = run_inference(g_composingText, g_contextText);

    // Build JSON array: [{"text": "...", "correspondingCount": N}, ...]
    std::string json = "[";
    for (size_t i = 0; i < g_candidates.size(); i++) {
        if (i > 0) json += ",";
        json += "{\"text\":\"";
        // Escape JSON string
        for (char c : g_candidates[i].text) {
            if (c == '"') json += "\\\"";
            else if (c == '\\') json += "\\\\";
            else json += c;
        }
        json += "\",\"correspondingCount\":";
        json += std::to_string(g_candidates[i].correspondingCount);
        json += "}";
    }
    json += "]";

    return _strdup(json.c_str());
}

EXPORT void SelectCandidate(int index) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (index >= 0 && index < (int)g_candidates.size()) {
        // Update context with selected text
        g_contextText = g_candidates[index].text;
    }
    g_composingText.clear();
    g_candidates.clear();
}

EXPORT void ShrinkText(void) {
    RemoveText(1);
}

EXPORT void ExpandText(void) {
    // TODO: Expand segment boundary
}

EXPORT void SetContext(const char* precedingText) {
    if (!precedingText) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    g_contextText = precedingText;
}

EXPORT void SetZenzaiEnabled(bool enabled) {
    // Always enabled for our model (no toggle needed)
    (void)enabled;
}

EXPORT void SetZenzaiInferenceLimit(int limit) {
    // Not applicable to our single-pass inference
    (void)limit;
}

EXPORT void FreeString(const char* str) {
    free(const_cast<char*>(str));
}

}  // extern "C"


// --- Inference stub (replace with ONNX Runtime) ---

static int count_utf8_chars(const std::string& s) {
    int count = 0;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        if (c < 0x80) i += 1;
        else if (c < 0xE0) i += 2;
        else if (c < 0xF0) i += 3;
        else i += 4;
        count++;
    }
    return count;
}

static std::vector<Candidate> run_inference(const std::string& hiragana, const std::string& context) {
    if (hiragana.empty()) return {};

    // TODO: Replace with actual ONNX Runtime inference
    // 1. Tokenize hiragana + context
    // 2. Run encoder → decoder → CTC head (or AR greedy)
    // 3. CTC beam search / greedy decode
    // 4. Return top-K candidates

    // Stub: return hiragana as-is (no conversion)
    int charCount = count_utf8_chars(hiragana);
    return {{hiragana, charCount}};
}
