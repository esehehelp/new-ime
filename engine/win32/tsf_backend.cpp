#include "tsf_backend.h"

#include <vector>

namespace {

std::wstring module_dir() {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&module_dir), &module)) {
        return L"";
    }
    wchar_t buf[MAX_PATH];
    DWORD len = GetModuleFileNameW(module, buf, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        return L"";
    }
    std::wstring path(buf, len);
    auto slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : path.substr(0, slash);
}

} // namespace

bool EngineBackend::load() {
    if (module_) return true;
    std::wstring dll_path = module_dir() + L"\\new-ime-engine.dll";
    module_ = LoadLibraryW(dll_path.c_str());
    if (!module_) return false;

    initialize_ = reinterpret_cast<FnVoidStrStr>(GetProcAddress(module_, "Initialize"));
    shutdown_ = reinterpret_cast<FnVoid>(GetProcAddress(module_, "Shutdown"));
    append_text_ = reinterpret_cast<FnVoidStr>(GetProcAddress(module_, "AppendText"));
    set_composition_text_ = reinterpret_cast<FnVoidStrInt>(GetProcAddress(module_, "SetCompositionText"));
    remove_text_ = reinterpret_cast<FnVoidInt>(GetProcAddress(module_, "RemoveText"));
    move_cursor_ = reinterpret_cast<FnVoidInt>(GetProcAddress(module_, "MoveCursor"));
    clear_text_ = reinterpret_cast<FnVoid>(GetProcAddress(module_, "ClearText"));
    get_preedit_ = reinterpret_cast<FnStrVoid>(GetProcAddress(module_, "GetPreedit"));
    commit_selected_candidate_ = reinterpret_cast<FnStrVoid>(GetProcAddress(module_, "CommitSelectedCandidate"));
    commit_raw_text_ = reinterpret_cast<FnStrVoid>(GetProcAddress(module_, "CommitRawText"));
    get_state_json_ = reinterpret_cast<FnStrVoid>(GetProcAddress(module_, "GetStateJson"));
    free_string_ = reinterpret_cast<FnFreeStr>(GetProcAddress(module_, "FreeString"));
    is_initialized_ = reinterpret_cast<FnIntVoid>(GetProcAddress(module_, "IsInitialized"));

    if (!initialize_ || !shutdown_ || !append_text_ || !set_composition_text_ || !remove_text_ || !move_cursor_ ||
        !clear_text_ || !get_preedit_ || !commit_selected_candidate_ || !commit_raw_text_ ||
        !get_state_json_ || !free_string_ || !is_initialized_) {
        unload();
        return false;
    }
    return true;
}

void EngineBackend::unload() {
    if (module_) {
        FreeLibrary(module_);
        module_ = nullptr;
    }
    initialize_ = nullptr;
    shutdown_ = nullptr;
    append_text_ = nullptr;
    set_composition_text_ = nullptr;
    remove_text_ = nullptr;
    move_cursor_ = nullptr;
    clear_text_ = nullptr;
    get_preedit_ = nullptr;
    commit_selected_candidate_ = nullptr;
    commit_raw_text_ = nullptr;
    get_state_json_ = nullptr;
    free_string_ = nullptr;
    is_initialized_ = nullptr;
}

bool EngineBackend::initialize(const std::wstring& model_path) {
    if (!load()) return false;
    std::string utf8 = wide_to_utf8(model_path);
    initialize_(utf8.c_str(), nullptr);
    // Even if internal state did not fully initialize (sandboxed FS, missing model,
    // etc.), keep the TSF service activated so the IME is still selectable. Worst
    // case the EXPORTs return empty / echo, caught inside each try block.
    return true;
}

void EngineBackend::shutdown() {
    if (shutdown_) shutdown_();
}

bool EngineBackend::append_utf8(const std::string& text) {
    if (!append_text_) return false;
    append_text_(text.c_str());
    return true;
}

bool EngineBackend::set_composition_utf8(const std::string& text, int cursor_chars) {
    if (!set_composition_text_) return false;
    set_composition_text_(text.c_str(), cursor_chars);
    return true;
}

bool EngineBackend::backspace() {
    if (!remove_text_) return false;
    remove_text_(1);
    return true;
}

bool EngineBackend::move_cursor(int delta) {
    if (!move_cursor_) return false;
    move_cursor_(delta);
    return true;
}

bool EngineBackend::clear() {
    if (!clear_text_) return false;
    clear_text_();
    return true;
}

std::string EngineBackend::preedit() {
    return call_string(get_preedit_);
}

std::string EngineBackend::commit_selected() {
    return call_string(commit_selected_candidate_);
}

std::string EngineBackend::commit_raw() {
    return call_string(commit_raw_text_);
}

std::string EngineBackend::state_json() {
    return call_string(get_state_json_);
}

std::string EngineBackend::call_string(FnStrVoid fn) {
    if (!fn || !free_string_) return {};
    const char* ptr = fn();
    if (!ptr) return {};
    std::string out(ptr);
    free_string_(ptr);
    return out;
}

std::string EngineBackend::wide_to_utf8(const std::wstring& input) {
    if (input.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, input.c_str(), -1, out.data(), len, nullptr, nullptr);
    if (!out.empty() && out.back() == '\0') out.pop_back();
    return out;
}
