#pragma once

#include <windows.h>

#include <string>

class EngineBackend {
public:
    bool load();
    void unload();

    bool initialize(const std::wstring& model_path);
    void shutdown();

    bool append_utf8(const std::string& text);
    bool set_composition_utf8(const std::string& text, int cursor_chars);
    bool backspace();
    bool move_cursor(int delta);
    bool clear();

    std::string preedit();
    std::string commit_selected();
    std::string commit_raw();
    std::string state_json();

private:
    using FnVoidStrStr = void (*)(const char*, const char*);
    using FnVoidStr = void (*)(const char*);
    using FnVoidStrInt = void (*)(const char*, int);
    using FnVoid = void (*)();
    using FnVoidInt = void (*)(int);
    using FnStrVoid = const char* (*)();
    using FnFreeStr = void (*)(const char*);
    using FnIntVoid = int (*)();

    std::string call_string(FnStrVoid fn);
    static std::string wide_to_utf8(const std::wstring& input);

    HMODULE module_ = nullptr;
    FnVoidStrStr initialize_ = nullptr;
    FnVoid shutdown_ = nullptr;
    FnVoidStr append_text_ = nullptr;
    FnVoidStrInt set_composition_text_ = nullptr;
    FnVoidInt remove_text_ = nullptr;
    FnVoidInt move_cursor_ = nullptr;
    FnVoid clear_text_ = nullptr;
    FnStrVoid get_preedit_ = nullptr;
    FnStrVoid commit_selected_candidate_ = nullptr;
    FnStrVoid commit_raw_text_ = nullptr;
    FnStrVoid get_state_json_ = nullptr;
    FnFreeStr free_string_ = nullptr;
    FnIntVoid is_initialized_ = nullptr;
};
