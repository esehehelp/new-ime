/**
 * Interactive console IME demo using new-ime-engine.dll
 * Type hiragana, get kanji conversion in real-time.
 *
 * Build: cl /EHsc /std:c++17 /utf-8 interactive.cpp /Fe:interactive.exe
 * Run: interactive.exe (make sure new-ime-engine.dll and onnxruntime.dll are in same dir)
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <windows.h>

typedef void (*FnVoidStrStr)(const char*, const char*);
typedef void (*FnVoidStr)(const char*);
typedef void (*FnVoid)(void);
typedef void (*FnVoidInt)(int);
typedef const char* (*FnStrVoid)(void);
typedef void (*FnFreeStr)(const char*);

int main() {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    HMODULE dll = LoadLibraryA("new-ime-engine.dll");
    if (!dll) { printf("Failed to load DLL\n"); return 1; }

    auto Initialize = (FnVoidStrStr)GetProcAddress(dll, "Initialize");
    auto Shutdown = (FnVoid)GetProcAddress(dll, "Shutdown");
    auto AppendText = (FnVoidStr)GetProcAddress(dll, "AppendText");
    auto ClearText = (FnVoid)GetProcAddress(dll, "ClearText");
    auto GetComposedText = (FnStrVoid)GetProcAddress(dll, "GetComposedText");
    auto SetContext = (FnVoidStr)GetProcAddress(dll, "SetContext");
    auto FreeString = (FnFreeStr)GetProcAddress(dll, "FreeString");

    printf("Initializing model...\n");
    Initialize("..\\..\\models", nullptr);
    printf("Ready!\n\n");

    printf("=== new-ime Interactive Demo ===\n");
    printf("Type hiragana and press Enter to convert.\n");
    printf("Type 'q' to quit.\n\n");

    std::string context;
    char buf[1024];

    while (true) {
        if (!context.empty()) {
            printf("[context: ...%s]\n", context.substr(context.size() > 20 ? context.size() - 20 : 0).c_str());
        }
        printf("> ");
        fflush(stdout);

        if (!fgets(buf, sizeof(buf), stdin)) break;

        // Remove trailing newline
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = '\0';

        if (strcmp(buf, "q") == 0 || strcmp(buf, "quit") == 0) break;
        if (len == 0) continue;

        // Set context and convert
        if (!context.empty()) {
            SetContext(context.c_str());
        }
        ClearText();
        AppendText(buf);

        LARGE_INTEGER freq, t0, t1;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t0);

        const char* result = GetComposedText();

        QueryPerformanceCounter(&t1);
        double ms = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart * 1000.0;

        printf("  => %s  (%.0fms)\n\n", result ? result : "(null)", ms);

        // Update context
        if (result) {
            context = result;
            FreeString(result);
        }
    }

    Shutdown();
    FreeLibrary(dll);
    printf("Bye!\n");
    return 0;
}
