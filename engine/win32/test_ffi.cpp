/**
 * Test program for new-ime-engine.dll FFI
 * Build: cl /EHsc /std:c++17 test_ffi.cpp /link new-ime-engine.lib
 */

#include <cstdio>
#include <cstdlib>
#include <windows.h>

// FFI function types
typedef void (*FnVoidStr)(const char*);
typedef void (*FnVoidStrStr)(const char*, const char*);
typedef void (*FnVoid)(void);
typedef void (*FnVoidInt)(int);
typedef void (*FnVoidBool)(bool);
typedef const char* (*FnStrVoid)(void);
typedef void (*FnFreeStr)(const char*);

int main() {
    SetConsoleOutputCP(65001); // UTF-8

    HMODULE dll = LoadLibraryA("new-ime-engine.dll");
    if (!dll) {
        printf("Failed to load new-ime-engine.dll\n");
        return 1;
    }
    printf("DLL loaded OK\n");

    auto pInitialize = (FnVoidStrStr)GetProcAddress(dll, "Initialize");
    auto pShutdown = (FnVoid)GetProcAddress(dll, "Shutdown");
    auto pAppendText = (FnVoidStr)GetProcAddress(dll, "AppendText");
    auto pClearText = (FnVoid)GetProcAddress(dll, "ClearText");
    auto pGetComposedText = (FnStrVoid)GetProcAddress(dll, "GetComposedText");
    auto pGetCandidates = (FnStrVoid)GetProcAddress(dll, "GetCandidates");
    auto pSetContext = (FnVoidStr)GetProcAddress(dll, "SetContext");
    auto pSelectCandidate = (FnVoidInt)GetProcAddress(dll, "SelectCandidate");
    auto pFreeString = (FnFreeStr)GetProcAddress(dll, "FreeString");

    if (!pInitialize || !pAppendText || !pGetComposedText || !pGetCandidates) {
        printf("Failed to find functions in DLL\n");
        FreeLibrary(dll);
        return 1;
    }
    printf("Functions loaded OK\n");

    // Initialize with model directory
    printf("\nInitializing...\n");
    pInitialize("C:\\Users\\admin\\Dev\\new-ime\\models", nullptr);
    printf("Initialized\n");

    // Test cases
    const char* tests[][2] = {
        {"", "\xe3\x81\x8c\xe3\x81\xa3\xe3\x81\x93\xe3\x81\x86\xe3\x81\xab\xe3\x81\x84\xe3\x81\x8f"}, // がっこうにいく
        {"", "\xe3\x81\x97\xe3\x82\x93\xe3\x81\xb6\xe3\x82\x93\xe3\x82\x92\xe3\x82\x88\xe3\x82\x80"}, // しんぶんをよむ
        {"", "\xe3\x81\xa8\xe3\x81\x86\xe3\x81\x8d\xe3\x82\x87\xe3\x81\x86\xe3\x81\xa8\xe3\x81\x97\xe3\x81\xb6\xe3\x82\x84\xe3\x81\x8f"}, // とうきょうとしぶやく
        {"", "\xe3\x81\x8d\xe3\x82\x87\xe3\x81\x86\xe3\x81\xaf\xe3\x81\x84\xe3\x81\x84\xe3\x81\xa6\xe3\x82\x93\xe3\x81\x8d\xe3\x81\xa7\xe3\x81\x99\xe3\x81\xad"}, // きょうはいいてんきですね
    };

    for (int i = 0; i < 4; i++) {
        pClearText();
        if (strlen(tests[i][0]) > 0) {
            pSetContext(tests[i][0]);
        }
        pAppendText(tests[i][1]);

        const char* result = pGetComposedText();
        printf("  in:  %s\n", tests[i][1]);
        printf("  out: %s\n", result ? result : "(null)");
        if (result) pFreeString(result);

        const char* cands = pGetCandidates();
        printf("  candidates: %s\n\n", cands ? cands : "(null)");
        if (cands) pFreeString(cands);
    }

    pShutdown();
    FreeLibrary(dll);
    printf("Done\n");
    return 0;
}
