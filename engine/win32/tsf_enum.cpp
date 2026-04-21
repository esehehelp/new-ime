#include "tsf_guids.h"

#include <msctf.h>
#include <windows.h>

#include <cstdio>

int wmain() {
    HRESULT hr = CoInitialize(nullptr);
    if (FAILED(hr)) {
        std::printf("CoInitialize failed: 0x%08X\n", static_cast<unsigned>(hr));
        return 1;
    }

    ITfInputProcessorProfileMgr* mgr = nullptr;
    hr = CoCreateInstance(CLSID_TF_InputProcessorProfiles, nullptr, CLSCTX_INPROC_SERVER,
                          IID_ITfInputProcessorProfileMgr, reinterpret_cast<void**>(&mgr));
    if (FAILED(hr)) {
        std::printf("CoCreateInstance failed: 0x%08X\n", static_cast<unsigned>(hr));
        CoUninitialize();
        return 1;
    }

    IEnumTfInputProcessorProfiles* en = nullptr;
    hr = mgr->EnumProfiles(MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN), &en);
    if (FAILED(hr)) {
        std::printf("EnumProfiles failed: 0x%08X\n", static_cast<unsigned>(hr));
        mgr->Release();
        CoUninitialize();
        return 1;
    }

    bool found = false;
    TF_INPUTPROCESSORPROFILE profile{};
    ULONG fetched = 0;
    while (en->Next(1, &profile, &fetched) == S_OK && fetched == 1) {
        LPOLESTR clsid = nullptr;
        LPOLESTR guid = nullptr;
        StringFromCLSID(profile.clsid, &clsid);
        StringFromCLSID(profile.guidProfile, &guid);
        wprintf(L"type=%lu lang=0x%04X clsid=%ls profile=%ls\n",
                profile.dwProfileType,
                profile.langid,
                clsid ? clsid : L"(null)",
                guid ? guid : L"(null)");
        if (profile.clsid == CLSID_NewImeTextService ||
            profile.guidProfile == GUID_PROFILE_NewIme) {
            found = true;
        }
        if (clsid) CoTaskMemFree(clsid);
        if (guid) CoTaskMemFree(guid);
    }

    std::printf("new-ime-found=%s\n", found ? "true" : "false");

    en->Release();
    mgr->Release();
    CoUninitialize();
    return found ? 0 : 2;
}
