#include "tsf_backend.h"
#include "tsf_guids.h"
#include "../src/composing_text.h"

#include <msctf.h>
#include <strsafe.h>
#include <windows.h>

#include <atomic>
#include <memory>
#include <string>

namespace {

std::atomic<ULONG> g_dllRefCount = 0;
HINSTANCE g_hInstance = nullptr;

constexpr wchar_t kServiceName[] = L"new-ime";
constexpr wchar_t kDescription[] = L"new-ime TSF Text Service";
constexpr wchar_t kThreadingModel[] = L"Apartment";
constexpr wchar_t kClsidRoot[] = L"Software\\Classes\\CLSID\\";

std::wstring utf8_to_wide(const std::string& input) {
    if (input.empty()) return L"";
    int len = MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, nullptr, 0);
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, input.c_str(), -1, out.data(), len);
    if (!out.empty() && out.back() == L'\0') out.pop_back();
    return out;
}

void tsf_log(const char* line) {
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(g_hInstance, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return;
    std::wstring p(path, len);
    auto slash = p.find_last_of(L"\\/");
    std::wstring dir = slash == std::wstring::npos ? L"." : p.substr(0, slash);
    std::wstring log_path = dir + L"\\new-ime-tsf.log";
    HANDLE file = CreateFileW(log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                              nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    std::string out(line);
    out += "\r\n";
    DWORD written = 0;
    WriteFile(file, out.data(), static_cast<DWORD>(out.size()), &written, nullptr);
    CloseHandle(file);
}

std::wstring module_dir(HMODULE module) {
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(module, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return L".";
    std::wstring out(path, len);
    auto slash = out.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : out.substr(0, slash);
}

std::wstring model_root_for_tsf() {
    wchar_t base[MAX_PATH];
    DWORD len = GetFullPathNameW((module_dir(g_hInstance) + L"\\..\\..\\models").c_str(),
                                 MAX_PATH, base, nullptr);
    if (len == 0 || len >= MAX_PATH) {
        return module_dir(g_hInstance) + L"\\..\\..\\models";
    }
    return std::wstring(base, len);
}

std::wstring module_path(HMODULE module) {
    wchar_t path[MAX_PATH];
    DWORD len = GetModuleFileNameW(module, path, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) return L"";
    return std::wstring(path, len);
}

std::wstring guid_to_string(REFGUID guid) {
    wchar_t buf[64];
    StringFromGUID2(guid, buf, ARRAYSIZE(buf));
    return buf;
}

LONG set_reg_string(HKEY root,
                    const std::wstring& subkey,
                    const wchar_t* name,
                    const std::wstring& value) {
    HKEY key = nullptr;
    DWORD disposition = 0;
    LONG rc = RegCreateKeyExW(root, subkey.c_str(), 0, nullptr, 0,
                              KEY_WRITE, nullptr, &key, &disposition);
    if (rc != ERROR_SUCCESS) return rc;
    rc = RegSetValueExW(key, name, 0, REG_SZ,
                        reinterpret_cast<const BYTE*>(value.c_str()),
                        static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
    return rc;
}

void delete_reg_tree_if_exists(HKEY root, const std::wstring& subkey) {
    RegDeleteTreeW(root, subkey.c_str());
}

class TextService;

class SimpleEditSession : public ITfEditSession {
public:
    using Handler = HRESULT (*)(TextService*, TfEditCookie);

    SimpleEditSession(TextService* service, Handler handler)
        : ref_(1), service_(service), handler_(handler) {}

    STDMETHODIMP QueryInterface(REFIID riid, void** ppvObj) override;
    STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG value = --ref_;
        if (!value) delete this;
        return value;
    }
    STDMETHODIMP DoEditSession(TfEditCookie ec) override;

private:
    std::atomic<ULONG> ref_;
    TextService* service_;
    Handler handler_;
};

class TextService final : public ITfTextInputProcessor,
                          public ITfKeyEventSink,
                          public ITfCompositionSink {
public:
    TextService() : ref_(1) { ++g_dllRefCount; }
    ~TextService() {
        if (composition_) composition_->Release();
        if (thread_mgr_) thread_mgr_->Release();
        --g_dllRefCount;
    }

    STDMETHODIMP QueryInterface(REFIID riid, void** ppvObj) override;
    STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG value = --ref_;
        if (!value) delete this;
        return value;
    }

    STDMETHODIMP Activate(ITfThreadMgr* ptim, TfClientId tid) override;
    STDMETHODIMP Deactivate() override;

    STDMETHODIMP OnSetFocus(BOOL) override { return S_OK; }
    STDMETHODIMP OnTestKeyDown(ITfContext*, WPARAM wParam, LPARAM, BOOL* pfEaten) override;
    STDMETHODIMP OnTestKeyUp(ITfContext*, WPARAM, LPARAM, BOOL* pfEaten) override {
        *pfEaten = FALSE;
        return S_OK;
    }
    STDMETHODIMP OnKeyDown(ITfContext*, WPARAM wParam, LPARAM, BOOL* pfEaten) override;
    STDMETHODIMP OnKeyUp(ITfContext*, WPARAM, LPARAM, BOOL* pfEaten) override {
        *pfEaten = FALSE;
        return S_OK;
    }
    STDMETHODIMP OnPreservedKey(ITfContext*, REFGUID, BOOL* pfEaten) override {
        *pfEaten = FALSE;
        return S_OK;
    }

    STDMETHODIMP OnCompositionTerminated(TfEditCookie, ITfComposition*) override;

    HRESULT update_preedit();
    HRESULT commit_text(const std::wstring& text);
    HRESULT ensure_composition(TfEditCookie ec, ITfContext* context);
    HRESULT set_composition_text(TfEditCookie ec, const std::wstring& text);
    HRESULT end_composition(TfEditCookie ec);

    static HRESULT StartCompositionSession(TextService* self, TfEditCookie ec);
    static HRESULT UpdateCompositionSession(TextService* self, TfEditCookie ec);
    static HRESULT CommitCompositionSession(TextService* self, TfEditCookie ec);

private:
    bool should_handle_key(WPARAM wParam) const;
    std::string key_to_utf8(WPARAM wParam) const;
    void sync_backend_reading();
    HRESULT request_edit_session(ITfContext* context, SimpleEditSession::Handler handler);
    HRESULT current_context(ITfContext** context) const;
    HRESULT unadvise_key_sink();

    std::atomic<ULONG> ref_;
    TfClientId client_id_ = TF_CLIENTID_NULL;
    ITfThreadMgr* thread_mgr_ = nullptr;
    ITfComposition* composition_ = nullptr;
    DWORD key_event_sink_cookie_ = TF_INVALID_COOKIE;
    EngineBackend backend_;
    newime::ComposingText romaji_;
};

class ClassFactory final : public IClassFactory {
public:
    ClassFactory() : ref_(1) { ++g_dllRefCount; }
    ~ClassFactory() { --g_dllRefCount; }

    STDMETHODIMP QueryInterface(REFIID riid, void** ppvObj) override;
    STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG value = --ref_;
        if (!value) delete this;
        return value;
    }
    STDMETHODIMP CreateInstance(IUnknown* outer, REFIID riid, void** ppvObj) override;
    STDMETHODIMP LockServer(BOOL lock) override {
        if (lock) ++g_dllRefCount;
        else --g_dllRefCount;
        return S_OK;
    }

private:
    std::atomic<ULONG> ref_;
};

STDMETHODIMP SimpleEditSession::QueryInterface(REFIID riid, void** ppvObj) {
    if (!ppvObj) return E_INVALIDARG;
    *ppvObj = nullptr;
    if (riid == IID_IUnknown || riid == IID_ITfEditSession) {
        *ppvObj = static_cast<ITfEditSession*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

STDMETHODIMP SimpleEditSession::DoEditSession(TfEditCookie ec) {
    return handler_(service_, ec);
}

STDMETHODIMP TextService::QueryInterface(REFIID riid, void** ppvObj) {
    if (!ppvObj) return E_INVALIDARG;
    *ppvObj = nullptr;
    if (riid == IID_IUnknown || riid == IID_ITfTextInputProcessor) {
        *ppvObj = static_cast<ITfTextInputProcessor*>(this);
    } else if (riid == IID_ITfKeyEventSink) {
        *ppvObj = static_cast<ITfKeyEventSink*>(this);
    } else if (riid == IID_ITfCompositionSink) {
        *ppvObj = static_cast<ITfCompositionSink*>(this);
    } else {
        return E_NOINTERFACE;
    }
    AddRef();
    return S_OK;
}

STDMETHODIMP ClassFactory::QueryInterface(REFIID riid, void** ppvObj) {
    if (!ppvObj) return E_INVALIDARG;
    *ppvObj = nullptr;
    if (riid == IID_IUnknown || riid == IID_IClassFactory) {
        *ppvObj = static_cast<IClassFactory*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

STDMETHODIMP ClassFactory::CreateInstance(IUnknown* outer, REFIID riid, void** ppvObj) {
    if (outer) return CLASS_E_NOAGGREGATION;
    auto* service = new TextService();
    HRESULT hr = service->QueryInterface(riid, ppvObj);
    service->Release();
    return hr;
}

STDMETHODIMP TextService::Activate(ITfThreadMgr* ptim, TfClientId tid) {
    if (!ptim) return E_INVALIDARG;
    thread_mgr_ = ptim;
    thread_mgr_->AddRef();
    client_id_ = tid;

    if (!backend_.load()) return E_FAIL;
    if (!backend_.initialize(model_root_for_tsf())) return E_FAIL;

    ITfKeystrokeMgr* key_mgr = nullptr;
    HRESULT hr = thread_mgr_->QueryInterface(IID_ITfKeystrokeMgr, reinterpret_cast<void**>(&key_mgr));
    if (FAILED(hr)) return hr;
    hr = key_mgr->AdviseKeyEventSink(client_id_, static_cast<ITfKeyEventSink*>(this), TRUE);
    if (SUCCEEDED(hr)) key_event_sink_cookie_ = 1;
    key_mgr->Release();
    return hr;
}

HRESULT TextService::unadvise_key_sink() {
    if (!thread_mgr_ || key_event_sink_cookie_ == TF_INVALID_COOKIE) return S_OK;
    ITfKeystrokeMgr* key_mgr = nullptr;
    HRESULT hr = thread_mgr_->QueryInterface(IID_ITfKeystrokeMgr, reinterpret_cast<void**>(&key_mgr));
    if (FAILED(hr)) return hr;
    hr = key_mgr->UnadviseKeyEventSink(client_id_);
    key_mgr->Release();
    key_event_sink_cookie_ = TF_INVALID_COOKIE;
    return hr;
}

STDMETHODIMP TextService::Deactivate() {
    unadvise_key_sink();
    backend_.shutdown();
    if (composition_) {
        composition_->Release();
        composition_ = nullptr;
    }
    if (thread_mgr_) {
        thread_mgr_->Release();
        thread_mgr_ = nullptr;
    }
    client_id_ = TF_CLIENTID_NULL;
    return S_OK;
}

bool TextService::should_handle_key(WPARAM wParam) const {
    if ((wParam >= 'A' && wParam <= 'Z') || (wParam >= '0' && wParam <= '9')) return true;
    switch (wParam) {
    case VK_BACK:
    case VK_LEFT:
    case VK_RIGHT:
    case VK_SPACE:
    case VK_RETURN:
    case VK_ESCAPE:
    case VK_OEM_MINUS:
    case VK_OEM_COMMA:
    case VK_OEM_PERIOD:
        return true;
    default:
        return false;
    }
}

std::string TextService::key_to_utf8(WPARAM wParam) const {
    BYTE state[256] = {};
    GetKeyboardState(state);
    WCHAR buf[8] = {};
    int rc = ToUnicode(static_cast<UINT>(wParam), 0, state, buf, 8, 0);
    if (rc <= 0) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, buf, rc, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, rc, out.data(), len, nullptr, nullptr);
    return out;
}

STDMETHODIMP TextService::OnTestKeyDown(ITfContext*, WPARAM wParam, LPARAM, BOOL* pfEaten) {
    if (!pfEaten) return E_INVALIDARG;
    *pfEaten = should_handle_key(wParam) ? TRUE : FALSE;
    return S_OK;
}

HRESULT TextService::current_context(ITfContext** context) const {
    if (!thread_mgr_ || !context) return E_FAIL;
    *context = nullptr;
    ITfDocumentMgr* doc_mgr = nullptr;
    HRESULT hr = thread_mgr_->GetFocus(&doc_mgr);
    if (FAILED(hr) || !doc_mgr) return FAILED(hr) ? hr : E_FAIL;
    hr = doc_mgr->GetTop(context);
    doc_mgr->Release();
    return hr;
}

HRESULT TextService::request_edit_session(ITfContext* context, SimpleEditSession::Handler handler) {
    if (!context) return E_INVALIDARG;
    auto* edit = new SimpleEditSession(this, handler);
    HRESULT session_hr = E_FAIL;
    HRESULT hr = context->RequestEditSession(client_id_, edit, TF_ES_SYNC | TF_ES_READWRITE, &session_hr);
    edit->Release();
    return FAILED(hr) ? hr : session_hr;
}

void TextService::sync_backend_reading() {
    backend_.set_composition_utf8(romaji_.hiragana(), romaji_.cursor());
}

STDMETHODIMP TextService::OnKeyDown(ITfContext*, WPARAM wParam, LPARAM, BOOL* pfEaten) {
    if (!pfEaten) return E_INVALIDARG;
    *pfEaten = FALSE;
    if (!should_handle_key(wParam)) return S_OK;

    switch (wParam) {
    case VK_BACK:
        romaji_.delete_left();
        sync_backend_reading();
        break;
    case VK_LEFT:
        romaji_.move_cursor(-1);
        sync_backend_reading();
        break;
    case VK_RIGHT:
        romaji_.move_cursor(1);
        sync_backend_reading();
        break;
    case VK_SPACE:
    case VK_RETURN: {
        ITfContext* context = nullptr;
        HRESULT hr = current_context(&context);
        if (FAILED(hr)) return hr;
        hr = request_edit_session(context, &TextService::CommitCompositionSession);
        context->Release();
        if (SUCCEEDED(hr)) *pfEaten = TRUE;
        return hr;
    }
    case VK_ESCAPE:
        romaji_.reset();
        backend_.clear();
        break;
    default: {
        std::string text = key_to_utf8(wParam);
        if (text.empty()) return S_OK;
        if (text.size() == 1) {
            char c = text[0];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '.' || c == ',') {
                romaji_.input_char(c);
                sync_backend_reading();
            }
        }
        break;
    }
    }

    ITfContext* context = nullptr;
    HRESULT hr = current_context(&context);
    if (FAILED(hr)) return hr;
    hr = request_edit_session(context, composition_ ? &TextService::UpdateCompositionSession
                                                    : &TextService::StartCompositionSession);
    context->Release();
    if (SUCCEEDED(hr)) *pfEaten = TRUE;
    return hr;
}

STDMETHODIMP TextService::OnCompositionTerminated(TfEditCookie, ITfComposition*) {
    tsf_log("OnCompositionTerminated enter");
    if (composition_) {
        composition_->Release();
        composition_ = nullptr;
    }
    romaji_.reset();
    tsf_log("OnCompositionTerminated before backend_.clear");
    backend_.clear();
    tsf_log("OnCompositionTerminated end");
    return S_OK;
}

HRESULT TextService::ensure_composition(TfEditCookie ec, ITfContext* context) {
    if (composition_) return S_OK;
    ITfInsertAtSelection* insert = nullptr;
    HRESULT hr = context->QueryInterface(IID_ITfInsertAtSelection, reinterpret_cast<void**>(&insert));
    if (FAILED(hr)) return hr;
    ITfRange* range = nullptr;
    hr = insert->InsertTextAtSelection(ec, TF_IAS_QUERYONLY, L"", 0, &range);
    insert->Release();
    if (FAILED(hr)) return hr;
    ITfContextComposition* composition_mgr = nullptr;
    hr = context->QueryInterface(IID_ITfContextComposition, reinterpret_cast<void**>(&composition_mgr));
    if (SUCCEEDED(hr)) {
        hr = composition_mgr->StartComposition(ec, range, static_cast<ITfCompositionSink*>(this), &composition_);
        composition_mgr->Release();
    }
    range->Release();
    return hr;
}

HRESULT TextService::set_composition_text(TfEditCookie ec, const std::wstring& text) {
    if (!composition_) return E_FAIL;
    ITfRange* range = nullptr;
    HRESULT hr = composition_->GetRange(&range);
    if (FAILED(hr)) return hr;
    hr = range->SetText(ec, 0, text.c_str(), static_cast<LONG>(text.size()));
    range->Release();
    return hr;
}

HRESULT TextService::end_composition(TfEditCookie ec) {
    tsf_log("end_composition enter");
    ITfComposition* comp = composition_;
    if (!comp) {
        tsf_log("end_composition no composition");
        return S_OK;
    }
    composition_ = nullptr;
    tsf_log("end_composition before EndComposition");
    HRESULT hr = comp->EndComposition(ec);
    tsf_log("end_composition after EndComposition");
    comp->Release();
    tsf_log("end_composition after Release");
    return hr;
}

HRESULT TextService::commit_text(const std::wstring& text) {
    ITfContext* context = nullptr;
    HRESULT hr = current_context(&context);
    if (FAILED(hr)) return hr;
    ITfInsertAtSelection* insert = nullptr;
    hr = context->QueryInterface(IID_ITfInsertAtSelection, reinterpret_cast<void**>(&insert));
    if (FAILED(hr)) {
        context->Release();
        return hr;
    }

    class CommitSession final : public ITfEditSession {
    public:
        CommitSession(TextService* service, ITfInsertAtSelection* insert, std::wstring text)
            : ref_(1), service_(service), insert_(insert), text_(std::move(text)) {
            insert_->AddRef();
        }
        ~CommitSession() { insert_->Release(); }
        STDMETHODIMP QueryInterface(REFIID riid, void** ppvObj) override {
            if (!ppvObj) return E_INVALIDARG;
            *ppvObj = nullptr;
            if (riid == IID_IUnknown || riid == IID_ITfEditSession) {
                *ppvObj = static_cast<ITfEditSession*>(this);
                AddRef();
                return S_OK;
            }
            return E_NOINTERFACE;
        }
        STDMETHODIMP_(ULONG) AddRef() override { return ++ref_; }
        STDMETHODIMP_(ULONG) Release() override {
            ULONG value = --ref_;
            if (!value) delete this;
            return value;
        }
        STDMETHODIMP DoEditSession(TfEditCookie ec) override {
            service_->end_composition(ec);
            ITfRange* range = nullptr;
            HRESULT hr = insert_->InsertTextAtSelection(ec, TF_IAS_NOQUERY, text_.c_str(),
                                                        static_cast<LONG>(text_.size()), &range);
            if (range) range->Release();
            return hr;
        }
    private:
        std::atomic<ULONG> ref_;
        TextService* service_;
        ITfInsertAtSelection* insert_;
        std::wstring text_;
    };

    auto* session = new CommitSession(this, insert, text);
    HRESULT session_hr = E_FAIL;
    hr = context->RequestEditSession(client_id_, session, TF_ES_SYNC | TF_ES_READWRITE, &session_hr);
    session->Release();
    insert->Release();
    context->Release();
    return FAILED(hr) ? hr : session_hr;
}

HRESULT TextService::StartCompositionSession(TextService* self, TfEditCookie ec) {
    ITfContext* context = nullptr;
    HRESULT hr = self->current_context(&context);
    if (FAILED(hr)) return hr;
    hr = self->ensure_composition(ec, context);
    if (SUCCEEDED(hr)) {
        hr = self->set_composition_text(ec, utf8_to_wide(self->backend_.preedit()));
    }
    context->Release();
    return hr;
}

HRESULT TextService::UpdateCompositionSession(TextService* self, TfEditCookie ec) {
    return self->set_composition_text(ec, utf8_to_wide(self->backend_.preedit()));
}

HRESULT TextService::CommitCompositionSession(TextService* self, TfEditCookie ec) {
    tsf_log("CommitCompositionSession begin");
    std::wstring commit = utf8_to_wide(self->backend_.commit_selected());
    tsf_log("CommitCompositionSession after commit_selected");
    self->romaji_.reset();
    tsf_log("CommitCompositionSession before end_composition");
    self->end_composition(ec);
    tsf_log("CommitCompositionSession after end_composition");
    if (commit.empty()) {
        tsf_log("CommitCompositionSession empty commit");
        return S_OK;
    }
    ITfContext* context = nullptr;
    HRESULT hr = self->current_context(&context);
    tsf_log("CommitCompositionSession after current_context");
    if (FAILED(hr)) return hr;
    ITfInsertAtSelection* insert = nullptr;
    hr = context->QueryInterface(IID_ITfInsertAtSelection, reinterpret_cast<void**>(&insert));
    if (FAILED(hr)) {
        tsf_log("CommitCompositionSession QI ITfInsertAtSelection failed");
        context->Release();
        return hr;
    }
    tsf_log("CommitCompositionSession before InsertTextAtSelection");
    ITfRange* range = nullptr;
    hr = insert->InsertTextAtSelection(ec, TF_IAS_NOQUERY, commit.c_str(),
                                       static_cast<LONG>(commit.size()), &range);
    tsf_log("CommitCompositionSession after InsertTextAtSelection");
    if (range) range->Release();
    insert->Release();
    context->Release();
    tsf_log("CommitCompositionSession end");
    return hr;
}

} // namespace

STDAPI DllCanUnloadNow() {
    return g_dllRefCount.load() == 0 ? S_OK : S_FALSE;
}

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv) {
    if (rclsid != CLSID_NewImeTextService) return CLASS_E_CLASSNOTAVAILABLE;
    auto* factory = new ClassFactory();
    HRESULT hr = factory->QueryInterface(riid, ppv);
    factory->Release();
    return hr;
}

STDAPI DllRegisterServer() {
    HRESULT hr = CoInitialize(nullptr);
    bool coinit = SUCCEEDED(hr);

    std::wstring clsid = guid_to_string(CLSID_NewImeTextService);
    std::wstring dll_path = module_path(g_hInstance);
    std::wstring clsid_key = std::wstring(kClsidRoot) + clsid;
    std::wstring inproc_key = clsid_key + L"\\InprocServer32";

    LONG rc = set_reg_string(HKEY_CURRENT_USER, clsid_key, nullptr, kDescription);
    if (rc == ERROR_SUCCESS) {
        rc = set_reg_string(HKEY_CURRENT_USER, inproc_key, nullptr, dll_path);
    }
    if (rc == ERROR_SUCCESS) {
        rc = set_reg_string(HKEY_CURRENT_USER, inproc_key, L"ThreadingModel", kThreadingModel);
    }
    if (rc != ERROR_SUCCESS) {
        if (coinit) CoUninitialize();
        return HRESULT_FROM_WIN32(rc);
    }

    ITfInputProcessorProfiles* profiles = nullptr;
    hr = CoCreateInstance(CLSID_TF_InputProcessorProfiles, nullptr, CLSCTX_INPROC_SERVER,
                          IID_ITfInputProcessorProfiles, reinterpret_cast<void**>(&profiles));
    if (FAILED(hr)) {
        if (coinit) CoUninitialize();
        return hr;
    }

    hr = profiles->Register(CLSID_NewImeTextService);
    LANGID lang = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
    ITfInputProcessorProfileMgr* profile_mgr = nullptr;
    if (SUCCEEDED(hr)) {
        hr = profiles->QueryInterface(IID_ITfInputProcessorProfileMgr,
                                      reinterpret_cast<void**>(&profile_mgr));
    }
    if (SUCCEEDED(hr)) {
        hr = profiles->AddLanguageProfile(CLSID_NewImeTextService, lang, GUID_PROFILE_NewIme,
                                          kDescription, static_cast<ULONG>(wcslen(kDescription)),
                                          dll_path.c_str(), static_cast<ULONG>(dll_path.size()), 0);
        if (SUCCEEDED(hr)) {
            hr = profiles->EnableLanguageProfile(CLSID_NewImeTextService, lang, GUID_PROFILE_NewIme, TRUE);
        }
    }

    if (SUCCEEDED(hr) && profile_mgr) {
        hr = profile_mgr->RegisterProfile(CLSID_NewImeTextService, lang, GUID_PROFILE_NewIme,
                                          kDescription, static_cast<ULONG>(wcslen(kDescription)),
                                          dll_path.c_str(), static_cast<ULONG>(dll_path.size()),
                                          0, nullptr, 0, TRUE, 0);
    }
    if (profile_mgr) profile_mgr->Release();
    profiles->Release();

    if (SUCCEEDED(hr)) {
        ITfCategoryMgr* cat_mgr = nullptr;
        hr = CoCreateInstance(CLSID_TF_CategoryMgr, nullptr, CLSCTX_INPROC_SERVER,
                              IID_ITfCategoryMgr, reinterpret_cast<void**>(&cat_mgr));
        if (SUCCEEDED(hr)) {
            cat_mgr->RegisterCategory(CLSID_NewImeTextService, GUID_TFCAT_TIP_KEYBOARD, CLSID_NewImeTextService);
            cat_mgr->RegisterCategory(CLSID_NewImeTextService, GUID_TFCAT_TIPCAP_INPUTMODECOMPARTMENT,
                                      CLSID_NewImeTextService);
            cat_mgr->Release();
        }
    }

    if (coinit) CoUninitialize();
    return hr;
}

STDAPI DllUnregisterServer() {
    HRESULT hr = CoInitialize(nullptr);
    bool coinit = SUCCEEDED(hr);
    ITfInputProcessorProfiles* profiles = nullptr;
    hr = CoCreateInstance(CLSID_TF_InputProcessorProfiles, nullptr, CLSCTX_INPROC_SERVER,
                          IID_ITfInputProcessorProfiles, reinterpret_cast<void**>(&profiles));
    if (SUCCEEDED(hr)) {
        LANGID lang = MAKELANGID(LANG_JAPANESE, SUBLANG_JAPANESE_JAPAN);
        profiles->RemoveLanguageProfile(CLSID_NewImeTextService, lang, GUID_PROFILE_NewIme);
        ITfInputProcessorProfileMgr* profile_mgr = nullptr;
        if (SUCCEEDED(profiles->QueryInterface(IID_ITfInputProcessorProfileMgr,
                                              reinterpret_cast<void**>(&profile_mgr)))) {
            profile_mgr->UnregisterProfile(CLSID_NewImeTextService, lang, GUID_PROFILE_NewIme, 0);
            profile_mgr->Release();
        }
        profiles->Unregister(CLSID_NewImeTextService);
        profiles->Release();
    }
    std::wstring clsid = guid_to_string(CLSID_NewImeTextService);
    delete_reg_tree_if_exists(HKEY_CURRENT_USER, std::wstring(kClsidRoot) + clsid);
    if (coinit) CoUninitialize();
    return S_OK;
}

BOOL APIENTRY DllMain(HINSTANCE hinstDLL, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hInstance = hinstDLL;
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}
