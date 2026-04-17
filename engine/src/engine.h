#pragma once

/// fcitx5 engine plugin for new-ime.
///
/// This is the main entry point registered with fcitx5.
/// On Linux with fcitx5 installed, this compiles into a shared library
/// loaded by fcitx5 at runtime. The actual implementation uses fcitx5 headers;
/// this file defines the interface that will be connected to fcitx5 APIs.
///
/// Build dependency: fcitx5 (Fcitx5Core, Fcitx5Config)
///
/// Key fcitx5 classes to inherit:
///   - InputMethodEngineV2 → NewImeEngine
///   - InputContextProperty → NewImeState
///   - CommonCandidateList → NewImeCandidateList
///   - CandidateWord → NewImeCandidateWord
///   - AddonFactory → NewImeEngineFactory

// When building without fcitx5 (e.g., for testing), we provide stub declarations.
// The actual fcitx5 integration is guarded by HAS_FCITX5.

#ifdef HAS_FCITX5

#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextproperty.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputpanel.h>
#include <fcitx-utils/i18n.h>

#include "composing_text.h"
#include "preedit.h"
#include "server_connector.h"

namespace newime {

class NewImeEngine;

class NewImeState : public fcitx::InputContextProperty {
public:
    NewImeState(NewImeEngine* engine, fcitx::InputContext* ic);

    void keyEvent(fcitx::KeyEvent& event);
    void reset();
    void commitPreedit();

private:
    enum class Mode { Direct, Composing, CandidateSelection };

    void handleDirectMode(fcitx::KeyEvent& event);
    void handleComposingMode(fcitx::KeyEvent& event);
    void handleCandidateMode(fcitx::KeyEvent& event);

    void showCandidateList();
    void updatePreedit();

    NewImeEngine* engine_;
    fcitx::InputContext* ic_;
    Mode mode_ = Mode::Direct;
    ComposingText composing_;
    Preedit preedit_;
    std::vector<Candidate> candidates_;
    int selected_index_ = 0;
    std::string left_context_;
};

class NewImeEngine : public fcitx::InputMethodEngineV2 {
public:
    NewImeEngine(fcitx::Instance* instance);

    void keyEvent(const fcitx::InputMethodEntry& entry,
                  fcitx::KeyEvent& event) override;
    void activate(const fcitx::InputMethodEntry& entry,
                  fcitx::InputContextEvent& event) override;
    void deactivate(const fcitx::InputMethodEntry& entry,
                    fcitx::InputContextEvent& event) override;
    void save() override;

    ServerConnector& server() { return server_; }

private:
    fcitx::Instance* instance_;
    fcitx::FactoryFor<NewImeState> factory_;
    ServerConnector server_;
};

class NewImeEngineFactory : public fcitx::AddonFactory {
public:
    fcitx::AddonInstance* create(fcitx::AddonManager* manager) override {
        return new NewImeEngine(manager->instance());
    }
};

} // namespace newime

#else // !HAS_FCITX5

// Stub mode for testing without fcitx5 headers
namespace newime {

class NewImeEngine {
public:
    ServerConnector& server() { return server_; }
private:
    ServerConnector server_;
};

} // namespace newime

#endif // HAS_FCITX5
