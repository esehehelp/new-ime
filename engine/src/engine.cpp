#ifdef HAS_FCITX5

#include "engine.h"

namespace newime {

// ---- NewImeState ----

NewImeState::NewImeState(NewImeEngine* engine, fcitx::InputContext* ic)
    : engine_(engine), ic_(ic) {}

void NewImeState::keyEvent(fcitx::KeyEvent& event) {
    switch (mode_) {
    case Mode::Direct:
        handleDirectMode(event);
        break;
    case Mode::Composing:
        handleComposingMode(event);
        break;
    case Mode::CandidateSelection:
        handleCandidateMode(event);
        break;
    }
}

void NewImeState::handleDirectMode(fcitx::KeyEvent& event) {
    auto key = event.key();

    // Only handle printable ASCII for romaji input
    if (key.isSimple() && !event.isRelease()) {
        char c = static_cast<char>(key.sym());
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            composing_.input_char(c);
            mode_ = Mode::Composing;
            updatePreedit();
            event.filterAndAccept();
            return;
        }
    }
}

void NewImeState::handleComposingMode(fcitx::KeyEvent& event) {
    if (event.isRelease()) return;

    auto key = event.key();

    // Space/Return → request conversion
    if (key.check(FcitxKey_space) || key.check(FcitxKey_Return)) {
        if (!composing_.empty()) {
            showCandidateList();
            event.filterAndAccept();
            return;
        }
    }

    // Escape → cancel composing
    if (key.check(FcitxKey_Escape)) {
        composing_.reset();
        preedit_.reset();
        mode_ = Mode::Direct;
        ic_->inputPanel().reset();
        ic_->updatePreedit();
        ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        event.filterAndAccept();
        return;
    }

    // Backspace → delete left
    if (key.check(FcitxKey_BackSpace)) {
        composing_.delete_left();
        if (composing_.empty()) {
            mode_ = Mode::Direct;
            preedit_.reset();
            ic_->inputPanel().reset();
            ic_->updatePreedit();
            ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        } else {
            updatePreedit();
        }
        event.filterAndAccept();
        return;
    }

    // Printable characters → add to composing
    if (key.isSimple()) {
        char c = static_cast<char>(key.sym());
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            c == '-' || c == '.' || c == ',') {
            composing_.input_char(c);
            updatePreedit();
            event.filterAndAccept();
            return;
        }
    }
}

void NewImeState::handleCandidateMode(fcitx::KeyEvent& event) {
    if (event.isRelease()) return;

    auto key = event.key();

    // Number keys 1-9,0 → select candidate
    for (int i = 0; i < 10 && i < static_cast<int>(candidates_.size()); i++) {
        int num = (i + 1) % 10;  // 1,2,...,9,0
        if (key.check(FcitxKey_1 + (num == 0 ? 9 : num - 1))) {
            ic_->commitString(candidates_[i].text);
            left_context_ = candidates_[i].text;
            composing_.reset();
            preedit_.reset();
            candidates_.clear();
            mode_ = Mode::Direct;
            ic_->inputPanel().reset();
            ic_->updatePreedit();
            ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
            event.filterAndAccept();
            return;
        }
    }

    // Space → next candidate
    if (key.check(FcitxKey_space)) {
        selected_index_ = (selected_index_ + 1) % static_cast<int>(candidates_.size());
        // Update candidate list highlight
        if (auto* cl = ic_->inputPanel().candidateList()) {
            cl->setGlobalCursorIndex(selected_index_);
        }
        ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        event.filterAndAccept();
        return;
    }

    // Enter → commit first/selected candidate
    if (key.check(FcitxKey_Return)) {
        if (!candidates_.empty()) {
            ic_->commitString(candidates_[selected_index_].text);
            left_context_ = candidates_[selected_index_].text;
        }
        composing_.reset();
        preedit_.reset();
        candidates_.clear();
        mode_ = Mode::Direct;
        ic_->inputPanel().reset();
        ic_->updatePreedit();
        ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
        event.filterAndAccept();
        return;
    }

    // Escape → back to composing
    if (key.check(FcitxKey_Escape)) {
        candidates_.clear();
        mode_ = Mode::Composing;
        ic_->inputPanel().reset();
        updatePreedit();
        event.filterAndAccept();
        return;
    }
}

void NewImeState::showCandidateList() {
    candidates_ = engine_->server().convert(
        composing_.hiragana(), left_context_, 5, false);

    if (candidates_.empty()) {
        // Fallback: commit hiragana as-is
        ic_->commitString(composing_.hiragana());
        left_context_ = composing_.hiragana();
        composing_.reset();
        preedit_.reset();
        mode_ = Mode::Direct;
        return;
    }

    selected_index_ = 0;
    mode_ = Mode::CandidateSelection;

    // Show first candidate as preedit
    preedit_.set_highlighted(candidates_[0].text);

    // Build candidate list
    auto candidateList = std::make_unique<fcitx::CommonCandidateList>();
    candidateList->setPageSize(10);
    candidateList->setLayoutHint(fcitx::CandidateLayoutHint::Vertical);

    for (const auto& cand : candidates_) {
        std::unique_ptr<fcitx::CandidateWord> word =
            std::make_unique<fcitx::DisplayOnlyCandidateWord>(
                fcitx::Text(cand.text));
        candidateList->append(std::move(word));
    }

    ic_->inputPanel().setCandidateList(std::move(candidateList));

    // Update preedit
    fcitx::Text preeditText;
    preeditText.append(preedit_.text(),
                       fcitx::TextFormatFlag::HighLight);
    ic_->inputPanel().setClientPreedit(preeditText);
    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void NewImeState::updatePreedit() {
    std::string display = composing_.display();
    preedit_.set_simple(display, composing_.cursor());

    fcitx::Text preeditText;
    preeditText.append(display, fcitx::TextFormatFlag::Underline);
    preeditText.setCursor(preedit_.cursor_byte_pos());

    ic_->inputPanel().setClientPreedit(preeditText);
    ic_->updatePreedit();
    ic_->updateUserInterface(fcitx::UserInterfaceComponent::InputPanel);
}

void NewImeState::reset() {
    composing_.reset();
    preedit_.reset();
    candidates_.clear();
    mode_ = Mode::Direct;
}

void NewImeState::commitPreedit() {
    if (!composing_.empty()) {
        ic_->commitString(composing_.hiragana());
        left_context_ = composing_.hiragana();
    }
    reset();
}

// ---- NewImeEngine ----

NewImeEngine::NewImeEngine(fcitx::Instance* instance)
    : instance_(instance),
      factory_([this](fcitx::InputContext& ic) {
          return new NewImeState(this, &ic);
      }) {
    instance_->inputContextManager().registerProperty("newImeState", &factory_);
    server_.connect();
}

void NewImeEngine::keyEvent(const fcitx::InputMethodEntry& /*entry*/,
                            fcitx::KeyEvent& event) {
    auto* ic = event.inputContext();
    auto* state = ic->propertyFor(&factory_);
    state->keyEvent(event);
}

void NewImeEngine::activate(const fcitx::InputMethodEntry& /*entry*/,
                            fcitx::InputContextEvent& event) {
    auto* state = event.inputContext()->propertyFor(&factory_);
    state->reset();
}

void NewImeEngine::deactivate(const fcitx::InputMethodEntry& /*entry*/,
                              fcitx::InputContextEvent& event) {
    auto* state = event.inputContext()->propertyFor(&factory_);
    state->commitPreedit();
}

void NewImeEngine::save() {
    // Save any persistent state (e.g., learning data)
}

} // namespace newime

FCITX_ADDON_FACTORY(newime::NewImeEngineFactory);

#endif // HAS_FCITX5
