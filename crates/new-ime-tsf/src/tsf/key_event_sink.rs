//! ITfKeyEventSink — turns Windows key events into engine actions.

use std::cell::RefCell;
use std::rc::Rc;

use windows::core::*;
use windows::Win32::Foundation::{BOOL, FALSE, LPARAM, TRUE, WPARAM};
use windows::Win32::UI::TextServices::*;

use crate::engine_bridge::{Action, KeyKind};
use crate::keymap::classify_vk;
use crate::tsf::edit_session::ActionEditSession;
use crate::tsf::text_input_processor::NewImeTextService_Impl;

impl ITfKeyEventSink_Impl for NewImeTextService_Impl {
    fn OnSetFocus(&self, _fforeground: BOOL) -> Result<()> {
        Ok(())
    }

    fn OnTestKeyDown(
        &self,
        _pic: Option<&ITfContext>,
        wparam: WPARAM,
        _lparam: LPARAM,
    ) -> Result<BOOL> {
        let vk = wparam.0 as u32;

        // Hankaku/Zenkaku (JP kbd) / VK_KANJI: always eat so we can toggle.
        if is_toggle_key(vk, alt_down()) {
            return Ok(TRUE);
        }

        // Ctrl/Alt/Win never consumed — preserves Ctrl+C, Alt+Tab, Win shortcuts.
        if ctrl_down() || alt_down() || win_down() {
            return Ok(FALSE);
        }

        let inner = self.inner.borrow();
        if !inner.enabled {
            return Ok(FALSE);
        }

        let shift = shift_down();
        let Some(kind) = classify_vk(vk, shift) else {
            return Ok(FALSE);
        };
        Ok(BOOL::from(inner.engine.would_consume(kind)))
    }

    fn OnKeyDown(&self, pic: Option<&ITfContext>, wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        let vk = wparam.0 as u32;

        if is_toggle_key(vk, alt_down()) {
            return self.toggle_enabled(pic);
        }

        if ctrl_down() || alt_down() || win_down() {
            return Ok(FALSE);
        }

        {
            let inner = self.inner.borrow();
            if !inner.enabled {
                return Ok(FALSE);
            }
        }

        let shift = shift_down();
        let Some(kind) = classify_vk(vk, shift) else {
            return Ok(FALSE);
        };

        let outcome = {
            let mut inner = self.inner.borrow_mut();
            match kind {
                KeyKind::Char(c) if c.is_ascii_uppercase() => {
                    // Shift+letter: enter (or extend) the temporary
                    // alphabet mode. Uppercase-only check is safe because
                    // `classify_vk` only returns uppercase when the Shift
                    // modifier was held at the time of the keypress.
                    inner.engine.handle_shifted_alpha(c)
                }
                KeyKind::Char(c) => inner.engine.handle_char(c),
                KeyKind::Backspace => inner.engine.handle_backspace(),
                KeyKind::Enter => inner.engine.handle_enter(),
                KeyKind::Escape => inner.engine.handle_escape(),
                KeyKind::Space => inner.engine.handle_space(),
                KeyKind::Left => inner.engine.handle_move_cursor(-1),
                KeyKind::Right => inner.engine.handle_move_cursor(1),
                KeyKind::Up => inner.engine.handle_up(),
                KeyKind::Down => inner.engine.handle_down(),
                KeyKind::PageUp => inner.engine.handle_page_up(),
                KeyKind::PageDown => inner.engine.handle_page_down(),
                KeyKind::Comma => inner.engine.handle_punctuation("、"),
                KeyKind::Period => inner.engine.handle_punctuation("。"),
                KeyKind::Exclaim => inner.engine.handle_punctuation("！"),
                KeyKind::Question => inner.engine.handle_punctuation("？"),
            }
        };

        if !outcome.consumed {
            return Ok(FALSE);
        }

        if let Some(context) = pic {
            apply_actions(self, context, outcome.actions)?;
            return Ok(TRUE);
        }
        Ok(BOOL::from(outcome.consumed))
    }

    fn OnTestKeyUp(
        &self,
        _pic: Option<&ITfContext>,
        _wparam: WPARAM,
        _lparam: LPARAM,
    ) -> Result<BOOL> {
        Ok(FALSE)
    }

    fn OnKeyUp(&self, _pic: Option<&ITfContext>, _wparam: WPARAM, _lparam: LPARAM) -> Result<BOOL> {
        Ok(FALSE)
    }

    fn OnPreservedKey(&self, pic: Option<&ITfContext>, rguid: *const GUID) -> Result<BOOL> {
        use crate::globals::GUID_PRESERVED_KEY_ONOFF;
        unsafe {
            if rguid.is_null() {
                return Ok(FALSE);
            }
            if *rguid == GUID_PRESERVED_KEY_ONOFF {
                return self.toggle_enabled(pic);
            }
        }
        Ok(FALSE)
    }
}

/// IME on/off toggle via the OnKeyDown path.
///   * `VK_KANJI` (0x19) — JP Hankaku/Zenkaku key (usual mapping)
///   * `VK_OEM_ATTN` / `VK_DBE_SBCSCHAR` (0xF3) — alt mapping on some JP
///     keyboards that put the same physical key on a different VK code
///   * `VK_DBE_DBCSCHAR` (0xF4) — same family
///   * `Alt+\``  (VK_OEM_3 = 0xC0 with Alt held) — portable fallback for
///     non-JP keyboards
///
/// We register 0x19 as a PreservedKey in ActivateEx too, but some hosts
/// dispatch it through OnKeyDown instead; accepting both paths avoids the
/// "needs two presses" class of bugs.
fn is_toggle_key(vk: u32, alt: bool) -> bool {
    matches!(vk, 0x19 | 0xF3 | 0xF4) || (alt && vk == 0xC0)
}

impl NewImeTextService_Impl {
    pub(crate) fn toggle_enabled(&self, pic: Option<&ITfContext>) -> Result<BOOL> {
        // Flush any in-flight composition as committed text before switching
        // modes — otherwise pending romaji vanishes silently on toggle.
        let actions = {
            let mut inner = self.inner.borrow_mut();
            let flush = inner.engine.handle_enter();
            inner.enabled = !inner.enabled;
            tracing::info!("IME toggle: enabled={}", inner.enabled);
            flush.actions
        };
        if let Some(context) = pic {
            if !actions.is_empty() {
                apply_actions(self, context, actions)?;
            }
        }
        Ok(TRUE)
    }
}

fn shift_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x10) < 0 }
}

fn ctrl_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x11) < 0 }
}

fn alt_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x12) < 0 }
}

fn win_down() -> bool {
    use windows::Win32::UI::Input::KeyboardAndMouse::GetKeyState;
    unsafe { GetKeyState(0x5B) < 0 || GetKeyState(0x5C) < 0 }
}

fn apply_actions(
    service: &NewImeTextService_Impl,
    context: &ITfContext,
    actions: Vec<Action>,
) -> Result<()> {
    // Split: edit-session-bound actions vs candidate-window-bound actions.
    // The edit session needs synchronous TSF access (TF_ES_SYNC); candidate
    // window operations are plain Win32 and run afterwards.
    //
    // Additionally: drop `UpdatePreedit` actions whose text matches the
    // last one we rendered. Some TSF hosts repaint the composition on
    // every SetText even when the content is identical, which is visible
    // flicker. Commit/End actions are never dropped (they change
    // composition lifetime, not just the rendered text).
    let mut edit_actions: Vec<Action> = Vec::with_capacity(actions.len());
    let mut window_actions: Vec<Action> = Vec::new();
    {
        let last = service.inner.borrow().last_preedit.clone();
        let mut running_last = last;
        for a in actions {
            match a {
                Action::ShowCandidates { .. } | Action::HideCandidates => window_actions.push(a),
                Action::UpdatePreedit { ref text } if *text == running_last => {
                    // No-op — skip.
                }
                Action::UpdatePreedit { ref text } => {
                    running_last = text.clone();
                    edit_actions.push(a);
                }
                _ => edit_actions.push(a),
            }
        }
    }

    let (composition_snapshot, client_id, atom_input, atom_converted, candidate_window) = {
        let inner = service.inner.borrow();
        (
            inner.composition.clone(),
            inner.client_id,
            inner.atom_input,
            inner.atom_converted,
            inner.candidate_window.clone(),
        )
    };
    let composition_cell = Rc::new(RefCell::new(composition_snapshot));
    let composition_sink: ITfCompositionSink = service.cast_interface()?;

    let session_actions_snapshot: Vec<Action> = edit_actions.clone();
    if !edit_actions.is_empty() {
        let session = ActionEditSession::new(
            context.clone(),
            edit_actions,
            composition_cell.clone(),
            composition_sink,
            atom_input,
            atom_converted,
        );
        let edit_session: ITfEditSession = session.into();
        unsafe {
            let hr = context.RequestEditSession(
                client_id,
                &edit_session,
                TF_ES_SYNC | TF_ES_READWRITE,
            )?;
            if hr.is_err() {
                tracing::warn!("RequestEditSession failed: {:?}", hr);
            }
        }
    }

    {
        let mut inner = service.inner.borrow_mut();
        inner.composition = composition_cell.borrow().clone();
        // Track the last-rendered preedit so the next apply_actions call
        // can dedup no-op SetText. Walk session_actions in order so the
        // latest UpdatePreedit / Commit wins.
        for a in &session_actions_snapshot {
            match a {
                Action::UpdatePreedit { text } => inner.last_preedit = text.clone(),
                Action::Commit { .. } | Action::EndComposition => inner.last_preedit.clear(),
                _ => {}
            }
        }
    }

    if let Some(cw) = candidate_window {
        for a in window_actions {
            match a {
                Action::ShowCandidates { list, selected } => {
                    let caret = current_caret_screen_pos(context, client_id, &composition_cell);
                    let mut w = cw.borrow_mut();
                    w.show(&list, selected);
                    if let Some((x, y)) = caret {
                        w.move_to(x, y);
                    }
                }
                Action::HideCandidates => {
                    cw.borrow_mut().hide();
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Query the composition's on-screen caret rect so the candidate window can
/// position itself just below the current preedit. Returns `None` if any
/// step fails — the caller then falls back to the last-known position.
fn current_caret_screen_pos(
    context: &ITfContext,
    client_id: u32,
    composition: &Rc<RefCell<Option<ITfComposition>>>,
) -> Option<(i32, i32)> {
    use windows::Win32::Foundation::BOOL;
    // We need a ReadOnly edit session to call GetTextExt. Define a small
    // one-shot session that captures the result into a shared slot.
    struct ExtSession {
        composition: Rc<RefCell<Option<ITfComposition>>>,
        context: ITfContext,
        result: Rc<RefCell<Option<(i32, i32)>>>,
    }
    #[implement(ITfEditSession)]
    struct Sess(ExtSession);
    impl ITfEditSession_Impl for Sess_Impl {
        fn DoEditSession(&self, ec: u32) -> Result<()> {
            unsafe {
                let comp = self.0.composition.borrow();
                let Some(composition) = comp.as_ref() else {
                    return Ok(());
                };
                let range = composition.GetRange()?;
                let view = self.0.context.GetActiveView()?;
                let mut rect = windows::Win32::Foundation::RECT::default();
                let mut clipped = BOOL::default();
                view.GetTextExt(ec, &range, &mut rect, &mut clipped)?;
                *self.0.result.borrow_mut() = Some((rect.left, rect.bottom));
            }
            Ok(())
        }
    }

    let result: Rc<RefCell<Option<(i32, i32)>>> = Rc::new(RefCell::new(None));
    let session = Sess(ExtSession {
        composition: composition.clone(),
        context: context.clone(),
        result: result.clone(),
    });
    let edit_session: ITfEditSession = session.into();
    unsafe {
        let _ = context.RequestEditSession(client_id, &edit_session, TF_ES_SYNC | TF_ES_READ);
    }
    let r = result.borrow();
    *r
}
