//! Phase state machine with **async** live conversion.
//!
//! Threading model:
//!   * UI thread: advances `ComposingText` and fires an immediate preedit
//!     update with the RAW hiragana (no blocking inference).
//!   * Worker thread (`AsyncConverter`): runs prefix beam + KenLM for each
//!     dispatch, then `PostMessage`s `WM_APP_CONVERT_DONE` to the notify
//!     HWND owned by the service. That wakes the UI thread, which calls
//!     [`EngineBridge::apply_async_result`] and fires a follow-up edit
//!     session to swap the preedit to the kanji top-1.
//!
//! Phases:
//!   * `Empty`      — no composition
//!   * `Composing`  — typing; candidates is whatever the last async run
//!                    delivered (may be empty while the first job is in
//!                    flight). Preedit shows `candidates[0]` if present,
//!                    else raw hiragana + pending romaji.
//!   * `Converting` — Space was pressed; the candidate window is visible.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use new_ime_engine_core::EngineSession;

use crate::async_convert::{AsyncConverter, Request as ConvertReq};
use crate::composing_text::ComposingText;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    UpdatePreedit { text: String },
    Commit { text: String },
    EndComposition,
    ShowCandidates { list: Vec<String>, selected: usize },
    HideCandidates,
}

pub struct KeyOutcome {
    pub consumed: bool,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Copy)]
pub enum KeyKind {
    Char(char),
    Backspace,
    Enter,
    Escape,
    Space,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Comma,
    Period,
    Exclaim,
    Question,
}

enum Phase {
    Empty,
    Composing {
        candidates: Vec<String>,
    },
    Converting {
        candidates: Vec<String>,
        selected: usize,
    },
    /// Temporary raw-ASCII mode. Entered via Shift+letter, exited by
    /// Enter (commits buffer) or Escape (discards). Mirrors MS-IME /
    /// karukan: letters (case preserved), digits, space and `-` append
    /// verbatim; any other key commits the buffer first and then re-
    /// enters the normal flow.
    Alphabet {
        buffer: String,
    },
}

pub struct EngineBridge {
    composing: ComposingText,
    phase: Phase,
    engine: Option<Arc<Mutex<EngineSession>>>,
    async_conv: Option<Arc<AsyncConverter>>,
    context: String,
    req_counter: AtomicU64,
    latest_req_id: u64,
    /// Reading that the current `Phase::Composing.candidates` matches.
    /// Prevents redundant dispatches while the user types mid-kana and the
    /// hiragana portion is unchanged.
    last_applied_reading: String,
    /// Kanji rendering of the last successfully-converted **prefix** of
    /// hiragana (i.e. `hiragana[..H-TAIL_KEEP]`). Used to render the
    /// preedit as `cached_kanji + raw_tail + pending` instantly on every
    /// keystroke — the user's just-typed characters stay visible as
    /// hiragana while earlier chars settle into kanji once the worker
    /// returns.
    cached_prefix_reading: String,
    cached_prefix_kanji: String,
}

impl EngineBridge {
    pub fn new(
        engine: Option<Arc<Mutex<EngineSession>>>,
        async_conv: Option<Arc<AsyncConverter>>,
    ) -> Self {
        Self {
            composing: ComposingText::new(),
            phase: Phase::Empty,
            engine,
            async_conv,
            context: String::new(),
            req_counter: AtomicU64::new(0),
            latest_req_id: 0,
            last_applied_reading: String::new(),
            cached_prefix_reading: String::new(),
            cached_prefix_kanji: String::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.phase, Phase::Empty)
    }

    const CONTEXT_WINDOW: usize = 80;

    fn append_context(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.context.push_str(text);
        let n = self.context.chars().count();
        if n > Self::CONTEXT_WINDOW {
            let skip = n - Self::CONTEXT_WINDOW;
            self.context = self.context.chars().skip(skip).collect();
        }
    }

    pub fn would_consume(&self, kind: KeyKind) -> bool {
        match kind {
            KeyKind::Char(c) => c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '-',
            KeyKind::Comma | KeyKind::Period | KeyKind::Exclaim | KeyKind::Question => true,
            KeyKind::Backspace
            | KeyKind::Enter
            | KeyKind::Space
            | KeyKind::Escape
            | KeyKind::Left
            | KeyKind::Right
            | KeyKind::Up
            | KeyKind::Down
            | KeyKind::PageUp
            | KeyKind::PageDown => self.is_active(),
        }
    }

    /// Called from the notify-HWND WndProc when the worker posts a
    /// completed convert. We accept the result as long as its `reading`
    /// is still a **prefix** of the current hiragana — if the user kept
    /// typing while the worker was running, the newly-typed tail will
    /// show as raw kana after the cached prefix (no whole-string
    /// flicker). If the user back-spaced past the cached prefix the
    /// result is dropped.
    pub fn apply_async_result(&mut self) -> Option<String> {
        let conv = self.async_conv.as_ref()?;
        let result = conv.take_latest()?;
        if !matches!(self.phase, Phase::Composing { .. }) {
            return None;
        }
        let current_hira = self.composing.hiragana().to_string();
        if !current_hira.starts_with(&result.reading) {
            return None;
        }
        let top = result.candidates.first().cloned().unwrap_or_default();
        if top.is_empty() {
            return None;
        }
        if let Phase::Composing { candidates } = &mut self.phase {
            *candidates = result.candidates.clone();
        }
        self.cached_prefix_reading = result.reading.clone();
        self.cached_prefix_kanji = top;
        self.last_applied_reading = current_hira;
        Some(self.live_preedit())
    }

    /// Shift+letter: enter (or extend) the temporary `Alphabet` phase.
    /// Commits any pending hiragana/kanji composition first so the raw
    /// ASCII buffer is cleanly separated from what came before.
    pub fn handle_shifted_alpha(&mut self, c: char) -> KeyOutcome {
        if !c.is_ascii_alphabetic() {
            return not_consumed();
        }
        let mut actions = match &self.phase {
            Phase::Empty => Vec::new(),
            Phase::Composing { .. } => {
                let outcome = self.handle_enter();
                outcome.actions
            }
            Phase::Converting { .. } => {
                let mut a = self.commit_converting();
                a.push(Action::HideCandidates);
                a
            }
            Phase::Alphabet { buffer } => {
                // Already in alphabet mode — just append.
                let mut buf = buffer.clone();
                buf.push(c);
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                return KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                };
            }
        };
        let buffer = c.to_string();
        let preedit = buffer.clone();
        self.phase = Phase::Alphabet { buffer };
        actions.push(Action::UpdatePreedit { text: preedit });
        KeyOutcome {
            consumed: true,
            actions,
        }
    }

    pub fn handle_char(&mut self, c: char) -> KeyOutcome {
        if !c.is_ascii_alphabetic() && !c.is_ascii_digit() && c != '-' {
            return not_consumed();
        }
        // In Alphabet phase, letters / digits / hyphen append verbatim
        // (case preserved) to the buffer.
        if let Phase::Alphabet { buffer } = &self.phase {
            let mut buf = buffer.clone();
            buf.push(c);
            let preedit = buf.clone();
            self.phase = Phase::Alphabet { buffer: buf };
            return KeyOutcome {
                consumed: true,
                actions: vec![Action::UpdatePreedit { text: preedit }],
            };
        }
        match &self.phase {
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.composing.reset();
                self.composing.input_char(c);
                self.phase = Phase::Composing {
                    candidates: Vec::new(),
                };
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            _ => {
                self.composing.input_char(c);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_backspace(&mut self) -> KeyOutcome {
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.pop();
                if buf.is_empty() {
                    self.phase = Phase::Empty;
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. } => {
                self.composing.delete_left();
                if self.composing.is_empty() {
                    self.phase = Phase::Empty;
                    self.last_applied_reading.clear();
                    self.cached_prefix_reading.clear();
                    self.cached_prefix_kanji.clear();
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
        }
    }

    pub fn handle_space(&mut self) -> KeyOutcome {
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let mut buf = buffer.clone();
                buf.push(' ');
                let preedit = buf.clone();
                self.phase = Phase::Alphabet { buffer: buf };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => self.handle_down(),
            Phase::Composing { .. } => {
                let list = self.refresh_candidates_full();
                if list.is_empty() {
                    return self.handle_enter();
                }
                let selected = 0;
                let first = list.first().cloned().unwrap_or_default();
                self.phase = Phase::Converting {
                    candidates: list.clone(),
                    selected,
                };
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::UpdatePreedit { text: first },
                        Action::ShowCandidates { list, selected },
                    ],
                }
            }
        }
    }

    pub fn handle_enter(&mut self) -> KeyOutcome {
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { buffer } => {
                let text = buffer.clone();
                self.phase = Phase::Empty;
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                actions.push(Action::HideCandidates);
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
            Phase::Composing { candidates } => {
                let cand_top = candidates.first().cloned();
                let text = match cand_top {
                    Some(t) if !t.is_empty() => {
                        let _ = self.composing.commit();
                        t
                    }
                    _ => self.composing.commit(),
                };
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                if text.is_empty() {
                    return KeyOutcome {
                        consumed: true,
                        actions: vec![Action::EndComposition],
                    };
                }
                self.append_context(&text);
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::Commit { text }],
                }
            }
        }
    }

    pub fn handle_escape(&mut self) -> KeyOutcome {
        match &self.phase {
            Phase::Empty => not_consumed(),
            Phase::Alphabet { .. } => {
                self.phase = Phase::Empty;
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::EndComposition],
                }
            }
            Phase::Converting { .. } => {
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![
                        Action::HideCandidates,
                        Action::UpdatePreedit { text: preedit },
                    ],
                }
            }
            Phase::Composing { .. } => {
                self.composing.reset();
                self.phase = Phase::Empty;
                self.last_applied_reading.clear();
                self.cached_prefix_reading.clear();
                self.cached_prefix_kanji.clear();
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::EndComposition],
                }
            }
        }
    }

    pub fn handle_move_cursor(&mut self, delta: i32) -> KeyOutcome {
        match &self.phase {
            Phase::Composing { .. } => {
                self.composing.move_cursor(delta);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            _ => not_consumed(),
        }
    }

    pub fn handle_up(&mut self) -> KeyOutcome {
        self.move_selection(-1)
    }
    pub fn handle_down(&mut self) -> KeyOutcome {
        self.move_selection(1)
    }
    pub fn handle_page_up(&mut self) -> KeyOutcome {
        self.move_selection(-10)
    }
    pub fn handle_page_down(&mut self) -> KeyOutcome {
        self.move_selection(10)
    }

    fn move_selection(&mut self, delta: i32) -> KeyOutcome {
        let Phase::Converting {
            candidates,
            selected,
        } = &mut self.phase
        else {
            return not_consumed();
        };
        let n = candidates.len();
        if n == 0 {
            return not_consumed();
        }
        let cur = *selected as i32;
        let new_sel = ((cur + delta).rem_euclid(n as i32)) as usize;
        *selected = new_sel;
        let text = candidates[new_sel].clone();
        let list_snap = candidates.clone();
        KeyOutcome {
            consumed: true,
            actions: vec![
                Action::UpdatePreedit { text },
                Action::ShowCandidates {
                    list: list_snap,
                    selected: new_sel,
                },
            ],
        }
    }

    pub fn handle_punctuation(&mut self, kana: &str) -> KeyOutcome {
        // In Alphabet mode punctuation commits the buffer (same as Enter)
        // and then starts a fresh punctuation composition. Matches MS-IME:
        // typing `!` while in temp-alphabet mode ends it.
        if let Phase::Alphabet { buffer } = &self.phase {
            let text = buffer.clone();
            self.phase = Phase::Empty;
            let mut actions = Vec::new();
            if !text.is_empty() {
                self.append_context(&text);
                actions.push(Action::Commit { text });
            }
            self.composing.reset();
            self.composing.flush_romaji_and_insert(kana);
            let candidates = self.dispatch_live();
            let preedit = self.live_preedit();
            let _ = &candidates;
            self.phase = Phase::Composing { candidates };
            actions.push(Action::UpdatePreedit { text: preedit });
            return KeyOutcome {
                consumed: true,
                actions,
            };
        }
        match &self.phase {
            Phase::Alphabet { .. } => unreachable!(), // handled above
            Phase::Empty | Phase::Composing { .. } => {
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                KeyOutcome {
                    consumed: true,
                    actions: vec![Action::UpdatePreedit { text: preedit }],
                }
            }
            Phase::Converting { .. } => {
                let mut actions = self.commit_converting();
                self.composing.reset();
                self.composing.flush_romaji_and_insert(kana);
                let candidates = self.dispatch_live();
                let preedit = self.live_preedit();
                let _ = &candidates;
                self.phase = Phase::Composing { candidates };
                actions.push(Action::HideCandidates);
                actions.push(Action::UpdatePreedit { text: preedit });
                KeyOutcome {
                    consumed: true,
                    actions,
                }
            }
        }
    }

    /// Fire an async convert job for the current full hiragana. Returns
    /// the candidate list currently held by the Composing phase (possibly
    /// stale — the keystroke always renders immediately with raw hiragana
    /// and the kanji version lands ~100 ms later).
    fn dispatch_live(&mut self) -> Vec<String> {
        let hira = self.composing.hiragana().to_string();
        if hira.is_empty() {
            self.last_applied_reading.clear();
            self.cached_prefix_reading.clear();
            self.cached_prefix_kanji.clear();
            return Vec::new();
        }
        // Keep the cache even when the user backspaces — `live_preedit`
        // handles the shrink case by trimming the cached kanji tail,
        // which preserves visual continuity until the next async result
        // lands. We only drop the cache explicitly (in commit / escape /
        // empty-composition paths) elsewhere.
        if hira != self.cached_prefix_reading {
            if let Some(conv) = &self.async_conv {
                let req_id = self.req_counter.fetch_add(1, Ordering::Relaxed) + 1;
                self.latest_req_id = req_id;
                conv.dispatch(ConvertReq {
                    req_id,
                    context: self.context.clone(),
                    reading: hira,
                    dispatched_at: Instant::now(),
                });
            }
        }
        if let Phase::Composing { candidates } = &self.phase {
            candidates.clone()
        } else {
            Vec::new()
        }
    }

    /// Synchronous full-beam fetch used by Space only.
    fn refresh_candidates_full(&mut self) -> Vec<String> {
        let reading = self.composing.hiragana().to_string();
        if reading.is_empty() {
            return Vec::new();
        }
        self.last_applied_reading = reading.clone();
        self.run_convert(&reading)
    }

    /// Preedit for the current composition. Three cases:
    ///
    /// * **Grow** (`hira.starts_with(cached_reading)`, user kept typing):
    ///   `cached_kanji + (hira - cached_reading) + pending`. Everything
    ///   already converted stays kanji; only the delta is raw kana.
    ///
    /// * **Shrink** (`cached_reading.starts_with(hira)`, user back-spaced):
    ///   trim trailing chars from `cached_kanji` by the delta count in
    ///   chars. This is an approximation (1 hiragana char ≠ 1 kanji char
    ///   in general), but it keeps the preedit visually continuous until
    ///   the async convert returns and fixes up the cache. Avoids the
    ///   `kanji → raw → kanji` flash that the naive "drop cache" version
    ///   caused.
    ///
    /// * **Diverge** (cache unrelated to current hira): fall back to raw
    ///   `hira + pending`.
    fn live_preedit(&self) -> String {
        let hira = self.composing.hiragana();
        let pending = self.composing.pending_romaji();
        if hira.is_empty() {
            return pending.to_string();
        }
        let cached_r = &self.cached_prefix_reading;
        let cached_k = &self.cached_prefix_kanji;
        if !cached_r.is_empty() && !cached_k.is_empty() {
            if hira.starts_with(cached_r.as_str()) {
                let tail = &hira[cached_r.len()..];
                return format!("{}{}{}", cached_k, tail, pending);
            }
            if cached_r.starts_with(hira) {
                let shrink_chars = cached_r
                    .chars()
                    .count()
                    .saturating_sub(hira.chars().count());
                let kanji_chars: Vec<char> = cached_k.chars().collect();
                let keep = kanji_chars.len().saturating_sub(shrink_chars);
                let trimmed: String = kanji_chars[..keep].iter().collect();
                return format!("{}{}", trimmed, pending);
            }
        }
        format!("{}{}", hira, pending)
    }

    fn run_convert(&self, reading: &str) -> Vec<String> {
        let Some(engine) = &self.engine else {
            return Vec::new();
        };
        let Ok(mut guard) = engine.lock() else {
            return Vec::new();
        };
        match guard.convert(&self.context, reading) {
            Ok(list) => list,
            Err(e) => {
                tracing::warn!("engine convert failed: {}", e);
                Vec::new()
            }
        }
    }

    fn commit_converting(&mut self) -> Vec<Action> {
        let Phase::Converting {
            candidates,
            selected,
        } = &self.phase
        else {
            return vec![];
        };
        let text = candidates.get(*selected).cloned().unwrap_or_default();
        self.composing.reset();
        self.last_applied_reading.clear();
        self.cached_prefix_reading.clear();
        self.cached_prefix_kanji.clear();
        if !text.is_empty() {
            self.append_context(&text);
            vec![Action::Commit { text }]
        } else {
            vec![Action::EndComposition]
        }
    }
}

fn not_consumed() -> KeyOutcome {
    KeyOutcome {
        consumed: false,
        actions: vec![],
    }
}
