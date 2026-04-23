//! Background worker for running `EngineSession::convert` off the UI thread.
//!
//! Rationale: running the full CTC beam + KenLM inside `OnKeyDown` blocks
//! TSF for ~30-40 ms, which is noticeable while typing. Instead we update
//! the preedit to raw hiragana immediately (no inference needed) and queue
//! a conversion job; when the worker finishes, it posts `WM_APP_CONVERT_DONE`
//! to a hidden msg-only window owned by the service, and its WndProc triggers
//! a second edit session that swaps the preedit to the kanji top-1.
//!
//! Clobbering: only the *latest* request matters. The worker drains any
//! queued requests and processes only the tail, so a burst of keystrokes
//! produces one ONNX run, not N.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use new_ime_engine_core::EngineSession;
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::PostMessageW;

/// Message posted by the worker when a conversion result is ready.
/// `wparam` carries the request id; `lparam` is unused.
pub const WM_APP_CONVERT_DONE: u32 = 0x8000 + 1;

#[derive(Clone)]
pub struct Request {
    pub req_id: u64,
    pub context: String,
    pub reading: String,
    /// When the UI dispatched this job. The worker uses this to pace the
    /// post-back so the user gets a fixed perceptual delay between
    /// keystroke and the kanji preedit appearing (see
    /// `PERCEPTUAL_DELAY_MS` in `worker_loop`).
    pub dispatched_at: Instant,
}

pub struct Result {
    pub req_id: u64,
    pub reading: String,
    pub candidates: Vec<String>,
}

struct Shared {
    // Latest worker output (single slot). Older results are overwritten.
    latest: Mutex<Option<Result>>,
    // HWND stored as raw isize (sendable); 0 means "not set yet".
    notify_hwnd_raw: Mutex<isize>,
    alive: AtomicBool,
}

pub struct AsyncConverter {
    /// `Option` so `Drop` can take the sender, drop it, and unblock the
    /// worker's `rx.recv()` before `join()`. Without this the drop order
    /// was: `alive=false` → worker still blocked on recv → `join()`
    /// hangs forever (sender still alive on `self`) → Deactivate stuck
    /// → TSF hard-kills the host. That crash reproduced on rapid IME
    /// switches.
    tx: Option<mpsc::Sender<Request>>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl AsyncConverter {
    pub fn spawn(session: Arc<Mutex<EngineSession>>) -> Self {
        let (tx, rx) = mpsc::channel::<Request>();
        let shared = Arc::new(Shared {
            latest: Mutex::new(None),
            notify_hwnd_raw: Mutex::new(0),
            alive: AtomicBool::new(true),
        });
        let worker_shared = shared.clone();
        let worker = std::thread::Builder::new()
            .name("new-ime-convert".into())
            .spawn(move || worker_loop(rx, session, worker_shared))
            .expect("spawn convert worker");
        Self {
            tx: Some(tx),
            shared,
            worker: Some(worker),
        }
    }

    /// Point the worker at the msg-only window that should receive
    /// `WM_APP_CONVERT_DONE`. Must be called once after the window is
    /// created.
    pub fn set_notify_hwnd(&self, hwnd: HWND) {
        *self.shared.notify_hwnd_raw.lock().unwrap() = hwnd.0 as isize;
    }

    pub fn dispatch(&self, req: Request) {
        // Best-effort; if the worker is already shutting down (tx dropped
        // in our own `Drop`, or the channel was closed) we silently drop.
        if let Some(ref tx) = self.tx {
            let _ = tx.send(req);
        }
    }

    pub fn take_latest(&self) -> Option<Result> {
        self.shared.latest.lock().ok()?.take()
    }
}

impl Drop for AsyncConverter {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::SeqCst);
        // CRITICAL: take & drop the sender BEFORE joining. Otherwise the
        // worker's `rx.recv()` stays blocked because a sender is still
        // alive on `self`, and the join below hangs forever.
        self.tx.take();
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    rx: mpsc::Receiver<Request>,
    session: Arc<Mutex<EngineSession>>,
    shared: Arc<Shared>,
) {
    while shared.alive.load(Ordering::SeqCst) {
        // Block on the next request.
        let first = match rx.recv() {
            Ok(r) => r,
            Err(_) => return, // sender dropped
        };
        let mut latest = first;
        while let Ok(next) = rx.try_recv() {
            latest = next;
        }

        if latest.reading.is_empty() {
            continue;
        }
        let candidates = match session.lock() {
            Ok(mut guard) => guard
                .convert(&latest.context, &latest.reading)
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        };

        // Perceptual-delay pacing: humans can't cleanly register a
        // preedit that flips from hiragana to kanji in <~80 ms. We pad
        // the time between dispatch and PostMessage out to
        // `PERCEPTUAL_DELAY_MS` so the user always sees their raw
        // hiragana first, then the kanji rendering lands after a
        // predictable beat. If ONNX already took longer than the budget
        // we post immediately.
        const PERCEPTUAL_DELAY_MS: u64 = 100;
        let budget = Duration::from_millis(PERCEPTUAL_DELAY_MS);
        if let Some(left) = budget.checked_sub(latest.dispatched_at.elapsed()) {
            if !left.is_zero() {
                thread::sleep(left);
            }
        }

        // Hand the result to the UI thread.
        {
            let mut slot = shared.latest.lock().unwrap();
            *slot = Some(Result {
                req_id: latest.req_id,
                reading: latest.reading.clone(),
                candidates,
            });
        }
        let hwnd_raw = *shared.notify_hwnd_raw.lock().unwrap();
        if hwnd_raw != 0 {
            // SAFETY: the service's Deactivate tears the HWND down and
            // replaces `notify_hwnd_raw` with 0 before dropping the
            // AsyncConverter, so this handle is valid while we see it.
            unsafe {
                let _ = PostMessageW(
                    HWND(hwnd_raw as *mut core::ffi::c_void),
                    WM_APP_CONVERT_DONE,
                    WPARAM(latest.req_id as usize),
                    LPARAM(0),
                );
            }
        }
    }
}
