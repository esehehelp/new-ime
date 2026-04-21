//! ITfTextInputProcessor(Ex) — main TSF entry point.
//!
//! Implements the interfaces that together give us romaji → hiragana input:
//!   * `ITfTextInputProcessor(Ex)` — activation lifecycle
//!   * `ITfKeyEventSink` — key event dispatch (see `key_event_sink.rs`)
//!   * `ITfCompositionSink` — reentrant-safe composition termination
//!     (see `composition_sink.rs`)
//!
//! Shape copied from karukan-tsf (MIT/Apache-2.0).

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use new_ime_engine_core::EngineSession;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::TextServices::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::*;

use crate::async_convert::{AsyncConverter, WM_APP_CONVERT_DONE};
use crate::engine_bridge::EngineBridge;
use crate::globals::{dll_add_ref, dll_release};
use crate::tsf::candidate_window::CandidateWindow;

#[implement(
    ITfTextInputProcessorEx,
    ITfTextInputProcessor,
    ITfKeyEventSink,
    ITfCompositionSink,
    ITfDisplayAttributeProvider
)]
pub struct NewImeTextService {
    pub(crate) inner: RefCell<Inner>,
}

pub(crate) struct Inner {
    pub(crate) thread_mgr: Option<ITfThreadMgr>,
    pub(crate) client_id: u32,
    pub(crate) activate_flags: u32,
    pub(crate) keystroke_advised: bool,
    pub(crate) engine: EngineBridge,
    pub(crate) composition: Option<ITfComposition>,
    /// IME ON = romaji→hiragana conversion active; OFF = pass-through so the
    /// user can type English (including capital letters). Toggled by
    /// Hankaku/Zenkaku (VK_KANJI / VK_OEM_ATTN) or Alt+`.
    pub(crate) enabled: bool,
    /// Atom registered via ITfCategoryMgr for the input (dotted-underline)
    /// display attribute. 0 until ActivateEx succeeds.
    pub(crate) atom_input: u32,
    /// Atom for the converted (solid-underline) display attribute.
    pub(crate) atom_converted: u32,
    /// The last preedit string we actually wrote to the composition range.
    /// Consulted before every `UpdatePreedit` so a SetText that would be
    /// a no-op (new text == last text) can be skipped — some TSF hosts
    /// repaint the composition on every SetText even when the content is
    /// identical, which shows up as visible flicker.
    pub(crate) last_preedit: String,
    /// Floating candidate window. None until ActivateEx creates one.
    pub(crate) candidate_window: Option<Rc<RefCell<CandidateWindow>>>,
    /// Background ONNX worker. None in sandboxed hosts where engine load
    /// failed — live conversion then silently falls back to raw hiragana
    /// preedit with no kanji replacement.
    pub(crate) async_conv: Option<Arc<AsyncConverter>>,
    /// Msg-only HWND the worker `PostMessage`s to when a result is ready.
    /// Destroyed FIRST in Deactivate so stray async callbacks can't reach
    /// a half-torn service.
    pub(crate) notify_hwnd: HWND,
}

#[allow(clippy::new_without_default)]
impl NewImeTextService {
    pub fn new() -> Self {
        dll_add_ref();
        Self {
            inner: RefCell::new(Inner {
                thread_mgr: None,
                client_id: 0,
                activate_flags: 0,
                keystroke_advised: false,
                engine: EngineBridge::new(None, None),
                composition: None,
                enabled: true,
                atom_input: 0,
                atom_converted: 0,
                last_preedit: String::new(),
                candidate_window: None,
                async_conv: None,
                notify_hwnd: HWND::default(),
            }),
        }
    }
}

struct ModelPaths {
    onnx: PathBuf,
    /// `HashMap<domain, path>`. Contains `"general"` always when any LM
    /// was found, plus `"tech"` / `"entity"` when their bin files exist —
    /// in that case the engine activates KenLM-MoE.
    lms: std::collections::HashMap<String, PathBuf>,
}

fn find_model_paths() -> Option<ModelPaths> {
    let candidates: Vec<PathBuf> = {
        let mut v = Vec::new();
        if let Ok(env) = std::env::var("NEWIME_MODEL_DIR") {
            v.push(PathBuf::from(env));
        }
        if let Some(dll_dir) = dll_dir() {
            v.push(dll_dir.join("..").join("..").join("models"));
        }
        v.push(PathBuf::from("D:\\Dev\\new-ime\\models"));
        v
    };
    let onnx_name = "onnx/ctc-nat-30m-student-step160000.int8.onnx";
    let lm_general = "kenlm/kenlm_general_train_4gram.bin";
    let lm_tech = "kenlm/kenlm_tech_4gram.bin";
    let lm_entity = "kenlm/kenlm_entity_4gram.bin";
    for base in &candidates {
        let onnx = base.join(onnx_name);
        if !onnx.exists() {
            continue;
        }
        let mut lms = std::collections::HashMap::new();
        let g = base.join(lm_general);
        if g.exists() {
            lms.insert("general".to_string(), g);
        }
        let t = base.join(lm_tech);
        if t.exists() {
            lms.insert("tech".to_string(), t);
        }
        let e = base.join(lm_entity);
        if e.exists() {
            lms.insert("entity".to_string(), e);
        }
        return Some(ModelPaths { onnx, lms });
    }
    None
}

#[cfg(target_os = "windows")]
fn dll_dir() -> Option<PathBuf> {
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;
    use crate::globals::DLL_INSTANCE;
    let hmodule = DLL_INSTANCE.get().map(|s| s.0).unwrap_or_default();
    let mut buf = [0u16; 260];
    let len = unsafe { GetModuleFileNameW(hmodule, &mut buf) } as usize;
    if len == 0 {
        return None;
    }
    let full = String::from_utf16_lossy(&buf[..len]);
    PathBuf::from(full).parent().map(|p| p.to_path_buf())
}

#[cfg(not(target_os = "windows"))]
fn dll_dir() -> Option<PathBuf> { None }

// ---- notify HWND for async convert results ----

const NOTIFY_CLASS: PCWSTR = w!("NewImeAsyncNotify");

fn create_notify_window(service_ptr: *const core::ffi::c_void) -> HWND {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| unsafe {
        let wc = WNDCLASSW {
            lpfnWndProc: Some(notify_wnd_proc),
            lpszClassName: NOTIFY_CLASS,
            ..Default::default()
        };
        RegisterClassW(&wc);
    });
    unsafe {
        let hwnd = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            NOTIFY_CLASS,
            w!(""),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            HWND(-3isize as *mut _), // HWND_MESSAGE
            None,
            None,
            None,
        )
        .unwrap_or_default();
        if hwnd.0 as usize != 0 {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, service_ptr as isize);
        }
        hwnd
    }
}

unsafe extern "system" fn notify_wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_APP_CONVERT_DONE {
        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const NewImeTextService_Impl;
        if !ptr.is_null() {
            // Safety: the service outlives its notify HWND — Deactivate
            // tears the HWND down (and clears USERDATA) BEFORE letting COM
            // release the service.
            let svc = unsafe { &*ptr };
            on_async_done(svc);
        }
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

/// Drain the latest async result and, if still relevant, fire an edit
/// session that swaps the preedit to the kanji top-1.
fn on_async_done(svc: &NewImeTextService_Impl) {
    // Pull the result + current state out of the service under a borrow,
    // then drop the borrow before requesting the edit session so nothing
    // re-enters while the TSF callback runs.
    let update = {
        let mut inner = match svc.inner.try_borrow_mut() {
            Ok(i) => i,
            Err(_) => return, // already mid-edit; the next keystroke will try again
        };
        inner.engine.apply_async_result()
    };
    let Some(preedit) = update else { return; };
    if preedit.is_empty() {
        return;
    }
    // Dedup: if the async result would render the exact same preedit we
    // already painted, skip the entire edit session. This cuts one of
    // the common flicker sources where the cached kanji happened to
    // already match what the async worker returned.
    if svc.inner.borrow().last_preedit == preedit {
        return;
    }

    // We need a context to edit. Grab the current focused context via
    // thread_mgr (same path `current_context` uses in key_event_sink).
    let (thread_mgr, client_id, atom_input, atom_converted) = {
        let inner = svc.inner.borrow();
        (
            inner.thread_mgr.clone(),
            inner.client_id,
            inner.atom_input,
            inner.atom_converted,
        )
    };
    let Some(tm) = thread_mgr else { return; };

    let context: ITfContext = unsafe {
        let doc = match tm.GetFocus() {
            Ok(d) => d,
            Err(_) => return,
        };
        match doc.GetTop() {
            Ok(c) => c,
            Err(_) => return,
        }
    };

    // Build an edit session carrying a single UpdatePreedit action.
    use crate::engine_bridge::Action;
    use crate::tsf::edit_session::ActionEditSession;
    use std::rc::Rc as StdRc;
    let composition_snapshot = svc.inner.borrow().composition.clone();
    let composition_cell = StdRc::new(RefCell::new(composition_snapshot));
    let composition_sink: ITfCompositionSink = match svc.cast_interface() {
        Ok(s) => s,
        Err(_) => return,
    };
    let session = ActionEditSession::new(
        context.clone(),
        vec![Action::UpdatePreedit { text: preedit.clone() }],
        composition_cell.clone(),
        composition_sink,
        atom_input,
        atom_converted,
    );
    let edit_session: ITfEditSession = session.into();
    unsafe {
        let _ = context.RequestEditSession(
            client_id,
            &edit_session,
            TF_ES_SYNC | TF_ES_READWRITE,
        );
    }
    // Propagate any composition-lifecycle change back to the service and
    // remember what we just painted so the next update can dedup.
    let mut inner = svc.inner.borrow_mut();
    inner.last_preedit = preedit;
    inner.composition = composition_cell.borrow().clone();
}

fn resolve_display_atoms() -> (u32, u32) {
    use crate::globals::{GUID_DISPLAY_ATTRIBUTE_CONVERTED, GUID_DISPLAY_ATTRIBUTE_INPUT};
    use windows::Win32::System::Com::{CoCreateInstance, CLSCTX_INPROC_SERVER};
    unsafe {
        let cat_mgr: Result<ITfCategoryMgr> =
            CoCreateInstance(&CLSID_TF_CategoryMgr, None, CLSCTX_INPROC_SERVER);
        match cat_mgr {
            Ok(m) => {
                let input = m.RegisterGUID(&GUID_DISPLAY_ATTRIBUTE_INPUT).unwrap_or(0);
                let converted = m.RegisterGUID(&GUID_DISPLAY_ATTRIBUTE_CONVERTED).unwrap_or(0);
                (input, converted)
            }
            Err(e) => {
                tracing::warn!("CategoryMgr unavailable, no display attributes: {}", e);
                (0, 0)
            }
        }
    }
}

fn load_engine() -> Option<Arc<Mutex<EngineSession>>> {
    let paths = find_model_paths()?;
    let mut session = match EngineSession::load(&paths.onnx) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("engine load failed: {}", e);
            return None;
        }
    };
    // Prefer MoE when 2+ domain LMs are present; fall back to the single
    // general LM otherwise.
    if paths.lms.len() >= 2 {
        if let Err(e) = session.attach_kenlm_moe(&paths.lms) {
            tracing::warn!("kenlm-moe load failed, falling back to single: {}", e);
            if let Some(g) = paths.lms.get("general") {
                let _ = session.attach_kenlm(g);
            }
        } else {
            let keys: Vec<&String> = paths.lms.keys().collect();
            tracing::info!("kenlm-moe loaded: {:?}", keys);
        }
    } else if let Some(g) = paths.lms.get("general") {
        if let Err(e) = session.attach_kenlm(g) {
            tracing::warn!("kenlm load failed (continuing without): {}", e);
        }
    }
    tracing::info!("engine loaded from {}", paths.onnx.display());
    Some(Arc::new(Mutex::new(session)))
}

impl NewImeTextService_Impl {
    /// Expose `cast` so sibling modules can reach the other interfaces
    /// implemented on this object (e.g. key_event_sink needs the
    /// `ITfCompositionSink` interface to hand to `StartComposition`).
    pub(crate) fn cast_interface<T: windows::core::Interface>(&self) -> Result<T> {
        unsafe { self.cast() }
    }
}

impl Drop for NewImeTextService {
    fn drop(&mut self) {
        // Defensive teardown: if the service drops without a prior
        // `Deactivate` (rare but seen during rapid IME switches when TSF
        // releases the object as part of its own unwind) we still need
        // to kill the notify HWND and worker thread, otherwise:
        //   * WndProc receives WM_APP_CONVERT_DONE on a freed service →
        //     use-after-free,
        //   * worker thread stays alive past DLL unload → access
        //     violation when the loader zero-fills our code section.
        if let Ok(mut inner) = self.inner.try_borrow_mut() {
            if inner.notify_hwnd.0 as usize != 0 {
                unsafe {
                    SetWindowLongPtrW(inner.notify_hwnd, GWLP_USERDATA, 0);
                    let _ = DestroyWindow(inner.notify_hwnd);
                }
                inner.notify_hwnd = HWND::default();
            }
            inner.async_conv = None;
            inner.candidate_window = None;
        }
        dll_release();
    }
}

impl ITfTextInputProcessor_Impl for NewImeTextService_Impl {
    fn Activate(&self, ptim: Option<&ITfThreadMgr>, tid: u32) -> Result<()> {
        self.ActivateEx(ptim, tid, 0)
    }

    fn Deactivate(&self) -> Result<()> {
        // Tear the notify HWND down FIRST so any in-flight worker results
        // have nowhere to land. `async_conv` is then dropped (below) which
        // joins the worker thread.
        let (thread_mgr, client_id, advised) = {
            let mut inner = self.inner.borrow_mut();
            if inner.notify_hwnd.0 as usize != 0 {
                unsafe {
                    SetWindowLongPtrW(inner.notify_hwnd, GWLP_USERDATA, 0);
                    let _ = DestroyWindow(inner.notify_hwnd);
                }
                inner.notify_hwnd = HWND::default();
            }
            inner.async_conv = None;
            inner.engine.handle_escape();
            inner.composition = None;
            if let Some(ref cw) = inner.candidate_window {
                cw.borrow_mut().hide();
            }
            inner.candidate_window = None;
            let advised = inner.keystroke_advised;
            inner.keystroke_advised = false;
            let tm = inner.thread_mgr.take();
            let cid = inner.client_id;
            inner.client_id = 0;
            inner.activate_flags = 0;
            (tm, cid, advised)
        };

        if advised {
            if let Some(tm) = &thread_mgr {
                unsafe {
                    if let Ok(km) = tm.cast::<ITfKeystrokeMgr>() {
                        use crate::globals::GUID_PRESERVED_KEY_ONOFF;
                        let key = TF_PRESERVEDKEY {
                            uVKey: 0x19,
                            uModifiers: 0,
                        };
                        let _ = km.UnpreserveKey(&GUID_PRESERVED_KEY_ONOFF, &key);
                        let _ = km.UnadviseKeyEventSink(client_id);
                    }
                }
            }
        }
        tracing::info!("new-ime TSF Deactivate");
        Ok(())
    }
}

impl ITfTextInputProcessorEx_Impl for NewImeTextService_Impl {
    fn ActivateEx(&self, ptim: Option<&ITfThreadMgr>, tid: u32, dwflags: u32) -> Result<()> {
        let Some(thread_mgr) = ptim else {
            return Err(Error::from(windows::Win32::Foundation::E_INVALIDARG));
        };

        // Defensive: if this service instance is already active (TSF
        // reused it without a matching Deactivate — rare but possible in
        // quick switching scenarios), tear the previous activation down
        // first so we never double-create the worker / notify HWND.
        if self.inner.borrow().keystroke_advised {
            let _ = self.Deactivate();
        }

        // Engine load happens outside any borrow so the (potentially slow)
        // ONNX session setup doesn't hold the RefCell.
        let engine = load_engine();

        // Resolve display attribute atoms once per activation.
        let (atom_input, atom_converted) = resolve_display_atoms();

        // Create candidate window (hidden). Drop happens on Deactivate via
        // `Option::take` so the HWND doesn't leak across activations.
        let candidate_window = Rc::new(RefCell::new(CandidateWindow::new()));

        // Spawn the async converter + its notify window if the engine
        // loaded successfully.
        let (async_conv, notify_hwnd) = if let Some(eng) = engine.clone() {
            let conv = Arc::new(AsyncConverter::spawn(eng));
            let hwnd = create_notify_window(self as *const _ as *const core::ffi::c_void);
            conv.set_notify_hwnd(hwnd);
            (Some(conv), hwnd)
        } else {
            (None, HWND::default())
        };

        {
            let mut inner = self.inner.borrow_mut();
            inner.thread_mgr = Some(thread_mgr.clone());
            inner.client_id = tid;
            inner.activate_flags = dwflags;
            inner.engine = EngineBridge::new(engine, async_conv.clone());
            inner.atom_input = atom_input;
            inner.atom_converted = atom_converted;
            inner.candidate_window = Some(candidate_window);
            inner.async_conv = async_conv;
            inner.notify_hwnd = notify_hwnd;
        }

        unsafe {
            let key_sink: ITfKeyEventSink = self.cast_interface()?;
            let keystroke_mgr: ITfKeystrokeMgr = thread_mgr.cast()?;
            keystroke_mgr.AdviseKeyEventSink(tid, &key_sink, true)?;

            // Register Hankaku/Zenkaku as a preserved key so our
            // `OnPreservedKey` fires on a single press regardless of the
            // host's own dispatch routing. Description uses UTF-16 "IME
            // On/Off".
            use crate::globals::GUID_PRESERVED_KEY_ONOFF;
            let key = TF_PRESERVEDKEY {
                uVKey: 0x19,  // VK_KANJI (Hankaku/Zenkaku on JP keyboards)
                uModifiers: 0,
            };
            let desc: Vec<u16> = "IME On/Off".encode_utf16().collect();
            let _ =
                keystroke_mgr.PreserveKey(tid, &GUID_PRESERVED_KEY_ONOFF, &key, &desc);
        }
        {
            let mut inner = self.inner.borrow_mut();
            inner.keystroke_advised = true;
        }

        tracing::info!("new-ime TSF ActivateEx (tid={}, flags={:#x})", tid, dwflags);
        Ok(())
    }
}
