//! Win32 candidate window.
//!
//! Topmost popup child window rendered with GDI. Ported from karukan-tsf
//! `candidate/window.rs` (MIT/Apache-2.0); shape simplified to what v1
//! needs: Meiryo UI font, DPI-scaled, selection highlight, and — new —
//! 10-per-page pagination so wider beams don't drown the UI.

use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::HiDpi::GetDpiForWindow;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::*;

const PAGE_SIZE: usize = 10;
const ITEM_HEIGHT: i32 = 24;
const PADDING: i32 = 4;
const CHAR_WIDTH: i32 = 14;
const LABEL_WIDTH: i32 = 28;
const FOOTER_HEIGHT: i32 = 18;
const MIN_WIDTH: i32 = 160;
const FONT_SIZE: i32 = 16;
const FOOTER_FONT_SIZE: i32 = 12;
const CLASS_NAME: PCWSTR = w!("NewImeCandidateWindow");

#[derive(Default)]
struct RenderData {
    candidates: Vec<String>,
    selected: usize,
}

pub struct CandidateWindow {
    hwnd: HWND,
}

impl CandidateWindow {
    pub fn new() -> Self {
        register_class();
        let hwnd = create_window();
        if hwnd.0 as usize != 0 {
            let data = Box::new(RenderData::default());
            unsafe {
                SetWindowLongPtrW(hwnd, GWLP_USERDATA, Box::into_raw(data) as isize);
            }
        }
        Self { hwnd }
    }

    pub fn show(&mut self, candidates: &[String], selected: usize) {
        if self.hwnd.0 as usize == 0 {
            return;
        }
        unsafe {
            let ptr = GetWindowLongPtrW(self.hwnd, GWLP_USERDATA) as *mut RenderData;
            if !ptr.is_null() {
                (*ptr).candidates = candidates.to_vec();
                (*ptr).selected = selected;
            }
        }
        let s = dpi_scale(self.hwnd);
        let total = candidates.len();
        // Only the current page is drawn; clamp the drawn row count to the
        // number of items on that page so short last pages don't render a
        // giant blank stripe.
        let page = selected / PAGE_SIZE;
        let start = page * PAGE_SIZE;
        let visible_rows = total.saturating_sub(start).min(PAGE_SIZE).max(1);

        let height = visible_rows as i32 * scale(ITEM_HEIGHT, s)
            + scale(PADDING, s) * 2
            + if total > PAGE_SIZE { scale(FOOTER_HEIGHT, s) } else { 0 };
        let width = calc_width(candidates, s, total > PAGE_SIZE);
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                0,
                0,
                width,
                height,
                SWP_NOMOVE | SWP_NOACTIVATE,
            );
            let _ = InvalidateRect(self.hwnd, None, true);
            let _ = ShowWindow(self.hwnd, SW_SHOWNOACTIVATE);
        }
    }

    pub fn hide(&mut self) {
        if self.hwnd.0 as usize == 0 {
            return;
        }
        unsafe {
            let _ = ShowWindow(self.hwnd, SW_HIDE);
        }
    }

    pub fn move_to(&mut self, x: i32, y: i32) {
        if self.hwnd.0 as usize == 0 {
            return;
        }
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                HWND_TOPMOST,
                x,
                y,
                0,
                0,
                SWP_NOSIZE | SWP_NOACTIVATE,
            );
        }
    }
}

impl Drop for CandidateWindow {
    fn drop(&mut self) {
        if self.hwnd.0 as usize != 0 {
            unsafe {
                let ptr = GetWindowLongPtrW(self.hwnd, GWLP_USERDATA) as *mut RenderData;
                if !ptr.is_null() {
                    SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, 0);
                    let _ = Box::from_raw(ptr);
                }
                let _ = DestroyWindow(self.hwnd);
            }
            self.hwnd = HWND::default();
        }
    }
}

fn register_class() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| unsafe {
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wnd_proc),
            hbrBackground: GetSysColorBrush(COLOR_WINDOW),
            lpszClassName: CLASS_NAME,
            ..Default::default()
        };
        RegisterClassW(&wc);
    });
}

fn create_window() -> HWND {
    unsafe {
        CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            CLASS_NAME,
            w!(""),
            WS_POPUP | WS_BORDER,
            0,
            0,
            200,
            100,
            None,
            None,
            None,
            None,
        )
        .unwrap_or_default()
    }
}

fn dpi_scale(hwnd: HWND) -> f64 {
    let dpi = unsafe { GetDpiForWindow(hwnd) };
    if dpi == 0 {
        1.0
    } else {
        dpi as f64 / 96.0
    }
}

fn scale(v: i32, s: f64) -> i32 {
    (v as f64 * s).round() as i32
}

fn calc_width(candidates: &[String], s: f64, paginated: bool) -> i32 {
    let max_chars = candidates
        .iter()
        .map(|c| c.chars().count())
        .max()
        .unwrap_or(4);
    let base = max_chars as i32 * scale(CHAR_WIDTH, s)
        + scale(LABEL_WIDTH, s)
        + scale(PADDING, s) * 2;
    let footer = if paginated { scale(60, s) } else { 0 };
    base.max(scale(MIN_WIDTH, s)).max(footer)
}

unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        WM_PAINT => {
            unsafe { paint(hwnd) };
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

unsafe fn paint(hwnd: HWND) {
    unsafe {
        let s = dpi_scale(hwnd);
        let item_h = scale(ITEM_HEIGHT, s);
        let pad = scale(PADDING, s);
        let text_y = scale(3, s);
        let font_size = scale(FONT_SIZE, s);
        let footer_size = scale(FOOTER_FONT_SIZE, s);

        let mut ps = PAINTSTRUCT::default();
        let hdc = BeginPaint(hwnd, &mut ps);

        let ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const RenderData;
        if !ptr.is_null() {
            let data = &*ptr;
            let _ = SetBkMode(hdc, TRANSPARENT);

            let font = CreateFontW(
                -font_size,
                0,
                0,
                0,
                FW_NORMAL.0 as i32,
                0,
                0,
                0,
                DEFAULT_CHARSET.0 as u32,
                OUT_DEFAULT_PRECIS.0 as u32,
                CLIP_DEFAULT_PRECIS.0 as u32,
                CLEARTYPE_QUALITY.0 as u32,
                (FF_DONTCARE.0 | FIXED_PITCH.0) as u32,
                w!("Meiryo UI"),
            );
            let old_font = SelectObject(hdc, font);
            let highlight = CreateSolidBrush(COLORREF(0x00D77800));

            let total = data.candidates.len();
            let page = data.selected / PAGE_SIZE;
            let total_pages = (total + PAGE_SIZE - 1) / PAGE_SIZE;
            let start = page * PAGE_SIZE;
            let end = (start + PAGE_SIZE).min(total);

            for (draw_i, abs_i) in (start..end).enumerate() {
                let y = pad + draw_i as i32 * item_h;
                if abs_i == data.selected {
                    let rect = RECT {
                        left: 0,
                        top: y,
                        right: ps.rcPaint.right,
                        bottom: y + item_h,
                    };
                    FillRect(hdc, &rect, highlight);
                    SetTextColor(hdc, COLORREF(0x00FFFFFF));
                } else {
                    SetTextColor(hdc, COLORREF(0x00000000));
                }
                // Label is the 1-based index on the current page (1-10),
                // matching the familiar MS-IME "1. ... 0. ..." layout.
                let shown = draw_i + 1;
                let shown = if shown == 10 { 0 } else { shown };
                let label = format!("{}. {}", shown, data.candidates[abs_i]);
                let wide: Vec<u16> = label.encode_utf16().collect();
                let _ = TextOutW(hdc, pad, y + text_y, &wide);
            }

            // Footer: "page/total" when there is more than one page.
            if total_pages > 1 {
                let foot_y = pad + (end - start) as i32 * item_h;
                let foot_font = CreateFontW(
                    -footer_size,
                    0,
                    0,
                    0,
                    FW_NORMAL.0 as i32,
                    0,
                    0,
                    0,
                    DEFAULT_CHARSET.0 as u32,
                    OUT_DEFAULT_PRECIS.0 as u32,
                    CLIP_DEFAULT_PRECIS.0 as u32,
                    CLEARTYPE_QUALITY.0 as u32,
                    (FF_DONTCARE.0 | FIXED_PITCH.0) as u32,
                    w!("Meiryo UI"),
                );
                let old_foot = SelectObject(hdc, foot_font);
                SetTextColor(hdc, COLORREF(0x00707070));
                let footer = format!("{}/{}", page + 1, total_pages);
                let wide: Vec<u16> = footer.encode_utf16().collect();
                let _ = TextOutW(hdc, pad, foot_y + text_y, &wide);
                SelectObject(hdc, old_foot);
                let _ = DeleteObject(foot_font);
            }

            let _ = DeleteObject(highlight);
            SelectObject(hdc, old_font);
            let _ = DeleteObject(font);
        }
        let _ = EndPaint(hwnd, &ps);
    }
}
