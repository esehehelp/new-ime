//! Global state and GUID definitions for the new-ime TSF IME.
//!
//! Structure and helpers adapted from karukan-tsf (MIT/Apache-2.0).
//! CLSID / profile GUID reused from the existing C++ new-ime-tsf so the
//! rewrite drops into the same registration slot without colliding.

#[cfg(target_os = "windows")]
use once_cell::sync::OnceCell;
#[cfg(target_os = "windows")]
use windows::core::GUID;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::HMODULE;

/// Stable CLSID for the Rust TSF service.
/// `{7498E5B1-16C1-4C16-9A7E-1F6AC4B798C1}`
#[cfg(target_os = "windows")]
pub const CLSID_NEW_IME_TEXT_SERVICE: GUID =
    GUID::from_u128(0x7498_E5B1_16C1_4C16_9A7E_1F6A_C4B7_98C1);

/// Stable language profile GUID for the Rust TSF service.
/// `{B4819601-2CF2-42A0-9A1C-8A16E1B7994A}`
#[cfg(target_os = "windows")]
pub const GUID_NEW_IME_PROFILE: GUID = GUID::from_u128(0xB481_9601_2CF2_42A0_9A1C_8A16_E1B7_994A);

/// Preedit input display attribute (underline). New GUID.
/// `{F4A1B2C3-D4E5-4F06-8A17-2B3C4D5E6F78}`
#[cfg(target_os = "windows")]
pub const GUID_DISPLAY_ATTRIBUTE_INPUT: GUID =
    GUID::from_u128(0xF4A1_B2C3_D4E5_4F06_8A17_2B3C_4D5E_6F78);

/// Converted text display attribute (bold). New GUID.
/// `{F5B2C3D4-E5F6-4017-8B28-3C4D5E6F7089}`
#[cfg(target_os = "windows")]
pub const GUID_DISPLAY_ATTRIBUTE_CONVERTED: GUID =
    GUID::from_u128(0xF5B2_C3D4_E5F6_4017_8B28_3C4D_5E6F_7089);

/// Immersive (modern/UWP) support — required so UWP hosts load us.
/// `{13A016DF-560B-46CD-947A-4C3AF1E0E35D}`
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_IMMERSIVESUPPORT: GUID =
    GUID::from_u128(0x13A0_16DF_560B_46CD_947A_4C3A_F1E0_E35D);

/// System tray support — required for proper system tray integration.
/// `{25504FB4-7BAB-4BC1-9C69-CF81890F0EF5}`
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_SYSTRAYSUPPORT: GUID =
    GUID::from_u128(0x2550_4FB4_7BAB_4BC1_9C69_CF81_890F_0EF5);

/// Secure-mode capability — required for key delivery in UWP apps.
/// `{49D2F9CE-1F5E-11D7-A6D3-00065B84435C}`
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_SECUREMODE: GUID =
    GUID::from_u128(0x49D2_F9CE_1F5E_11D7_A6D3_0006_5B84_435C);

/// UI element support.
/// `{49D2F9CF-1F5E-11D7-A6D3-00065B84435C}`
#[cfg(target_os = "windows")]
pub const GUID_TFCAT_TIPCAP_UIELEMENTENABLED: GUID =
    GUID::from_u128(0x49D2_F9CF_1F5E_11D7_A6D3_0006_5B84_435C);

/// PreservedKey GUID for the Hankaku/Zenkaku (IME on/off) toggle.
/// `{5A7E36C3-8ADE-4FC1-9B2F-3CAE12DBCD48}`
#[cfg(target_os = "windows")]
pub const GUID_PRESERVED_KEY_ONOFF: GUID =
    GUID::from_u128(0x5A7E_36C3_8ADE_4FC1_9B2F_3CAE_12DB_CD48);

/// Japanese LANGID (0x0411).
pub const LANGID_JAPANESE: u16 = 0x0411;

/// Newtype so HMODULE can live in a OnceCell (HMODULE is `*mut void` and
/// therefore `!Send`/`!Sync` by default).
#[cfg(target_os = "windows")]
#[derive(Clone, Copy)]
pub struct SyncHmodule(pub HMODULE);

#[cfg(target_os = "windows")]
unsafe impl Send for SyncHmodule {}
#[cfg(target_os = "windows")]
unsafe impl Sync for SyncHmodule {}

#[cfg(target_os = "windows")]
pub static DLL_INSTANCE: OnceCell<SyncHmodule> = OnceCell::new();

#[cfg(target_os = "windows")]
pub static DLL_REF_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(target_os = "windows")]
pub fn dll_add_ref() {
    DLL_REF_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(target_os = "windows")]
pub fn dll_release() {
    let prev = DLL_REF_COUNT.load(std::sync::atomic::Ordering::SeqCst);
    if prev == 0 {
        tracing::error!("dll_release underflow");
        return;
    }
    DLL_REF_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(target_os = "windows")]
pub fn dll_can_unload() -> bool {
    DLL_REF_COUNT.load(std::sync::atomic::Ordering::SeqCst) == 0
}
