//! new-ime TSF DLL (Rust port).
//!
//! Replacement for the C++ `new-ime-tsf.dll`. Structure mirrors karukan-tsf
//! (MIT/Apache-2.0 © togatoga / anosatsuk124): DLL exports, CLSID/profile
//! GUIDs, a ClassFactory, and a TextInputProcessor + KeyEventSink stack.
//!
//! For the initial scaffold only the COM registration + minimal class
//! factory + TextInputProcessor (Activate/Deactivate no-op) are wired up.
//! Key handling, composition, and candidate rendering land in follow-up
//! commits.

pub mod async_convert;
pub mod composing_text;
pub mod engine_bridge;
pub mod globals;
pub mod keymap;
pub mod registration;

#[cfg(target_os = "windows")]
pub mod tsf;

#[cfg(target_os = "windows")]
mod dll_exports {
    use windows::core::*;
    use windows::Win32::Foundation::{
        BOOL, CLASS_E_CLASSNOTAVAILABLE, E_POINTER, HMODULE, S_FALSE, S_OK, TRUE,
    };

    use crate::globals::*;
    use crate::tsf::class_factory::NewImeClassFactory;

    const DLL_PROCESS_ATTACH: u32 = 1;

    #[unsafe(no_mangle)]
    extern "system" fn DllMain(
        hinstance: HMODULE,
        reason: u32,
        _reserved: *mut core::ffi::c_void,
    ) -> BOOL {
        if reason == DLL_PROCESS_ATTACH {
            let _ = DLL_INSTANCE.set(SyncHmodule(hinstance));
            std::panic::set_hook(Box::new(|info| {
                let _ = std::fs::OpenOptions::new();
                eprintln!("new-ime-tsf panic: {}", info);
            }));
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
                )
                .with_writer(std::io::stderr)
                .try_init();
        }
        TRUE
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllGetClassObject(
        rclsid: *const GUID,
        riid: *const GUID,
        ppv: *mut *mut core::ffi::c_void,
    ) -> HRESULT {
        unsafe {
            if ppv.is_null() || rclsid.is_null() || riid.is_null() {
                return E_POINTER;
            }
            *ppv = core::ptr::null_mut();
            if *rclsid != CLSID_NEW_IME_TEXT_SERVICE {
                return CLASS_E_CLASSNOTAVAILABLE;
            }
            let factory: IUnknown = NewImeClassFactory::new().into();
            factory.query(&*riid, ppv)
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllCanUnloadNow() -> HRESULT {
        if dll_can_unload() {
            S_OK
        } else {
            S_FALSE
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllRegisterServer() -> HRESULT {
        match crate::registration::register_server() {
            Ok(()) => S_OK,
            Err(e) => e.code(),
        }
    }

    #[unsafe(no_mangle)]
    extern "system" fn DllUnregisterServer() -> HRESULT {
        match crate::registration::unregister_server() {
            Ok(()) => S_OK,
            Err(e) => e.code(),
        }
    }
}
