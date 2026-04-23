//! IClassFactory for the new-ime text service.
//!
//! Pattern taken from karukan-tsf (MIT/Apache-2.0). The only allocation
//! path the DLL ever hands out is `NewImeTextService` — no aggregation.

use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::System::Com::*;

use crate::globals::{dll_add_ref, dll_release};
use crate::tsf::text_input_processor::NewImeTextService;

#[implement(IClassFactory)]
pub struct NewImeClassFactory;

#[allow(clippy::new_without_default)]
impl NewImeClassFactory {
    pub fn new() -> Self {
        dll_add_ref();
        Self
    }
}

impl Drop for NewImeClassFactory {
    fn drop(&mut self) {
        dll_release();
    }
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl IClassFactory_Impl for NewImeClassFactory_Impl {
    fn CreateInstance(
        &self,
        punkouter: Option<&IUnknown>,
        riid: *const GUID,
        ppvobject: *mut *mut core::ffi::c_void,
    ) -> Result<()> {
        unsafe {
            if ppvobject.is_null() {
                return Err(E_POINTER.into());
            }
            *ppvobject = core::ptr::null_mut();
            if punkouter.is_some() {
                return Err(CLASS_E_NOAGGREGATION.into());
            }
            let service: IUnknown = NewImeTextService::new().into();
            service.query(&*riid, ppvobject).ok()
        }
    }

    fn LockServer(&self, flock: BOOL) -> Result<()> {
        if flock.as_bool() {
            dll_add_ref();
        } else {
            dll_release();
        }
        Ok(())
    }
}
