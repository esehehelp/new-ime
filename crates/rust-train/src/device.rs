//! Compute device selection.
//!
//! Kept intentionally free of `tch` so the default (no-cuda) build can still
//! parse device specs, validate config, and route the training loop. With
//! `--features cuda`, [`Device::resolve_tch_device`] maps to an actual
//! `tch::Device`.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Device {
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    pub fn label(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(idx) => format!("cuda:{idx}"),
        }
    }

    /// Hint for the prefetch queue size. GPU training benefits from a bit more
    /// look-ahead because H2D copy can overlap with compute; CPU stays
    /// conservative to keep RAM bounded. Exposed for config-less callers;
    /// the TOML path overrides this directly.
    #[allow(dead_code)]
    pub fn default_prefetch_queue(&self) -> usize {
        match self {
            Device::Cpu => 2,
            Device::Cuda(_) => 4,
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label())
    }
}

impl FromStr for Device {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            bail!("device spec is empty");
        }
        let lower = trimmed.to_ascii_lowercase();
        if lower == "cpu" {
            return Ok(Device::Cpu);
        }
        if lower == "cuda" || lower == "gpu" {
            return Ok(Device::Cuda(0));
        }
        if let Some(rest) = lower.strip_prefix("cuda:") {
            let idx: usize = rest
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid cuda index in device spec: {raw}"))?;
            return Ok(Device::Cuda(idx));
        }
        bail!("unknown device spec: {raw} (expected cpu | cuda | cuda:N)")
    }
}

/// Backend kinds that make sense on a CUDA device. Used by the CLI to
/// reject obvious mistakes (e.g. asking for `cuda` with the `mock`
/// backend, which cannot exercise the GPU path).
///
/// Only `tch-ctc-nat` qualifies: the CPU `ctc` / `surrogate` / `toy` /
/// `mock` backends are all f64 Rust implementations with no GPU path.
/// Listing them here would be a lie — `new_backend` in `main.rs` rejects
/// them outright on `--device cuda`.
pub fn backend_supports_cuda(kind: &str) -> bool {
    matches!(kind, "tch-ctc-nat")
}

#[cfg(feature = "cuda")]
pub fn resolve_tch_device(device: Device) -> Result<tch::Device> {
    match device {
        Device::Cpu => Ok(tch::Device::Cpu),
        Device::Cuda(idx) => {
            prime_windows_torch_dlls()?;
            let count = tch::Cuda::device_count();
            if count <= 0 {
                let is_available = tch::Cuda::is_available();
                let cudnn_is_available = tch::Cuda::cudnn_is_available();
                let has_cuda = tch::utils::has_cuda();
                let has_cudart = tch::utils::has_cudart();
                bail!(
                    "requested {device} but libtorch reports no CUDA device \
                     (device_count={count}, is_available={is_available}, \
                     cudnn_is_available={cudnn_is_available}, has_cuda={has_cuda}, \
                     has_cudart={has_cudart}). \
                     set LIBTORCH to a CUDA-enabled libtorch install before \
                     cargo build/run"
                );
            }
            let count = count as usize;
            if idx >= count {
                bail!("cuda:{idx} requested but only {count} device(s) visible");
            }
            Ok(tch::Device::Cuda(idx))
        }
    }
}

#[cfg(all(feature = "cuda", windows))]
fn prime_windows_torch_dlls() -> Result<()> {
    use std::ffi::{c_void, OsStr};
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
    const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x0000_0400;

    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
        fn LoadLibraryW(file_name: *const u16) -> *mut c_void;
    }

    fn discover_torch_lib_dir() -> Option<PathBuf> {
        if let Ok(root) = std::env::var("LIBTORCH") {
            let path = PathBuf::from(root).join("lib");
            if path.is_dir() {
                return Some(path);
            }
        }
        None
    }

    fn to_wide(value: &OsStr) -> Vec<u16> {
        value.encode_wide().chain(std::iter::once(0)).collect()
    }

    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        let Some(torch_lib) = discover_torch_lib_dir() else {
            return Ok(());
        };
        if !torch_lib.is_dir() {
            return Ok(());
        }

        let torch_lib_wide = to_wide(torch_lib.as_os_str());
        unsafe {
            let _ = SetDefaultDllDirectories(
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS,
            );
            let cookie = AddDllDirectory(torch_lib_wide.as_ptr());
            if cookie.is_null() {
                return Err(format!(
                    "AddDllDirectory failed for {}",
                    torch_lib.display()
                ));
            }
        }

        for dll in [
            "c10.dll",
            "c10_cuda.dll",
            "torch.dll",
            "torch_cpu.dll",
            "torch_cuda.dll",
            "cudart64_12.dll",
            "caffe2_nvrtc.dll",
            "nvrtc64_120_0.dll",
        ] {
            let path = torch_lib.join(dll);
            if !path.is_file() {
                continue;
            }
            let wide = to_wide(path.as_os_str());
            unsafe {
                if LoadLibraryW(wide.as_ptr()).is_null() {
                    return Err(format!("LoadLibraryW failed for {}", path.display()));
                }
            }
        }

        Ok(())
    })
    .as_ref()
    .map_err(|err| anyhow::anyhow!(err.clone()))?;
    Ok(())
}

#[cfg(all(feature = "cuda", not(windows)))]
fn prime_windows_torch_dlls() -> Result<()> {
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn require_cuda_built(device: Device) -> Result<()> {
    if device.is_cuda() {
        bail!(
            "{device} requested but this binary was built without the `cuda` feature. \
             rebuild with `--features cuda` and set LIBTORCH to a libtorch install"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn require_cuda_built(_device: Device) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_and_cuda_specs() {
        assert_eq!(Device::from_str("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("CPU").unwrap(), Device::Cpu);
        assert_eq!(Device::from_str("cuda").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("gpu").unwrap(), Device::Cuda(0));
        assert_eq!(Device::from_str("cuda:3").unwrap(), Device::Cuda(3));
    }

    #[test]
    fn rejects_unknown_specs() {
        assert!(Device::from_str("").is_err());
        assert!(Device::from_str("tpu").is_err());
        assert!(Device::from_str("cuda:xx").is_err());
    }

    #[test]
    fn round_trips_through_display() {
        for raw in ["cpu", "cuda:0", "cuda:2"] {
            let dev = Device::from_str(raw).unwrap();
            assert_eq!(dev.label(), raw);
        }
    }

    #[test]
    fn prefetch_queue_defaults_differ() {
        assert!(Device::Cuda(0).default_prefetch_queue() > Device::Cpu.default_prefetch_queue());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn require_cuda_built_rejects_cuda_without_feature() {
        assert!(require_cuda_built(Device::Cuda(0)).is_err());
        assert!(require_cuda_built(Device::Cpu).is_ok());
    }
}
