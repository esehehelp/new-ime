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
    /// conservative to keep RAM bounded.
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

/// Backend kinds that make sense with each device. Used by the CLI to reject
/// obvious mistakes (e.g. asking for `cuda` with the `mock` backend, which
/// cannot exercise the GPU path).
pub fn backend_supports_cuda(kind: &str) -> bool {
    matches!(kind, "tch-ctc-nat" | "ctc")
}

#[cfg(feature = "cuda")]
pub fn resolve_tch_device(device: Device) -> Result<tch::Device> {
    match device {
        Device::Cpu => Ok(tch::Device::Cpu),
        Device::Cuda(idx) => {
            if !tch::Cuda::is_available() {
                bail!("requested {device} but libtorch reports no CUDA device");
            }
            let count = tch::Cuda::device_count() as usize;
            if idx >= count {
                bail!("cuda:{idx} requested but only {count} device(s) visible");
            }
            Ok(tch::Device::Cuda(idx))
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn require_cuda_built(device: Device) -> Result<()> {
    if device.is_cuda() {
        bail!(
            "{device} requested but this binary was built without the `cuda` feature. \
             rebuild with `--features cuda` (requires LIBTORCH)"
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
