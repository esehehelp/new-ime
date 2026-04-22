//! Safetensors serialization for `VarStore` weights.
//!
//! Saves and loads model weights via the `safetensors` crate so
//! checkpoints are portable (inspectable from Python, stable across
//! libtorch versions). A companion JSON metadata file stores training
//! bookkeeping (step, loss, kind, param count).
//!
//! **Scope note**: optimizer state (AdamW m/v) is NOT persisted in this
//! revision. `tch::nn::Optimizer` in tch 0.18 does not expose per-param
//! moment buffers, so resume restarts AdamW from zero. Acceptable for
//! the next Suiko run — the m/v recover within a few hundred steps of
//! training — and tracked as a follow-up for full bit-for-bit resume.

use super::backend::TchCtcNatBackend;
use crate::pipeline::CheckpointWrite;
use anyhow::{bail, Context, Result};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

pub const WEIGHTS_SUFFIX: &str = ".weights.safetensors";
pub const OPTIM_SUFFIX: &str = ".optim.safetensors";
/// The trainer ledger + resume check both require `<step>.backend.json`
/// to exist next to `<step>.ckpt.json`. The tch backend treats this
/// anchor file AS the metadata payload (step, loss, weight file name,
/// format version) so there is no second `.meta.json` sidecar to keep
/// in sync with the anchor.
pub const BACKEND_SUFFIX: &str = ".backend.json";

#[derive(Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub kind: String,
    pub step: usize,
    pub last_loss: Option<f64>,
    pub param_count: i64,
    pub preset_hint: String,
    pub vocab_size: i64,
    pub hidden_size: i64,
    /// Safetensors filename relative to meta path.
    pub weights_file: String,
    /// Semantic version of this checkpoint layout, bumped whenever the
    /// on-disk format changes so loaders can reject mismatched files.
    pub format_version: u32,
}

pub const CHECKPOINT_FORMAT_VERSION: u32 = 1;

/// Build a `Tensor` from raw safetensors bytes. Allocates the target
/// tensor with `kind`+`shape` first, then copies bytes in — this path
/// sidesteps the `from_slice` / `from_data_size` size assumptions in
/// tch 0.18.
fn tensor_from_view_typed(bytes: &[u8], shape: &[i64], kind: Kind) -> Result<Tensor> {
    let elem_bytes = match kind {
        Kind::Float | Kind::Int => 4usize,
        Kind::Double | Kind::Int64 => 8,
        Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
        Kind::Int8 | Kind::Uint8 | Kind::Bool => 1,
        other => bail!("tensor_from_view_typed: unsupported dtype {:?}", other),
    };
    if bytes.len() % elem_bytes != 0 {
        bail!(
            "payload length {} not divisible by elem_bytes {}",
            bytes.len(),
            elem_bytes
        );
    }
    let numel = bytes.len() / elem_bytes;
    let expected: i64 = shape.iter().product();
    if expected as usize != numel {
        bail!(
            "payload numel {numel} does not match shape {shape:?} (expected {expected})"
        );
    }
    let dst = Tensor::zeros(shape, (kind, Device::Cpu));
    // Safety: dst was just allocated with this kind+shape, is contiguous
    // and owns its storage. We're writing exactly `bytes.len()` bytes
    // starting at the data pointer.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            dst.data_ptr() as *mut u8,
            bytes.len(),
        );
    }
    Ok(dst)
}

fn kind_to_dtype(kind: Kind) -> Result<Dtype> {
    Ok(match kind {
        Kind::Float => Dtype::F32,
        Kind::Double => Dtype::F64,
        Kind::Half => Dtype::F16,
        Kind::BFloat16 => Dtype::BF16,
        Kind::Int64 => Dtype::I64,
        Kind::Int => Dtype::I32,
        Kind::Int16 => Dtype::I16,
        Kind::Int8 => Dtype::I8,
        Kind::Uint8 => Dtype::U8,
        Kind::Bool => Dtype::BOOL,
        other => bail!("unsupported tensor kind for safetensors: {:?}", other),
    })
}

fn dtype_to_kind(dtype: Dtype) -> Result<Kind> {
    Ok(match dtype {
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        Dtype::F16 => Kind::Half,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::I64 => Kind::Int64,
        Dtype::I32 => Kind::Int,
        Dtype::I16 => Kind::Int16,
        Dtype::I8 => Kind::Int8,
        Dtype::U8 => Kind::Uint8,
        Dtype::BOOL => Kind::Bool,
        other => bail!("unsupported safetensors dtype: {:?}", other),
    })
}

/// Copy `tensor` to CPU and return its raw bytes + shape + dtype.
fn tensor_to_bytes(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>, Dtype)> {
    let cpu = tensor.to_device(Device::Cpu).contiguous();
    let dtype = kind_to_dtype(cpu.kind())?;
    let shape: Vec<usize> = cpu.size().iter().map(|v| *v as usize).collect();
    let numel = cpu.numel();
    let elem_bytes = match dtype {
        Dtype::F32 | Dtype::I32 => 4,
        Dtype::F64 | Dtype::I64 => 8,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
        Dtype::I8 | Dtype::U8 | Dtype::BOOL => 1,
        _ => bail!("unsupported dtype size"),
    };
    let mut bytes = vec![0u8; numel * elem_bytes];
    // tch `copy_data_u8` takes element count (not byte count); it
    // multiplies by `elt_size_in_bytes` internally.
    cpu.copy_data_u8(&mut bytes, numel);
    Ok((bytes, shape, dtype))
}

/// Serialize every variable of `vs` into a safetensors byte vector.
fn var_store_bytes(vs: &VarStore) -> Result<Vec<u8>> {
    let vars = vs.variables();
    let mut buffers: BTreeMap<String, (Vec<u8>, Vec<usize>, Dtype)> = BTreeMap::new();
    for (name, tensor) in vars.into_iter() {
        buffers.insert(name, tensor_to_bytes(&tensor)?);
    }
    serialize_buffers(&buffers)
}

/// Serialize a `(name, tensor)` map (e.g. optimizer state) into
/// safetensors bytes. Same encoding as `var_store_bytes` but pulls
/// from an arbitrary map instead of a VarStore.
pub(super) fn tensors_to_bytes(
    tensors: &BTreeMap<String, Tensor>,
) -> Result<Vec<u8>> {
    let mut buffers: BTreeMap<String, (Vec<u8>, Vec<usize>, Dtype)> = BTreeMap::new();
    for (name, tensor) in tensors.iter() {
        buffers.insert(name.clone(), tensor_to_bytes(tensor)?);
    }
    serialize_buffers(&buffers)
}

fn serialize_buffers(
    buffers: &BTreeMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
) -> Result<Vec<u8>> {
    let views: Vec<(String, TensorView)> = buffers
        .iter()
        .map(|(name, (bytes, shape, dtype))| {
            let view = TensorView::new(*dtype, shape.clone(), bytes)
                .with_context(|| format!("build safetensors view for {name}"))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize(views, &None).context("safetensors serialize")
}

/// Deserialize a safetensors blob into a BTreeMap of materialized
/// tensors on CPU. Complements `tensors_to_bytes`.
pub(super) fn tensors_from_bytes(bytes: &[u8]) -> Result<BTreeMap<String, Tensor>> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;
    let mut out = BTreeMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let kind = dtype_to_kind(view.dtype())?;
        let shape: Vec<i64> = view.shape().iter().map(|v| *v as i64).collect();
        let t = tensor_from_view_typed(view.data(), &shape, kind)?;
        out.insert(name.clone(), t);
    }
    Ok(out)
}

/// Serialize every variable (trainable and non-trainable) of `vs` into
/// a safetensors file at `path`.
pub fn save_var_store(vs: &VarStore, path: &Path) -> Result<()> {
    let bytes = var_store_bytes(vs)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Load every named tensor from `path` into `vs` in place. Shape and
/// dtype must match; unknown names and missing names both error so
/// silent drift is caught immediately.
pub fn load_var_store(vs: &mut VarStore, path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parse safetensors {}", path.display()))?;
    let mut vars = vs.variables();
    let mut seen = std::collections::HashSet::new();

    for name in st.names() {
        seen.insert(name.clone());
        let view = st.tensor(name)?;
        let kind = dtype_to_kind(view.dtype())?;
        let shape_i64: Vec<i64> = view.shape().iter().map(|v| *v as i64).collect();
        let Some(var) = vars.get_mut(name) else {
            bail!("safetensors file has unknown variable `{name}`");
        };
        if var.size() != shape_i64 {
            bail!(
                "variable `{name}` shape mismatch: vs={:?} safetensors={:?}",
                var.size(),
                shape_i64
            );
        }
        // Materialize the new tensor from typed slices so we bypass the
        // alignment check in tch::Tensor::from_data_size (which rejects
        // payloads <64 bytes). Supports the dtypes we use for weights.
        let device = var.device();
        let new_t = tensor_from_view_typed(view.data(), &shape_i64, kind)?.to_device(device);
        tch::no_grad(|| {
            let _ = var.copy_(&new_t);
        });
    }

    // Missing variables are a hard error — we don't want a silent partial
    // restore on resume.
    for name in vars.keys() {
        if !seen.contains(name) {
            bail!(
                "variable `{name}` is present in VarStore but missing from {}",
                path.display()
            );
        }
    }

    Ok(())
}

pub fn weights_path_for(anchor: &Path) -> std::path::PathBuf {
    anchor.with_file_name(sibling_name_for(anchor, "weights.safetensors"))
}
pub fn optim_path_for(anchor: &Path) -> std::path::PathBuf {
    anchor.with_file_name(sibling_name_for(anchor, "optim.safetensors"))
}

/// Build a sibling filename by trimming `.backend.json` (or `.json`)
/// off `anchor` and appending `suffix`. Callers use
/// `anchor.with_file_name(...)` so the directory stays intact. The
/// prune path in the trainer uses the same convention via
/// `checkpoint_sidecar_path`.
fn sibling_name_for(anchor: &Path, suffix: &str) -> String {
    let name = anchor.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let stem = name
        .strip_suffix(".backend.json")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name);
    format!("{stem}.{suffix}")
        .trim_start_matches(|c: char| c == '.')
        .to_string()
}

/// Wire `TchCtcNatBackend` to the safetensors format.
///
/// On-disk layout per checkpoint step is:
/// - `<step>.backend.json`          — the metadata anchor the trainer
///   ledger keys on (step, loss, vocab, weights filename, format version)
/// - `<step>.weights.safetensors`   — model weights
///
/// The trainer's `<step>.ckpt.json` sidecar (trainer step summary) is
/// written by the caller, not here.
///
/// If the backend has an async checkpoint sender attached, both files
/// are submitted to the writer thread — the calling training step
/// returns immediately while the bytes land on disk in the background.
/// Without a sender we fall back to atomic synchronous writes.
pub fn save_backend(backend: &TchCtcNatBackend, anchor: &Path) -> Result<()> {
    let weights_path: PathBuf = weights_path_for(anchor);
    let weights_bytes = var_store_bytes(backend.var_store())?;
    // Optimizer moments ride next to the weights. Without this the
    // AdamW m/v reset on every resume, which silently alters the loss
    // curve. Only present when the backend actually has an optim
    // attached (eval runs pass a backend with no optim).
    let optim_payload: Option<(PathBuf, Vec<u8>)> = if let Some(state) =
        backend.optim_state_dict()
    {
        Some((optim_path_for(anchor), tensors_to_bytes(&state)?))
    } else {
        None
    };
    let meta_blob = CheckpointMeta {
        kind: "tch-ctc-nat".to_string(),
        step: backend.step_count(),
        last_loss: backend.last_loss(),
        param_count: backend.trainable_param_count(),
        preset_hint: String::new(),
        vocab_size: backend.config().output_size as i64,
        hidden_size: backend.config().hidden_size as i64,
        weights_file: weights_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string(),
        format_version: CHECKPOINT_FORMAT_VERSION,
    };
    let anchor_bytes =
        serde_json::to_vec_pretty(&meta_blob).context("serialize checkpoint meta")?;
    let anchor_path: PathBuf = anchor.to_path_buf();

    if let Some(sender) = backend.ckpt_sender() {
        sender
            .send(CheckpointWrite::File {
                path: anchor_path.clone(),
                bytes: anchor_bytes,
                sidecar: Some((weights_path.clone(), weights_bytes)),
            })
            .map_err(|_| anyhow::anyhow!("checkpoint writer thread is gone"))?;
        if let Some((optim_path, optim_bytes)) = optim_payload {
            sender
                .send(CheckpointWrite::File {
                    path: optim_path,
                    bytes: optim_bytes,
                    sidecar: None,
                })
                .map_err(|_| anyhow::anyhow!("checkpoint writer thread is gone"))?;
        }
        return Ok(());
    }

    if let Some(parent) = anchor_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
    }
    std::fs::write(&anchor_path, anchor_bytes)
        .with_context(|| format!("write {}", anchor_path.display()))?;
    std::fs::write(&weights_path, weights_bytes)
        .with_context(|| format!("write {}", weights_path.display()))?;
    if let Some((optim_path, optim_bytes)) = optim_payload {
        std::fs::write(&optim_path, optim_bytes)
            .with_context(|| format!("write {}", optim_path.display()))?;
    }
    Ok(())
}

pub fn load_backend(backend: &mut TchCtcNatBackend, anchor: &Path) -> Result<()> {
    let bytes =
        std::fs::read(anchor).with_context(|| format!("read {}", anchor.display()))?;
    let meta: CheckpointMeta = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", anchor.display()))?;
    if meta.format_version != CHECKPOINT_FORMAT_VERSION {
        bail!(
            "checkpoint {} has format version {}, expected {}",
            anchor.display(),
            meta.format_version,
            CHECKPOINT_FORMAT_VERSION
        );
    }
    if meta.vocab_size != backend.config().output_size as i64 {
        bail!(
            "checkpoint vocab_size {} does not match backend config {}",
            meta.vocab_size,
            backend.config().output_size
        );
    }
    if meta.hidden_size != backend.config().hidden_size as i64 {
        bail!(
            "checkpoint hidden_size {} does not match backend config {}",
            meta.hidden_size,
            backend.config().hidden_size
        );
    }
    let weights_path = weights_path_for(anchor);
    load_var_store(backend.var_store_mut(), &weights_path)?;
    backend.set_step_count(meta.step);
    backend.set_last_loss(meta.last_loss);
    // Restore AdamW m/v from the optim sidecar if present. Pre-fix
    // checkpoints have no `.optim.safetensors` — those still resume,
    // but AdamW starts from zero state and the trainer logs a warning
    // so the behavior isn't silent.
    let optim_path = optim_path_for(anchor);
    if optim_path.exists() {
        let optim_bytes = std::fs::read(&optim_path)
            .with_context(|| format!("read {}", optim_path.display()))?;
        let state = tensors_from_bytes(&optim_bytes)?;
        backend.load_optim_state_dict(&state)?;
    } else if backend.has_optimizer() {
        eprintln!(
            "[kkc-train] warning: resumed from {} but no optim sidecar \
             ({}) — AdamW restarts from zero state. Expect a brief \
             loss bump while momentum rebuilds.",
            anchor.display(),
            optim_path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendConfig;
    use crate::device::Device as KkcDevice;
    use tempfile::tempdir;

    fn tiny_config() -> BackendConfig {
        BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            hidden_size: 16,
            encoder_layers: 2,
            num_heads: 4,
            ffn_size: 32,
            decoder_layers: 2,
            decoder_heads: 4,
            decoder_ffn_size: 32,
            output_size: 12,
            blank_id: 4,
            max_positions: 8,
            mask_token_id: 5,
            ..BackendConfig::default()
        }
    }

    #[test]
    fn weights_round_trip_bitwise_identical() {
        let dir = tempdir().unwrap();
        let anchor = dir.path().join("ckpt.backend.json");
        let cfg = tiny_config();

        let mut source = TchCtcNatBackend::new(&cfg, KkcDevice::Cpu).unwrap();
        // Vary every parameter from zero-init so weights are non-trivial.
        for var in source.var_store().trainable_variables() {
            tch::no_grad(|| {
                let mut v = var;
                let _ = v.uniform_(-0.1, 0.1);
            });
        }
        save_backend(&source, &anchor).unwrap();

        let mut loaded = TchCtcNatBackend::new(&cfg, KkcDevice::Cpu).unwrap();
        load_backend(&mut loaded, &anchor).unwrap();

        let src_vars = source.var_store().variables();
        let loaded_vars = loaded.var_store().variables();
        assert_eq!(src_vars.len(), loaded_vars.len());
        for (name, sv) in src_vars.iter() {
            let lv = loaded_vars
                .get(name)
                .unwrap_or_else(|| panic!("missing var {name}"));
            let diff = (sv - lv).abs().max().double_value(&[]);
            assert!(
                diff < 1e-6,
                "var {name} not bitwise equal: max_abs_diff={diff}"
            );
        }
    }

    /// Guard the resume contract: trainer + eval + check_resume assume
    /// that `<step>.backend.json` exists next to `<step>.ckpt.json`. If
    /// the GPU save stops producing it, resume breaks silently.
    #[test]
    fn save_backend_writes_anchor_and_weights_sync_path() {
        let dir = tempdir().unwrap();
        let anchor = dir.path().join("step_00000010.backend.json");
        let backend = TchCtcNatBackend::new(&tiny_config(), KkcDevice::Cpu).unwrap();
        save_backend(&backend, &anchor).unwrap();
        assert!(anchor.exists(), "anchor {} not written", anchor.display());
        let weights = weights_path_for(&anchor);
        assert!(
            weights.exists(),
            "weights sidecar {} not written",
            weights.display()
        );
        // Anchor payload must be parseable as CheckpointMeta.
        let meta: CheckpointMeta =
            serde_json::from_slice(&std::fs::read(&anchor).unwrap()).unwrap();
        assert_eq!(meta.format_version, CHECKPOINT_FORMAT_VERSION);
    }

    #[test]
    fn load_rejects_mismatched_vocab() {
        let dir = tempdir().unwrap();
        let anchor = dir.path().join("ckpt.backend.json");
        let source = TchCtcNatBackend::new(&tiny_config(), KkcDevice::Cpu).unwrap();
        save_backend(&source, &anchor).unwrap();
        let mut wrong_cfg = tiny_config();
        wrong_cfg.output_size = 16; // mismatch
        let mut loaded = TchCtcNatBackend::new(&wrong_cfg, KkcDevice::Cpu).unwrap();
        assert!(load_backend(&mut loaded, &anchor).is_err());
    }
}
