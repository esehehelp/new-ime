//! `TrainBackend` implementation that drives [`CtcNatModel`] end-to-end.
//!
//! Step 1 (this file): the skeleton holds a live `VarStore` + `CtcNatModel`
//! on the configured tch device and wires `TrainBackend::step` through to
//! a forward-only run for the accounting fields. Loss + backward + optim
//! lands in step 2; this file stays small so step 2 can replace `step` in
//! place.

use super::batch::{GpuBatch, StagedHostBatch};
use super::model::CtcNatModel;
use crate::backend::{BackendConfig, TrainBackend};
use crate::device::{resolve_tch_device, Device};
use crate::trainer::TrainerStep;
use anyhow::{Context, Result};
use kkc_data::PackedBatch;
use std::path::Path;
use tch::nn::VarStore;
use tch::{Device as TchDevice, Kind};

pub struct TchCtcNatBackend {
    vs: VarStore,
    model: CtcNatModel,
    device: TchDevice,
    config: BackendConfig,
    last_loss: Option<f64>,
    step_count: usize,
}

impl TchCtcNatBackend {
    pub fn new(config: &BackendConfig, device: Device) -> Result<Self> {
        let tch_device = resolve_tch_device(device)?;
        let vs = VarStore::new(tch_device);
        let model = CtcNatModel::new(&vs.root(), config)?;
        Ok(Self {
            vs,
            model,
            device: tch_device,
            config: config.clone(),
            last_loss: None,
            step_count: 0,
        })
    }

    pub fn device(&self) -> TchDevice {
        self.device
    }
    pub fn var_store(&self) -> &VarStore {
        &self.vs
    }
    pub fn model(&self) -> &CtcNatModel {
        &self.model
    }
    pub fn config(&self) -> &BackendConfig {
        &self.config
    }
    pub fn last_loss(&self) -> Option<f64> {
        self.last_loss
    }
    pub fn step_count(&self) -> usize {
        self.step_count
    }
    pub fn trainable_param_count(&self) -> i64 {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum()
    }

    /// Consume a pre-uploaded [`GpuBatch`] directly. Step 1 runs a
    /// forward-only proposal pass so the device is exercised and a
    /// meaningful loss surrogate is returned; real CTC loss + backward
    /// lands in step 2.
    pub fn step_gpu(&mut self, step: usize, batch: &GpuBatch) -> Result<TrainerStep> {
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let out = self
            .model
            .forward_proposal_only(&batch.input_ids, &mask_bool);
        // Placeholder loss: mean of |logits|. Any finite scalar keeps the
        // training loop healthy until step 2 replaces this with CTC loss.
        let loss = out.proposal_logits.abs().mean(Kind::Float);
        let loss = loss.double_value(&[]);
        self.last_loss = Some(loss);
        self.step_count = step;
        Ok(TrainerStep {
            loss,
            rows: batch.batch_size,
            bytes: batch.bytes,
            input_tokens: batch.non_padding_input_tokens,
            target_tokens: batch.non_padding_target_tokens,
        })
    }
}

impl TrainBackend for TchCtcNatBackend {
    fn kind(&self) -> &'static str {
        "tch-ctc-nat"
    }

    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        self.step_gpu(step, &gpu)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        // Step 4 replaces this with safetensors. For now persist a marker
        // so the ledger stays valid across runs.
        let body = serde_json::json!({
            "kind": "tch-ctc-nat",
            "step": self.step_count,
            "last_loss": self.last_loss,
            "param_count": self.trainable_param_count(),
            "note": "weights pending: safetensors snapshot lands in step 4",
        });
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&body).context("serialize tch backend marker")?,
        )
        .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        if let Some(s) = parsed.get("step").and_then(|v| v.as_u64()) {
            self.step_count = s as usize;
        }
        if let Some(l) = parsed.get("last_loss").and_then(|v| v.as_f64()) {
            self.last_loss = Some(l);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_packed() -> PackedBatch {
        PackedBatch {
            input_ids: vec![1, 2, 0, 3, 4, 5],
            attention_mask: vec![1, 1, 0, 1, 1, 1],
            target_ids: vec![8, 9, 0, 7],
            input_lengths: vec![2, 3],
            target_lengths: vec![2, 1],
            source_ids: vec![0, 1],
            batch_size: 2,
            max_input_len: 3,
            max_target_len: 2,
            order_cursor: 2,
        }
    }

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
    fn tch_backend_step_produces_finite_loss_on_cpu() {
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        let packed = tiny_packed();
        let step = backend.step(1, &packed).unwrap();
        assert_eq!(step.rows, 2);
        assert!(step.loss.is_finite() && step.loss >= 0.0);
        assert!(backend.trainable_param_count() > 0);
    }

    #[test]
    fn tch_backend_checkpoint_marker_round_trips() {
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        let packed = tiny_packed();
        let _ = backend.step(5, &packed).unwrap();
        let tmp = tempfile::NamedTempFile::new().unwrap();
        backend.save_checkpoint(tmp.path()).unwrap();
        let mut restored = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        restored.load_checkpoint(tmp.path()).unwrap();
        assert_eq!(restored.step_count(), 5);
        assert!(restored.last_loss().is_some());
    }
}
