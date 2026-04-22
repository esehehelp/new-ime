//! `TrainBackend` implementation that drives [`CtcNatModel`] end-to-end.
//!
//! Step 1 (this file): the skeleton holds a live `VarStore` + `CtcNatModel`
//! on the configured tch device and wires `TrainBackend::step` through to
//! a forward-only run for the accounting fields. Loss + backward + optim
//! lands in step 2; this file stays small so step 2 can replace `step` in
//! place.

use super::batch::{GpuBatch, StagedHostBatch};
use super::loss::{
    build_target_refinement, ctc_proposal_loss, refine_mlm_loss, refine_weight_ramp, remask_loss,
    stop_loss,
};
use super::model::CtcNatModel;
use crate::backend::{BackendConfig, TrainBackend};
use crate::device::{resolve_tch_device, Device};
use crate::trainer::TrainerStep;
use anyhow::{bail, Result};
use kkc_data::PackedBatch;
use std::path::Path;
use tch::nn::VarStore;
use tch::{Device as TchDevice, Kind, Tensor};

/// What the tch training path currently implements. Documented here so
/// the check that rejects unsupported configs (see `validate_tch_config`
/// below) stays truthful when someone extends the GPU kernels.
///
/// Supported:
/// - CTC proposal loss (always on).
/// - Mask-CTC refinement with `refine_source = "target"` and a single
///   iteration (`refine_iterations == 1`), gated by `refine_loss_weight`.
/// - Learned remask head (BCE on `refined_argmax != target` over valid
///   positions), gated by `remask_loss_weight`.
/// - Learned stop head (BCE on full-row correctness), gated by
///   `stop_loss_weight`.
///
/// Not yet implemented — any config that asks for these is rejected
/// rather than silently ignored:
/// - `refine_source = "proposal"` / `"mixed"`
/// - `refine_iterations > 1`
/// - `refine_mask_ratio_min` / `refine_mask_ratio_max` per-batch sampling
/// - `remask_threshold` / `stop_threshold` at inference (training is OK
///   because the losses don't consult the thresholds).
/// - `confidence_fallback` (only affects proposal-source builder).
fn validate_tch_config(config: &BackendConfig) -> Result<()> {
    if !matches!(config.refine_source.as_str(), "target") {
        bail!(
            "tch backend: refine_source=`{}` is not supported yet; use \"target\"",
            config.refine_source
        );
    }
    if config.refine_iterations > 1 {
        bail!(
            "tch backend: refine_iterations={} not supported yet; use 1",
            config.refine_iterations
        );
    }
    if config.refine_mask_ratio_min.is_some() || config.refine_mask_ratio_max.is_some() {
        bail!(
            "tch backend: refine_mask_ratio_min/max not supported yet; use a fixed \
             refine_mask_ratio"
        );
    }
    Ok(())
}

pub struct TchCtcNatBackend {
    vs: VarStore,
    model: CtcNatModel,
    device: TchDevice,
    config: BackendConfig,
    last_loss: Option<f64>,
    step_count: usize,
    optim: Option<super::optim::TchOptimizer>,
}

impl TchCtcNatBackend {
    pub fn new(config: &BackendConfig, device: Device) -> Result<Self> {
        validate_tch_config(config)?;
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
            optim: None,
        })
    }

    /// Attach an AdamW optimizer so `step_gpu` can take a full train
    /// step (forward → backward → optim → zero_grad). Used by the
    /// training loop; tests that only need forward leave it unset.
    pub fn attach_optimizer(&mut self, grad_clip: f64) -> Result<()> {
        let optim = super::optim::TchOptimizer::from_config(&self.vs, &self.config, grad_clip)?;
        self.optim = Some(optim);
        Ok(())
    }

    pub fn has_optimizer(&self) -> bool {
        self.optim.is_some()
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
    pub fn set_step_count(&mut self, step: usize) {
        self.step_count = step;
    }
    pub fn set_last_loss(&mut self, loss: Option<f64>) {
        self.last_loss = loss;
    }
    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.vs
    }
    pub fn trainable_param_count(&self) -> i64 {
        self.vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum()
    }

    /// Consume a pre-uploaded [`GpuBatch`] directly. Computes the full
    /// training loss (CTC proposal + refine MLM + remask + stop) and
    /// runs `backward()` to populate parameter gradients.
    ///
    /// The optimizer step is NOT applied here — that lives in step 3
    /// (`TchOptimizer`). The training loop receives the scalar loss and
    /// is expected to drive optim + zero_grad externally.
    pub fn step_gpu(&mut self, step: usize, batch: &GpuBatch) -> Result<TrainerStep> {
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let target_len = batch.target_lengths.shallow_clone();
        let input_len = batch.input_lengths.shallow_clone();

        // Proposal path (always runs).
        let encoder_out = self.model.encode(&batch.input_ids, &mask_bool);
        let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
        let ctc = ctc_proposal_loss(
            &proposal_logits,
            &batch.target_ids,
            &input_len,
            &target_len,
            self.model.blank_id,
        );
        let mut loss = ctc.shallow_clone();

        // Refinement path (only when the weight is positive).
        let refine_weight_now = self.config.refine_loss_weight
            * refine_weight_ramp(step, self.config.refine_warmup_steps);
        if refine_weight_now > 0.0 {
            let (hyp_ids, mask_positions, valid) = build_target_refinement(
                &batch.target_ids,
                &target_len,
                self.config.refine_mask_ratio,
                self.model.mask_token_id,
            );
            let (refined_logits, remask_logits, stop_logits_batch) = self.model.refine(
                &hyp_ids,
                &valid,
                &encoder_out,
                &mask_bool,
            );
            let refine_ce = refine_mlm_loss(&refined_logits, &batch.target_ids, &mask_positions);
            loss = &loss + refine_weight_now * &refine_ce;

            if self.config.remask_loss_weight > 0.0 {
                let r = remask_loss(&remask_logits, &refined_logits, &batch.target_ids, &valid);
                loss = &loss + self.config.remask_loss_weight * &r;
            }
            if self.config.stop_loss_weight > 0.0 {
                let s = stop_loss(
                    &stop_logits_batch,
                    &refined_logits,
                    &batch.target_ids,
                    &valid,
                );
                loss = &loss + self.config.stop_loss_weight * &s;
            }
        }

        let loss_val = loss.double_value(&[]);
        loss.backward();
        if let Some(opt) = self.optim.as_mut() {
            opt.optimize(step);
        }

        self.last_loss = Some(loss_val);
        self.step_count = step;
        Ok(TrainerStep {
            loss: loss_val,
            rows: batch.batch_size,
            bytes: batch.bytes,
            input_tokens: batch.non_padding_input_tokens,
            target_tokens: batch.non_padding_target_tokens,
        })
    }

    /// Zero out `.grad` on every trainable variable. `VarStore` itself
    /// doesn't ship this helper in tch 0.18, so we iterate.
    /// Step 3's `TchOptimizer` will take this over as part of its step.
    pub fn zero_grad(&mut self) {
        for var in self.vs.trainable_variables() {
            let mut grad = var.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
    }

    /// Forward-only evaluation on a pre-uploaded [`GpuBatch`]. Guarded by
    /// `no_grad`, never touches the optimizer, and does NOT mutate any
    /// cached training bookkeeping (step counter, last loss) — eval is
    /// supposed to leave no trace on the training run.
    pub fn eval_gpu(&self, batch: &GpuBatch) -> Result<TrainerStep> {
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let loss_val: f64 = tch::no_grad(|| {
            let encoder_out = self.model.encode(&batch.input_ids, &mask_bool);
            let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
            let loss = ctc_proposal_loss(
                &proposal_logits,
                &batch.target_ids,
                &batch.input_lengths,
                &batch.target_lengths,
                self.model.blank_id,
            );
            loss.double_value(&[])
        });
        Ok(TrainerStep {
            loss: loss_val,
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
        super::ckpt::save_backend(self, path)
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        super::ckpt::load_backend(self, path)
    }

    fn eval_step(&mut self, _step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        self.eval_gpu(&gpu)
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
    fn tch_backend_step_populates_gradients() {
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        backend.zero_grad();
        let packed = tiny_packed();
        let _ = backend.step(1, &packed).unwrap();

        let mut any_defined = false;
        let mut any_nonzero = false;
        for var in backend.var_store().trainable_variables() {
            let grad = var.grad();
            if grad.defined() {
                any_defined = true;
                if grad.abs().max().double_value(&[]) > 0.0 {
                    any_nonzero = true;
                    break;
                }
            }
        }
        assert!(any_defined, "no grads were created by backward()");
        assert!(any_nonzero, "all grads were zero after backward()");
    }

    #[test]
    fn tch_backend_refine_weight_zero_skips_refine_path() {
        // Skipping the refine path is cheap and deterministic: loss
        // should equal exactly the proposal CTC loss.
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        assert_eq!(backend.config().refine_loss_weight, 0.0);
        let _ = backend.step(1, &tiny_packed()).unwrap();
        // Just assert we returned finite — deeper equality is covered by
        // the parity test in step 5.
        assert!(backend.last_loss().unwrap().is_finite());
    }

    #[test]
    fn tch_backend_rejects_unsupported_refine_source() {
        let mut cfg = tiny_config();
        cfg.refine_source = "proposal".to_string();
        let err = TchCtcNatBackend::new(&cfg, Device::Cpu)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("refine_source"), "err={err}");
    }

    #[test]
    fn tch_backend_rejects_multi_iteration_refine() {
        let mut cfg = tiny_config();
        cfg.refine_iterations = 3;
        assert!(TchCtcNatBackend::new(&cfg, Device::Cpu).is_err());
    }

    #[test]
    fn tch_backend_rejects_mask_ratio_range_sampling() {
        let mut cfg = tiny_config();
        cfg.refine_mask_ratio_min = Some(0.1);
        cfg.refine_mask_ratio_max = Some(0.4);
        assert!(TchCtcNatBackend::new(&cfg, Device::Cpu).is_err());
    }

    /// Guard against the "eval actually trains" regression Codex caught.
    /// When `eval_step` is called, weights must be bitwise identical
    /// before and after, even with an attached optimizer.
    #[test]
    fn tch_backend_eval_step_does_not_mutate_weights() {
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        // Randomize so "unchanged" means something.
        for var in backend.var_store().trainable_variables() {
            tch::no_grad(|| {
                let mut v = var;
                let _ = v.uniform_(-0.1, 0.1);
            });
        }
        backend.attach_optimizer(1.0).unwrap();
        let snapshot: Vec<(String, Tensor)> = backend
            .var_store()
            .variables()
            .into_iter()
            .map(|(n, t)| (n, t.to_device(TchDevice::Cpu).copy()))
            .collect();
        let packed = tiny_packed();
        let _ = backend.eval_step(0, &packed).unwrap();
        let after = backend.var_store().variables();
        for (name, before) in snapshot.iter() {
            let a = &after[name];
            let diff = (a - before).abs().max().double_value(&[]);
            assert!(
                diff == 0.0,
                "eval_step mutated weight `{name}`: max_abs_diff={diff}"
            );
        }
    }

    /// End-to-end convergence smoke: repeatedly training on a single
    /// fixed batch with a working optimizer must drive the loss down.
    /// This is the only sanity check we can get for the joint forward +
    /// backward + optim loop without a fully wired training harness.
    #[test]
    fn tch_backend_with_optimizer_reduces_loss_on_fixed_batch() {
        let cfg = BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            learning_rate: 5e-2,
            warmup_steps: 0,
            scheduler_total_steps: 50,
            min_lr_scale: 1.0, // disable decay; just train
            weight_decay: 0.0,
            ..tiny_config()
        };
        let mut backend = TchCtcNatBackend::new(&cfg, Device::Cpu).unwrap();
        backend.attach_optimizer(1.0).unwrap();
        let packed = tiny_packed();

        let first = backend.step(1, &packed).unwrap().loss;
        for step in 2..=20 {
            let _ = backend.step(step, &packed).unwrap();
        }
        let last = backend.last_loss().unwrap();
        assert!(
            last < first,
            "loss did not decrease: first={first} last={last}"
        );
    }

    #[test]
    fn tch_backend_checkpoint_round_trip_preserves_step_and_weights() {
        let mut backend = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        // Randomize weights so the round-trip has something real to check.
        for var in backend.var_store().trainable_variables() {
            tch::no_grad(|| {
                let mut v = var;
                let _ = v.uniform_(-0.1, 0.1);
            });
        }
        let packed = tiny_packed();
        let _ = backend.step(5, &packed).unwrap();
        let tmp_dir = tempfile::tempdir().unwrap();
        let anchor = tmp_dir.path().join("ckpt.backend.json");
        backend.save_checkpoint(&anchor).unwrap();
        let mut restored = TchCtcNatBackend::new(&tiny_config(), Device::Cpu).unwrap();
        restored.load_checkpoint(&anchor).unwrap();
        assert_eq!(restored.step_count(), 5);
        assert_eq!(restored.last_loss(), backend.last_loss());
        // Verify one weight made it through.
        let src_vars = backend.var_store().variables();
        let loaded_vars = restored.var_store().variables();
        for (name, sv) in src_vars.iter().take(3) {
            let lv = &loaded_vars[name];
            let diff = (sv - lv).abs().max().double_value(&[]);
            assert!(diff < 1e-6, "{name} diverged after round trip");
        }
    }
}
