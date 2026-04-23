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
use crate::backend::{BackendConfig, EvalBatchOutput, TrainBackend};
use crate::device::{resolve_tch_device, Device};
use crate::trainer::TrainerStep;
use anyhow::{bail, Result};
use kkc_data::PackedBatch;
use std::path::Path;
use std::time::Instant;
use tch::nn::VarStore;
use tch::{Device as TchDevice, Kind, Tensor};

fn collapse_ctc_argmax(
    argmax: &tch::Tensor,
    attention_mask: &tch::Tensor,
    blank_id: i64,
) -> Vec<Vec<u32>> {
    let sizes = argmax.size();
    let batch = sizes[0] as usize;
    let time = sizes[1] as usize;
    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::new();
        let mut prev: Option<i64> = None;
        for t in 0..time {
            if attention_mask.int64_value(&[b as i64, t as i64]) == 0 {
                continue;
            }
            let token = argmax.int64_value(&[b as i64, t as i64]);
            if token == blank_id {
                prev = None;
                continue;
            }
            if prev == Some(token) {
                continue;
            }
            row.push(token as u32);
            prev = Some(token);
        }
        out.push(row);
    }
    out
}

fn logsumexp_pair(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

fn prefix_beam_search(
    log_probs: &tch::Tensor,
    blank_id: i64,
    beam_width: usize,
    top_k_per_step: usize,
) -> Vec<(Vec<u32>, f64)> {
    let sizes = log_probs.size();
    let time = sizes[0] as usize;
    let vocab = sizes[1] as usize;
    let mut beam: std::collections::BTreeMap<Vec<i64>, (f64, f64)> =
        std::collections::BTreeMap::from([(Vec::new(), (0.0, f64::NEG_INFINITY))]);

    for t in 0..time {
        let blank_logp = log_probs.double_value(&[t as i64, blank_id]);
        let mut top = Vec::with_capacity(vocab);
        for v in 0..vocab {
            top.push((v as i64, log_probs.double_value(&[t as i64, v as i64])));
        }
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        top.truncate(top_k_per_step.min(vocab));

        let mut next_beam: std::collections::BTreeMap<Vec<i64>, (f64, f64)> =
            std::collections::BTreeMap::new();
        let update = |next_beam: &mut std::collections::BTreeMap<Vec<i64>, (f64, f64)>,
                      prefix: Vec<i64>,
                      new_pb: f64,
                      new_pnb: f64| {
            let entry = next_beam
                .entry(prefix)
                .or_insert((f64::NEG_INFINITY, f64::NEG_INFINITY));
            entry.0 = logsumexp_pair(entry.0, new_pb);
            entry.1 = logsumexp_pair(entry.1, new_pnb);
        };

        for (prefix, (pb, pnb)) in &beam {
            let new_blank = logsumexp_pair(*pb, *pnb) + blank_logp;
            update(&mut next_beam, prefix.clone(), new_blank, f64::NEG_INFINITY);

            for (token, token_logp) in &top {
                if *token == blank_id {
                    continue;
                }
                if prefix.last().copied() == Some(*token) {
                    let mut extended = prefix.clone();
                    extended.push(*token);
                    update(
                        &mut next_beam,
                        extended,
                        f64::NEG_INFINITY,
                        *pb + *token_logp,
                    );
                    update(
                        &mut next_beam,
                        prefix.clone(),
                        f64::NEG_INFINITY,
                        *pnb + *token_logp,
                    );
                } else {
                    let mut extended = prefix.clone();
                    extended.push(*token);
                    let new_pnb = logsumexp_pair(*pb, *pnb) + *token_logp;
                    update(&mut next_beam, extended, f64::NEG_INFINITY, new_pnb);
                }
            }
        }

        let mut scored: Vec<_> = next_beam
            .into_iter()
            .map(|(prefix, (pb, pnb))| {
                let score = logsumexp_pair(pb, pnb);
                (prefix, (pb, pnb), score)
            })
            .collect();
        scored.sort_by(|a, b| b.2.total_cmp(&a.2));
        scored.truncate(beam_width.max(1));
        beam = scored
            .into_iter()
            .map(|(prefix, (pb, pnb), _)| (prefix, (pb, pnb)))
            .collect();
    }

    let mut final_beam: Vec<_> = beam
        .into_iter()
        .map(|(prefix, (pb, pnb))| {
            (
                prefix
                    .into_iter()
                    .map(|token| token as u32)
                    .collect::<Vec<_>>(),
                logsumexp_pair(pb, pnb),
            )
        })
        .collect();
    final_beam.sort_by(|a, b| b.1.total_cmp(&a.1));
    final_beam
}

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
    ckpt_sender: Option<std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>>,
    debug_timing: bool,
}

impl TchCtcNatBackend {
    fn refine_weight_now(&self, step: usize) -> f64 {
        self.config.refine_loss_weight * refine_weight_ramp(step, self.config.refine_warmup_steps)
    }

    fn decode_proposal_greedy(
        &self,
        proposal_logits: &Tensor,
        attention_mask: &Tensor,
    ) -> Vec<Vec<u32>> {
        let argmax_cpu = proposal_logits.argmax(-1, false).to_device(TchDevice::Cpu);
        let mask_cpu = attention_mask.to_device(TchDevice::Cpu);
        collapse_ctc_argmax(&argmax_cpu, &mask_cpu, self.model.blank_id)
    }

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
            ckpt_sender: None,
            debug_timing: false,
        })
    }

    pub(super) fn ckpt_sender(
        &self,
    ) -> Option<&std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>> {
        self.ckpt_sender.as_ref()
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

    /// Snapshot the attached optimizer's state (m/v buffers + adam step)
    /// so it can ride along in the checkpoint sidecar. Returns `None`
    /// when no optimizer is attached — eval-only backends produce no
    /// optim artifact.
    pub fn optim_state_dict(&self) -> Option<std::collections::BTreeMap<String, tch::Tensor>> {
        self.optim.as_ref().map(|o| o.state_dict())
    }

    /// Restore the optimizer from a previously saved state. Errors if
    /// the attached optimizer doesn't share the same variable set.
    pub fn load_optim_state_dict(
        &mut self,
        dict: &std::collections::BTreeMap<String, tch::Tensor>,
    ) -> Result<()> {
        match self.optim.as_mut() {
            Some(opt) => opt.load_state_dict(dict),
            None => Ok(()),
        }
    }

    pub fn var_store(&self) -> &VarStore {
        &self.vs
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

    pub fn predict_candidates_from_ids(
        &mut self,
        input_ids: &[u32],
        num_beams: usize,
        num_return: usize,
    ) -> Result<Vec<Vec<u32>>> {
        if input_ids.is_empty() {
            return Ok(vec![Vec::new()]);
        }
        if input_ids.len() > self.config.max_positions {
            bail!(
                "input length {} exceeds max_positions {}",
                input_ids.len(),
                self.config.max_positions
            );
        }
        let input_ids_i64: Vec<i64> = input_ids.iter().map(|v| *v as i64).collect();
        let t = input_ids.len() as i64;
        let device = self.device;
        let input = tch::Tensor::from_slice(&input_ids_i64)
            .view([1, t])
            .to_device(device);
        let mask = tch::Tensor::ones([1, t], (Kind::Bool, device));

        tch::no_grad(|| {
            let encoder_out = self.model.encode(&input, &mask);
            let proposal_logits = self.model.proposal(&encoder_out, &mask);
            let log_probs = proposal_logits
                .log_softmax(-1, Kind::Float)
                .squeeze_dim(0)
                .to_device(TchDevice::Cpu);
            if num_beams <= 1 || num_return <= 1 {
                let argmax = proposal_logits.argmax(-1, false).to_device(TchDevice::Cpu);
                let cpu_mask = mask.to_kind(Kind::Int64).to_device(TchDevice::Cpu);
                let greedy = collapse_ctc_argmax(&argmax, &cpu_mask, self.model.blank_id);
                return Ok(greedy.into_iter().take(num_return.max(1)).collect());
            }
            Ok(
                prefix_beam_search(&log_probs, self.model.blank_id, num_beams, 16)
                    .into_iter()
                    .take(num_return)
                    .map(|(tokens, _)| tokens)
                    .collect(),
            )
        })
    }

    /// Full training step on a pre-uploaded [`GpuBatch`]:
    ///   encode → proposal → (optional) refine → sum losses
    ///   → `backward()` → (if an optimizer is attached) `optim.optimize(step)`
    ///
    /// Attaching an optimizer is the norm via `attach_optimizer`; tests
    /// that only need forward/backward can skip it and the optim step
    /// is silently elided.
    ///
    /// Eval callers must use [`Self::eval_gpu`] or `TrainBackend::eval_step`,
    /// both of which run `no_grad` and never mutate weights — this method
    /// is NOT safe for evaluation.
    pub fn step_gpu(
        &mut self,
        step: usize,
        batch: &GpuBatch,
        upload_ms: Option<f64>,
    ) -> Result<TrainerStep> {
        let total_started = Instant::now();
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let target_len = batch.target_lengths.shallow_clone();
        let input_len = batch.input_lengths.shallow_clone();

        // Proposal path (always runs).
        let encode_started = Instant::now();
        let encoder_out = self.model.encode(&batch.input_ids, &mask_bool);
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let proposal_started = Instant::now();
        let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
        let ctc = ctc_proposal_loss(
            &proposal_logits,
            &batch.target_ids,
            &input_len,
            &target_len,
            self.model.blank_id,
        );
        let proposal_ms = proposal_started.elapsed().as_secs_f64() * 1000.0;
        let mut loss = ctc.shallow_clone();

        // Refinement path (only when the weight is positive).
        let refine_started = Instant::now();
        let refine_weight_now = self.refine_weight_now(step);
        if refine_weight_now > 0.0 {
            let (hyp_ids, mask_positions, valid) = build_target_refinement(
                &batch.target_ids,
                &target_len,
                self.config.refine_mask_ratio,
                self.model.mask_token_id,
                step as u64,
            );
            let (refined_logits, remask_logits, stop_logits_batch) =
                self.model
                    .refine(&hyp_ids, &valid, &encoder_out, &mask_bool);
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
        let refine_ms = refine_started.elapsed().as_secs_f64() * 1000.0;

        let sync_started = Instant::now();
        let loss_val = loss.double_value(&[]);
        let sync_ms = sync_started.elapsed().as_secs_f64() * 1000.0;

        let backward_started = Instant::now();
        loss.backward();
        let backward_ms = backward_started.elapsed().as_secs_f64() * 1000.0;

        let optim_started = Instant::now();
        if let Some(opt) = self.optim.as_mut() {
            opt.optimize(&self.vs, step);
        }
        let optim_ms = optim_started.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_started.elapsed().as_secs_f64() * 1000.0;

        if self.debug_timing {
            eprintln!(
                "[debug step {}] upload={:.1}ms encode={:.1}ms proposal={:.1}ms refine={:.1}ms sync={:.1}ms backward={:.1}ms optim={:.1}ms total={:.1}ms rows={} input_tokens={} target_tokens={}",
                step,
                upload_ms.unwrap_or(0.0),
                encode_ms,
                proposal_ms,
                refine_ms,
                sync_ms,
                backward_ms,
                optim_ms,
                total_ms + upload_ms.unwrap_or(0.0),
                batch.batch_size,
                batch.non_padding_input_tokens,
                batch.non_padding_target_tokens,
            );
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
    #[cfg(test)]
    pub fn zero_grad(&mut self) {
        for var in self.vs.trainable_variables() {
            let mut grad = var.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
        }
    }

    /// Forward-only evaluation on a pre-uploaded [`GpuBatch`]. Guarded
    /// by `no_grad`, never touches the optimizer, and does NOT mutate
    /// any cached training bookkeeping (step counter, last loss).
    ///
    /// Returns proposal CTC loss only, matching Python's
    /// `evaluate_model` path (`model(...).loss` + `greedy_decode`).
    pub fn eval_gpu(&self, batch: &GpuBatch) -> Result<TrainerStep> {
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let target_len = batch.target_lengths.shallow_clone();
        let input_len = batch.input_lengths.shallow_clone();
        let loss_val: f64 = tch::no_grad(|| {
            let encoder_out = self.model.encode(&batch.input_ids, &mask_bool);
            let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
            let ctc = ctc_proposal_loss(
                &proposal_logits,
                &batch.target_ids,
                &input_len,
                &target_len,
                self.model.blank_id,
            );
            ctc.double_value(&[])
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
        let upload_started = Instant::now();
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        let upload_ms = upload_started.elapsed().as_secs_f64() * 1000.0;
        self.step_gpu(step, &gpu, Some(upload_ms))
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

    fn decode_top1(&mut self, batch: &PackedBatch) -> Result<Vec<Vec<u32>>> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        let mask_bool = gpu.attention_mask.to_kind(Kind::Bool);
        let decoded = tch::no_grad(|| {
            let encoder_out = self.model.encode(&gpu.input_ids, &mask_bool);
            let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
            self.decode_proposal_greedy(&proposal_logits, &mask_bool)
        });
        Ok(decoded)
    }

    fn set_debug(&mut self, enabled: bool) {
        self.debug_timing = enabled;
    }

    fn eval_batch_output(&mut self, _step: usize, batch: &PackedBatch) -> Result<EvalBatchOutput> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        let mask_bool = gpu.attention_mask.to_kind(Kind::Bool);
        let target_len = gpu.target_lengths.shallow_clone();
        let input_len = gpu.input_lengths.shallow_clone();
        let (loss_val, decoded_ids, blank_fraction) = tch::no_grad(|| {
            let encoder_out = self.model.encode(&gpu.input_ids, &mask_bool);
            let proposal_logits = self.model.proposal(&encoder_out, &mask_bool);
            let argmax = proposal_logits.argmax(-1, false);
            let valid = gpu.attention_mask.to_kind(Kind::Bool);
            let blank_mask = argmax.eq(self.model.blank_id).logical_and(&valid);
            let blank_count = blank_mask
                .to_kind(Kind::Float)
                .sum(Kind::Float)
                .double_value(&[]);
            let valid_count = valid
                .to_kind(Kind::Float)
                .sum(Kind::Float)
                .double_value(&[])
                .max(1.0);
            let ctc = ctc_proposal_loss(
                &proposal_logits,
                &gpu.target_ids,
                &input_len,
                &target_len,
                self.model.blank_id,
            );
            (
                ctc.double_value(&[]),
                self.decode_proposal_greedy(&proposal_logits, &mask_bool),
                blank_count / valid_count,
            )
        });
        Ok(EvalBatchOutput {
            step: TrainerStep {
                loss: loss_val,
                rows: gpu.batch_size,
                bytes: gpu.bytes,
                input_tokens: gpu.non_padding_input_tokens,
                target_tokens: gpu.non_padding_target_tokens,
            },
            decoded_ids: Some(decoded_ids),
            blank_fraction: Some(blank_fraction),
        })
    }

    fn attach_ckpt_sender(
        &mut self,
        sender: std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>,
    ) {
        self.ckpt_sender = Some(sender);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

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
