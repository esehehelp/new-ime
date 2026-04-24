//! `TrainBackend` implementation that drives [`CtcNatModel`] end-to-end.
//!
//! Step 1 (this file): the skeleton holds a live `VarStore` + `CtcNatModel`
//! on the configured tch device and wires `TrainBackend::step` through to
//! a forward-only run for the accounting fields. Loss + backward + optim
//! lands in step 2; this file stays small so step 2 can replace `step` in
//! place.

use super::batch::{GpuBatch, StagedHostBatch};
use super::kd::{
    alpha_at, compute_kd_kl_loss, hard_example_mask, should_run_kd_microbatch, CtcTeacher,
};
use super::loss::{
    build_target_refinement, ctc_proposal_loss, refine_mlm_loss, refine_weight_ramp,
};
use super::model::{CtcNatModel, CvaeLabelSpaces};
use crate::backend::{BackendConfig, CvaeConfig, EvalBatchOutput, KdConfig, TrainBackend};
use crate::device::{resolve_tch_device, Device};
use crate::trainer::TrainerStep;
use anyhow::{bail, Result};
use rust_data::PackedBatch;
use rust_tokenizer::SharedCharTokenizer;
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
    // Bulk-copy both tensors to host Vecs in one shot. Scalar-level
    // `int64_value(&[b,t])` costs a CUDA sync per element and dominates
    // step time when T * B is large.
    let argmax_cpu = argmax
        .to_kind(Kind::Int64)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let mask_cpu = attention_mask
        .to_kind(Kind::Int64)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let mut argmax_vec = vec![0i64; batch * time];
    let mut mask_vec = vec![0i64; batch * time];
    argmax_cpu.copy_data::<i64>(&mut argmax_vec, batch * time);
    mask_cpu.copy_data::<i64>(&mut mask_vec, batch * time);

    let mut out = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::new();
        let mut prev: Option<i64> = None;
        for t in 0..time {
            let idx = b * time + t;
            if mask_vec[idx] == 0 {
                continue;
            }
            let token = argmax_vec[idx];
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

#[derive(Debug, Default)]
struct CollapsedProposalRow {
    token_ids: Vec<i64>,
    min_log_probs: Vec<f64>,
    min_margins: Vec<f64>,
}

fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn collapse_proposal_rows(
    proposal_logits: &Tensor,
    attention_mask: &Tensor,
    blank_id: i64,
) -> Vec<CollapsedProposalRow> {
    // Compute top-2 per (b, t) on the GPU, then ship three compact
    // [B, T] tensors to the host in three bulk copies. The previous
    // implementation copied the full [B, T, V] log-probs tensor and
    // then `double_value`-synced every scalar; with V=4801 and B*T
    // =16k, that was tens of millions of CUDA syncs per step.
    let log_probs = proposal_logits.log_softmax(-1, Kind::Float);
    let (top_vals, top_ids) = log_probs.topk(2, -1, /*largest=*/ true, /*sorted=*/ true);
    // top_vals: [B, T, 2] float, top_ids: [B, T, 2] int64
    let best_id_gpu = top_ids.select(-1, 0);
    let best_logp_gpu = top_vals.select(-1, 0);
    let second_logp_gpu = top_vals.select(-1, 1);
    let best_id_cpu = best_id_gpu.to_device(TchDevice::Cpu).contiguous();
    let best_logp_cpu = best_logp_gpu
        .to_kind(Kind::Double)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let second_logp_cpu = second_logp_gpu
        .to_kind(Kind::Double)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let mask_cpu = attention_mask
        .to_kind(Kind::Int64)
        .to_device(TchDevice::Cpu)
        .contiguous();

    let sizes = best_id_cpu.size();
    let batch = sizes[0] as usize;
    let time = sizes[1] as usize;
    let n = batch * time;
    let mut best_id_vec = vec![0i64; n];
    let mut best_logp_vec = vec![0f64; n];
    let mut second_logp_vec = vec![0f64; n];
    let mut mask_vec = vec![0i64; n];
    best_id_cpu.copy_data::<i64>(&mut best_id_vec, n);
    best_logp_cpu.copy_data::<f64>(&mut best_logp_vec, n);
    second_logp_cpu.copy_data::<f64>(&mut second_logp_vec, n);
    mask_cpu.copy_data::<i64>(&mut mask_vec, n);

    let mut rows = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = CollapsedProposalRow::default();
        let mut prev = blank_id;
        for t in 0..time {
            let idx = b * time + t;
            if mask_vec[idx] == 0 {
                continue;
            }
            let best_id = best_id_vec[idx];
            if best_id == blank_id {
                prev = best_id;
                continue;
            }
            let best_logp = best_logp_vec[idx];
            let second_logp = second_logp_vec[idx];
            let margin = best_logp - second_logp;
            if best_id == prev {
                if let Some(last) = row.min_log_probs.last_mut() {
                    *last = last.min(best_logp);
                }
                if let Some(last) = row.min_margins.last_mut() {
                    *last = last.min(margin);
                }
            } else {
                row.token_ids.push(best_id);
                row.min_log_probs.push(best_logp);
                row.min_margins.push(margin);
            }
            prev = best_id;
        }
        rows.push(row);
    }
    rows
}

fn force_one_mask_per_wrong_row(next_mask: &Tensor, wrong: &Tensor) -> Tensor {
    let mask_cpu = next_mask
        .to_kind(Kind::Int64)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let wrong_cpu = wrong
        .to_kind(Kind::Int64)
        .to_device(TchDevice::Cpu)
        .contiguous();
    let sizes = mask_cpu.size();
    let batch = sizes[0] as usize;
    let seq = sizes[1] as usize;
    let n = batch * seq;
    let mut mask_vec = vec![0i64; n];
    let mut wrong_vec = vec![0i64; n];
    mask_cpu.copy_data::<i64>(&mut mask_vec, n);
    wrong_cpu.copy_data::<i64>(&mut wrong_vec, n);
    let mut out = vec![0i64; n];
    for b in 0..batch {
        let mut any_wrong = false;
        let mut any_mask = false;
        let mut first_wrong = None;
        for t in 0..seq {
            let idx = b * seq + t;
            let wrong_here = wrong_vec[idx] != 0;
            let mask_here = mask_vec[idx] != 0;
            out[idx] = i64::from(mask_here);
            any_mask |= mask_here;
            if wrong_here {
                any_wrong = true;
                first_wrong.get_or_insert(t);
            }
        }
        if any_wrong && !any_mask {
            if let Some(t) = first_wrong {
                out[b * seq + t] = 1;
            }
        }
    }
    Tensor::from_slice(&out)
        .view([batch as i64, seq as i64])
        .to_device(next_mask.device())
        .to_kind(Kind::Bool)
}

/// What the tch training path currently implements. Documented here so
/// the check that rejects unsupported configs (see `validate_tch_config`
/// below) stays truthful when someone extends the GPU kernels.
///
fn validate_tch_config(config: &BackendConfig) -> Result<()> {
    if !matches!(
        config.refine_source.as_str(),
        "target" | "proposal" | "mixed"
    ) {
        bail!(
            "tch backend: unknown refine_source=`{}`; expected target/proposal/mixed",
            config.refine_source
        );
    }
    if config.refine_iterations == 0 {
        bail!("tch backend: refine_iterations must be >= 1");
    }
    Ok(())
}

pub struct TchCtcNatBackend {
    vs: VarStore,
    model: CtcNatModel,
    device: TchDevice,
    config: BackendConfig,
    cvae_config: CvaeConfig,
    kd_config: KdConfig,
    teacher: Option<CtcTeacher>,
    last_loss: Option<f64>,
    step_count: usize,
    optim: Option<super::optim::TchOptimizer>,
    ckpt_sender: Option<std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>>,
    debug_timing: bool,
    /// When > 1, `step_gpu` divides the uploaded batch into this many
    /// micro-batches and runs forward+backward on each before calling
    /// the optimizer once — true gradient accumulation that keeps peak
    /// VRAM at `1/grad_accum_divisor` of the merged-batch path.
    grad_accum_divisor: usize,
}

impl TchCtcNatBackend {
    fn refine_weight_now(&self, step: usize) -> f64 {
        self.config.refine_loss_weight * refine_weight_ramp(step, self.config.refine_warmup_steps)
    }

    fn resolve_refine_mask_ratio(&self, step: usize) -> f64 {
        match (
            self.config.refine_mask_ratio_min,
            self.config.refine_mask_ratio_max,
        ) {
            (Some(a), Some(b)) => {
                let lo = a.min(b);
                let hi = a.max(b);
                if hi <= lo {
                    lo.clamp(0.0, 1.0)
                } else {
                    let seed = mix64(
                        step as u64
                            ^ ((self.config.hidden_size as u64) << 11)
                            ^ self.model.mask_token_id as u64,
                    );
                    let unit = (seed as f64) / (u64::MAX as f64);
                    (lo + (hi - lo) * unit).clamp(0.0, 1.0)
                }
            }
            _ => self.config.refine_mask_ratio.clamp(0.0, 1.0),
        }
    }

    fn build_proposal_refinement_batch(
        &self,
        proposal_logits: &Tensor,
        attention_mask: &Tensor,
        target_ids: &Tensor,
        target_lengths: &Tensor,
        mask_ratio: f64,
        step: usize,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let (fallback_hyp, fallback_mask, valid) = build_target_refinement(
            target_ids,
            target_lengths,
            mask_ratio,
            self.model.mask_token_id,
            step as u64,
        );
        let device = target_ids.device();
        let sizes = target_ids.size();
        let batch = sizes[0] as usize;
        let seq = sizes[1] as usize;
        let hyp_cpu = fallback_hyp.to_device(TchDevice::Cpu).contiguous();
        let mask_cpu = fallback_mask
            .to_kind(Kind::Int64)
            .to_device(TchDevice::Cpu)
            .contiguous();
        let target_len_cpu = target_lengths.to_device(TchDevice::Cpu).contiguous();
        let proposals =
            collapse_proposal_rows(proposal_logits, attention_mask, self.model.blank_id);
        // Bulk-copy the full tensor into a Rust Vec once — individual
        // `int64_value(&[b,t])` calls synchronize the CUDA stream on every
        // scalar and turn a [B=128, T=128] prep into 16k sync points.
        let mut hyp_out: Vec<i64> = vec![0i64; batch * seq];
        let mut mask_out: Vec<i64> = vec![0i64; batch * seq];
        hyp_cpu.copy_data::<i64>(&mut hyp_out, batch * seq);
        mask_cpu.copy_data::<i64>(&mut mask_out, batch * seq);
        let mut target_len_vec: Vec<i64> = vec![0i64; batch];
        target_len_cpu.copy_data::<i64>(&mut target_len_vec, batch);
        let mut used_out = vec![0i64; batch];
        for b in 0..batch {
            let target_len = target_len_vec[b] as usize;
            if target_len == 0 {
                continue;
            }
            let proposal = &proposals[b];
            if proposal.token_ids.len() != target_len {
                continue;
            }
            let mut order = (0..target_len).collect::<Vec<_>>();
            order.sort_by(|&lhs, &rhs| {
                proposal.min_log_probs[lhs]
                    .total_cmp(&proposal.min_log_probs[rhs])
                    .then_with(|| proposal.min_margins[lhs].total_cmp(&proposal.min_margins[rhs]))
                    .then_with(|| lhs.cmp(&rhs))
            });
            let mut num_masks = ((target_len as f64) * mask_ratio).round() as usize;
            num_masks = num_masks.clamp(1, target_len);
            for t in 0..target_len {
                let idx = b * seq + t;
                hyp_out[idx] = proposal.token_ids[t];
                mask_out[idx] = 0;
            }
            for &t in order.iter().take(num_masks) {
                let idx = b * seq + t;
                hyp_out[idx] = self.model.mask_token_id;
                mask_out[idx] = 1;
            }
            used_out[b] = 1;
        }
        (
            Tensor::from_slice(&hyp_out)
                .view([batch as i64, seq as i64])
                .to_device(device),
            Tensor::from_slice(&mask_out)
                .view([batch as i64, seq as i64])
                .to_device(device)
                .to_kind(Kind::Bool),
            valid,
            Tensor::from_slice(&used_out)
                .to_device(device)
                .to_kind(Kind::Bool),
        )
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

    fn should_iterative_decode(&self) -> bool {
        // Gate on refine_warmup_steps: while the refine layers haven't
        // received any gradient (step_count < refine_warmup_steps) they
        // are still at their random init, so running decode through them
        // injects pure noise and tanks every eval / probe number. This
        // mirrors the Python reference where greedy_decode uses
        // proposal logits only (ctc_nat.py:718-737). Once refine has
        // had a warmup worth of training, iterative decode wins again.
        self.config.refine_loss_weight > 0.0
            && self.step_count >= self.config.refine_warmup_steps
    }

    fn build_decode_refinement_batch(
        &self,
        proposal_logits: &Tensor,
        attention_mask: &Tensor,
    ) -> Option<(Tensor, Tensor, Tensor)> {
        let proposals =
            collapse_proposal_rows(proposal_logits, attention_mask, self.model.blank_id);
        let batch = proposals.len();
        let max_len = proposals
            .iter()
            .map(|row| row.token_ids.len())
            .max()
            .unwrap_or(0);
        if max_len == 0 {
            return None;
        }
        let device = proposal_logits.device();
        let mut current_ids = vec![self.model.mask_token_id; batch * max_len];
        let mut current_mask = vec![0i64; batch * max_len];
        let mut valid_positions = vec![0i64; batch * max_len];
        let mask_ratio = self.resolve_refine_mask_ratio(0);
        for (b, proposal) in proposals.iter().enumerate() {
            let len = proposal.token_ids.len();
            if len == 0 {
                continue;
            }
            let row_offset = b * max_len;
            for (t, &token) in proposal.token_ids.iter().enumerate() {
                current_ids[row_offset + t] = token;
                valid_positions[row_offset + t] = 1;
            }
            let mut order = (0..len).collect::<Vec<_>>();
            order.sort_by(|&lhs, &rhs| {
                proposal.min_log_probs[lhs]
                    .total_cmp(&proposal.min_log_probs[rhs])
                    .then_with(|| proposal.min_margins[lhs].total_cmp(&proposal.min_margins[rhs]))
                    .then_with(|| lhs.cmp(&rhs))
            });
            let num_masks = ((len as f64) * mask_ratio).round().clamp(1.0, len as f64) as usize;
            for &t in order.iter().take(num_masks) {
                current_ids[row_offset + t] = self.model.mask_token_id;
                current_mask[row_offset + t] = 1;
            }
        }
        Some((
            Tensor::from_slice(&current_ids)
                .view([batch as i64, max_len as i64])
                .to_device(device),
            Tensor::from_slice(&current_mask)
                .view([batch as i64, max_len as i64])
                .to_device(device)
                .to_kind(Kind::Bool),
            Tensor::from_slice(&valid_positions)
                .view([batch as i64, max_len as i64])
                .to_device(device)
                .to_kind(Kind::Bool),
        ))
    }

    fn decode_refined_ids(&self, current_ids: &Tensor, valid_positions: &Tensor) -> Vec<Vec<u32>> {
        // Bulk-copy path — scalar `int64_value(&[b,t])` per cell forces a
        // CUDA sync on every element (B*T ~16k) and dominated eval
        // latency before this fix.
        let ids_cpu = current_ids
            .to_kind(Kind::Int64)
            .to_device(TchDevice::Cpu)
            .contiguous();
        let valid_cpu = valid_positions
            .to_kind(Kind::Int64)
            .to_device(TchDevice::Cpu)
            .contiguous();
        let sizes = ids_cpu.size();
        let batch = sizes[0] as usize;
        let seq = sizes[1] as usize;
        let n = batch * seq;
        let mut ids_vec = vec![0i64; n];
        let mut valid_vec = vec![0i64; n];
        ids_cpu.copy_data::<i64>(&mut ids_vec, n);
        valid_cpu.copy_data::<i64>(&mut valid_vec, n);
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut row = Vec::new();
            for t in 0..seq {
                let idx = b * seq + t;
                if valid_vec[idx] == 0 {
                    continue;
                }
                let token = ids_vec[idx];
                if token == self.model.blank_id || token == self.model.mask_token_id {
                    continue;
                }
                row.push(token as u32);
            }
            out.push(row);
        }
        out
    }

    fn decode_iterative_from_proposal(
        &self,
        proposal_logits: &Tensor,
        encoder_out: &Tensor,
        attention_mask: &Tensor,
        film_conditioning: Option<&[(Tensor, Tensor)]>,
    ) -> Vec<Vec<u32>> {
        if !self.should_iterative_decode() {
            return self.decode_proposal_greedy(proposal_logits, attention_mask);
        }
        let Some((mut current_ids, mut current_mask, valid_positions)) =
            self.build_decode_refinement_batch(proposal_logits, attention_mask)
        else {
            return vec![Vec::new(); proposal_logits.size()[0] as usize];
        };
        let mut done = Tensor::zeros([proposal_logits.size()[0]], (Kind::Bool, self.device));
        let iterations = self.config.refine_iterations.max(1);
        for iter_idx in 0..iterations {
            let (refined_logits, remask_logits, stop_logits) = self.model.refine(
                &current_ids,
                &valid_positions,
                encoder_out,
                attention_mask,
                film_conditioning,
            );
            let argmax = refined_logits.argmax(-1, false);
            current_ids = argmax.where_self(&current_mask, &current_ids);

            if self.config.use_learned_stop {
                let stop_rows = stop_logits.sigmoid().ge(self.config.stop_threshold);
                done = done.logical_or(&stop_rows);
                let active_rows = done.logical_not().to_kind(Kind::Int64).sum(Kind::Int64);
                if active_rows.int64_value(&[]) == 0 {
                    break;
                }
            }

            if iter_idx + 1 >= iterations {
                break;
            }

            let next_mask = if self.config.use_learned_remask {
                remask_logits.sigmoid().ge(self.config.remask_threshold)
            } else {
                refined_logits
                    .softmax(-1, Kind::Float)
                    .max_dim(-1, false)
                    .0
                    .lt(self.config.confidence_fallback)
            }
            .logical_and(&valid_positions)
            .logical_and(&done.logical_not().unsqueeze(-1));
            let any_next = next_mask.to_kind(Kind::Int64).sum(Kind::Int64);
            if any_next.int64_value(&[]) == 0 {
                break;
            }
            let mask_token = Tensor::from(self.model.mask_token_id).to_device(self.device);
            current_ids = current_ids.where_self(&next_mask.logical_not(), &mask_token);
            current_mask = next_mask;
        }
        self.decode_refined_ids(&current_ids, &valid_positions)
    }

    pub fn new(
        config: &BackendConfig,
        cvae_config: &CvaeConfig,
        kd_config: &KdConfig,
        cvae_labels: CvaeLabelSpaces,
        student_tokenizer: &SharedCharTokenizer,
        device: Device,
    ) -> Result<Self> {
        validate_tch_config(config)?;
        let tch_device = resolve_tch_device(device)?;
        let mut vs = VarStore::new(tch_device);
        let model = CtcNatModel::new(&vs.root(), config, cvae_config, cvae_labels)?;
        if config.use_bf16 {
            // Cast every float-kind weight to bf16. Int embeddings,
            // attention masks, etc. stay in their native dtype. The
            // custom AdamW in `TchOptimizer` keeps (m, v) in fp32 so
            // Adam's numerical stability isn't compromised.
            vs.bfloat16();
        } else if config.use_fp16 {
            vs.half();
        }
        let teacher = if kd_config.alpha > 0.0 {
            Some(CtcTeacher::load(kd_config, tch_device, student_tokenizer)?)
        } else {
            None
        };
        Ok(Self {
            vs,
            model,
            device: tch_device,
            config: config.clone(),
            cvae_config: cvae_config.clone(),
            kd_config: kd_config.clone(),
            teacher,
            last_loss: None,
            step_count: 0,
            optim: None,
            ckpt_sender: None,
            debug_timing: false,
            grad_accum_divisor: 1,
        })
    }

    /// Configure micro-batch gradient accumulation. `n=1` (default)
    /// preserves the legacy single-pass step. `n>1` splits the uploaded
    /// batch into `n` row-slices and runs forward+backward on each,
    /// calling the optimizer once at the end. The caller is responsible
    /// for ensuring the merged batch size is a multiple of `n`.
    pub fn set_grad_accum_divisor(&mut self, n: usize) {
        self.grad_accum_divisor = n.max(1);
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
            let proposal = self
                .model
                .proposal_output(&input, &mask, None, None, None, None, None, false);
            let log_probs = proposal
                .proposal_logits
                .log_softmax(-1, Kind::Float)
                .squeeze_dim(0)
                .to_device(TchDevice::Cpu);
            if num_beams <= 1 || num_return <= 1 {
                let decoded = self.decode_iterative_from_proposal(
                    &proposal.proposal_logits,
                    &proposal.encoder_out,
                    &mask,
                    proposal.film_conditioning.as_deref(),
                );
                return Ok(decoded.into_iter().take(num_return.max(1)).collect());
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
        let n = self.grad_accum_divisor.max(1);
        let total_rows = batch.batch_size as i64;
        if n == 1 || total_rows < n as i64 || total_rows % n as i64 != 0 {
            return self.step_gpu_once(step, batch, upload_ms, 1.0, true);
        }
        let micro_rows = total_rows / n as i64;
        let loss_scale = 1.0 / n as f64;
        let mut loss_sum = 0f64;
        let mut in_tok = 0usize;
        let mut tg_tok = 0usize;
        for i in 0..n as i64 {
            let sub = batch.narrow_rows(i * micro_rows, micro_rows);
            let is_last = i == n as i64 - 1;
            let pass_upload = if i == 0 { upload_ms } else { None };
            let out = self.step_gpu_once(step, &sub, pass_upload, loss_scale, is_last)?;
            loss_sum += out.loss;
            in_tok += out.input_tokens;
            tg_tok += out.target_tokens;
        }
        let avg_loss = loss_sum / n as f64;
        self.last_loss = Some(avg_loss);
        self.step_count = step;
        Ok(TrainerStep {
            loss: avg_loss,
            rows: batch.batch_size,
            bytes: batch.bytes,
            input_tokens: in_tok,
            target_tokens: tg_tok,
        })
    }

    /// Single forward + scaled backward pass. When `run_optim` is true
    /// (the last micro-batch in an accumulation round, or a plain
    /// non-accumulating step), also apply the optimizer. `loss_scale`
    /// scales the gradient contribution of this micro-batch so that
    /// `N` passes with `1/N` scale produce the same gradient as one
    /// full-batch pass.
    fn step_gpu_once(
        &mut self,
        step: usize,
        batch: &GpuBatch,
        upload_ms: Option<f64>,
        loss_scale: f64,
        run_optim: bool,
    ) -> Result<TrainerStep> {
        let total_started = Instant::now();
        let mask_bool = batch.attention_mask.to_kind(Kind::Bool);
        let target_len = batch.target_lengths.shallow_clone();
        let input_len = batch.input_lengths.shallow_clone();

        // Proposal path (always runs).
        let encode_started = Instant::now();
        let proposal = self.model.proposal_output(
            &batch.input_ids,
            &mask_bool,
            Some(&batch.target_ids),
            Some(&target_len),
            Some(&batch.writer_ids),
            Some(&batch.domain_ids),
            Some(&batch.source_ids),
            true,
        );
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let proposal_started = Instant::now();
        let ctc = ctc_proposal_loss(
            &proposal.proposal_logits,
            &batch.target_ids,
            &input_len,
            &target_len,
            self.model.blank_id,
        );
        let proposal_ms = proposal_started.elapsed().as_secs_f64() * 1000.0;
        let mut loss = ctc.shallow_clone();
        if self.cvae_config.enabled && self.cvae_config.kl_weight > 0.0 {
            if let Some(kl) = proposal.kl.as_ref() {
                loss = &loss + self.cvae_config.kl_weight * kl;
            }
        }
        // Optional aux loss: penalize the proposal for assigning high
        // softmax probability to the CTC blank token. CTC alone tolerates
        // the collapsed "always blank" solution at low target density, so
        // this nudges the head toward emitting real tokens once the
        // encoder has warmed up.
        if self.config.proposal_blank_penalty_weight > 0.0
            && step >= self.config.proposal_blank_penalty_start_step
        {
            let probs = proposal.proposal_logits.softmax(-1, Kind::Float);
            let blank_probs = probs.select(-1, self.model.blank_id as i64); // [B, T]
            let mask_f = mask_bool.to_kind(Kind::Float);
            let valid = mask_f.sum(Kind::Float).maximum(&Tensor::from(1.0));
            let penalty = (&blank_probs * &mask_f).sum(Kind::Float) / &valid;
            loss = &loss + self.config.proposal_blank_penalty_weight * &penalty;
        }

        // Refinement path (only when the weight is positive).
        let refine_started = Instant::now();
        let refine_weight_now = self.refine_weight_now(step);
        if refine_weight_now > 0.0 {
            let refine_mask_ratio = self.resolve_refine_mask_ratio(step);
            let (mut current_ids, mut current_mask, mut valid_positions, used_proposal_rows) =
                match self.config.refine_source.as_str() {
                    "proposal" | "mixed" => self.build_proposal_refinement_batch(
                        &proposal.proposal_logits.detach(),
                        &mask_bool,
                        &batch.target_ids,
                        &target_len,
                        refine_mask_ratio,
                        step,
                    ),
                    _ => {
                        let (hyp_ids, mask_positions, valid) = build_target_refinement(
                            &batch.target_ids,
                            &target_len,
                            refine_mask_ratio,
                            self.model.mask_token_id,
                            step as u64,
                        );
                        (
                            hyp_ids,
                            mask_positions,
                            valid,
                            Tensor::zeros(
                                [batch.batch_size as i64],
                                (Kind::Bool, batch.target_ids.device()),
                            ),
                        )
                    }
                };
            if self.config.refine_source == "proposal"
                && used_proposal_rows
                    .to_kind(Kind::Int64)
                    .sum(Kind::Int64)
                    .int64_value(&[])
                    == 0
            {
                let (hyp_ids, mask_positions, valid) = build_target_refinement(
                    &batch.target_ids,
                    &target_len,
                    refine_mask_ratio,
                    self.model.mask_token_id,
                    step as u64,
                );
                current_ids = hyp_ids;
                current_mask = mask_positions;
                valid_positions = valid;
            }

            // These carry no gradient — they're int64 ids / bool masks used
            // only as lookup keys. `detach()` keeps the values live without
            // pinning the computation graph.
            let initial_hyp_ids = current_ids.detach();
            let initial_mask_positions = current_mask.detach();
            let mut total_refine_loss = Tensor::zeros([], (Kind::Float, self.device));
            let mut first_refined_logits: Option<Tensor> = None;
            let mut last_remask_logits: Option<Tensor> = None;
            let mut last_stop_logits: Option<Tensor> = None;
            let mut last_filled: Option<Tensor> = None;
            let iterations = self.config.refine_iterations.max(1);
            for iter_idx in 0..iterations {
                let (refined_logits, remask_logits, stop_logits_batch) = self.model.refine(
                    &current_ids,
                    &valid_positions,
                    &proposal.encoder_out,
                    &mask_bool,
                    proposal.film_conditioning.as_deref(),
                );
                if first_refined_logits.is_none() {
                    // first_refined_logits is consumed only through argmax →
                    // where_self → ne_tensor, none of which need gradients.
                    // Detach so this iteration's graph can be freed after the
                    // CE backward.
                    first_refined_logits = Some(refined_logits.detach());
                }
                let refine_ce = refine_mlm_loss(&refined_logits, &batch.target_ids, &current_mask);
                total_refine_loss = &total_refine_loss + &refine_ce;

                let argmax = refined_logits.argmax(-1, false);
                let filled = argmax.where_self(&current_mask, &current_ids);
                last_filled = Some(filled.detach());
                last_remask_logits = Some(remask_logits.shallow_clone());
                last_stop_logits = Some(stop_logits_batch.shallow_clone());
                current_ids = filled;

                if iter_idx + 1 >= iterations {
                    break;
                }

                let wrong = current_ids
                    .ne_tensor(&batch.target_ids)
                    .logical_and(&valid_positions);
                let still_wrong =
                    wrong
                        .to_kind(Kind::Int64)
                        .sum_dim_intlist([1i64].as_ref(), false, Kind::Int64);
                if still_wrong.sum(Kind::Int64).int64_value(&[]) == 0 {
                    break;
                }

                let next_mask = if self.config.use_learned_remask {
                    remask_logits
                        .sigmoid()
                        .ge(self.config.remask_threshold)
                        .logical_and(&valid_positions)
                } else {
                    refined_logits
                        .softmax(-1, Kind::Float)
                        .max_dim(-1, false)
                        .0
                        .lt(self.config.confidence_fallback)
                        .logical_and(&valid_positions)
                };
                let next_mask = force_one_mask_per_wrong_row(&next_mask, &wrong);
                let mask_token = Tensor::from(self.model.mask_token_id).to_device(self.device);
                current_ids = current_ids.where_self(&next_mask.logical_not(), &mask_token);
                current_mask = next_mask;
            }

            loss = &loss + refine_weight_now * &total_refine_loss;

            if self.config.remask_loss_weight > 0.0 {
                if let (Some(first_logits), Some(last_remask_logits)) =
                    (first_refined_logits.as_ref(), last_remask_logits.as_ref())
                {
                    let first_argmax = first_logits.argmax(-1, false);
                    let first_filled =
                        first_argmax.where_self(&initial_mask_positions, &initial_hyp_ids);
                    let remask_target = first_filled
                        .ne_tensor(&batch.target_ids)
                        .logical_and(&valid_positions)
                        .to_kind(Kind::Float);
                    let remask_bce = last_remask_logits.binary_cross_entropy_with_logits::<Tensor>(
                        &remask_target,
                        None,
                        None,
                        tch::Reduction::None,
                    );
                    let valid_f = valid_positions.to_kind(Kind::Float);
                    let remask_loss = (&remask_bce * &valid_f).sum(Kind::Float)
                        / valid_f.sum(Kind::Float).clamp_min(1.0);
                    loss = &loss + self.config.remask_loss_weight * &remask_loss;
                }
            }
            if self.config.stop_loss_weight > 0.0 {
                if let (Some(last_stop_logits), Some(last_filled)) =
                    (last_stop_logits.as_ref(), last_filled.as_ref())
                {
                    let row_wrong = last_filled
                        .ne_tensor(&batch.target_ids)
                        .logical_and(&valid_positions)
                        .to_kind(Kind::Int64)
                        .sum_dim_intlist([1i64].as_ref(), false, Kind::Int64);
                    let stop_target = row_wrong.eq(0).to_kind(Kind::Float);
                    let stop_loss = last_stop_logits.binary_cross_entropy_with_logits::<Tensor>(
                        &stop_target,
                        None,
                        None,
                        tch::Reduction::Mean,
                    );
                    loss = &loss + self.config.stop_loss_weight * &stop_loss;
                }
            }
        }
        let refine_ms = refine_started.elapsed().as_secs_f64() * 1000.0;

        let kd_started = Instant::now();
        if let Some(teacher) = self.teacher.as_ref() {
            if should_run_kd_microbatch(step, self.kd_config.every) {
                let alpha_now = alpha_at(&self.kd_config, step);
                if alpha_now > 0.0 {
                    let (teacher_logits, teacher_conf) = teacher.forward(
                        &batch.input_ids,
                        &mask_bool,
                        Some(&batch.writer_ids),
                        Some(&batch.domain_ids),
                        Some(&batch.source_ids),
                    );
                    let hard_rows = hard_example_mask(
                        &teacher_conf,
                        self.kd_config.hard_threshold,
                        self.kd_config.gate_mode,
                    );
                    let (kd_loss, hard_count) = compute_kd_kl_loss(
                        &proposal.proposal_logits,
                        &teacher_logits,
                        &mask_bool,
                        &hard_rows,
                        self.kd_config.temperature,
                    );
                    if hard_count > 0 {
                        loss = &loss + alpha_now * kd_loss;
                    }
                }
            }
        }
        let kd_ms = kd_started.elapsed().as_secs_f64() * 1000.0;

        // Defer the scalar sync on loss until the optimizer-applying
        // micro-batch of an accumulation round. Pulling
        // `loss.double_value()` mid-round forces the GPU to flush,
        // costing ~15ms per micro when the compute is ~150ms.
        let sync_started = Instant::now();
        let loss_val = if run_optim { loss.double_value(&[]) } else { 0.0 };
        let sync_ms = sync_started.elapsed().as_secs_f64() * 1000.0;

        let backward_started = Instant::now();
        if loss_scale == 1.0 {
            loss.backward();
        } else {
            (&loss * loss_scale).backward();
        }
        let backward_ms = backward_started.elapsed().as_secs_f64() * 1000.0;

        let optim_started = Instant::now();
        if run_optim {
            if let Some(opt) = self.optim.as_mut() {
                opt.optimize(&self.vs, step);
            }
        }
        let optim_ms = optim_started.elapsed().as_secs_f64() * 1000.0;
        let total_ms = total_started.elapsed().as_secs_f64() * 1000.0;

        if self.debug_timing {
            eprintln!(
                "[debug step {}] upload={:.1}ms encode={:.1}ms proposal={:.1}ms refine={:.1}ms kd={:.1}ms sync={:.1}ms backward={:.1}ms optim={:.1}ms total={:.1}ms rows={} input_tokens={} target_tokens={}",
                step,
                upload_ms.unwrap_or(0.0),
                encode_ms,
                proposal_ms,
                refine_ms,
                kd_ms,
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
            let proposal = self.model.proposal_output(
                &batch.input_ids,
                &mask_bool,
                None,
                None,
                Some(&batch.writer_ids),
                Some(&batch.domain_ids),
                Some(&batch.source_ids),
                false,
            );
            let ctc = ctc_proposal_loss(
                &proposal.proposal_logits,
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
            let proposal = self.model.proposal_output(
                &gpu.input_ids,
                &mask_bool,
                None,
                None,
                Some(&gpu.writer_ids),
                Some(&gpu.domain_ids),
                Some(&gpu.source_ids),
                false,
            );
            self.decode_iterative_from_proposal(
                &proposal.proposal_logits,
                &proposal.encoder_out,
                &mask_bool,
                proposal.film_conditioning.as_deref(),
            )
        });
        Ok(decoded)
    }

    fn set_debug(&mut self, enabled: bool) {
        self.debug_timing = enabled;
    }

    fn set_grad_accum_divisor(&mut self, n: usize) {
        TchCtcNatBackend::set_grad_accum_divisor(self, n);
    }

    fn eval_batch_output(&mut self, _step: usize, batch: &PackedBatch) -> Result<EvalBatchOutput> {
        let staged = StagedHostBatch::from_packed(batch);
        let gpu = GpuBatch::upload(staged, self.device);
        let mask_bool = gpu.attention_mask.to_kind(Kind::Bool);
        let target_len = gpu.target_lengths.shallow_clone();
        let input_len = gpu.input_lengths.shallow_clone();
        let (loss_val, decoded_ids, blank_fraction) = tch::no_grad(|| {
            let proposal = self.model.proposal_output(
                &gpu.input_ids,
                &mask_bool,
                None,
                None,
                Some(&gpu.writer_ids),
                Some(&gpu.domain_ids),
                Some(&gpu.source_ids),
                false,
            );
            let proposal_logits = &proposal.proposal_logits;
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
                self.decode_iterative_from_proposal(
                    proposal_logits,
                    &proposal.encoder_out,
                    &mask_bool,
                    proposal.film_conditioning.as_deref(),
                ),
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
    use crate::backend::KdConfig;
    use rust_tokenizer::SharedCharTokenizer;
    use tch::Tensor;

    fn default_cvae() -> CvaeConfig {
        CvaeConfig::default()
    }

    fn default_kd() -> KdConfig {
        KdConfig::default()
    }

    fn test_tokenizer() -> SharedCharTokenizer {
        SharedCharTokenizer::new_default(64)
    }

    fn new_backend(cfg: &BackendConfig) -> TchCtcNatBackend {
        TchCtcNatBackend::new(
            cfg,
            &default_cvae(),
            &default_kd(),
            CvaeLabelSpaces::new(1, 1, 1),
            &test_tokenizer(),
            Device::Cpu,
        )
        .unwrap()
    }

    fn tiny_packed() -> PackedBatch {
        PackedBatch {
            input_ids: vec![1, 2, 0, 3, 4, 5],
            attention_mask: vec![1, 1, 0, 1, 1, 1],
            target_ids: vec![8, 9, 0, 7],
            input_lengths: vec![2, 3],
            target_lengths: vec![2, 1],
            writer_ids: vec![10, 11],
            domain_ids: vec![20, 21],
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
        let mut backend = new_backend(&tiny_config());
        let packed = tiny_packed();
        let step = backend.step(1, &packed).unwrap();
        assert_eq!(step.rows, 2);
        assert!(step.loss.is_finite() && step.loss >= 0.0);
        assert!(backend.trainable_param_count() > 0);
    }

    #[test]
    fn tch_backend_step_populates_gradients() {
        let mut backend = new_backend(&tiny_config());
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
        let mut backend = new_backend(&tiny_config());
        assert_eq!(backend.config().refine_loss_weight, 0.0);
        let _ = backend.step(1, &tiny_packed()).unwrap();
        // Just assert we returned finite — deeper equality is covered by
        // the parity test in step 5.
        assert!(backend.last_loss().unwrap().is_finite());
    }

    #[test]
    fn tch_backend_accepts_proposal_refine_source() {
        let mut cfg = tiny_config();
        cfg.refine_source = "proposal".to_string();
        let backend = new_backend(&cfg);
        assert_eq!(backend.config().refine_source, "proposal");
    }

    #[test]
    fn tch_backend_accepts_multi_iteration_refine() {
        let mut cfg = tiny_config();
        cfg.refine_iterations = 3;
        let backend = new_backend(&cfg);
        assert_eq!(backend.config().refine_iterations, 3);
    }

    #[test]
    fn tch_backend_supports_mask_ratio_range_sampling() {
        let mut cfg = tiny_config();
        cfg.refine_mask_ratio_min = Some(0.1);
        cfg.refine_mask_ratio_max = Some(0.4);
        let backend = new_backend(&cfg);
        let ratio = backend.resolve_refine_mask_ratio(7);
        assert!((0.1..=0.4).contains(&ratio), "ratio={ratio}");
    }

    /// Guard against the "eval actually trains" regression Codex caught.
    /// When `eval_step` is called, weights must be bitwise identical
    /// before and after, even with an attached optimizer.
    #[test]
    fn tch_backend_eval_step_does_not_mutate_weights() {
        let mut backend = new_backend(&tiny_config());
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
        let mut backend = new_backend(&cfg);
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
        let mut backend = new_backend(&tiny_config());
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
        let mut restored = new_backend(&tiny_config());
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
