//! Loss functions for the tch CTC-NAT training step.
//!
//! Mirrors the Python trainer (`train_ctc_nat.py:1134-1443`):
//! - CTC loss on the proposal logits
//! - Cross-entropy on masked positions for the refine decoder
//! - BCE-with-logits on the learned remask head (targets = "this position
//!   is still wrong after the first iteration's argmax fill")
//! - BCE-with-logits on the learned stop head (targets = "the entire valid
//!   span now matches the gold target")
//!
//! The hypothesis builder matches the `refine_source = "target"` path in
//! the Python trainer: gold target with a sampled fraction of positions
//! replaced by `mask_token_id`.

use tch::{Kind, Reduction, Tensor};

/// CTC loss on the proposal logits.
///
/// - `proposal_logits`: `[B, T, V]`
/// - `target_ids`:      `[B, S]` (int64, padded with 0; true length in
///   `target_lengths`)
/// - `input_lengths`:   `[B]`   valid input lengths (int64)
/// - `target_lengths`:  `[B]`
///
/// Returns a scalar tensor.
pub fn ctc_proposal_loss(
    proposal_logits: &Tensor,
    target_ids: &Tensor,
    input_lengths: &Tensor,
    target_lengths: &Tensor,
    blank_id: i64,
) -> Tensor {
    // tch `ctc_loss` expects log_probs in `[T, B, V]` order.
    let log_probs = proposal_logits
        .log_softmax(-1, Kind::Float)
        .permute([1, 0, 2]);
    Tensor::ctc_loss_tensor(
        &log_probs,
        target_ids,
        input_lengths,
        target_lengths,
        blank_id,
        Reduction::Mean,
        /*zero_infinity=*/ true,
    )
}

/// splitmix64 — same body as `crate::backend::mix64`, duplicated here
/// so this module doesn't reach into the CPU backend's crate-private
/// helpers.
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Build a refinement hypothesis by masking a random fraction of the
/// valid target positions. Matches Python's `refine_source = "target"`
/// behavior (`backend::CtcBackend::build_target_refinement_batch`).
///
/// Returns `(hypothesis_ids [B, S], mask_positions [B, S] bool,
/// hypothesis_padding_mask [B, S] bool)`.
///
/// `step` seeds the deterministic forced-mask fallback so resume and
/// Python parity stay consistent — when no positions get sampled for a
/// row, we pick a `mix64(step, row_idx)`-seeded position inside the
/// valid span rather than forcing index 0 (which biases the model).
pub fn build_target_refinement(
    target_ids: &Tensor,
    target_lengths: &Tensor,
    mask_ratio: f64,
    mask_token_id: i64,
    step: u64,
) -> (Tensor, Tensor, Tensor) {
    let (b, s) = target_ids.size2().expect("target_ids must be 2-D");
    let device = target_ids.device();

    // valid: [B, S], true where position < target_lengths[row].
    let positions = Tensor::arange(s, (Kind::Int64, device)).unsqueeze(0); // [1, S]
    let valid = positions.lt_tensor(&target_lengths.unsqueeze(-1)); // [B, S] bool

    // draw: [B, S] uniform in [0,1)
    let draw = Tensor::rand([b, s], (Kind::Float, device));
    let mask_positions = draw.lt(mask_ratio).logical_and(&valid); // bool [B, S]

    // Guarantee at least one masked position per row (matches Python
    // fallback in backend.rs:783-787). For rows where nothing was
    // picked, pick a deterministic random position inside the valid
    // span via splitmix64(step, row_idx) — matches the CPU reference
    // and avoids a position-0 bias.
    let any = mask_positions.to_kind(Kind::Int64).sum_dim_intlist(
        [1i64].as_ref(),
        /*keepdim=*/ true,
        Kind::Int64,
    ); // [B, 1]
    let needs_force = any.eq(0).logical_and(&target_lengths.unsqueeze(-1).gt(0)); // [B, 1]

    // Build forced indices on CPU (tiny, [B]) then move to device.
    let lens_cpu = target_lengths.to_device(tch::Device::Cpu);
    let mut forced_indices = vec![0i64; b as usize];
    for row in 0..b as usize {
        let len = lens_cpu.int64_value(&[row as i64]).max(1);
        let seed = mix64(step ^ ((row as u64) << 13));
        forced_indices[row] = ((seed as i64).rem_euclid(len)) as i64;
    }
    let forced_idx = Tensor::from_slice(&forced_indices)
        .view([b, 1])
        .to_device(device); // [B, 1]
    let forced_mask = positions.eq_tensor(&forced_idx).logical_and(&needs_force); // [B, S]
    let mask_positions = mask_positions.logical_or(&forced_mask);

    let mask_token = Tensor::from(mask_token_id).to_device(device);
    let hypothesis = target_ids.where_self(&mask_positions.logical_not(), &mask_token);

    (hypothesis, mask_positions, valid)
}

/// Cross-entropy on refine decoder outputs over masked positions only.
/// `refined_logits [B, S, V]`, `target_ids [B, S]`, `mask_positions [B, S]`.
pub fn refine_mlm_loss(
    refined_logits: &Tensor,
    target_ids: &Tensor,
    mask_positions: &Tensor,
) -> Tensor {
    let (b, s, v) = refined_logits.size3().expect("refined_logits must be 3-D");
    let logits_flat = refined_logits.view([b * s, v]);
    let targets_flat = target_ids.view([b * s]);
    let mask_flat = mask_positions.view([b * s]);

    let log_probs = logits_flat.log_softmax(-1, Kind::Float); // [B*S, V]
    let neg_ll = -log_probs
        .gather(1, &targets_flat.unsqueeze(-1), false)
        .squeeze_dim(-1); // [B*S]
    let masked = &neg_ll * &mask_flat.to_kind(Kind::Float);
    let denom = mask_flat
        .to_kind(Kind::Float)
        .sum(Kind::Float)
        .clamp_min(1.0);
    masked.sum(Kind::Float) / denom
}

/// BCE-with-logits for the learned remask head. Target is 1 for
/// positions where the refined argmax differs from the gold target
/// (and the position is still valid).
pub fn remask_loss(
    remask_logits: &Tensor,
    refined_logits: &Tensor,
    target_ids: &Tensor,
    valid_positions: &Tensor,
) -> Tensor {
    let refined_argmax = refined_logits.argmax(-1, false); // [B, S]
    let wrong = refined_argmax.ne_tensor(target_ids); // [B, S] bool
    let target = wrong.logical_and(valid_positions).to_kind(Kind::Float); // [B, S]
    let losses = remask_logits.binary_cross_entropy_with_logits::<Tensor>(
        &target,
        None,
        None,
        Reduction::None,
    );
    let valid_f = valid_positions.to_kind(Kind::Float);
    let denom = valid_f.sum(Kind::Float).clamp_min(1.0);
    (losses * valid_f).sum(Kind::Float) / denom
}

/// BCE-with-logits for the learned stop head. Target is 1 when the
/// refined argmax matches the gold target over every valid position in
/// the row (full-sequence correctness).
pub fn stop_loss(
    stop_logit: &Tensor,
    refined_logits: &Tensor,
    target_ids: &Tensor,
    valid_positions: &Tensor,
) -> Tensor {
    let refined_argmax = refined_logits.argmax(-1, false); // [B, S]
    let correct = refined_argmax.eq_tensor(target_ids); // [B, S]
                                                        // A row is "fully correct" if every valid position matches.
    let wrong_positions = correct.logical_not().logical_and(valid_positions); // [B, S]
    let any_wrong =
        wrong_positions
            .to_kind(Kind::Int64)
            .sum_dim_intlist([1i64].as_ref(), false, Kind::Int64); // [B]
    let target = any_wrong.eq(0).to_kind(Kind::Float); // [B]
    stop_logit.binary_cross_entropy_with_logits::<Tensor>(&target, None, None, Reduction::Mean)
}

/// Linear warmup scalar in `[0, 1]` used to ramp `refine_loss_weight`
/// during training, matching `resolve_refine_loss_weight` (Python
/// train_ctc_nat.py:185-191).
pub fn refine_weight_ramp(step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps == 0 {
        1.0
    } else {
        ((step as f64) / (warmup_steps as f64)).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn ctc_proposal_loss_is_finite_and_nonnegative() {
        let proposal = Tensor::randn([2, 6, 8], (Kind::Float, Device::Cpu));
        // pad target to 3, with actual lengths 2/3; blank id = 0
        let target = Tensor::from_slice(&[1i64, 2, 0, 3, 4, 5]).view([2, 3]);
        let in_len = Tensor::from_slice(&[6i64, 6]);
        let tgt_len = Tensor::from_slice(&[2i64, 3]);
        let loss = ctc_proposal_loss(&proposal, &target, &in_len, &tgt_len, 0);
        let v = loss.double_value(&[]);
        assert!(v.is_finite() && v >= 0.0, "loss = {v}");
    }

    #[test]
    fn build_target_refinement_masks_only_valid_positions() {
        let target = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]).view([2, 5]);
        let lens = Tensor::from_slice(&[5i64, 3]);
        let (hyp, mask_positions, valid) =
            build_target_refinement(&target, &lens, 0.5, 99, /*step=*/ 42);
        assert_eq!(hyp.size(), vec![2, 5]);
        // no mask outside valid span
        let outside_mask = mask_positions
            .logical_and(&valid.logical_not())
            .to_kind(Kind::Int64)
            .sum(Kind::Int64)
            .int64_value(&[]);
        assert_eq!(outside_mask, 0);
        // at least one masked position per non-empty row
        let per_row = mask_positions.to_kind(Kind::Int64).sum_dim_intlist(
            [1i64].as_ref(),
            false,
            Kind::Int64,
        );
        let r0 = per_row.int64_value(&[0]);
        let r1 = per_row.int64_value(&[1]);
        assert!(r0 >= 1 && r1 >= 1, "rows had 0 masks: r0={r0} r1={r1}");
    }

    #[test]
    fn build_target_refinement_forced_mask_spreads_across_positions() {
        // With mask_ratio = 0, every row hits the forced-mask fallback.
        // Over many steps the chosen position must NOT always be 0.
        let target = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6, 7, 8]).view([1, 8]);
        let lens = Tensor::from_slice(&[8i64]);
        let mut seen = std::collections::HashSet::new();
        for step in 0..32 {
            let (_hyp, mask_positions, _valid) =
                build_target_refinement(&target, &lens, 0.0, 99, step);
            let chosen = (0..8)
                .find(|i| mask_positions.int64_value(&[0, *i]) != 0)
                .expect("forced mask not placed");
            seen.insert(chosen);
        }
        assert!(
            seen.len() > 1,
            "forced mask position did not vary across steps: seen={seen:?}"
        );
    }

    #[test]
    fn refine_mlm_loss_is_bounded_by_log_v() {
        // With uniform logits, CE = log(V); that's the maximum per-token
        // expected loss for a random init.
        let logits = Tensor::zeros([2, 4, 8], (Kind::Float, Device::Cpu));
        let target = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6, 7, 0]).view([2, 4]);
        let mask = Tensor::ones([2, 4], (Kind::Bool, Device::Cpu));
        let loss = refine_mlm_loss(&logits, &target, &mask).double_value(&[]);
        let expected = (8.0f64).ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn remask_and_stop_are_finite() {
        let refined = Tensor::randn([2, 4, 8], (Kind::Float, Device::Cpu));
        let target = Tensor::from_slice(&[1i64, 2, 3, 4, 5, 6, 7, 0]).view([2, 4]);
        let valid = Tensor::ones([2, 4], (Kind::Bool, Device::Cpu));
        let remask_logits = Tensor::randn([2, 4], (Kind::Float, Device::Cpu));
        let stop_logit = Tensor::randn([2], (Kind::Float, Device::Cpu));
        assert!(remask_loss(&remask_logits, &refined, &target, &valid)
            .double_value(&[])
            .is_finite());
        assert!(stop_loss(&stop_logit, &refined, &target, &valid)
            .double_value(&[])
            .is_finite());
    }
}
