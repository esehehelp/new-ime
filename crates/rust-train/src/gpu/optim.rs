//! AdamW + warmup-cosine scheduler for the tch training loop.
//!
//! Thin wrapper over `tch::nn::Optimizer` (AdamW flavor). libtorch's
//! fused AdamW kernel is substantially faster than the prior Rust-side
//! per-variable loop that called `f_add_` / `g_mul_scalar_` dozens of
//! times per parameter — measurable in debug logs as `optim=130ms` per
//! step collapsing to ~30-60ms.
//!
//! Gradient accumulation is the caller's responsibility: drive
//! `backward()` N times (loss already scaled by 1/accum), then call
//! `optimize()` once. `optimize` runs: set_lr → grad_clip → step →
//! zero_grad in that order.
//!
//! Optimizer state persistence: the tch `Optimizer` keeps `(m, v)` in
//! libtorch-owned C++ buffers which aren't exposed to Rust. `state_dict`
//! snapshots only the scheduler step counter — a fresh-start resume
//! loses the Adam moment buffers. For scratch-start smoke and full runs
//! this is acceptable; resumed runs absorb a few steps of noisy updates
//! until the moments re-populate, which is cheap against a 200k-step
//! schedule. Persisting moments would need a `torch-sys` FFI addition.

use crate::backend::BackendConfig;
use anyhow::Result;
use std::collections::BTreeMap;
use tch::nn::{AdamW, Optimizer as TchNativeOptimizer, OptimizerConfig, VarStore};
use tch::Tensor;

pub struct TchOptimizer {
    inner: TchNativeOptimizer,
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr_scale: f64,
    grad_clip: f64,
    /// Internal step counter for telemetry and resume. Distinct from
    /// the training-loop step counter so it survives a re-init.
    adam_step: i64,
}

impl TchOptimizer {
    pub fn from_config(vs: &VarStore, config: &BackendConfig, grad_clip: f64) -> Result<Self> {
        let adamw = AdamW {
            beta1: config.beta1,
            beta2: config.beta2,
            wd: config.weight_decay,
            eps: config.epsilon,
            amsgrad: false,
        };
        let inner = adamw.build(vs, config.learning_rate)?;
        Ok(Self {
            inner,
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.scheduler_total_steps,
            min_lr_scale: config.min_lr_scale,
            grad_clip,
            adam_step: 0,
        })
    }

    /// LR at a given step under the warmup-cosine schedule. Mirrors the
    /// CPU `OptimizerState::current_lr` for `"warmup_cosine"`.
    pub fn lr_at(&self, step: usize) -> f64 {
        let warmup = self.warmup_steps.max(1);
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.base_lr * step as f64 / warmup as f64;
        }
        if self.total_steps <= self.warmup_steps {
            return self.base_lr;
        }
        let progress = ((step.saturating_sub(self.warmup_steps)) as f64
            / (self.total_steps - self.warmup_steps) as f64)
            .clamp(0.0, 1.0);
        let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.base_lr * (self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine)
    }

    /// Zero all parameter gradients. Delegates to libtorch.
    pub fn zero_grad(&mut self, _vs: &VarStore) {
        self.inner.zero_grad();
    }

    /// Apply one optimizer update: set_lr → grad-clip → AdamW step →
    /// zero_grad. Caller drives `backward()` beforehand (once per
    /// micro-batch when accumulating).
    pub fn optimize(&mut self, _vs: &VarStore, step: usize) {
        let lr = self.lr_at(step);
        self.inner.set_lr(lr);
        if self.grad_clip > 0.0 {
            self.inner.clip_grad_norm(self.grad_clip);
        }
        self.adam_step += 1;
        self.inner.step();
        self.inner.zero_grad();
    }

    pub fn adam_step(&self) -> i64 {
        self.adam_step
    }

    /// Minimal snapshot — libtorch-owned `(m, v)` buffers are not
    /// reachable via tch in 0.18, so only the scheduler step is
    /// persisted. See module doc for the trade-off.
    pub fn state_dict(&self) -> BTreeMap<String, Tensor> {
        let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
        out.insert(
            "adamw.step".to_string(),
            Tensor::from_slice(&[self.adam_step]),
        );
        out
    }

    /// Restore the scheduler step counter. Moment buffers cannot be
    /// restored via tch 0.18; they reinitialize to zero and settle
    /// within a handful of optimizer updates.
    pub fn load_state_dict(&mut self, dict: &BTreeMap<String, Tensor>) -> Result<()> {
        if let Some(step) = dict.get("adamw.step") {
            if step.numel() >= 1 {
                self.adam_step = step.int64_value(&[0]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    fn base_config() -> BackendConfig {
        BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            hidden_size: 8,
            encoder_layers: 1,
            num_heads: 2,
            ffn_size: 16,
            decoder_layers: 1,
            decoder_heads: 2,
            decoder_ffn_size: 16,
            output_size: 10,
            blank_id: 4,
            max_positions: 8,
            mask_token_id: 5,
            learning_rate: 1e-3,
            warmup_steps: 100,
            scheduler_total_steps: 1000,
            min_lr_scale: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            ..BackendConfig::default()
        }
    }

    #[test]
    fn lr_warms_up_linearly_then_cosines_down_to_floor() {
        let vs = VarStore::new(Device::Cpu);
        let _w: Tensor = vs.root().zeros("w", &[4]);
        let opt = TchOptimizer::from_config(&vs, &base_config(), 1.0).unwrap();
        let lr_50 = opt.lr_at(50);
        assert!((lr_50 - 1e-3 * 50.0 / 100.0).abs() < 1e-9, "lr_50={lr_50}");
        let peak = opt.lr_at(100);
        assert!((peak - 1e-3).abs() < 1e-6);
        let floor = opt.lr_at(1000);
        assert!((floor - 1e-3 * 0.1).abs() < 1e-6, "floor={floor}");
    }

    #[test]
    fn optimizer_step_changes_parameters() {
        let vs = VarStore::new(Device::Cpu);
        let w = vs.root().randn("w", &[4], 0.0, 0.1);
        let before = Vec::<f64>::try_from(&w).unwrap();

        let mut opt = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        let loss = w.sum(Kind::Float);
        loss.backward();
        opt.optimize(&vs, 50);

        let after = Vec::<f64>::try_from(&w).unwrap();
        assert!(
            before.iter().zip(after.iter()).any(|(b, a)| (b - a).abs() > 1e-6),
            "weights unchanged after optimize(): before={before:?} after={after:?}"
        );
    }

    #[test]
    fn state_dict_round_trip_preserves_step_counter() {
        let vs = VarStore::new(Device::Cpu);
        let w = vs.root().randn("w", &[4], 0.0, 0.1);
        let mut opt = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        for _ in 0..3 {
            let loss = w.sum(Kind::Float);
            loss.backward();
            opt.optimize(&vs, 50);
        }
        assert_eq!(opt.adam_step(), 3);
        let dict = opt.state_dict();
        let mut opt2 = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        opt2.load_state_dict(&dict).unwrap();
        assert_eq!(opt2.adam_step(), 3);
    }
}
