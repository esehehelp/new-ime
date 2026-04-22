//! AdamW + warmup-cosine scheduler for the tch training loop.
//!
//! Wraps `tch::nn::AdamW` so the trainer gets a single object that knows
//! both the optimizer and the per-step learning rate. The schedule reuses
//! the CPU implementation from `crate::optim` so Python parity (Step 5)
//! can check they produce identical LR curves.
//!
//! Gradient accumulation is the caller's responsibility: drive backward()
//! N times (loss already scaled by 1/accum), then call `optimize()` once.
//!
//! AMP autocast is deferred to a follow-up. The Suiko-v1-small next run
//! will start in fp32; bf16/fp16 can be plugged in without changing the
//! optimizer interface.

use crate::backend::BackendConfig;
use anyhow::Result;
use tch::nn::{self, OptimizerConfig, VarStore};

/// Owned AdamW + scheduler pair. One instance per training run.
pub struct TchOptimizer {
    inner: nn::Optimizer,
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr_scale: f64,
    weight_decay: f64,
    grad_clip: f64,
}

impl TchOptimizer {
    pub fn from_config(vs: &VarStore, config: &BackendConfig, grad_clip: f64) -> Result<Self> {
        let inner = nn::AdamW {
            beta1: config.beta1,
            beta2: config.beta2,
            wd: config.weight_decay,
            eps: config.epsilon,
            amsgrad: false,
        }
        .build(vs, config.learning_rate)?;
        Ok(Self {
            inner,
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.scheduler_total_steps,
            min_lr_scale: config.min_lr_scale,
            weight_decay: config.weight_decay,
            grad_clip,
        })
    }

    /// LR at a given step under the warmup-cosine schedule. Mirrors the
    /// CPU `OptimizerState::current_lr` for `"warmup_cosine"` (optim.rs:64-76)
    /// so Python parity checks see identical LR curves.
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

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    /// Clip, update LR, take one step, and zero grads.
    ///
    /// Caller must have already driven `backward()` the appropriate number
    /// of times (once per micro-batch when accumulating).
    pub fn optimize(&mut self, step: usize) {
        if self.grad_clip > 0.0 {
            self.inner.clip_grad_norm(self.grad_clip);
        }
        let lr = self.lr_at(step);
        self.inner.set_lr(lr);
        self.inner.step();
        self.inner.zero_grad();
    }

    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }
    pub fn grad_clip(&self) -> f64 {
        self.grad_clip
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

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
        let _w: tch::Tensor = vs.root().zeros("w", &[4]);
        let opt = TchOptimizer::from_config(&vs, &base_config(), 1.0).unwrap();
        // mid-warmup: lr = base * step / warmup
        let lr_50 = opt.lr_at(50);
        assert!((lr_50 - 1e-3 * 50.0 / 100.0).abs() < 1e-9, "lr_50={lr_50}");
        // peak lr at warmup boundary (cosine progress=0 -> scale=1)
        let peak = opt.lr_at(100);
        assert!((peak - 1e-3).abs() < 1e-6);
        // at total_steps the scale hits min_lr_scale
        let floor = opt.lr_at(1000);
        assert!((floor - 1e-3 * 0.1).abs() < 1e-6, "floor={floor}");
    }

    #[test]
    fn optimizer_step_changes_parameters() {
        let vs = VarStore::new(Device::Cpu);
        let w = vs.root().randn("w", &[4], 0.0, 0.1);
        let before = Vec::<f64>::try_from(&w).unwrap();

        let mut opt = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        // synthetic loss = w.sum(); grad = 1 for every element.
        let loss = w.sum(tch::Kind::Float);
        loss.backward();
        opt.optimize(50);

        let after = Vec::<f64>::try_from(&w).unwrap();
        assert!(
            before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-8),
            "optim.step() did not change weights"
        );
    }
}
