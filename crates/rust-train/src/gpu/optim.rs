//! AdamW + warmup-cosine scheduler for the tch training loop.
//!
//! Custom AdamW implementation that owns its per-parameter
//! `(m, v)` tensors. The stock `tch::nn::AdamW` wraps a libtorch
//! optimizer whose moment buffers aren't exposed; that meant every
//! resume started AdamW from zero state, silently changing the
//! loss curve for long runs (codex flagged this repeatedly). Owning
//! the state in Rust means we can round-trip it through safetensors
//! next to the weights.
//!
//! Gradient accumulation is the caller's responsibility: drive
//! `backward()` N times (loss already scaled by 1/accum), then call
//! `optimize()` once.
//!
//! AMP autocast is deferred to a follow-up — the Suiko-v1-small next
//! run will start in fp32 and bf16 can be plugged in without changing
//! the optimizer interface.

use crate::backend::BackendConfig;
use anyhow::{bail, Result};
use std::collections::BTreeMap;
use tch::nn::VarStore;
use tch::{Kind, Tensor};

/// Per-variable momentum state carried by the optimizer. `m` and `v`
/// have the same shape + dtype + device as the trainable variable.
struct MomentState {
    m: Tensor,
    v: Tensor,
}

pub struct TchOptimizer {
    // Keyed by `VarStore::variables()` name so moments stay stable
    // across process restarts and match the safetensors layout.
    state: BTreeMap<String, MomentState>,
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr_scale: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    grad_clip: f64,
    /// Internal step counter for bias correction. Distinct from the
    /// training-loop step counter because bias correction must use
    /// the number of optimizer updates actually applied.
    adam_step: i64,
}

impl TchOptimizer {
    pub fn from_config(vs: &VarStore, config: &BackendConfig, grad_clip: f64) -> Result<Self> {
        let mut state = BTreeMap::new();
        for (name, var) in vs.variables() {
            // Only trainable vars participate in AdamW updates — but we
            // can check `requires_grad()` to be safe. Non-trainable
            // vars don't get m/v allocated.
            if !var.requires_grad() {
                continue;
            }
            let shape = var.size();
            let device = var.device();
            state.insert(
                name,
                MomentState {
                    m: Tensor::zeros(&shape, (Kind::Float, device)),
                    v: Tensor::zeros(&shape, (Kind::Float, device)),
                },
            );
        }
        Ok(Self {
            state,
            base_lr: config.learning_rate,
            warmup_steps: config.warmup_steps,
            total_steps: config.scheduler_total_steps,
            min_lr_scale: config.min_lr_scale,
            weight_decay: config.weight_decay,
            beta1: config.beta1,
            beta2: config.beta2,
            epsilon: config.epsilon,
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

    /// Zero all parameter gradients by iterating the VarStore.
    /// Called implicitly at the end of each `optimize()`.
    pub fn zero_grad(&self, vs: &VarStore) {
        for (_, var) in vs.variables() {
            if var.requires_grad() {
                let mut g = var.grad();
                if g.defined() {
                    let _ = g.zero_();
                }
            }
        }
    }

    /// Apply one optimizer update: grad-clip → AdamW step → zero_grad.
    ///
    /// Caller must have already driven `backward()` the appropriate
    /// number of times (once per micro-batch when accumulating).
    pub fn optimize(&mut self, vs: &VarStore, step: usize) {
        if self.grad_clip > 0.0 {
            clip_grad_norm(vs, self.grad_clip);
        }
        self.adam_step += 1;
        let lr = self.lr_at(step);
        let bc1 = 1.0 - self.beta1.powi(self.adam_step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.adam_step as i32);
        tch::no_grad(|| {
            for (name, var) in vs.variables() {
                if !var.requires_grad() {
                    continue;
                }
                let Some(moments) = self.state.get_mut(&name) else {
                    continue;
                };
                let grad = var.grad();
                if !grad.defined() {
                    continue;
                }
                // Decoupled weight decay on the parameter itself,
                // before the Adam update. Matches torch.optim.AdamW.
                if self.weight_decay > 0.0 {
                    let mut param = var.shallow_clone();
                    let _ = param.g_mul_scalar_(1.0 - lr * self.weight_decay);
                }
                // m = beta1 * m + (1 - beta1) * grad
                let _ = moments.m.g_mul_scalar_(self.beta1);
                let _ = moments.m.f_add_(&(&grad * (1.0 - self.beta1))).unwrap();
                // v = beta2 * v + (1 - beta2) * grad^2
                let _ = moments.v.g_mul_scalar_(self.beta2);
                let _ = moments
                    .v
                    .f_add_(&(&grad * &grad * (1.0 - self.beta2)))
                    .unwrap();
                // theta -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
                let m_hat = &moments.m / bc1;
                let v_hat = &moments.v / bc2;
                let denom = v_hat.sqrt() + self.epsilon;
                let update = m_hat / denom * lr;
                let mut param = var.shallow_clone();
                let _ = param.f_sub_(&update).unwrap();
            }
        });
        self.zero_grad(vs);
    }

    #[cfg(test)]
    pub fn adam_step(&self) -> i64 {
        self.adam_step
    }

    /// Snapshot every `(name, tensor)` needed to resume the optimizer:
    /// per-variable `m` + `v` plus the internal `adam_step` counter
    /// (encoded as a 0-D int64 tensor so the safetensors writer doesn't
    /// need a separate JSON sidecar).
    pub fn state_dict(&self) -> BTreeMap<String, Tensor> {
        let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
        for (name, moments) in self.state.iter() {
            out.insert(format!("adamw.m.{name}"), moments.m.shallow_clone());
            out.insert(format!("adamw.v.{name}"), moments.v.shallow_clone());
        }
        out.insert(
            "adamw.step".to_string(),
            Tensor::from_slice(&[self.adam_step]),
        );
        out
    }

    /// Restore from a previously saved `state_dict`. Missing entries
    /// default to zero — tolerant to a partial write or a resume from
    /// a run that didn't enable optim persistence.
    pub fn load_state_dict(&mut self, dict: &BTreeMap<String, Tensor>) -> Result<()> {
        for (name, moments) in self.state.iter_mut() {
            if let Some(m) = dict.get(&format!("adamw.m.{name}")) {
                if m.size() != moments.m.size() {
                    bail!("optim state shape mismatch for {name}");
                }
                tch::no_grad(|| {
                    let _ = moments.m.copy_(m);
                });
            }
            if let Some(v) = dict.get(&format!("adamw.v.{name}")) {
                tch::no_grad(|| {
                    let _ = moments.v.copy_(v);
                });
            }
        }
        if let Some(step) = dict.get("adamw.step") {
            if step.numel() >= 1 {
                self.adam_step = step.int64_value(&[0]);
            }
        }
        Ok(())
    }
}

/// Compute the global L2 norm of all trainable grads and rescale each
/// grad by `max_norm / total_norm` if it exceeds `max_norm`. Matches
/// `torch.nn.utils.clip_grad_norm_`.
fn clip_grad_norm(vs: &VarStore, max_norm: f64) {
    let mut total: f64 = 0.0;
    for (_, var) in vs.variables() {
        if !var.requires_grad() {
            continue;
        }
        let g = var.grad();
        if !g.defined() {
            continue;
        }
        total += (&g * &g).sum(Kind::Float).double_value(&[]);
    }
    let total_norm = total.sqrt();
    if total_norm <= max_norm || total_norm == 0.0 {
        return;
    }
    let scale = max_norm / (total_norm + 1e-6);
    tch::no_grad(|| {
        for (_, var) in vs.variables() {
            if !var.requires_grad() {
                continue;
            }
            let mut g = var.grad();
            if !g.defined() {
                continue;
            }
            let _ = g.g_mul_scalar_(scale);
        }
    });
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
        // synthetic loss = w.sum(); grad = 1 for every element.
        let loss = w.sum(Kind::Float);
        loss.backward();
        opt.optimize(&vs, 50);

        let after = Vec::<f64>::try_from(&w).unwrap();
        assert!(
            before
                .iter()
                .zip(after.iter())
                .any(|(a, b)| (a - b).abs() > 1e-8),
            "optim.step() did not change weights"
        );
    }

    #[test]
    fn state_dict_round_trip_preserves_m_v_and_step() {
        let vs = VarStore::new(Device::Cpu);
        let w = vs.root().randn("w", &[4], 0.0, 0.1);
        let mut opt = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        // Drive 3 steps so m/v are non-trivial.
        for _ in 0..3 {
            let loss = w.sum(Kind::Float);
            loss.backward();
            opt.optimize(&vs, 50);
        }
        let dict = opt.state_dict();
        assert_eq!(opt.adam_step(), 3);
        let mut opt2 = TchOptimizer::from_config(&vs, &base_config(), 0.0).unwrap();
        opt2.load_state_dict(&dict).unwrap();
        assert_eq!(opt2.adam_step(), 3);
        // m and v should match bitwise.
        let orig = &opt.state["w"];
        let restored = &opt2.state["w"];
        let dm = (&orig.m - &restored.m).abs().max().double_value(&[]);
        let dv = (&orig.v - &restored.v).abs().max().double_value(&[]);
        assert!(dm < 1e-12 && dv < 1e-12, "m/v diverged: dm={dm} dv={dv}");
    }
}
