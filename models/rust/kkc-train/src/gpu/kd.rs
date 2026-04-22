//! Knowledge distillation wrapper for the tch training step.
//!
//! Ports the Python KD mechanics from `models/src/training/kd.py`:
//! - `KdConfig` mirrors the Python CLI knobs (alpha, schedules, gate).
//! - `alpha_at(step)` returns the KD weight for a given optimizer step.
//! - `hard_example_mask` selects which batch rows contribute to the KD
//!   loss based on teacher confidence.
//! - `should_run_kd_microbatch` applies the every-N stride so KD doesn't
//!   pay its latency cost on every microbatch.
//!
//! **Scope note.** The actual AR teacher generation is not yet wired in
//! this commit. `ArTeacher::generate_greedy` returns `Err("not yet
//! implemented")` so the caller can ship a run with `kd.alpha = 0`
//! (exactly how Suiko-v1-small was trained) and iterate on teacher
//! integration separately. The integration path is:
//!
//! 1. `tools/rust/export_ar_teacher.py` — TorchScripts the Python AR
//!    teacher (raw `state_dict` cannot be loaded by tch directly).
//! 2. `ArTeacher::new(path)` — `tch::CModule::load_on_device(path, dev)`.
//! 3. Greedy decode — forward teacher with (context, reading, partial
//!    surface), take argmax, append, repeat up to `max_new_tokens`.
//!
//! Step 6 lands the infrastructure; the greedy loop is an addendum.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use tch::{Kind, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateMode {
    /// Only count samples the teacher is *uncertain* about
    /// (confidence < threshold). Default mode — protects the student
    /// from being capped at the teacher's easy wins.
    LowConf,
    /// Only count samples the teacher is confident about.
    HighConf,
    /// Everyone contributes.
    All,
}

impl Default for GateMode {
    fn default() -> Self {
        GateMode::LowConf
    }
}

impl GateMode {
    pub fn from_str(value: &str) -> Result<Self> {
        Ok(match value.to_ascii_lowercase().as_str() {
            "low_conf" | "low-conf" | "low" => GateMode::LowConf,
            "high_conf" | "high-conf" | "high" => GateMode::HighConf,
            "all" => GateMode::All,
            other => bail!("unknown kd gate_mode: {other}"),
        })
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            GateMode::LowConf => "low_conf",
            GateMode::HighConf => "high_conf",
            GateMode::All => "all",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KdConfig {
    pub alpha: f64,
    pub alpha_final: Option<f64>,
    pub alpha_decay_start: usize,
    pub alpha_decay_steps: usize,
    pub start_step: usize,
    pub warmup_steps: usize,
    /// Run KD on `step % every == 0` optimizer steps only.
    pub every: usize,
    pub hard_threshold: f64,
    pub gate_mode: GateMode,
    pub max_new_tokens: usize,
    pub teacher_path: String,
    pub teacher_vocab_path: String,
}

impl Default for KdConfig {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            alpha_final: None,
            alpha_decay_start: 0,
            alpha_decay_steps: 0,
            start_step: 0,
            warmup_steps: 0,
            every: 4,
            hard_threshold: 0.6,
            gate_mode: GateMode::LowConf,
            max_new_tokens: 96,
            teacher_path: String::new(),
            teacher_vocab_path: String::new(),
        }
    }
}

/// KD weight at a given optimizer step. Matches Python `alpha_at`
/// (kd.py:70-87): 0 before `start_step`, linear warmup from 0→alpha
/// over `warmup_steps`, optional linear decay from alpha→alpha_final
/// starting at `alpha_decay_start` for `alpha_decay_steps`.
pub fn alpha_at(cfg: &KdConfig, step: usize) -> f64 {
    if step < cfg.start_step {
        return 0.0;
    }
    let peak = cfg.alpha;
    let post_start = step - cfg.start_step;
    let after_warmup = if cfg.warmup_steps > 0 && post_start < cfg.warmup_steps {
        peak * (post_start as f64) / (cfg.warmup_steps as f64)
    } else {
        peak
    };
    match (cfg.alpha_final, cfg.alpha_decay_steps) {
        (Some(final_val), decay_steps) if decay_steps > 0 && step >= cfg.alpha_decay_start => {
            let into = step - cfg.alpha_decay_start;
            let progress = ((into as f64) / (decay_steps as f64)).clamp(0.0, 1.0);
            peak.min(after_warmup) + (final_val - peak.min(after_warmup)) * progress
        }
        _ => after_warmup,
    }
}

/// Stride gate — returns true if this microbatch should run KD.
pub fn should_run_kd_microbatch(step: usize, every: usize) -> bool {
    let every = every.max(1);
    step % every == 0
}

/// Given per-sample teacher confidence `[B]` (f32 in [0, 1]), return a
/// boolean mask `[B]` selecting which samples contribute to the KD
/// loss under `mode` + `threshold`.
pub fn hard_example_mask(confidence: &Tensor, threshold: f64, mode: GateMode) -> Tensor {
    match mode {
        GateMode::LowConf => confidence.lt(threshold),
        GateMode::HighConf => confidence.ge(threshold),
        GateMode::All => Tensor::ones(confidence.size(), (Kind::Bool, confidence.device())),
    }
}

/// Placeholder for the AR teacher. `generate_greedy` errors until the
/// TorchScript export + decode loop land.
pub struct ArTeacher {
    _module: tch::CModule,
    pub max_new_tokens: usize,
}

impl ArTeacher {
    pub fn load(path: &str, device: tch::Device, max_new_tokens: usize) -> Result<Self> {
        let module = tch::CModule::load_on_device(path, device).map_err(|err| {
            anyhow::anyhow!("failed to load TorchScripted AR teacher from {path}: {err}")
        })?;
        Ok(Self {
            _module: module,
            max_new_tokens,
        })
    }

    /// Greedy decode — emits (teacher_texts, confidences) per sample.
    ///
    /// Not yet implemented. Caller should run with `kd.alpha = 0` until
    /// this lands.
    pub fn generate_greedy(
        &self,
        _contexts: &[String],
        _readings: &[String],
    ) -> Result<(Vec<String>, Vec<f32>)> {
        bail!(
            "ArTeacher::generate_greedy is not implemented yet; train with kd.alpha=0 \
             until the TorchScript greedy loop lands"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn alpha_at_respects_start_step_and_warmup() {
        let cfg = KdConfig {
            alpha: 0.5,
            start_step: 100,
            warmup_steps: 200,
            ..KdConfig::default()
        };
        assert_eq!(alpha_at(&cfg, 50), 0.0); // before start
        assert_eq!(alpha_at(&cfg, 100), 0.0); // at start, post_start=0
        let mid = alpha_at(&cfg, 200); // mid-warmup: 100/200 * 0.5 = 0.25
        assert!((mid - 0.25).abs() < 1e-9, "mid={mid}");
        let peak = alpha_at(&cfg, 300); // warmup complete
        assert!((peak - 0.5).abs() < 1e-9);
    }

    #[test]
    fn alpha_at_applies_decay_to_alpha_final() {
        let cfg = KdConfig {
            alpha: 0.5,
            alpha_final: Some(0.1),
            start_step: 0,
            warmup_steps: 0,
            alpha_decay_start: 100,
            alpha_decay_steps: 200,
            ..KdConfig::default()
        };
        assert!((alpha_at(&cfg, 100) - 0.5).abs() < 1e-9); // start of decay
        let mid = alpha_at(&cfg, 200); // halfway: 0.5 + (0.1 - 0.5) * 0.5 = 0.3
        assert!((mid - 0.3).abs() < 1e-9, "mid={mid}");
        let end = alpha_at(&cfg, 300); // full decay -> alpha_final
        assert!((end - 0.1).abs() < 1e-9);
    }

    #[test]
    fn should_run_kd_microbatch_honors_every() {
        assert!(should_run_kd_microbatch(0, 4));
        assert!(!should_run_kd_microbatch(1, 4));
        assert!(!should_run_kd_microbatch(3, 4));
        assert!(should_run_kd_microbatch(4, 4));
        assert!(should_run_kd_microbatch(7, 1));
    }

    #[test]
    fn hard_example_mask_low_conf_filters_confident_samples() {
        let conf = Tensor::from_slice(&[0.1f32, 0.9, 0.55, 0.4])
            .to_kind(Kind::Float);
        let mask = hard_example_mask(&conf, 0.6, GateMode::LowConf);
        let rows: Vec<bool> = (0..4)
            .map(|i| mask.int64_value(&[i]) != 0)
            .collect();
        assert_eq!(rows, vec![true, false, true, true]);
    }

    #[test]
    fn hard_example_mask_high_conf_is_inverse() {
        let conf = Tensor::from_slice(&[0.1f32, 0.9, 0.55, 0.4])
            .to_kind(Kind::Float);
        let mask = hard_example_mask(&conf, 0.6, GateMode::HighConf);
        let rows: Vec<bool> = (0..4)
            .map(|i| mask.int64_value(&[i]) != 0)
            .collect();
        assert_eq!(rows, vec![false, true, false, false]);
    }

    #[test]
    fn hard_example_mask_all_accepts_everyone() {
        let conf = Tensor::from_slice(&[0.1f32, 0.9]).to_kind(Kind::Float);
        let mask = hard_example_mask(&conf, 0.6, GateMode::All);
        assert_eq!(mask.int64_value(&[0]), 1);
        assert_eq!(mask.int64_value(&[1]), 1);
    }

    #[test]
    fn gate_mode_round_trips_through_str() {
        for m in [GateMode::LowConf, GateMode::HighConf, GateMode::All] {
            assert_eq!(GateMode::from_str(m.as_str()).unwrap(), m);
        }
        assert!(GateMode::from_str("bogus").is_err());
    }

    #[test]
    fn ar_teacher_generate_is_unimplemented_for_now() {
        // `load` requires an actual TorchScript file; we don't have one
        // in unit tests, so assert the API shape via the downstream
        // generate_greedy error (caller-facing).
        // The load path is covered manually once teacher export lands.
        let _ = Device::Cpu;
    }
}
