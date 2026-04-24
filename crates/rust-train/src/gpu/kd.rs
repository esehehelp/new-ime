//! Rust-native CTC teacher KD utilities.
//!
//! This module intentionally supports only the CTC-teacher path:
//! - teacher is another `rust-train` run directory
//! - checkpoint format is the active Rust safetensors layout
//! - loss is soft-KL on proposal logits with hard-example gating

use super::ckpt::{load_var_store, CheckpointMeta};
use super::model::{CtcNatModel, CvaeLabelSpaces};
use crate::backend::{BackendConfig, CvaeConfig, GateMode, KdConfig};
use anyhow::{bail, Context, Result};
use rust_model::ctc_nat_preset;
use rust_tokenizer::SharedCharTokenizer;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tch::nn::VarStore;
use tch::{Device as TchDevice, Kind, Tensor};

#[derive(Debug, Deserialize)]
struct TeacherRunManifest {
    #[serde(default)]
    tokenizer_path: String,
    #[serde(default)]
    tokenizer_max_kanji: u32,
    model_preset: String,
    vocab_size: usize,
    #[serde(default)]
    backend_hidden_size: usize,
    #[serde(default)]
    backend_encoder_layers: usize,
    #[serde(default)]
    backend_num_heads: usize,
    #[serde(default)]
    backend_ffn_size: usize,
    #[serde(default)]
    backend_decoder_layers: usize,
    #[serde(default)]
    backend_decoder_heads: usize,
    #[serde(default)]
    backend_decoder_ffn_size: usize,
    #[serde(default)]
    backend_output_size: usize,
    #[serde(default)]
    backend_blank_id: usize,
    #[serde(default)]
    backend_max_positions: usize,
    #[serde(default)]
    backend_mask_token_id: usize,
    #[serde(default)]
    cvae_enabled: bool,
    #[serde(default)]
    cvae_latent_size: usize,
    #[serde(default)]
    cvae_posterior_hidden_size: usize,
    #[serde(default)]
    cvae_label_hidden_size: usize,
    #[serde(default)]
    cvae_writer_labels: usize,
    #[serde(default)]
    cvae_domain_labels: usize,
    #[serde(default)]
    cvae_source_labels: usize,
}

#[derive(Debug, Deserialize)]
struct TeacherState {
    #[serde(default)]
    best_checkpoint: Option<String>,
    #[serde(default)]
    last_checkpoint: Option<String>,
}

pub fn alpha_at(cfg: &KdConfig, step: usize) -> f64 {
    if step < cfg.start_step {
        return 0.0;
    }
    let peak = cfg.alpha;
    let post_start = step - cfg.start_step;
    let warmed = if cfg.warmup_steps > 0 && post_start < cfg.warmup_steps {
        peak * (post_start as f64) / (cfg.warmup_steps as f64)
    } else {
        peak
    };
    match (cfg.alpha_final, cfg.alpha_decay_steps) {
        (Some(final_val), decay_steps) if decay_steps > 0 && step >= cfg.alpha_decay_start => {
            let into = step - cfg.alpha_decay_start;
            let progress = ((into as f64) / (decay_steps as f64)).clamp(0.0, 1.0);
            warmed + (final_val - warmed) * progress
        }
        _ => warmed,
    }
}

pub fn should_run_kd_microbatch(step: usize, every: usize) -> bool {
    step % every.max(1) == 0
}

pub fn hard_example_mask(confidence: &Tensor, threshold: f64, mode: GateMode) -> Tensor {
    match mode {
        GateMode::LowConf => confidence.lt(threshold),
        GateMode::HighConf => confidence.ge(threshold),
        GateMode::All => Tensor::ones(confidence.size(), (Kind::Bool, confidence.device())),
    }
}

pub fn compute_kd_kl_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    attention_mask: &Tensor,
    hard_mask: &Tensor,
    temperature: f64,
) -> (Tensor, usize) {
    let device = student_logits.device();
    let valid = hard_mask.to_device(device);
    let num_hard = valid.to_kind(Kind::Int64).sum(Kind::Int64).int64_value(&[]) as usize;
    if num_hard == 0 {
        return (Tensor::zeros([], (student_logits.kind(), device)), 0);
    }

    let idx = valid.nonzero().squeeze_dim(-1);
    let s_log = (student_logits.index_select(0, &idx) / temperature).log_softmax(-1, Kind::Float);
    let t_log = (teacher_logits.index_select(0, &idx) / temperature).log_softmax(-1, Kind::Float);
    let t_prob = t_log.exp();
    let kl = (&t_prob * (&t_log - &s_log)).sum_dim_intlist([-1i64].as_ref(), false, Kind::Float);
    let mask = attention_mask
        .index_select(0, &idx)
        .to_kind(Kind::Float)
        .to_device(device);
    let loss = (&kl * &mask).sum(Kind::Float) / mask.sum(Kind::Float).clamp_min(1.0);
    (loss * (temperature * temperature), num_hard)
}

pub struct CtcTeacher {
    _vs: VarStore,
    model: CtcNatModel,
    device: TchDevice,
    blank_id: i64,
}

impl CtcTeacher {
    pub fn load(
        kd: &KdConfig,
        device: TchDevice,
        student_tokenizer: &SharedCharTokenizer,
    ) -> Result<Self> {
        let run_dir = PathBuf::from(&kd.teacher_run_dir);
        if kd.teacher_run_dir.is_empty() {
            bail!("kd.teacher_run_dir must be set when kd.alpha > 0");
        }
        let manifest = read_manifest(&run_dir.join("run_manifest.json"))?;
        let teacher_tokenizer = load_teacher_tokenizer(&manifest)?;
        validate_tokenizer_compatibility(student_tokenizer, &teacher_tokenizer)?;

        let preset = ctc_nat_preset(&manifest.model_preset)
            .with_context(|| format!("load preset {}", manifest.model_preset))?;
        let backend = BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            parameter_count: 1,
            hidden_size: manifest.backend_hidden_size.max(preset.hidden_size),
            encoder_layers: manifest.backend_encoder_layers.max(preset.encoder_layers),
            num_heads: manifest.backend_num_heads.max(preset.num_heads),
            ffn_size: manifest.backend_ffn_size.max(preset.ffn_size),
            decoder_layers: manifest.backend_decoder_layers.max(preset.decoder_layers),
            decoder_heads: manifest.backend_decoder_heads.max(preset.num_heads),
            decoder_ffn_size: manifest.backend_decoder_ffn_size.max(preset.ffn_size),
            output_size: manifest.backend_output_size.max(manifest.vocab_size),
            blank_id: manifest.backend_blank_id.max(1),
            max_positions: manifest.backend_max_positions.max(preset.max_positions),
            mask_token_id: manifest.backend_mask_token_id.max(1),
            ..BackendConfig::default()
        };
        let cvae = CvaeConfig {
            enabled: manifest.cvae_enabled,
            latent_size: manifest.cvae_latent_size.max(1),
            posterior_hidden_size: manifest.cvae_posterior_hidden_size.max(1),
            label_hidden_size: manifest.cvae_label_hidden_size.max(1),
            ..CvaeConfig::default()
        };
        let labels = CvaeLabelSpaces::new(
            manifest.cvae_writer_labels.max(1),
            manifest.cvae_domain_labels.max(1),
            manifest.cvae_source_labels.max(1),
        );

        let anchor = resolve_teacher_anchor(&run_dir, &kd.teacher_checkpoint)?;
        let meta = read_checkpoint_meta(&anchor)?;
        let weights_path = anchor.with_file_name(&meta.weights_file);
        let mut vs = VarStore::new(device);
        let model = CtcNatModel::new(&vs.root(), &backend, &cvae, labels)?;
        load_var_store(&mut vs, &weights_path)
            .with_context(|| format!("load teacher weights {}", weights_path.display()))?;

        Ok(Self {
            _vs: vs,
            model,
            device,
            blank_id: backend.blank_id as i64,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        writer_ids: Option<&Tensor>,
        domain_ids: Option<&Tensor>,
        source_ids: Option<&Tensor>,
    ) -> (Tensor, Tensor) {
        tch::no_grad(|| {
            let proposal = self.model.proposal_output(
                input_ids,
                attention_mask,
                None,
                None,
                writer_ids,
                domain_ids,
                source_ids,
                false,
            );
            let logits = proposal.proposal_logits;
            let probs = logits.softmax(-1, Kind::Float);
            let (top_probs, top_ids) = probs.max_dim(-1, false);
            let valid = attention_mask.to_kind(Kind::Bool);
            let nonblank = top_ids.ne(self.blank_id).logical_and(&valid);
            let nonblank_f = nonblank.to_kind(Kind::Float);
            let valid_f = valid.to_kind(Kind::Float);
            let nonblank_den = nonblank_f
                .sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                .clamp_min(1.0);
            let valid_den = valid_f
                .sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                .clamp_min(1.0);
            let conf_nonblank =
                (&top_probs * &nonblank_f).sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                    / nonblank_den;
            let conf_valid =
                (&top_probs * &valid_f).sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                    / valid_den;
            let has_nonblank = nonblank
                .to_kind(Kind::Int64)
                .sum_dim_intlist([1i64].as_ref(), false, Kind::Int64)
                .gt(0);
            let mean_conf = conf_nonblank.where_self(&has_nonblank, &conf_valid);
            (logits, mean_conf.to_device(self.device))
        })
    }
}

fn read_manifest(path: &Path) -> Result<TeacherRunManifest> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn read_state(path: &Path) -> Result<TeacherState> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn read_checkpoint_meta(path: &Path) -> Result<CheckpointMeta> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn resolve_teacher_anchor(run_dir: &Path, selector: &str) -> Result<PathBuf> {
    match selector {
        "" | "best" | "last" => {
            let state = read_state(&run_dir.join("trainer_state.json"))?;
            let raw = match selector {
                "last" => state.last_checkpoint,
                _ => state.best_checkpoint.or(state.last_checkpoint),
            }
            .ok_or_else(|| anyhow::anyhow!("teacher run has no checkpoint to resolve"))?;
            Ok(resolve_anchor_path(run_dir, &raw))
        }
        other => Ok(resolve_anchor_path(run_dir, other)),
    }
}

fn resolve_anchor_path(run_dir: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    let path = if path.is_absolute() {
        path
    } else {
        run_dir.join(path)
    };
    if raw.ends_with(".backend.json") {
        path
    } else if raw.ends_with(".ckpt.json") {
        sidecar_from_ckpt(&path)
    } else {
        path
    }
}

fn sidecar_from_ckpt(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    let replaced = file_name
        .strip_suffix(".ckpt.json")
        .map(|stem| format!("{stem}.backend.json"))
        .unwrap_or_else(|| format!("{file_name}.backend.json"));
    path.with_file_name(replaced)
}

fn load_teacher_tokenizer(manifest: &TeacherRunManifest) -> Result<SharedCharTokenizer> {
    if manifest.tokenizer_path.is_empty() {
        Ok(SharedCharTokenizer::new_default(
            manifest.tokenizer_max_kanji.max(1),
        ))
    } else {
        SharedCharTokenizer::load(&manifest.tokenizer_path)
    }
}

fn validate_tokenizer_compatibility(
    student: &SharedCharTokenizer,
    teacher: &SharedCharTokenizer,
) -> Result<()> {
    if teacher.vocab_size() != student.vocab_size() {
        bail!(
            "teacher vocab size {} does not match student {}",
            teacher.vocab_size(),
            student.vocab_size()
        );
    }
    for probe in [0usize, teacher.vocab_size() / 2, teacher.vocab_size() - 1] {
        let student_tok = student.decode(&[probe as u32]);
        let teacher_tok = teacher.decode(&[probe as u32]);
        if student_tok != teacher_tok {
            bail!(
                "tokenizer mismatch at id {}: student={:?} teacher={:?}",
                probe,
                student_tok,
                teacher_tok
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_at_respects_start_step_and_warmup() {
        let cfg = KdConfig {
            alpha: 0.5,
            start_step: 100,
            warmup_steps: 200,
            ..KdConfig::default()
        };
        assert_eq!(alpha_at(&cfg, 50), 0.0);
        assert_eq!(alpha_at(&cfg, 100), 0.0);
        let mid = alpha_at(&cfg, 200);
        assert!((mid - 0.25).abs() < 1e-9, "mid={mid}");
        let peak = alpha_at(&cfg, 300);
        assert!((peak - 0.5).abs() < 1e-9);
    }

    #[test]
    fn alpha_at_applies_decay_to_alpha_final() {
        let cfg = KdConfig {
            alpha: 0.5,
            alpha_final: Some(0.1),
            alpha_decay_start: 100,
            alpha_decay_steps: 200,
            ..KdConfig::default()
        };
        assert!((alpha_at(&cfg, 100) - 0.5).abs() < 1e-9);
        let mid = alpha_at(&cfg, 200);
        assert!((mid - 0.3).abs() < 1e-9, "mid={mid}");
        let end = alpha_at(&cfg, 300);
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
    fn hard_example_mask_respects_modes() {
        let conf = Tensor::from_slice(&[0.1f32, 0.9, 0.55, 0.4]).to_kind(Kind::Float);
        let low = hard_example_mask(&conf, 0.6, GateMode::LowConf);
        let high = hard_example_mask(&conf, 0.6, GateMode::HighConf);
        let all = hard_example_mask(&conf, 0.6, GateMode::All);
        let low_rows: Vec<bool> = (0..4).map(|i| low.int64_value(&[i]) != 0).collect();
        let high_rows: Vec<bool> = (0..4).map(|i| high.int64_value(&[i]) != 0).collect();
        let all_rows: Vec<bool> = (0..4).map(|i| all.int64_value(&[i]) != 0).collect();
        assert_eq!(low_rows, vec![true, false, true, true]);
        assert_eq!(high_rows, vec![false, true, false, false]);
        assert_eq!(all_rows, vec![true, true, true, true]);
    }

    #[test]
    fn compute_kd_kl_loss_skips_when_no_hard_rows() {
        let student = Tensor::randn([2, 4, 8], (Kind::Float, tch::Device::Cpu));
        let teacher = Tensor::randn([2, 4, 8], (Kind::Float, tch::Device::Cpu));
        let mask = Tensor::ones([2, 4], (Kind::Bool, tch::Device::Cpu));
        let hard = Tensor::zeros([2], (Kind::Bool, tch::Device::Cpu));
        let (loss, rows) = compute_kd_kl_loss(&student, &teacher, &mask, &hard, 2.0);
        assert_eq!(rows, 0);
        assert_eq!(loss.double_value(&[]), 0.0);
    }
}
