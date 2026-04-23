use crate::ctc;
use crate::nn;
use crate::optim::OptimizerState;
use crate::tensor::TensorBatch;
use crate::trainer::TrainerStep;
use anyhow::{Context, Result};
use rust_data::PackedBatch;
use rust_tokenizer::MASK_ID;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub kind: String,
    pub optimizer: String,
    pub scheduler: String,
    pub learning_rate: f64,
    pub init_scale: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub warmup_steps: usize,
    pub scheduler_total_steps: usize,
    pub min_lr_scale: f64,
    pub parameter_count: usize,
    pub hidden_size: usize,
    pub encoder_layers: usize,
    pub num_heads: usize,
    pub ffn_size: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ffn_size: usize,
    pub output_size: usize,
    pub blank_id: usize,
    pub max_positions: usize,
    #[serde(default = "default_refine_loss_weight")]
    pub refine_loss_weight: f64,
    #[serde(default = "default_refine_warmup_steps")]
    pub refine_warmup_steps: usize,
    #[serde(default = "default_refine_mask_ratio")]
    pub refine_mask_ratio: f64,
    #[serde(default)]
    pub refine_mask_ratio_min: Option<f64>,
    #[serde(default)]
    pub refine_mask_ratio_max: Option<f64>,
    #[serde(default = "default_refine_source")]
    pub refine_source: String,
    #[serde(default = "default_refine_iterations")]
    pub refine_iterations: usize,
    #[serde(default = "default_remask_loss_weight")]
    pub remask_loss_weight: f64,
    #[serde(default = "default_refine_threshold")]
    pub remask_threshold: f64,
    #[serde(default = "default_stop_loss_weight")]
    pub stop_loss_weight: f64,
    #[serde(default = "default_refine_threshold")]
    pub stop_threshold: f64,
    #[serde(default = "default_confidence_fallback")]
    pub confidence_fallback: f64,
    #[serde(default = "default_true")]
    pub use_learned_remask: bool,
    #[serde(default = "default_true")]
    pub use_learned_stop: bool,
    #[serde(default = "default_mask_token_id")]
    pub mask_token_id: usize,
    /// Global grad-norm clip. `0.0` disables clipping. Carried on the
    /// config (instead of only the CLI flag) so the resume manifest can
    /// detect silent changes between a run and its restart.
    #[serde(default)]
    pub grad_clip: f64,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            kind: "mock".to_string(),
            optimizer: "adamw".to_string(),
            scheduler: "constant".to_string(),
            learning_rate: 1e-3,
            init_scale: 1e-2,
            weight_decay: 0.0,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            warmup_steps: 0,
            scheduler_total_steps: 0,
            min_lr_scale: 0.1,
            parameter_count: 16,
            hidden_size: 32,
            encoder_layers: 2,
            num_heads: 4,
            ffn_size: 128,
            decoder_layers: 2,
            decoder_heads: 4,
            decoder_ffn_size: 128,
            output_size: 512,
            blank_id: 4,
            max_positions: 128,
            refine_loss_weight: 0.0,
            refine_warmup_steps: 0,
            refine_mask_ratio: 0.3,
            refine_mask_ratio_min: None,
            refine_mask_ratio_max: None,
            refine_source: "target".to_string(),
            refine_iterations: 1,
            remask_loss_weight: 0.0,
            remask_threshold: 0.5,
            stop_loss_weight: 0.0,
            stop_threshold: 0.5,
            confidence_fallback: 0.5,
            use_learned_remask: true,
            use_learned_stop: true,
            mask_token_id: MASK_ID as usize,
            grad_clip: 0.0,
        }
    }
}

fn default_refine_loss_weight() -> f64 {
    0.0
}

fn default_refine_warmup_steps() -> usize {
    0
}

fn default_refine_mask_ratio() -> f64 {
    0.3
}

fn default_refine_source() -> String {
    "target".to_string()
}

fn default_refine_iterations() -> usize {
    1
}

fn default_remask_loss_weight() -> f64 {
    0.0
}

fn default_stop_loss_weight() -> f64 {
    0.0
}

fn default_refine_threshold() -> f64 {
    0.5
}

fn default_confidence_fallback() -> f64 {
    0.5
}

fn default_true() -> bool {
    true
}

fn default_mask_token_id() -> usize {
    MASK_ID as usize
}

#[derive(Debug, Clone)]
pub struct EvalBatchOutput {
    pub step: TrainerStep,
    pub decoded_ids: Option<Vec<Vec<u32>>>,
    pub blank_fraction: Option<f64>,
}

pub trait TrainBackend {
    fn kind(&self) -> &'static str;
    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep>;
    fn save_checkpoint(&self, path: &Path) -> Result<()>;
    fn load_checkpoint(&mut self, path: &Path) -> Result<()>;
    fn set_debug(&mut self, _enabled: bool) {}

    /// Forward-only evaluation. Must NOT update weights.
    ///
    /// CPU backends default to calling `step` — their `step` doesn't
    /// auto-apply an optimizer so this is safe. GPU-side backends
    /// (`tch-ctc-nat`) override to run inside `no_grad` and skip the
    /// optimizer, since their `step` is a full training iteration.
    fn eval_step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        self.step(step, batch)
    }

    fn eval_batch_output(&mut self, step: usize, batch: &PackedBatch) -> Result<EvalBatchOutput> {
        Ok(EvalBatchOutput {
            step: self.eval_step(step, batch)?,
            decoded_ids: None,
            blank_fraction: None,
        })
    }

    /// Decode-only path for cheap probe-style evaluation.
    /// Default implementation routes through `eval_batch_output`.
    fn decode_top1(&mut self, batch: &PackedBatch) -> Result<Vec<Vec<u32>>> {
        Ok(self
            .eval_batch_output(0, batch)?
            .decoded_ids
            .unwrap_or_default())
    }

    /// Hook: wire an async checkpoint writer's sender into the backend
    /// so its `save_checkpoint` bytes can be drained off-thread. CPU
    /// backends ignore this (their save output is tiny and the sync
    /// path already dominates nothing); the tch backend overrides to
    /// store the sender and submit safetensors blobs through it.
    fn attach_ckpt_sender(
        &mut self,
        _sender: std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>,
    ) {
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MockBackend {
    pub steps_seen: usize,
    pub last_loss: Option<f64>,
}

impl TrainBackend for MockBackend {
    fn kind(&self) -> &'static str {
        "mock"
    }

    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let token_mass = batch.non_padding_input_tokens() + batch.non_padding_target_tokens();
        let out = TrainerStep {
            loss: (token_mass as f64 / batch.batch_size.max(1) as f64) / (step as f64 + 1.0),
            rows: batch.batch_size,
            bytes: batch.bytes(),
            input_tokens: batch.non_padding_input_tokens(),
            target_tokens: batch.non_padding_target_tokens(),
        };
        self.steps_seen = step;
        self.last_loss = Some(out.loss);
        Ok(out)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        std::fs::write(
            path,
            serde_json::to_vec(self).context("serialize mock backend state")?,
        )
        .with_context(|| format!("write {}", path.display()))
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        *self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyBackend {
    pub optimizer: OptimizerState,
    pub weights: Vec<f64>,
    pub last_loss: Option<f64>,
}

impl ToyBackend {
    pub fn new(config: &BackendConfig) -> Self {
        let n = config.parameter_count.max(1);
        let mut weights = Vec::with_capacity(n);
        for idx in 0..n {
            weights.push(config.init_scale * (idx as f64 + 1.0) / n as f64);
        }
        Self {
            optimizer: OptimizerState::new(config).expect("valid optimizer config"),
            weights,
            last_loss: None,
        }
    }
}

impl TrainBackend for ToyBackend {
    fn kind(&self) -> &'static str {
        "toy"
    }

    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let tensor = TensorBatch::from_packed(batch);
        let feature = mean_tokens(&tensor) / tensor.batch_size.max(1) as f64;
        let dim = self.weights.len().max(1) as f64;
        let prediction = self
            .weights
            .iter()
            .enumerate()
            .map(|(idx, w)| *w * ((idx + 1) as f64 / dim))
            .sum::<f64>();
        let target = feature / (step as f64 + 1.0);
        let error = prediction - target;
        let loss = error * error;
        let mut grads = vec![0.0; self.weights.len()];
        for (idx, grad_slot) in grads.iter_mut().enumerate() {
            let basis = (idx + 1) as f64 / dim;
            *grad_slot = 2.0 * error * basis;
        }
        self.optimizer
            .update(step, "toy.weights", &mut self.weights, &grads);
        self.last_loss = Some(loss);
        Ok(TrainerStep {
            loss,
            rows: batch.batch_size,
            bytes: batch.bytes(),
            input_tokens: batch.non_padding_input_tokens(),
            target_tokens: batch.non_padding_target_tokens(),
        })
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        std::fs::write(
            path,
            serde_json::to_vec(self).context("serialize toy backend state")?,
        )
        .with_context(|| format!("write {}", path.display()))
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        *self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(())
    }
}

fn mean_tokens(batch: &TensorBatch) -> f64 {
    batch
        .input_rows()
        .zip(batch.target_rows())
        .zip(batch.attention_mask.chunks(batch.max_input_len))
        .map(|((inp, tgt), mask)| {
            let masked_input = inp
                .iter()
                .zip(mask.iter())
                .map(|(v, m)| *v as f64 * *m as f64)
                .sum::<f64>();
            masked_input + tgt.iter().copied().map(|v| v as f64).sum::<f64>()
        })
        .sum::<f64>()
}

#[derive(Clone)]
pub enum BackendKind {
    Mock(MockBackend),
    Toy(ToyBackend),
    Surrogate(SurrogateCtcBackend),
    Ctc(CtcBackend),
}

impl BackendKind {
    pub fn new(config: &BackendConfig) -> Result<Self> {
        match config.kind.as_str() {
            "mock" => Ok(Self::Mock(MockBackend::default())),
            "toy" => Ok(Self::Toy(ToyBackend::new(config))),
            "surrogate" => Ok(Self::Surrogate(SurrogateCtcBackend::new(config))),
            "ctc" => Ok(Self::Ctc(CtcBackend::new(config))),
            other => anyhow::bail!("unknown backend kind: {}", other),
        }
    }
}

impl TrainBackend for BackendKind {
    fn kind(&self) -> &'static str {
        match self {
            BackendKind::Mock(inner) => inner.kind(),
            BackendKind::Toy(inner) => inner.kind(),
            BackendKind::Surrogate(inner) => inner.kind(),
            BackendKind::Ctc(inner) => inner.kind(),
        }
    }

    fn step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        match self {
            BackendKind::Mock(inner) => inner.step(step, batch),
            BackendKind::Toy(inner) => inner.step(step, batch),
            BackendKind::Surrogate(inner) => inner.step(step, batch),
            BackendKind::Ctc(inner) => inner.step(step, batch),
        }
    }

    fn eval_step(&mut self, step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        match self {
            BackendKind::Mock(inner) => inner.eval_step(step, batch),
            BackendKind::Toy(inner) => inner.eval_step(step, batch),
            BackendKind::Surrogate(inner) => inner.eval_step(step, batch),
            BackendKind::Ctc(inner) => inner.eval_step(step, batch),
        }
    }

    fn attach_ckpt_sender(
        &mut self,
        _sender: std::sync::mpsc::SyncSender<crate::pipeline::CheckpointWrite>,
    ) {
        // CPU backends don't benefit from async — their saves are small.
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        match self {
            BackendKind::Mock(inner) => inner.save_checkpoint(path),
            BackendKind::Toy(inner) => inner.save_checkpoint(path),
            BackendKind::Surrogate(inner) => inner.save_checkpoint(path),
            BackendKind::Ctc(inner) => inner.save_checkpoint(path),
        }
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        match self {
            BackendKind::Mock(inner) => inner.load_checkpoint(path),
            BackendKind::Toy(inner) => inner.load_checkpoint(path),
            BackendKind::Surrogate(inner) => inner.load_checkpoint(path),
            BackendKind::Ctc(inner) => inner.load_checkpoint(path),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateCtcBackend {
    pub optimizer: OptimizerState,
    pub hidden_size: usize,
    pub output_size: usize,
    pub embeddings: Vec<f64>,
    pub projection: Vec<f64>,
    pub bias: Vec<f64>,
    pub last_loss: Option<f64>,
}

impl SurrogateCtcBackend {
    pub fn new(config: &BackendConfig) -> Self {
        let hidden = config.hidden_size.max(1);
        let output = config.output_size.max(2);
        let mut embeddings = Vec::with_capacity(output * hidden);
        for idx in 0..(output * hidden) {
            embeddings.push(config.init_scale * (idx as f64 + 1.0) / (output * hidden) as f64);
        }
        let mut projection = Vec::with_capacity(hidden * output);
        for idx in 0..(hidden * output) {
            projection.push(config.init_scale * (idx as f64 + 1.0) / (hidden * output) as f64);
        }
        Self {
            optimizer: OptimizerState::new(config).expect("valid optimizer config"),
            hidden_size: hidden,
            output_size: output,
            embeddings,
            projection,
            bias: vec![0.0; output],
            last_loss: None,
        }
    }

    fn embedding_row(&self, token_id: usize) -> &[f64] {
        let idx = token_id * self.hidden_size;
        &self.embeddings[idx..idx + self.hidden_size]
    }
}

impl TrainBackend for SurrogateCtcBackend {
    fn kind(&self) -> &'static str {
        "surrogate"
    }

    fn step(&mut self, _step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let tensor = TensorBatch::from_packed(batch);
        let mut total_loss = 0.0;
        let mut sample_count = 0usize;
        let mut grad_projection = vec![0.0f64; self.projection.len()];
        let mut grad_bias = vec![0.0f64; self.bias.len()];

        for row_idx in 0..tensor.batch_size {
            let input_row = &tensor.input_ids
                [row_idx * tensor.max_input_len..(row_idx + 1) * tensor.max_input_len];
            let mask_row = &tensor.attention_mask
                [row_idx * tensor.max_input_len..(row_idx + 1) * tensor.max_input_len];
            let target_row = &tensor.target_ids
                [row_idx * tensor.max_target_len..(row_idx + 1) * tensor.max_target_len];

            let mut embedded = vec![0.0f32; tensor.max_input_len * self.hidden_size];
            for (tok_idx, token) in input_row.iter().copied().enumerate() {
                if token as usize >= self.output_size {
                    anyhow::bail!(
                        "surrogate backend input token {} exceeds output_size {}",
                        token,
                        self.output_size
                    );
                }
                let src = self.embedding_row(token as usize);
                let dst =
                    &mut embedded[tok_idx * self.hidden_size..(tok_idx + 1) * self.hidden_size];
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d = *s as f32;
                }
            }
            let hidden = nn::masked_mean_over_time(&embedded, mask_row, self.hidden_size);
            if hidden.iter().all(|v| *v == 0.0) {
                continue;
            }
            let target_id = target_row.first().copied().unwrap_or(0) as usize;
            if target_id >= self.output_size {
                anyhow::bail!(
                    "surrogate backend target token {} exceeds output_size {}",
                    target_id,
                    self.output_size
                );
            }
            let logits = nn::linear(&hidden, &self.projection, &self.bias, self.output_size);
            let probs = nn::softmax(&logits);
            let (loss, grad) = nn::cross_entropy_grad(&probs, target_id);
            total_loss += loss;
            sample_count += 1;

            for out_idx in 0..self.output_size {
                let diff = grad[out_idx];
                grad_bias[out_idx] += diff;
                let base = out_idx * self.hidden_size;
                for hid_idx in 0..self.hidden_size {
                    grad_projection[base + hid_idx] += diff * hidden[hid_idx];
                }
            }
        }

        let denom = sample_count.max(1) as f64;
        scale_grads(&mut grad_projection, denom);
        scale_grads(&mut grad_bias, denom);
        self.optimizer.update(
            _step,
            "surrogate.projection",
            &mut self.projection,
            &grad_projection,
        );
        self.optimizer
            .update(_step, "surrogate.bias", &mut self.bias, &grad_bias);
        let loss = total_loss / denom;
        self.last_loss = Some(loss);
        Ok(TrainerStep {
            loss,
            rows: batch.batch_size,
            bytes: batch.bytes(),
            input_tokens: batch.non_padding_input_tokens(),
            target_tokens: batch.non_padding_target_tokens(),
        })
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        std::fs::write(
            path,
            serde_json::to_vec(self).context("serialize surrogate backend state")?,
        )
        .with_context(|| format!("write {}", path.display()))
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        *self =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// CPU-side Transformer-style CTC/NAT backend used to replace the Python trainer incrementally.
/// Teacher/KD and full production features are still incomplete.
pub struct CtcBackend {
    pub optimizer: OptimizerState,
    pub hidden_size: usize,
    pub encoder_layers: usize,
    pub num_heads: usize,
    pub ffn_size: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_ffn_size: usize,
    pub output_size: usize,
    pub blank_id: usize,
    pub max_positions: usize,
    pub refine_loss_weight: f64,
    pub refine_warmup_steps: usize,
    pub refine_mask_ratio: f64,
    pub refine_mask_ratio_min: Option<f64>,
    pub refine_mask_ratio_max: Option<f64>,
    pub refine_source: String,
    pub refine_iterations: usize,
    pub remask_loss_weight: f64,
    pub remask_threshold: f64,
    pub stop_loss_weight: f64,
    pub stop_threshold: f64,
    pub confidence_fallback: f64,
    pub use_learned_remask: bool,
    pub use_learned_stop: bool,
    pub mask_token_id: usize,
    pub token_embeddings: Vec<f64>,
    pub pos_embeddings: Vec<f64>,
    pub blocks: Vec<EncoderBlock>,
    pub decoder_pos_embeddings: Vec<f64>,
    pub decoder_blocks: Vec<DecoderBlock>,
    pub refine_token_embeddings: Vec<f64>,
    pub refine_pos_embeddings: Vec<f64>,
    pub refine_blocks: Vec<DecoderBlock>,
    pub projection: Vec<f64>,
    pub refine_projection: Vec<f64>,
    pub remask_projection: Vec<f64>,
    pub stop_projection: Vec<f64>,
    pub bias: Vec<f64>,
    pub refine_bias: Vec<f64>,
    pub remask_bias: Vec<f64>,
    pub stop_bias: Vec<f64>,
    pub last_loss: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderBlock {
    pub q_proj: Vec<f64>,
    pub k_proj: Vec<f64>,
    pub v_proj: Vec<f64>,
    pub o_proj: Vec<f64>,
    pub ff_in: Vec<f64>,
    pub ff_in_bias: Vec<f64>,
    pub ff_out: Vec<f64>,
    pub ff_out_bias: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderBlock {
    pub self_q_proj: Vec<f64>,
    pub self_k_proj: Vec<f64>,
    pub self_v_proj: Vec<f64>,
    pub self_o_proj: Vec<f64>,
    pub cross_q_proj: Vec<f64>,
    pub cross_k_proj: Vec<f64>,
    pub cross_v_proj: Vec<f64>,
    pub cross_o_proj: Vec<f64>,
    pub ff_in: Vec<f64>,
    pub ff_in_bias: Vec<f64>,
    pub ff_out: Vec<f64>,
    pub ff_out_bias: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CollapsedProposal {
    token_ids: Vec<usize>,
    min_log_prob: Vec<f64>,
    min_margin: Vec<f64>,
}

#[allow(dead_code)] // fields read by CPU refine helpers not yet wired into `step`
#[derive(Debug, Clone)]
struct RefineForward {
    input_ids: Vec<usize>,
    seed: Vec<f64>,
    hidden: Vec<f64>,
    logits: Vec<f64>,
    remask_logits: Vec<f64>,
    stop_logit: f64,
    pooled: Vec<f64>,
    caches: Vec<nn::DecoderBlockCache>,
}

#[allow(dead_code)] // returned by `iterative_refine`, currently helper-only
#[derive(Debug, Clone)]
pub struct IterativeRefineResult {
    pub final_ids: Vec<usize>,
    pub stopped_rows: bool,
    pub iterations: usize,
}

// The refine/iterative helpers below are future integration surface for
// the CPU `ctc` backend — they mirror the Python reference and exist so
// the `tch-ctc-nat` GPU port has a CPU parity oracle when we need one.
// The current `step` method uses only a subset; warning suppression keeps
// that state explicit instead of tripping unused-code lints.
#[allow(dead_code)]
impl CtcBackend {
    pub fn new(config: &BackendConfig) -> Self {
        let hidden = config.hidden_size.max(1);
        let output = config.output_size.max(2);
        let positions = config.max_positions.max(1);
        let heads = config.num_heads.max(1);
        let ffn = config.ffn_size.max(hidden);
        let layers = config.encoder_layers.max(1);
        let decoder_layers = config.decoder_layers.max(1);
        let decoder_heads = config.decoder_heads.max(1);
        let decoder_ffn = config.decoder_ffn_size.max(hidden);
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(EncoderBlock {
                q_proj: init_table(hidden * hidden, config.init_scale),
                k_proj: init_table(hidden * hidden, config.init_scale),
                v_proj: init_table(hidden * hidden, config.init_scale),
                o_proj: init_table(hidden * hidden, config.init_scale),
                ff_in: init_table(ffn * hidden, config.init_scale),
                ff_in_bias: vec![0.0; ffn],
                ff_out: init_table(hidden * ffn, config.init_scale),
                ff_out_bias: vec![0.0; hidden],
            });
        }
        let mut decoder_blocks = Vec::with_capacity(decoder_layers);
        for _ in 0..decoder_layers {
            decoder_blocks.push(DecoderBlock {
                self_q_proj: init_table(hidden * hidden, config.init_scale),
                self_k_proj: init_table(hidden * hidden, config.init_scale),
                self_v_proj: init_table(hidden * hidden, config.init_scale),
                self_o_proj: init_table(hidden * hidden, config.init_scale),
                cross_q_proj: init_table(hidden * hidden, config.init_scale),
                cross_k_proj: init_table(hidden * hidden, config.init_scale),
                cross_v_proj: init_table(hidden * hidden, config.init_scale),
                cross_o_proj: init_table(hidden * hidden, config.init_scale),
                ff_in: init_table(decoder_ffn * hidden, config.init_scale),
                ff_in_bias: vec![0.0; decoder_ffn],
                ff_out: init_table(hidden * decoder_ffn, config.init_scale),
                ff_out_bias: vec![0.0; hidden],
            });
        }
        let mut refine_blocks = Vec::with_capacity(decoder_layers);
        for _ in 0..decoder_layers {
            refine_blocks.push(DecoderBlock {
                self_q_proj: init_table(hidden * hidden, config.init_scale),
                self_k_proj: init_table(hidden * hidden, config.init_scale),
                self_v_proj: init_table(hidden * hidden, config.init_scale),
                self_o_proj: init_table(hidden * hidden, config.init_scale),
                cross_q_proj: init_table(hidden * hidden, config.init_scale),
                cross_k_proj: init_table(hidden * hidden, config.init_scale),
                cross_v_proj: init_table(hidden * hidden, config.init_scale),
                cross_o_proj: init_table(hidden * hidden, config.init_scale),
                ff_in: init_table(decoder_ffn * hidden, config.init_scale),
                ff_in_bias: vec![0.0; decoder_ffn],
                ff_out: init_table(hidden * decoder_ffn, config.init_scale),
                ff_out_bias: vec![0.0; hidden],
            });
        }
        Self {
            optimizer: OptimizerState::new(config).expect("valid optimizer config"),
            hidden_size: hidden,
            encoder_layers: layers,
            num_heads: heads,
            ffn_size: ffn,
            decoder_layers,
            decoder_heads,
            decoder_ffn_size: decoder_ffn,
            output_size: output,
            blank_id: config.blank_id % output,
            max_positions: positions,
            refine_loss_weight: config.refine_loss_weight.max(0.0),
            refine_warmup_steps: config.refine_warmup_steps,
            refine_mask_ratio: config.refine_mask_ratio.clamp(0.0, 1.0),
            refine_mask_ratio_min: config.refine_mask_ratio_min,
            refine_mask_ratio_max: config.refine_mask_ratio_max,
            refine_source: config.refine_source.clone(),
            refine_iterations: config.refine_iterations.max(1),
            remask_loss_weight: config.remask_loss_weight.max(0.0),
            remask_threshold: config.remask_threshold.clamp(0.0, 1.0),
            stop_loss_weight: config.stop_loss_weight.max(0.0),
            stop_threshold: config.stop_threshold.clamp(0.0, 1.0),
            confidence_fallback: config.confidence_fallback.clamp(0.0, 1.0),
            use_learned_remask: config.use_learned_remask,
            use_learned_stop: config.use_learned_stop,
            mask_token_id: config.mask_token_id % output,
            token_embeddings: init_table(output * hidden, config.init_scale),
            pos_embeddings: init_table(positions * hidden, config.init_scale),
            blocks,
            decoder_pos_embeddings: init_table(positions * hidden, config.init_scale),
            decoder_blocks,
            refine_token_embeddings: init_table(output * hidden, config.init_scale),
            refine_pos_embeddings: init_table(positions * hidden, config.init_scale),
            refine_blocks,
            projection: init_table(hidden * output, config.init_scale),
            refine_projection: init_table(hidden * output, config.init_scale),
            remask_projection: init_table(hidden, config.init_scale),
            stop_projection: init_table(hidden, config.init_scale),
            bias: vec![0.0; output],
            refine_bias: vec![0.0; output],
            remask_bias: vec![0.0; 1],
            stop_bias: vec![0.0; 1],
            last_loss: None,
        }
    }

    fn token_row(&self, token_id: usize) -> &[f64] {
        let base = token_id * self.hidden_size;
        &self.token_embeddings[base..base + self.hidden_size]
    }

    fn pos_row(&self, pos: usize) -> &[f64] {
        let base = (pos % self.max_positions) * self.hidden_size;
        &self.pos_embeddings[base..base + self.hidden_size]
    }

    fn refine_token_row(&self, token_id: usize) -> &[f64] {
        let base = token_id * self.hidden_size;
        &self.refine_token_embeddings[base..base + self.hidden_size]
    }

    fn refine_pos_row(&self, pos: usize) -> &[f64] {
        let base = (pos % self.max_positions) * self.hidden_size;
        &self.refine_pos_embeddings[base..base + self.hidden_size]
    }

    fn resolve_refine_loss_weight(&self, step: usize) -> f64 {
        if self.refine_loss_weight <= 0.0 || self.refine_warmup_steps == 0 {
            return self.refine_loss_weight;
        }
        let scale = (step as f64 / self.refine_warmup_steps as f64).clamp(0.0, 1.0);
        self.refine_loss_weight * scale
    }

    fn resolve_refine_mask_ratio(&self, step: usize, row_idx: usize) -> f64 {
        match (self.refine_mask_ratio_min, self.refine_mask_ratio_max) {
            (Some(a), Some(b)) => {
                let lo = a.min(b);
                let hi = a.max(b);
                if hi <= lo {
                    return lo.clamp(0.0, 1.0);
                }
                let seed = mix64(
                    step as u64
                        ^ ((row_idx as u64) << 21)
                        ^ ((self.hidden_size as u64) << 7)
                        ^ self.mask_token_id as u64,
                );
                let unit = (seed as f64) / (u64::MAX as f64);
                (lo + (hi - lo) * unit).clamp(0.0, 1.0)
            }
            _ => self.refine_mask_ratio.clamp(0.0, 1.0),
        }
    }

    fn build_target_refinement_batch(
        &self,
        target_ids: &[usize],
        step: usize,
        row_idx: usize,
        mask_ratio: f64,
    ) -> (Vec<usize>, Vec<bool>) {
        let target_len = target_ids.len();
        let mut hypothesis = target_ids.to_vec();
        let mut mask_positions = vec![false; target_len];
        if target_len == 0 {
            return (hypothesis, mask_positions);
        }
        let ratio = mask_ratio.clamp(0.0, 1.0);
        for pos in 0..target_len {
            let seed = mix64(
                (step as u64)
                    ^ ((row_idx as u64) << 16)
                    ^ ((pos as u64) << 32)
                    ^ ((target_ids[pos] as u64) << 9),
            );
            let sample = (seed as f64) / (u64::MAX as f64);
            if sample < ratio {
                mask_positions[pos] = true;
                hypothesis[pos] = self.mask_token_id;
            }
        }
        if !mask_positions.iter().any(|flag| *flag) {
            let forced = (mix64(step as u64 ^ ((row_idx as u64) << 13)) as usize) % target_len;
            mask_positions[forced] = true;
            hypothesis[forced] = self.mask_token_id;
        }
        (hypothesis, mask_positions)
    }

    fn collapse_proposal(&self, probs: &[f64], input_len: usize) -> CollapsedProposal {
        let mut token_ids: Vec<usize> = Vec::new();
        let mut min_log_prob: Vec<f64> = Vec::new();
        let mut min_margin: Vec<f64> = Vec::new();
        let mut prev = self.blank_id;
        for t in 0..input_len {
            let row = &probs[t * self.output_size..(t + 1) * self.output_size];
            let mut best_idx = self.blank_id;
            let mut best_prob = 0.0;
            let mut second_prob = 0.0;
            for (idx, &prob) in row.iter().enumerate() {
                if prob > best_prob {
                    second_prob = best_prob;
                    best_prob = prob;
                    best_idx = idx;
                } else if prob > second_prob {
                    second_prob = prob;
                }
            }
            if best_idx == self.blank_id {
                prev = best_idx;
                continue;
            }
            let log_prob = best_prob.max(1e-12).ln();
            let margin = best_prob.max(1e-12).ln() - second_prob.max(1e-12).ln();
            if best_idx == prev {
                if let Some(last) = min_log_prob.last_mut() {
                    *last = last.min(log_prob);
                }
                if let Some(last) = min_margin.last_mut() {
                    *last = last.min(margin);
                }
            } else {
                token_ids.push(best_idx);
                min_log_prob.push(log_prob);
                min_margin.push(margin);
            }
            prev = best_idx;
        }
        CollapsedProposal {
            token_ids,
            min_log_prob,
            min_margin,
        }
    }

    fn build_proposal_refinement_batch(
        &self,
        target_ids: &[usize],
        proposal: &CollapsedProposal,
        step: usize,
        row_idx: usize,
        mask_ratio: f64,
    ) -> (Vec<usize>, Vec<bool>, bool) {
        let (mut fallback_ids, mut fallback_mask) =
            self.build_target_refinement_batch(target_ids, step, row_idx, mask_ratio);
        if proposal.token_ids.len() != target_ids.len() || target_ids.is_empty() {
            return (fallback_ids, fallback_mask, false);
        }
        let mut order: Vec<usize> = (0..target_ids.len()).collect();
        order.sort_by(|&a, &b| {
            proposal.min_log_prob[a]
                .partial_cmp(&proposal.min_log_prob[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    proposal.min_margin[a]
                        .partial_cmp(&proposal.min_margin[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.cmp(&b))
        });
        let mut num_masks = ((target_ids.len() as f64) * mask_ratio).round() as usize;
        num_masks = num_masks.clamp(1, target_ids.len());
        fallback_ids.copy_from_slice(&proposal.token_ids);
        fallback_mask.fill(false);
        for &idx in order.iter().take(num_masks) {
            fallback_ids[idx] = self.mask_token_id;
            fallback_mask[idx] = true;
        }
        (fallback_ids, fallback_mask, true)
    }

    fn refine_forward(
        &self,
        encoder_hidden: &[f64],
        input_len: usize,
        hypothesis_ids: &[usize],
    ) -> RefineForward {
        let refine_len = hypothesis_ids.len().min(self.max_positions);
        let mut hidden = vec![0.0f64; refine_len * self.hidden_size];
        for t in 0..refine_len {
            let tok = self.refine_token_row(hypothesis_ids[t]);
            let pos = self.refine_pos_row(t);
            let row = &mut hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
            for hid_idx in 0..self.hidden_size {
                row[hid_idx] = (tok[hid_idx] + pos[hid_idx]).tanh();
            }
        }
        let seed = hidden.clone();
        let mut caches = Vec::with_capacity(self.refine_blocks.len());
        for block in &self.refine_blocks {
            let (next_hidden, cache) = nn::decoder_block_forward_cached(
                &hidden,
                encoder_hidden,
                refine_len,
                input_len,
                self.hidden_size,
                self.decoder_heads,
                block,
            );
            caches.push(cache);
            hidden = next_hidden;
        }
        let mut logits = vec![0.0f64; refine_len * self.output_size];
        let mut remask_logits = vec![0.0f64; refine_len];
        for t in 0..refine_len {
            let hidden_row = &hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
            let row_logits = nn::linear(
                hidden_row,
                &self.refine_projection,
                &self.refine_bias,
                self.output_size,
            );
            logits[t * self.output_size..(t + 1) * self.output_size].copy_from_slice(&row_logits);
            remask_logits[t] =
                nn::linear(hidden_row, &self.remask_projection, &self.remask_bias, 1)[0];
        }
        let pooled = nn::mean_over_rows(&hidden, refine_len, self.hidden_size);
        let stop_logit = nn::linear(&pooled, &self.stop_projection, &self.stop_bias, 1)[0];
        RefineForward {
            input_ids: hypothesis_ids[..refine_len].to_vec(),
            seed,
            hidden,
            logits,
            remask_logits,
            stop_logit,
            pooled,
            caches,
        }
    }

    pub fn iterative_refine(
        &self,
        encoder_hidden: &[f64],
        input_len: usize,
        hypothesis_ids: &[usize],
        max_iterations: usize,
    ) -> IterativeRefineResult {
        let mut current_ids = hypothesis_ids.to_vec();
        let mut done = false;
        let mut seen_iterations = 0usize;
        for it in 0..max_iterations.max(1) {
            seen_iterations = it + 1;
            let forward = self.refine_forward(encoder_hidden, input_len, &current_ids);
            let refine_len = forward.input_ids.len();
            if refine_len == 0 {
                break;
            }
            let mut argmax_ids = vec![0usize; refine_len];
            let mut max_probs = vec![0.0; refine_len];
            for t in 0..refine_len {
                let probs =
                    nn::softmax(&forward.logits[t * self.output_size..(t + 1) * self.output_size]);
                let (token, prob) = probs
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((self.blank_id, 0.0));
                argmax_ids[t] = token;
                max_probs[t] = prob;
            }
            for t in 0..refine_len {
                if current_ids[t] == self.mask_token_id {
                    current_ids[t] = argmax_ids[t];
                }
            }
            if self.use_learned_stop {
                let stop_prob = sigmoid(forward.stop_logit);
                done |= stop_prob >= self.stop_threshold;
                if done {
                    break;
                }
            }
            if it + 1 >= max_iterations.max(1) {
                break;
            }
            let mut next_mask = vec![false; refine_len];
            if self.use_learned_remask {
                for (dst, &logit) in next_mask.iter_mut().zip(forward.remask_logits.iter()) {
                    *dst = sigmoid(logit) >= self.remask_threshold;
                }
            } else {
                for (dst, &prob) in next_mask.iter_mut().zip(max_probs.iter()) {
                    *dst = prob < self.confidence_fallback;
                }
            }
            if !next_mask.iter().any(|flag| *flag) {
                break;
            }
            for (token, mask) in current_ids.iter_mut().zip(next_mask.iter().copied()) {
                if mask {
                    *token = self.mask_token_id;
                }
            }
        }
        IterativeRefineResult {
            final_ids: current_ids,
            stopped_rows: done,
            iterations: seen_iterations,
        }
    }
}

impl TrainBackend for CtcBackend {
    fn kind(&self) -> &'static str {
        "ctc"
    }

    fn step(&mut self, _step: usize, batch: &PackedBatch) -> Result<TrainerStep> {
        let mut grad_token_embeddings = vec![0.0f64; self.token_embeddings.len()];
        let mut grad_pos_embeddings = vec![0.0f64; self.pos_embeddings.len()];
        let mut grad_decoder_pos_embeddings = vec![0.0f64; self.decoder_pos_embeddings.len()];
        let mut grad_refine_token_embeddings = vec![0.0f64; self.refine_token_embeddings.len()];
        let mut grad_refine_pos_embeddings = vec![0.0f64; self.refine_pos_embeddings.len()];
        let mut grad_projection = vec![0.0f64; self.projection.len()];
        let mut grad_refine_projection = vec![0.0f64; self.refine_projection.len()];
        let mut grad_remask_projection = vec![0.0f64; self.remask_projection.len()];
        let mut grad_stop_projection = vec![0.0f64; self.stop_projection.len()];
        let mut grad_bias = vec![0.0f64; self.bias.len()];
        let mut grad_refine_bias = vec![0.0f64; self.refine_bias.len()];
        let mut grad_remask_bias = vec![0.0f64; self.remask_bias.len()];
        let mut grad_stop_bias = vec![0.0f64; self.stop_bias.len()];
        let mut grad_block_o = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.o_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_block_q = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.q_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_block_k = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.k_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_block_v = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.v_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_block_ff_in = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in.len()])
            .collect::<Vec<_>>();
        let mut grad_block_ff_in_bias = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in_bias.len()])
            .collect::<Vec<_>>();
        let mut grad_block_ff_out = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out.len()])
            .collect::<Vec<_>>();
        let mut grad_block_ff_out_bias = self
            .blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out_bias.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_self_q = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.self_q_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_self_k = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.self_k_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_self_v = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.self_v_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_self_o = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.self_o_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_cross_q = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_q_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_cross_k = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_k_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_cross_v = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_v_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_cross_o = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_o_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_ff_in = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_ff_in_bias = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in_bias.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_ff_out = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out.len()])
            .collect::<Vec<_>>();
        let mut grad_decoder_ff_out_bias = self
            .decoder_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out_bias.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_self_q = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.self_q_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_self_k = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.self_k_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_self_v = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.self_v_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_self_o = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.self_o_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_cross_q = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_q_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_cross_k = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_k_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_cross_v = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_v_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_cross_o = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.cross_o_proj.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_ff_in = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_ff_in_bias = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_in_bias.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_ff_out = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out.len()])
            .collect::<Vec<_>>();
        let mut grad_refine_ff_out_bias = self
            .refine_blocks
            .iter()
            .map(|block| vec![0.0; block.ff_out_bias.len()])
            .collect::<Vec<_>>();
        let mut total_loss = 0.0f64;
        let mut sample_count = 0usize;

        for row_idx in 0..batch.batch_size {
            let input_len = batch.input_lengths.get(row_idx).copied().unwrap_or(0) as usize;
            let target_len = batch.target_lengths.get(row_idx).copied().unwrap_or(0) as usize;
            if input_len == 0 {
                continue;
            }
            let input_row = &batch.input_ids
                [row_idx * batch.max_input_len..(row_idx + 1) * batch.max_input_len];
            let target_row = &batch.target_ids
                [row_idx * batch.max_target_len..(row_idx + 1) * batch.max_target_len];

            let mut hidden = vec![0.0f64; input_len * self.hidden_size];
            let mut probs = vec![0.0f64; input_len * self.output_size];
            let mut token_ids = Vec::with_capacity(input_len);
            let mut pos_ids = Vec::with_capacity(input_len);
            let mut block_caches = Vec::with_capacity(self.blocks.len());
            let mut decoder_caches = Vec::with_capacity(self.decoder_blocks.len());

            for t in 0..input_len {
                let token_id = input_row[t] as usize;
                if token_id >= self.output_size {
                    anyhow::bail!(
                        "ctc backend input token {} exceeds output_size {}",
                        token_id,
                        self.output_size
                    );
                }
                if t >= self.max_positions {
                    anyhow::bail!(
                        "ctc backend position {} exceeds max_positions {}",
                        t,
                        self.max_positions
                    );
                }
                let pos_id = t;
                token_ids.push(token_id);
                pos_ids.push(pos_id);
                let src_token = self.token_row(token_id);
                let src_pos = self.pos_row(pos_id);
                let hidden_row = &mut hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    hidden_row[hid_idx] = (src_token[hid_idx] + src_pos[hid_idx]).tanh();
                }
            }
            for block in &self.blocks {
                let (next_hidden, cache) = nn::encoder_block_forward_cached(
                    &hidden,
                    input_len,
                    self.hidden_size,
                    self.num_heads,
                    block,
                );
                block_caches.push(cache);
                hidden = next_hidden;
            }
            let encoder_hidden = hidden.clone();
            let mut decoder_hidden = vec![0.0; encoder_hidden.len()];
            for t in 0..input_len {
                let src_pos = self.pos_row(t);
                let dst_pos =
                    &self.decoder_pos_embeddings[t * self.hidden_size..(t + 1) * self.hidden_size];
                let row = &mut decoder_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    row[hid_idx] = (encoder_hidden[t * self.hidden_size + hid_idx]
                        + src_pos[hid_idx]
                        + dst_pos[hid_idx])
                        .tanh();
                }
            }
            let decoder_seed = decoder_hidden.clone();
            for block in &self.decoder_blocks {
                let (next_hidden, cache) = nn::decoder_block_forward_cached(
                    &decoder_hidden,
                    &encoder_hidden,
                    input_len,
                    input_len,
                    self.hidden_size,
                    self.decoder_heads,
                    block,
                );
                decoder_caches.push(cache);
                decoder_hidden = next_hidden;
            }
            for t in 0..input_len {
                let hidden_row = &decoder_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                let logits = nn::linear(hidden_row, &self.projection, &self.bias, self.output_size);
                let prob_row = nn::softmax(&logits);
                probs[t * self.output_size..(t + 1) * self.output_size].copy_from_slice(&prob_row);
            }

            let mut hypothesis_ids = Vec::with_capacity(input_len);
            let mut prev = self.blank_id;
            for t in 0..input_len {
                let row = &probs[t * self.output_size..(t + 1) * self.output_size];
                let (token, _) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((self.blank_id, 0.0));
                if token != self.blank_id && token != prev {
                    hypothesis_ids.push(token);
                }
                prev = token;
            }
            if hypothesis_ids.is_empty() {
                hypothesis_ids.push(self.blank_id);
            }

            let refine_len = hypothesis_ids.len().min(self.max_positions);
            let mut refine_hidden = vec![0.0f64; refine_len * self.hidden_size];
            let mut refine_token_ids = Vec::with_capacity(refine_len);
            let mut refine_caches = Vec::with_capacity(self.refine_blocks.len());
            for t in 0..refine_len {
                let token_id = hypothesis_ids[t];
                refine_token_ids.push(token_id);
                let tok = self.refine_token_row(token_id);
                let pos = self.refine_pos_row(t);
                let row = &mut refine_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    row[hid_idx] = (tok[hid_idx] + pos[hid_idx]).tanh();
                }
            }
            let refine_seed = refine_hidden.clone();
            for block in &self.refine_blocks {
                let (next_hidden, cache) = nn::decoder_block_forward_cached(
                    &refine_hidden,
                    &encoder_hidden,
                    refine_len,
                    input_len,
                    self.hidden_size,
                    self.decoder_heads,
                    block,
                );
                refine_caches.push(cache);
                refine_hidden = next_hidden;
            }
            let mut refine_logits = vec![0.0f64; refine_len * self.output_size];
            let mut remask_logits = vec![0.0f64; refine_len];
            for t in 0..refine_len {
                let hidden_row = &refine_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                let logits = nn::linear(
                    hidden_row,
                    &self.refine_projection,
                    &self.refine_bias,
                    self.output_size,
                );
                refine_logits[t * self.output_size..(t + 1) * self.output_size]
                    .copy_from_slice(&logits);
                remask_logits[t] =
                    nn::linear(hidden_row, &self.remask_projection, &self.remask_bias, 1)[0];
            }
            let pooled = nn::mean_over_rows(&refine_hidden, refine_len, self.hidden_size);
            let stop_logit = nn::linear(&pooled, &self.stop_projection, &self.stop_bias, 1)[0];

            let targets: Vec<usize> = target_row
                .iter()
                .copied()
                .take(target_len)
                .map(|v| v as usize)
                .collect();
            if let Some(&bad) = targets.iter().find(|&&v| v >= self.output_size) {
                anyhow::bail!(
                    "ctc backend target token {} exceeds output_size {}",
                    bad,
                    self.output_size
                );
            }
            let ctc_out = ctc::ctc_loss_and_grad(
                &probs,
                input_len,
                &targets,
                self.blank_id,
                self.output_size,
            );
            if ctc_out.loss == 0.0 && ctc_out.grad_logits.iter().all(|v| *v == 0.0) {
                continue;
            }
            total_loss += ctc_out.loss;
            sample_count += 1;

            let mut grad_hidden_total = vec![0.0f64; decoder_hidden.len()];
            for t in 0..input_len {
                let grad_row =
                    &ctc_out.grad_logits[t * self.output_size..(t + 1) * self.output_size];
                let hidden_row = &decoder_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                let mut grad_hidden = vec![0.0f64; self.hidden_size];

                for out_idx in 0..self.output_size {
                    let diff = grad_row[out_idx];
                    grad_bias[out_idx] += diff;
                    let base = out_idx * self.hidden_size;
                    for hid_idx in 0..self.hidden_size {
                        grad_projection[base + hid_idx] += diff * hidden_row[hid_idx];
                        grad_hidden[hid_idx] += diff * self.projection[base + hid_idx];
                    }
                }
                grad_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size]
                    .copy_from_slice(&grad_hidden);
            }

            let aligned_refine_len = refine_len.min(target_len);
            let mut grad_refine_hidden_total = vec![0.0f64; refine_hidden.len()];
            for t in 0..aligned_refine_len {
                let target_id = targets[t];
                let logits = &refine_logits[t * self.output_size..(t + 1) * self.output_size];
                let probs = nn::softmax(logits);
                let (loss, grad) = nn::cross_entropy_grad(&probs, target_id);
                total_loss += 0.5 * loss;
                let hidden_row = &refine_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                let grad_hidden_row =
                    &mut grad_refine_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size];
                for out_idx in 0..self.output_size {
                    let diff = 0.5 * grad[out_idx];
                    grad_refine_bias[out_idx] += diff;
                    let base = out_idx * self.hidden_size;
                    for hid_idx in 0..self.hidden_size {
                        grad_refine_projection[base + hid_idx] += diff * hidden_row[hid_idx];
                        grad_hidden_row[hid_idx] += diff * self.refine_projection[base + hid_idx];
                    }
                }
            }
            for t in 0..refine_len {
                let remask_target = if t < target_len
                    && t < hypothesis_ids.len()
                    && hypothesis_ids[t] == targets[t]
                {
                    0.0
                } else {
                    1.0
                };
                let prob = sigmoid(remask_logits[t]);
                total_loss += 0.1 * binary_cross_entropy(prob, remask_target);
                let grad_logit = 0.1 * (prob - remask_target);
                grad_remask_bias[0] += grad_logit;
                let hidden_row = &refine_hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
                let grad_hidden_row =
                    &mut grad_refine_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    grad_remask_projection[hid_idx] += grad_logit * hidden_row[hid_idx];
                    grad_hidden_row[hid_idx] += grad_logit * self.remask_projection[hid_idx];
                }
            }
            if refine_len > 0 {
                let stop_target = if refine_len == target_len { 1.0 } else { 0.0 };
                let stop_prob = sigmoid(stop_logit);
                total_loss += 0.1 * binary_cross_entropy(stop_prob, stop_target);
                let stop_grad = 0.1 * (stop_prob - stop_target);
                grad_stop_bias[0] += stop_grad;
                let mut grad_pooled = vec![0.0f64; self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    grad_stop_projection[hid_idx] += stop_grad * pooled[hid_idx];
                    grad_pooled[hid_idx] += stop_grad * self.stop_projection[hid_idx];
                }
                let inv_len = 1.0 / refine_len as f64;
                for t in 0..refine_len {
                    let grad_hidden_row = &mut grad_refine_hidden_total
                        [t * self.hidden_size..(t + 1) * self.hidden_size];
                    for hid_idx in 0..self.hidden_size {
                        grad_hidden_row[hid_idx] += grad_pooled[hid_idx] * inv_len;
                    }
                }
            }

            let mut grad_encoder_from_decoder = vec![0.0f64; encoder_hidden.len()];
            for block_idx in (0..self.decoder_blocks.len()).rev() {
                let block = &self.decoder_blocks[block_idx];
                let grads = nn::decoder_block_backward(
                    &decoder_caches[block_idx],
                    block,
                    &grad_hidden_total,
                    input_len,
                    self.hidden_size,
                );
                accumulate(&mut grad_decoder_self_q[block_idx], &grads.self_q_proj);
                accumulate(&mut grad_decoder_self_k[block_idx], &grads.self_k_proj);
                accumulate(&mut grad_decoder_self_v[block_idx], &grads.self_v_proj);
                accumulate(&mut grad_decoder_self_o[block_idx], &grads.self_o_proj);
                accumulate(&mut grad_decoder_cross_q[block_idx], &grads.cross_q_proj);
                accumulate(&mut grad_decoder_cross_k[block_idx], &grads.cross_k_proj);
                accumulate(&mut grad_decoder_cross_v[block_idx], &grads.cross_v_proj);
                accumulate(&mut grad_decoder_cross_o[block_idx], &grads.cross_o_proj);
                accumulate(&mut grad_decoder_ff_in[block_idx], &grads.ff_in);
                accumulate(&mut grad_decoder_ff_in_bias[block_idx], &grads.ff_in_bias);
                accumulate(&mut grad_decoder_ff_out[block_idx], &grads.ff_out);
                accumulate(&mut grad_decoder_ff_out_bias[block_idx], &grads.ff_out_bias);
                accumulate(&mut grad_encoder_from_decoder, &grads.grad_memory);
                grad_hidden_total = grads.grad_input;
            }
            for t in 0..input_len {
                let pos_base = t * self.hidden_size;
                let grad_hidden_row =
                    &grad_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    let grad_pre =
                        grad_hidden_row[hid_idx] * (1.0 - decoder_seed[pos_base + hid_idx].powi(2));
                    grad_pos_embeddings[pos_base + hid_idx] += grad_pre;
                    grad_decoder_pos_embeddings[pos_base + hid_idx] += grad_pre;
                    grad_encoder_from_decoder[pos_base + hid_idx] += grad_pre;
                }
            }

            let mut grad_encoder_from_refine = vec![0.0f64; encoder_hidden.len()];
            for block_idx in (0..self.refine_blocks.len()).rev() {
                let block = &self.refine_blocks[block_idx];
                let grads = nn::decoder_block_backward(
                    &refine_caches[block_idx],
                    block,
                    &grad_refine_hidden_total,
                    refine_len,
                    self.hidden_size,
                );
                accumulate(&mut grad_refine_self_q[block_idx], &grads.self_q_proj);
                accumulate(&mut grad_refine_self_k[block_idx], &grads.self_k_proj);
                accumulate(&mut grad_refine_self_v[block_idx], &grads.self_v_proj);
                accumulate(&mut grad_refine_self_o[block_idx], &grads.self_o_proj);
                accumulate(&mut grad_refine_cross_q[block_idx], &grads.cross_q_proj);
                accumulate(&mut grad_refine_cross_k[block_idx], &grads.cross_k_proj);
                accumulate(&mut grad_refine_cross_v[block_idx], &grads.cross_v_proj);
                accumulate(&mut grad_refine_cross_o[block_idx], &grads.cross_o_proj);
                accumulate(&mut grad_refine_ff_in[block_idx], &grads.ff_in);
                accumulate(&mut grad_refine_ff_in_bias[block_idx], &grads.ff_in_bias);
                accumulate(&mut grad_refine_ff_out[block_idx], &grads.ff_out);
                accumulate(&mut grad_refine_ff_out_bias[block_idx], &grads.ff_out_bias);
                accumulate(&mut grad_encoder_from_refine, &grads.grad_memory);
                grad_refine_hidden_total = grads.grad_input;
            }
            for t in 0..refine_len {
                let pos_base = t * self.hidden_size;
                let token_base = refine_token_ids[t] * self.hidden_size;
                let grad_hidden_row =
                    &grad_refine_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size];
                for hid_idx in 0..self.hidden_size {
                    let grad_pre =
                        grad_hidden_row[hid_idx] * (1.0 - refine_seed[pos_base + hid_idx].powi(2));
                    grad_refine_token_embeddings[token_base + hid_idx] += grad_pre;
                    grad_refine_pos_embeddings[pos_base + hid_idx] += grad_pre;
                }
            }
            accumulate(&mut grad_encoder_from_decoder, &grad_encoder_from_refine);
            grad_hidden_total = grad_encoder_from_decoder;

            for block_idx in (0..self.blocks.len()).rev() {
                let block = &self.blocks[block_idx];
                let grads = nn::encoder_block_backward(
                    &block_caches[block_idx],
                    block,
                    &grad_hidden_total,
                    input_len,
                    self.hidden_size,
                );
                accumulate(&mut grad_block_q[block_idx], &grads.q_proj);
                accumulate(&mut grad_block_k[block_idx], &grads.k_proj);
                accumulate(&mut grad_block_v[block_idx], &grads.v_proj);
                accumulate(&mut grad_block_o[block_idx], &grads.o_proj);
                accumulate(&mut grad_block_ff_in[block_idx], &grads.ff_in);
                accumulate(&mut grad_block_ff_in_bias[block_idx], &grads.ff_in_bias);
                accumulate(&mut grad_block_ff_out[block_idx], &grads.ff_out);
                accumulate(&mut grad_block_ff_out_bias[block_idx], &grads.ff_out_bias);
                grad_hidden_total = grads.grad_input;
            }

            for t in 0..input_len {
                let grad_hidden =
                    &grad_hidden_total[t * self.hidden_size..(t + 1) * self.hidden_size];
                let token_base = token_ids[t] * self.hidden_size;
                let pos_base = pos_ids[t] * self.hidden_size;
                for hid_idx in 0..self.hidden_size {
                    let activation = hidden[t * self.hidden_size + hid_idx];
                    let grad_pre = grad_hidden[hid_idx] * (1.0 - activation * activation);
                    grad_token_embeddings[token_base + hid_idx] += grad_pre;
                    grad_pos_embeddings[pos_base + hid_idx] += grad_pre;
                }
            }
        }

        let denom = sample_count.max(1) as f64;
        scale_grads(&mut grad_token_embeddings, denom);
        scale_grads(&mut grad_pos_embeddings, denom);
        scale_grads(&mut grad_decoder_pos_embeddings, denom);
        scale_grads(&mut grad_refine_token_embeddings, denom);
        scale_grads(&mut grad_refine_pos_embeddings, denom);
        scale_grads(&mut grad_projection, denom);
        scale_grads(&mut grad_refine_projection, denom);
        scale_grads(&mut grad_remask_projection, denom);
        scale_grads(&mut grad_stop_projection, denom);
        scale_grads(&mut grad_bias, denom);
        scale_grads(&mut grad_refine_bias, denom);
        scale_grads(&mut grad_remask_bias, denom);
        scale_grads(&mut grad_stop_bias, denom);
        self.optimizer.update(
            _step,
            "ctc.token_embeddings",
            &mut self.token_embeddings,
            &grad_token_embeddings,
        );
        self.optimizer.update(
            _step,
            "ctc.pos_embeddings",
            &mut self.pos_embeddings,
            &grad_pos_embeddings,
        );
        self.optimizer.update(
            _step,
            "ctc.decoder_pos_embeddings",
            &mut self.decoder_pos_embeddings,
            &grad_decoder_pos_embeddings,
        );
        self.optimizer.update(
            _step,
            "ctc.refine_token_embeddings",
            &mut self.refine_token_embeddings,
            &grad_refine_token_embeddings,
        );
        self.optimizer.update(
            _step,
            "ctc.refine_pos_embeddings",
            &mut self.refine_pos_embeddings,
            &grad_refine_pos_embeddings,
        );
        self.optimizer.update(
            _step,
            "ctc.projection",
            &mut self.projection,
            &grad_projection,
        );
        self.optimizer.update(
            _step,
            "ctc.refine_projection",
            &mut self.refine_projection,
            &grad_refine_projection,
        );
        self.optimizer.update(
            _step,
            "ctc.remask_projection",
            &mut self.remask_projection,
            &grad_remask_projection,
        );
        self.optimizer.update(
            _step,
            "ctc.stop_projection",
            &mut self.stop_projection,
            &grad_stop_projection,
        );
        self.optimizer
            .update(_step, "ctc.bias", &mut self.bias, &grad_bias);
        self.optimizer.update(
            _step,
            "ctc.refine_bias",
            &mut self.refine_bias,
            &grad_refine_bias,
        );
        self.optimizer.update(
            _step,
            "ctc.remask_bias",
            &mut self.remask_bias,
            &grad_remask_bias,
        );
        self.optimizer
            .update(_step, "ctc.stop_bias", &mut self.stop_bias, &grad_stop_bias);
        for (idx, block) in self.blocks.iter_mut().enumerate() {
            scale_grads(&mut grad_block_q[idx], denom);
            scale_grads(&mut grad_block_k[idx], denom);
            scale_grads(&mut grad_block_v[idx], denom);
            scale_grads(&mut grad_block_o[idx], denom);
            scale_grads(&mut grad_block_ff_in[idx], denom);
            scale_grads(&mut grad_block_ff_in_bias[idx], denom);
            scale_grads(&mut grad_block_ff_out[idx], denom);
            scale_grads(&mut grad_block_ff_out_bias[idx], denom);
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.q_proj", idx),
                &mut block.q_proj,
                &grad_block_q[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.k_proj", idx),
                &mut block.k_proj,
                &grad_block_k[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.v_proj", idx),
                &mut block.v_proj,
                &grad_block_v[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.o_proj", idx),
                &mut block.o_proj,
                &grad_block_o[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.ff_in", idx),
                &mut block.ff_in,
                &grad_block_ff_in[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.ff_in_bias", idx),
                &mut block.ff_in_bias,
                &grad_block_ff_in_bias[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.ff_out", idx),
                &mut block.ff_out,
                &grad_block_ff_out[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.blocks.{}.ff_out_bias", idx),
                &mut block.ff_out_bias,
                &grad_block_ff_out_bias[idx],
            );
        }
        for (idx, block) in self.decoder_blocks.iter_mut().enumerate() {
            scale_grads(&mut grad_decoder_self_q[idx], denom);
            scale_grads(&mut grad_decoder_self_k[idx], denom);
            scale_grads(&mut grad_decoder_self_v[idx], denom);
            scale_grads(&mut grad_decoder_self_o[idx], denom);
            scale_grads(&mut grad_decoder_cross_q[idx], denom);
            scale_grads(&mut grad_decoder_cross_k[idx], denom);
            scale_grads(&mut grad_decoder_cross_v[idx], denom);
            scale_grads(&mut grad_decoder_cross_o[idx], denom);
            scale_grads(&mut grad_decoder_ff_in[idx], denom);
            scale_grads(&mut grad_decoder_ff_in_bias[idx], denom);
            scale_grads(&mut grad_decoder_ff_out[idx], denom);
            scale_grads(&mut grad_decoder_ff_out_bias[idx], denom);
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.self_q_proj", idx),
                &mut block.self_q_proj,
                &grad_decoder_self_q[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.self_k_proj", idx),
                &mut block.self_k_proj,
                &grad_decoder_self_k[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.self_v_proj", idx),
                &mut block.self_v_proj,
                &grad_decoder_self_v[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.self_o_proj", idx),
                &mut block.self_o_proj,
                &grad_decoder_self_o[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.cross_q_proj", idx),
                &mut block.cross_q_proj,
                &grad_decoder_cross_q[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.cross_k_proj", idx),
                &mut block.cross_k_proj,
                &grad_decoder_cross_k[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.cross_v_proj", idx),
                &mut block.cross_v_proj,
                &grad_decoder_cross_v[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.cross_o_proj", idx),
                &mut block.cross_o_proj,
                &grad_decoder_cross_o[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.ff_in", idx),
                &mut block.ff_in,
                &grad_decoder_ff_in[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.ff_in_bias", idx),
                &mut block.ff_in_bias,
                &grad_decoder_ff_in_bias[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.ff_out", idx),
                &mut block.ff_out,
                &grad_decoder_ff_out[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.decoder_blocks.{}.ff_out_bias", idx),
                &mut block.ff_out_bias,
                &grad_decoder_ff_out_bias[idx],
            );
        }
        for (idx, block) in self.refine_blocks.iter_mut().enumerate() {
            scale_grads(&mut grad_refine_self_q[idx], denom);
            scale_grads(&mut grad_refine_self_k[idx], denom);
            scale_grads(&mut grad_refine_self_v[idx], denom);
            scale_grads(&mut grad_refine_self_o[idx], denom);
            scale_grads(&mut grad_refine_cross_q[idx], denom);
            scale_grads(&mut grad_refine_cross_k[idx], denom);
            scale_grads(&mut grad_refine_cross_v[idx], denom);
            scale_grads(&mut grad_refine_cross_o[idx], denom);
            scale_grads(&mut grad_refine_ff_in[idx], denom);
            scale_grads(&mut grad_refine_ff_in_bias[idx], denom);
            scale_grads(&mut grad_refine_ff_out[idx], denom);
            scale_grads(&mut grad_refine_ff_out_bias[idx], denom);
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.self_q_proj", idx),
                &mut block.self_q_proj,
                &grad_refine_self_q[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.self_k_proj", idx),
                &mut block.self_k_proj,
                &grad_refine_self_k[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.self_v_proj", idx),
                &mut block.self_v_proj,
                &grad_refine_self_v[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.self_o_proj", idx),
                &mut block.self_o_proj,
                &grad_refine_self_o[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.cross_q_proj", idx),
                &mut block.cross_q_proj,
                &grad_refine_cross_q[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.cross_k_proj", idx),
                &mut block.cross_k_proj,
                &grad_refine_cross_k[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.cross_v_proj", idx),
                &mut block.cross_v_proj,
                &grad_refine_cross_v[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.cross_o_proj", idx),
                &mut block.cross_o_proj,
                &grad_refine_cross_o[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.ff_in", idx),
                &mut block.ff_in,
                &grad_refine_ff_in[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.ff_in_bias", idx),
                &mut block.ff_in_bias,
                &grad_refine_ff_in_bias[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.ff_out", idx),
                &mut block.ff_out,
                &grad_refine_ff_out[idx],
            );
            self.optimizer.update(
                _step,
                &format!("ctc.refine_blocks.{}.ff_out_bias", idx),
                &mut block.ff_out_bias,
                &grad_refine_ff_out_bias[idx],
            );
        }
        let loss = if sample_count == 0 {
            0.0
        } else {
            total_loss / sample_count as f64
        };
        self.last_loss = Some(loss);
        Ok(TrainerStep {
            loss,
            rows: batch.batch_size,
            bytes: batch.bytes(),
            input_tokens: batch.non_padding_input_tokens(),
            target_tokens: batch.non_padding_target_tokens(),
        })
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("write {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(b"KTCB1")?;
        self.optimizer.write_binary(&mut writer)?;
        write_usize(&mut writer, self.hidden_size)?;
        write_usize(&mut writer, self.encoder_layers)?;
        write_usize(&mut writer, self.num_heads)?;
        write_usize(&mut writer, self.ffn_size)?;
        write_usize(&mut writer, self.decoder_layers)?;
        write_usize(&mut writer, self.decoder_heads)?;
        write_usize(&mut writer, self.decoder_ffn_size)?;
        write_usize(&mut writer, self.output_size)?;
        write_usize(&mut writer, self.blank_id)?;
        write_usize(&mut writer, self.max_positions)?;
        write_vec_f32(&mut writer, &self.token_embeddings)?;
        write_vec_f32(&mut writer, &self.pos_embeddings)?;
        write_encoder_blocks(&mut writer, &self.blocks)?;
        write_vec_f32(&mut writer, &self.decoder_pos_embeddings)?;
        write_decoder_blocks(&mut writer, &self.decoder_blocks)?;
        write_vec_f32(&mut writer, &self.refine_token_embeddings)?;
        write_vec_f32(&mut writer, &self.refine_pos_embeddings)?;
        write_decoder_blocks(&mut writer, &self.refine_blocks)?;
        write_vec_f32(&mut writer, &self.projection)?;
        write_vec_f32(&mut writer, &self.refine_projection)?;
        write_vec_f32(&mut writer, &self.remask_projection)?;
        write_vec_f32(&mut writer, &self.stop_projection)?;
        write_vec_f32(&mut writer, &self.bias)?;
        write_vec_f32(&mut writer, &self.refine_bias)?;
        write_vec_f32(&mut writer, &self.remask_bias)?;
        write_vec_f32(&mut writer, &self.stop_bias)?;
        write_option_f64(&mut writer, self.last_loss)?;
        writer.flush()?;
        Ok(())
    }

    fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).with_context(|| format!("read {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic)?;
        if &magic == b"KTCB1" {
            self.optimizer = OptimizerState::read_binary(&mut reader)?;
            self.hidden_size = read_usize(&mut reader)?;
            self.encoder_layers = read_usize(&mut reader)?;
            self.num_heads = read_usize(&mut reader)?;
            self.ffn_size = read_usize(&mut reader)?;
            self.decoder_layers = read_usize(&mut reader)?;
            self.decoder_heads = read_usize(&mut reader)?;
            self.decoder_ffn_size = read_usize(&mut reader)?;
            self.output_size = read_usize(&mut reader)?;
            self.blank_id = read_usize(&mut reader)?;
            self.max_positions = read_usize(&mut reader)?;
            self.token_embeddings = read_vec_f32(&mut reader)?;
            self.pos_embeddings = read_vec_f32(&mut reader)?;
            self.blocks = read_encoder_blocks(&mut reader)?;
            self.decoder_pos_embeddings = read_vec_f32(&mut reader)?;
            self.decoder_blocks = read_decoder_blocks(&mut reader)?;
            self.refine_token_embeddings = read_vec_f32(&mut reader)?;
            self.refine_pos_embeddings = read_vec_f32(&mut reader)?;
            self.refine_blocks = read_decoder_blocks(&mut reader)?;
            self.projection = read_vec_f32(&mut reader)?;
            self.refine_projection = read_vec_f32(&mut reader)?;
            self.remask_projection = read_vec_f32(&mut reader)?;
            self.stop_projection = read_vec_f32(&mut reader)?;
            self.bias = read_vec_f32(&mut reader)?;
            self.refine_bias = read_vec_f32(&mut reader)?;
            self.remask_bias = read_vec_f32(&mut reader)?;
            self.stop_bias = read_vec_f32(&mut reader)?;
            self.last_loss = read_option_f64(&mut reader)?;
        } else {
            let mut bytes = magic.to_vec();
            reader.read_to_end(&mut bytes)?;
            *self = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", path.display()))?;
        }
        Ok(())
    }
}

/// splitmix64 hash. Deterministic per-seed scalar used for refine mask
/// sampling so the same (step, row, position) triple always produces the
/// same mask decision — important for resume + Python parity.
///
/// Currently referenced only by CPU refine helpers that are queued for
/// wiring into `CtcBackend::step`. Warning suppression documents that
/// this is intentional future integration surface, not stale code.
#[allow(dead_code)]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn init_table(len: usize, scale: f64) -> Vec<f64> {
    let denom = len.max(1) as f64;
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(scale * (idx as f64 + 1.0) / denom);
    }
    out
}

fn scale_grads(grads: &mut [f64], denom: f64) {
    if denom <= 0.0 {
        return;
    }
    for grad in grads {
        *grad /= denom;
    }
}

fn accumulate(dst: &mut [f64], src: &[f64]) {
    for (dst, src) in dst.iter_mut().zip(src.iter().copied()) {
        *dst += src;
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn binary_cross_entropy(prob: f64, target: f64) -> f64 {
    let clipped = prob.clamp(1e-8, 1.0 - 1e-8);
    -(target * clipped.ln() + (1.0 - target) * (1.0 - clipped).ln())
}

fn write_usize<W: Write>(writer: &mut W, value: usize) -> Result<()> {
    writer.write_all(&(value as u64).to_le_bytes())?;
    Ok(())
}

fn read_usize<R: Read>(reader: &mut R) -> Result<usize> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes) as usize)
}

fn write_vec_f32<W: Write>(writer: &mut W, values: &[f64]) -> Result<()> {
    write_usize(writer, values.len())?;
    for value in values {
        writer.write_all(&(*value as f32).to_le_bytes())?;
    }
    Ok(())
}

fn read_vec_f32<R: Read>(reader: &mut R) -> Result<Vec<f64>> {
    let len = read_usize(reader)?;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        out.push(f32::from_le_bytes(bytes) as f64);
    }
    Ok(out)
}

fn write_option_f64<W: Write>(writer: &mut W, value: Option<f64>) -> Result<()> {
    match value {
        Some(v) => {
            writer.write_all(&[1])?;
            writer.write_all(&v.to_le_bytes())?;
        }
        None => writer.write_all(&[0])?,
    }
    Ok(())
}

fn read_option_f64<R: Read>(reader: &mut R) -> Result<Option<f64>> {
    let mut tag = [0u8; 1];
    reader.read_exact(&mut tag)?;
    if tag[0] == 0 {
        return Ok(None);
    }
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(Some(f64::from_le_bytes(bytes)))
}

fn write_encoder_blocks<W: Write>(writer: &mut W, blocks: &[EncoderBlock]) -> Result<()> {
    write_usize(writer, blocks.len())?;
    for block in blocks {
        write_vec_f32(writer, &block.q_proj)?;
        write_vec_f32(writer, &block.k_proj)?;
        write_vec_f32(writer, &block.v_proj)?;
        write_vec_f32(writer, &block.o_proj)?;
        write_vec_f32(writer, &block.ff_in)?;
        write_vec_f32(writer, &block.ff_in_bias)?;
        write_vec_f32(writer, &block.ff_out)?;
        write_vec_f32(writer, &block.ff_out_bias)?;
    }
    Ok(())
}

fn read_encoder_blocks<R: Read>(reader: &mut R) -> Result<Vec<EncoderBlock>> {
    let len = read_usize(reader)?;
    let mut blocks = Vec::with_capacity(len);
    for _ in 0..len {
        blocks.push(EncoderBlock {
            q_proj: read_vec_f32(reader)?,
            k_proj: read_vec_f32(reader)?,
            v_proj: read_vec_f32(reader)?,
            o_proj: read_vec_f32(reader)?,
            ff_in: read_vec_f32(reader)?,
            ff_in_bias: read_vec_f32(reader)?,
            ff_out: read_vec_f32(reader)?,
            ff_out_bias: read_vec_f32(reader)?,
        });
    }
    Ok(blocks)
}

fn write_decoder_blocks<W: Write>(writer: &mut W, blocks: &[DecoderBlock]) -> Result<()> {
    write_usize(writer, blocks.len())?;
    for block in blocks {
        write_vec_f32(writer, &block.self_q_proj)?;
        write_vec_f32(writer, &block.self_k_proj)?;
        write_vec_f32(writer, &block.self_v_proj)?;
        write_vec_f32(writer, &block.self_o_proj)?;
        write_vec_f32(writer, &block.cross_q_proj)?;
        write_vec_f32(writer, &block.cross_k_proj)?;
        write_vec_f32(writer, &block.cross_v_proj)?;
        write_vec_f32(writer, &block.cross_o_proj)?;
        write_vec_f32(writer, &block.ff_in)?;
        write_vec_f32(writer, &block.ff_in_bias)?;
        write_vec_f32(writer, &block.ff_out)?;
        write_vec_f32(writer, &block.ff_out_bias)?;
    }
    Ok(())
}

fn read_decoder_blocks<R: Read>(reader: &mut R) -> Result<Vec<DecoderBlock>> {
    let len = read_usize(reader)?;
    let mut blocks = Vec::with_capacity(len);
    for _ in 0..len {
        blocks.push(DecoderBlock {
            self_q_proj: read_vec_f32(reader)?,
            self_k_proj: read_vec_f32(reader)?,
            self_v_proj: read_vec_f32(reader)?,
            self_o_proj: read_vec_f32(reader)?,
            cross_q_proj: read_vec_f32(reader)?,
            cross_k_proj: read_vec_f32(reader)?,
            cross_v_proj: read_vec_f32(reader)?,
            cross_o_proj: read_vec_f32(reader)?,
            ff_in: read_vec_f32(reader)?,
            ff_in_bias: read_vec_f32(reader)?,
            ff_out: read_vec_f32(reader)?,
            ff_out_bias: read_vec_f32(reader)?,
        });
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn backend_kind_names_match_config_values() {
        let mock = BackendKind::new(&BackendConfig {
            kind: "mock".to_string(),
            ..BackendConfig::default()
        })
        .unwrap();
        let toy = BackendKind::new(&BackendConfig {
            kind: "toy".to_string(),
            ..BackendConfig::default()
        })
        .unwrap();
        let surrogate = BackendKind::new(&BackendConfig {
            kind: "surrogate".to_string(),
            ..BackendConfig::default()
        })
        .unwrap();
        let ctc = BackendKind::new(&BackendConfig {
            kind: "ctc".to_string(),
            output_size: 16,
            blank_id: 4,
            max_positions: 16,
            ..BackendConfig::default()
        })
        .unwrap();
        assert_eq!(mock.kind(), "mock");
        assert_eq!(toy.kind(), "toy");
        assert_eq!(surrogate.kind(), "surrogate");
        assert_eq!(ctc.kind(), "ctc");
    }

    #[test]
    fn ctc_backend_step_returns_finite_loss() {
        let mut backend = CtcBackend::new(&BackendConfig {
            kind: "ctc".to_string(),
            hidden_size: 8,
            output_size: 12,
            blank_id: 4,
            max_positions: 8,
            ..BackendConfig::default()
        });
        let batch = PackedBatch {
            input_ids: vec![1, 2, 3, 0],
            attention_mask: vec![1, 1, 1, 0],
            target_ids: vec![5, 6, 0],
            input_lengths: vec![3],
            target_lengths: vec![2],
            source_ids: vec![0],
            batch_size: 1,
            max_input_len: 4,
            max_target_len: 3,
            order_cursor: 1,
        };
        let step = backend.step(1, &batch).unwrap();
        assert!(step.loss.is_finite());
    }

    #[test]
    fn ctc_backend_checkpoint_round_trip() {
        let mut backend = CtcBackend::new(&BackendConfig {
            kind: "ctc".to_string(),
            hidden_size: 8,
            output_size: 12,
            blank_id: 4,
            max_positions: 8,
            ..BackendConfig::default()
        });
        backend.last_loss = Some(1.25);
        let path = NamedTempFile::new().unwrap();
        backend.save_checkpoint(path.path()).unwrap();

        let mut loaded = CtcBackend::new(&BackendConfig {
            kind: "ctc".to_string(),
            hidden_size: 8,
            output_size: 12,
            blank_id: 4,
            max_positions: 8,
            ..BackendConfig::default()
        });
        loaded.load_checkpoint(path.path()).unwrap();

        assert_eq!(loaded.hidden_size, backend.hidden_size);
        assert_eq!(loaded.output_size, backend.output_size);
        assert_eq!(loaded.blocks.len(), backend.blocks.len());
        assert_eq!(loaded.decoder_blocks.len(), backend.decoder_blocks.len());
        assert_eq!(loaded.last_loss, backend.last_loss);
        assert_eq!(
            loaded.token_embeddings.len(),
            backend.token_embeddings.len()
        );
    }
}
