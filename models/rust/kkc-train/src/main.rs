mod backend;
mod ctc;
mod device;
mod nn;
mod optim;
mod pipeline;
mod tensor;
mod trainer;

#[cfg(feature = "cuda")]
mod gpu;

use crate::device::Device;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kkc_data::{
    compile_jsonl_to_shard, inspect_shard_batches, BatchIter, BatchIterConfig, CompileOptions,
    PrefetchedBatchIter, SequenceBudget, ShardMetadata,
};
use kkc_model::{ctc_nat_preset, estimate_ctc_nat_resources, BatchShape, RuntimeAssumptions};
use kkc_tokenizer::SharedCharTokenizer;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

const MAX_SEQUENCE_U16: usize = u16::MAX as usize;

#[derive(Parser)]
#[command(
    name = "kkc-train",
    about = "Rust-first trainer utilities for kana-kanji conversion"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Plan {
        #[arg(long)]
        config: PathBuf,
    },
    CompileShard {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 6000)]
        max_kanji: u32,
    },
    PeekBatches {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 3)]
        batches: usize,
    },
    ScanEpoch {
        #[arg(long)]
        config: PathBuf,
    },
    DryTrain {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
    },
    InitRun {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    RecordCheckpoint {
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long)]
        step: usize,
        #[arg(long)]
        epoch: usize,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        metric: Option<f64>,
        #[arg(long, default_value = "regular")]
        kind: String,
        #[arg(long, default_value = "maximize")]
        metric_mode: String,
    },
    ShowRun {
        #[arg(long)]
        run_dir: PathBuf,
    },
    CheckResume {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
    },
    Eval {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        batches: Option<usize>,
        /// Compute device: `cpu`, `cuda`, or `cuda:N`. `cuda*` requires
        /// the binary to have been built with `--features cuda`.
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    Fit {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        checkpoint_every: usize,
        /// Compute device: `cpu`, `cuda`, or `cuda:N`. `cuda*` requires the
        /// binary to have been built with `--features cuda`.
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Bound on how many checkpoint writes can be in flight before the
        /// training loop blocks. 0 disables the async writer and falls back
        /// to the synchronous path.
        #[arg(long, default_value_t = 2)]
        async_ckpt_queue: usize,
        /// Global grad-norm clip. `0.0` disables clipping. Only honored
        /// by the `tch-ctc-nat` backend for now; other backends ignore.
        #[arg(long, default_value_t = 0.0)]
        grad_clip: f64,
    },
    MockFit {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 50)]
        checkpoint_every: usize,
    },
}

#[derive(Debug, Deserialize)]
struct TrainConfig {
    dataset: DatasetConfig,
    tokenizer: TokenizerConfig,
    model: ModelConfig,
    runtime: RuntimeConfig,
    #[serde(default)]
    eval: EvalSection,
    #[serde(default)]
    backend: backend::BackendConfig,
    train: TrainSection,
}

#[derive(Debug, Deserialize)]
struct DatasetConfig {
    train_shard: PathBuf,
    #[serde(default)]
    eval_shard: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    path: Option<PathBuf>,
    #[serde(default = "default_max_kanji")]
    max_kanji: u32,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    preset: String,
}

#[derive(Debug, Deserialize)]
struct RuntimeConfig {
    #[serde(default = "default_param_dtype_bytes")]
    param_dtype_bytes: usize,
    #[serde(default = "default_grad_dtype_bytes")]
    grad_dtype_bytes: usize,
    #[serde(default = "default_adam_state_bytes")]
    adam_state_bytes: usize,
    #[serde(default = "default_activation_dtype_bytes")]
    activation_dtype_bytes: usize,
    #[serde(default)]
    prefetch_queue: usize,
}

#[derive(Debug, Deserialize)]
struct TrainSection {
    batch_size: usize,
    max_input_len: usize,
    max_target_len: usize,
    #[serde(default = "default_accum")]
    grad_accum: usize,
    #[serde(default = "default_block_rows")]
    block_rows: usize,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_checkpoint_keep_last")]
    checkpoint_keep_last: usize,
}

#[derive(Debug, Deserialize, Default)]
struct EvalSection {
    shard: Option<PathBuf>,
    #[serde(default)]
    batches: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunManifest {
    train_shard: String,
    model_preset: String,
    vocab_size: usize,
    backend_kind: String,
    backend_hidden_size: usize,
    backend_output_size: usize,
    backend_blank_id: usize,
    backend_max_positions: usize,
    batch_size: usize,
    max_input_len: usize,
    max_target_len: usize,
    grad_accum: usize,
    block_rows: usize,
    seed: u64,
    prefetch_queue: usize,
    parameter_count: usize,
    parameter_bytes: usize,
    gradient_bytes: usize,
    optimizer_bytes: usize,
    activation_bytes: usize,
    logits_bytes: usize,
    total_step_bytes: usize,
    estimated_step_upper_bound: usize,
    // Optim + schedule knobs. Persisted so a resume that silently
    // changes LR / decay / AdamW betas / weight decay / grad clip
    // trips `collect_resume_mismatches` instead of quietly altering
    // the training dynamics.
    #[serde(default)]
    optimizer: String,
    #[serde(default)]
    scheduler: String,
    #[serde(default)]
    learning_rate: f64,
    #[serde(default)]
    weight_decay: f64,
    #[serde(default)]
    beta1: f64,
    #[serde(default)]
    beta2: f64,
    #[serde(default)]
    epsilon: f64,
    #[serde(default)]
    warmup_steps: usize,
    #[serde(default)]
    scheduler_total_steps: usize,
    #[serde(default)]
    min_lr_scale: f64,
    #[serde(default)]
    refine_loss_weight: f64,
    #[serde(default)]
    refine_warmup_steps: usize,
    #[serde(default)]
    refine_mask_ratio: f64,
    #[serde(default)]
    refine_source: String,
    #[serde(default)]
    remask_loss_weight: f64,
    #[serde(default)]
    stop_loss_weight: f64,
    #[serde(default)]
    grad_clip: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointEntry {
    step: usize,
    epoch: usize,
    checkpoint: String,
    metric: Option<f64>,
    kind: String,
    metric_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainerState {
    step: usize,
    epoch: usize,
    data_cursor: usize,
    best_metric: Option<f64>,
    #[serde(default)]
    best_checkpoint: Option<String>,
    last_checkpoint: Option<String>,
    checkpoints: Vec<CheckpointEntry>,
}

fn default_max_kanji() -> u32 {
    6000
}

fn default_vocab_size() -> usize {
    4801
}

fn default_accum() -> usize {
    1
}

fn default_block_rows() -> usize {
    4096
}

fn default_seed() -> u64 {
    42
}

fn default_checkpoint_keep_last() -> usize {
    2
}

fn default_param_dtype_bytes() -> usize {
    2
}

fn default_grad_dtype_bytes() -> usize {
    4
}

fn default_adam_state_bytes() -> usize {
    8
}

fn default_activation_dtype_bytes() -> usize {
    2
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Plan { config } => plan(&config),
        Command::CompileShard {
            input,
            output,
            tokenizer,
            max_kanji,
        } => compile_shard(&input, &output, tokenizer.as_deref(), max_kanji),
        Command::PeekBatches { config, batches } => peek_batches(&config, batches),
        Command::ScanEpoch { config } => scan_epoch(&config),
        Command::DryTrain { config, steps } => dry_train(&config, steps),
        Command::InitRun { config, output } => init_run(&config, &output),
        Command::RecordCheckpoint {
            run_dir,
            step,
            epoch,
            checkpoint,
            metric,
            kind,
            metric_mode,
        } => record_checkpoint(
            &run_dir,
            step,
            epoch,
            &checkpoint,
            metric,
            &kind,
            &metric_mode,
        ),
        Command::ShowRun { run_dir } => show_run(&run_dir),
        Command::CheckResume { config, run_dir } => check_resume(&config, &run_dir),
        Command::Eval {
            config,
            run_dir,
            batches,
            device,
        } => {
            let device = Device::from_str(&device)?;
            device::require_cuda_built(device)?;
            eval_command(&config, run_dir.as_deref(), batches, device)
        }
        Command::Fit {
            config,
            run_dir,
            steps,
            checkpoint_every,
            device,
            async_ckpt_queue,
            grad_clip,
        } => {
            let device = Device::from_str(&device)?;
            device::require_cuda_built(device)?;
            fit(
                &config,
                &run_dir,
                steps,
                checkpoint_every,
                device,
                async_ckpt_queue,
                grad_clip,
            )
        }
        Command::MockFit {
            config,
            run_dir,
            steps,
            checkpoint_every,
        } => fit(
            &config,
            &run_dir,
            steps,
            checkpoint_every,
            Device::Cpu,
            0,
            0.0,
        ),
    }
}

fn plan(config_path: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let shard_stats = inspect_shard_batches(&config.dataset.train_shard, iter_config(&config), 8)?;
    let plan = sequence_budget(&config).estimate();
    let estimate = model_estimate(&config)?;
    println!("train shard: {}", config.dataset.train_shard.display());
    println!("model preset: {}", config.model.preset);
    match &config.tokenizer.path {
        Some(path) => println!("tokenizer: {}", path.display()),
        None => println!(
            "tokenizer: default(max_kanji={})",
            config.tokenizer.max_kanji
        ),
    }
    println!("vocab_size: {}", config.tokenizer.vocab_size);
    println!("rows: {}", shard_stats.rows);
    println!(
        "sample mean tokens: input={:.2} target={:.2}",
        shard_stats.mean_input_tokens, shard_stats.mean_target_tokens
    );
    println!(
        "sample max tokens: input={} target={}",
        shard_stats.max_input_tokens, shard_stats.max_target_tokens
    );
    println!(
        "sample packed-batch bytes: {}",
        shard_stats.sample_batch_bytes
    );
    println!(
        "batch bytes: input={} target={} attention={} total={}",
        plan.input_token_bytes, plan.target_token_bytes, plan.attention_bytes, plan.per_batch_bytes
    );
    println!("grad_accum: {}", config.train.grad_accum);
    println!("block_rows: {}", config.train.block_rows);
    println!("seed: {}", config.train.seed);
    println!("prefetch_queue: {}", config.runtime.prefetch_queue);
    println!("backend_kind: {}", config.backend.kind);
    println!(
        "optimizer: kind={} scheduler={} lr={} weight_decay={} warmup_steps={} total_steps={}",
        config.backend.optimizer,
        config.backend.scheduler,
        config.backend.learning_rate,
        config.backend.weight_decay,
        config.backend.warmup_steps,
        config.backend.scheduler_total_steps
    );
    println!(
        "model bytes: params={} grads={} optimizer={} activations={} logits={} total_step={}",
        estimate.parameter_bytes,
        estimate.gradient_bytes,
        estimate.optimizer_bytes,
        estimate.activation_bytes,
        estimate.logits_bytes,
        estimate.total_step_bytes
    );
    println!("parameter_count: {}", estimate.parameter_count);
    println!(
        "step bytes upper bound: {}",
        plan.per_batch_bytes * config.train.grad_accum + estimate.total_step_bytes
    );
    Ok(())
}

fn compile_shard(
    input: &Path,
    output: &Path,
    tokenizer_path: Option<&Path>,
    max_kanji: u32,
) -> Result<()> {
    let tokenizer = match tokenizer_path {
        Some(path) => SharedCharTokenizer::load(path)?,
        None => SharedCharTokenizer::new_default(max_kanji),
    };
    let metadata = compile_jsonl_to_shard(input, output, &tokenizer, &CompileOptions::default())?;
    let sidecar = PathBuf::from(format!("{}.meta.json", output.display()));
    std::fs::write(
        &sidecar,
        serde_json::to_vec_pretty(&metadata).context("serialize metadata")?,
    )
    .with_context(|| format!("write metadata {}", sidecar.display()))?;
    println!("compiled shard: {}", output.display());
    println!("metadata: {}", sidecar.display());
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("rows: {}", metadata.row_count);
    println!("distinct sources: {}", metadata.sources.len());
    Ok(())
}

fn peek_batches(config_path: &Path, batches: usize) -> Result<()> {
    let config = load_config(config_path)?;
    let mut iter = open_batch_source(&config)?;
    for idx in 0..batches {
        let Some(batch) = iter.next_batch()? else {
            println!("batch stream ended at {}", idx);
            break;
        };
        println!(
            "batch {}: rows={} max_input={} max_target={} bytes={} nonpad_input={} nonpad_target={}",
            idx,
            batch.batch_size,
            batch.max_input_len,
            batch.max_target_len,
            batch.bytes(),
            batch.non_padding_input_tokens(),
            batch.non_padding_target_tokens()
        );
    }
    Ok(())
}

fn scan_epoch(config_path: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let mut iter = open_batch_source(&config)?;
    let mut batches = 0usize;
    let mut rows = 0usize;
    let mut total_bytes = 0usize;
    let mut max_bytes = 0usize;
    let mut total_input_tokens = 0usize;
    let mut total_target_tokens = 0usize;
    while let Some(batch) = iter.next_batch()? {
        batches += 1;
        rows += batch.batch_size;
        total_bytes += batch.bytes();
        max_bytes = max_bytes.max(batch.bytes());
        total_input_tokens += batch.non_padding_input_tokens();
        total_target_tokens += batch.non_padding_target_tokens();
    }
    if batches == 0 {
        println!("no batches");
        return Ok(());
    }
    println!("epoch batches: {}", batches);
    println!("epoch rows: {}", rows);
    println!("avg batch bytes: {}", total_bytes / batches);
    println!("max batch bytes: {}", max_bytes);
    println!(
        "avg input tokens per batch: {}",
        total_input_tokens / batches
    );
    println!(
        "avg target tokens per batch: {}",
        total_target_tokens / batches
    );
    Ok(())
}

fn dry_train(config_path: &Path, steps: usize) -> Result<()> {
    let config = load_config(config_path)?;
    let mut iter = open_batch_source(&config)?;
    let started = Instant::now();
    let mut seen_steps = 0usize;
    let mut seen_rows = 0usize;
    let mut seen_bytes = 0usize;
    let mut seen_input_tokens = 0usize;
    let mut seen_target_tokens = 0usize;
    let mut max_batch_bytes = 0usize;
    while seen_steps < steps {
        let Some(batch) = iter.next_batch()? else {
            break;
        };
        seen_steps += 1;
        seen_rows += batch.batch_size;
        seen_bytes += batch.bytes();
        seen_input_tokens += batch.non_padding_input_tokens();
        seen_target_tokens += batch.non_padding_target_tokens();
        max_batch_bytes = max_batch_bytes.max(batch.bytes());
    }
    let elapsed = started.elapsed();
    let secs = elapsed.as_secs_f64().max(1e-9);
    println!("dry_train_steps: {}", seen_steps);
    println!("dry_train_rows: {}", seen_rows);
    println!("dry_train_elapsed_sec: {:.6}", secs);
    println!("dry_train_batches_per_sec: {:.2}", seen_steps as f64 / secs);
    println!("dry_train_rows_per_sec: {:.2}", seen_rows as f64 / secs);
    println!(
        "dry_train_batch_bytes_per_sec: {:.2}",
        seen_bytes as f64 / secs
    );
    println!(
        "dry_train_input_tokens_per_sec: {:.2}",
        seen_input_tokens as f64 / secs
    );
    println!(
        "dry_train_target_tokens_per_sec: {:.2}",
        seen_target_tokens as f64 / secs
    );
    println!(
        "dry_train_avg_batch_bytes: {}",
        seen_bytes / seen_steps.max(1)
    );
    println!("dry_train_max_batch_bytes: {}", max_batch_bytes);
    Ok(())
}

fn init_run(config_path: &Path, output: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let estimate = model_estimate(&config)?;
    let plan = sequence_budget(&config).estimate();
    std::fs::create_dir_all(output)
        .with_context(|| format!("create run dir {}", output.display()))?;
    let manifest = RunManifest {
        train_shard: config.dataset.train_shard.display().to_string(),
        model_preset: config.model.preset.clone(),
        vocab_size: config.tokenizer.vocab_size,
        backend_kind: config.backend.kind.clone(),
        backend_hidden_size: config.backend.hidden_size,
        backend_output_size: config.backend.output_size,
        backend_blank_id: config.backend.blank_id,
        backend_max_positions: config.backend.max_positions,
        batch_size: config.train.batch_size,
        max_input_len: config.train.max_input_len,
        max_target_len: config.train.max_target_len,
        grad_accum: config.train.grad_accum,
        block_rows: config.train.block_rows,
        seed: config.train.seed,
        prefetch_queue: config.runtime.prefetch_queue,
        parameter_count: estimate.parameter_count,
        parameter_bytes: estimate.parameter_bytes,
        gradient_bytes: estimate.gradient_bytes,
        optimizer_bytes: estimate.optimizer_bytes,
        activation_bytes: estimate.activation_bytes,
        logits_bytes: estimate.logits_bytes,
        total_step_bytes: estimate.total_step_bytes,
        estimated_step_upper_bound: plan.per_batch_bytes * config.train.grad_accum
            + estimate.total_step_bytes,
        optimizer: config.backend.optimizer.clone(),
        scheduler: config.backend.scheduler.clone(),
        learning_rate: config.backend.learning_rate,
        weight_decay: config.backend.weight_decay,
        beta1: config.backend.beta1,
        beta2: config.backend.beta2,
        epsilon: config.backend.epsilon,
        warmup_steps: config.backend.warmup_steps,
        scheduler_total_steps: config.backend.scheduler_total_steps,
        min_lr_scale: config.backend.min_lr_scale,
        refine_loss_weight: config.backend.refine_loss_weight,
        refine_warmup_steps: config.backend.refine_warmup_steps,
        refine_mask_ratio: config.backend.refine_mask_ratio,
        refine_source: config.backend.refine_source.clone(),
        remask_loss_weight: config.backend.remask_loss_weight,
        stop_loss_weight: config.backend.stop_loss_weight,
        grad_clip: config.backend.grad_clip,
    };
    let manifest_path = output.join("run_manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).context("serialize run manifest")?,
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;
    let state_path = output.join("trainer_state.json");
    write_state(
        &state_path,
        &TrainerState {
            step: 0,
            epoch: 0,
            data_cursor: 0,
            best_metric: None,
            best_checkpoint: None,
            last_checkpoint: None,
            checkpoints: Vec::new(),
        },
    )?;
    println!("run dir: {}", output.display());
    println!("manifest: {}", manifest_path.display());
    println!("trainer_state: {}", state_path.display());
    Ok(())
}

fn record_checkpoint(
    run_dir: &Path,
    step: usize,
    epoch: usize,
    checkpoint: &Path,
    metric: Option<f64>,
    kind: &str,
    metric_mode: &str,
) -> Result<()> {
    let state_path = run_dir.join("trainer_state.json");
    let manifest = read_manifest(&run_dir.join("run_manifest.json"))?;
    let mut state = read_state(&state_path)?;
    if !checkpoint.exists() {
        anyhow::bail!("checkpoint does not exist: {}", checkpoint.display());
    }
    let metric_mode = parse_metric_mode(metric_mode)?;
    let checkpoint_str = checkpoint.display().to_string();
    state.step = step;
    state.epoch = epoch;
    state.data_cursor = step.saturating_mul(manifest.batch_size);
    state.last_checkpoint = Some(checkpoint_str.clone());
    if let Some(metric_value) = metric {
        state.best_metric = match state.best_metric {
            Some(current) if !metric_mode.is_better(metric_value, current) => Some(current),
            _ => Some(metric_value),
        };
    }
    state.checkpoints.push(CheckpointEntry {
        step,
        epoch,
        checkpoint: checkpoint_str,
        metric,
        kind: kind.to_string(),
        metric_mode: metric_mode.as_str().to_string(),
    });
    write_state(&state_path, &state)?;
    println!("updated trainer state: {}", state_path.display());
    println!("step: {}", state.step);
    println!("epoch: {}", state.epoch);
    println!("data_cursor: {}", state.data_cursor);
    println!("checkpoints: {}", state.checkpoints.len());
    Ok(())
}

fn show_run(run_dir: &Path) -> Result<()> {
    let manifest_path = run_dir.join("run_manifest.json");
    let state_path = run_dir.join("trainer_state.json");
    let manifest = read_manifest(&manifest_path)?;
    let state = read_state(&state_path)?;
    println!("run_dir: {}", run_dir.display());
    println!("train_shard: {}", manifest.train_shard);
    println!("model_preset: {}", manifest.model_preset);
    println!("vocab_size: {}", manifest.vocab_size);
    println!("batch_size: {}", manifest.batch_size);
    println!("backend_kind: {}", manifest.backend_kind);
    println!("step: {}", state.step);
    println!("epoch: {}", state.epoch);
    println!("data_cursor: {}", state.data_cursor);
    println!(
        "best_metric: {}",
        state
            .best_metric
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "null".to_string())
    );
    println!(
        "last_checkpoint: {}",
        state.last_checkpoint.as_deref().unwrap_or("null")
    );
    println!(
        "best_checkpoint: {}",
        state.best_checkpoint.as_deref().unwrap_or("null")
    );
    println!("checkpoint_entries: {}", state.checkpoints.len());
    Ok(())
}

fn check_resume(config_path: &Path, run_dir: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let manifest = read_manifest(&run_dir.join("run_manifest.json"))?;
    let state = read_state(&run_dir.join("trainer_state.json"))?;
    let mismatches = collect_resume_mismatches(&config, &manifest, &state);
    if mismatches.is_empty() {
        println!("resume_ok: true");
        println!("step: {}", state.step);
        println!("epoch: {}", state.epoch);
        println!("data_cursor: {}", state.data_cursor);
        println!(
            "last_checkpoint: {}",
            state.last_checkpoint.as_deref().unwrap_or("null")
        );
    } else {
        println!("resume_ok: false");
        for mismatch in mismatches {
            println!("mismatch: {}", mismatch);
        }
    }
    Ok(())
}

fn fit(
    config_path: &Path,
    run_dir: &Path,
    steps: usize,
    checkpoint_every: usize,
    device: Device,
    async_ckpt_queue: usize,
    grad_clip: f64,
) -> Result<()> {
    let config = load_config(config_path)?;
    let manifest = read_manifest(&run_dir.join("run_manifest.json"))?;
    let state_path = run_dir.join("trainer_state.json");
    let mut state = read_state(&state_path)?;
    let mismatches = collect_resume_mismatches(&config, &manifest, &state);
    if !mismatches.is_empty() && state.step > 0 {
        anyhow::bail!("run is not resume-compatible; use check-resume first");
    }
    if device.is_cuda() && !device::backend_supports_cuda(&config.backend.kind) {
        anyhow::bail!(
            "backend kind `{}` cannot run on {device}; use `tch-ctc-nat` or pick a CPU backend",
            config.backend.kind,
        );
    }
    let mut source = open_batch_source_at_cursor(&config, state.data_cursor)?;
    // CLI --grad-clip overrides the config-resident value when set to a
    // nonzero number. Log the override so resume drift is visible in
    // the training log, not just the manifest.
    let effective_grad_clip = if grad_clip > 0.0 {
        if grad_clip != config.backend.grad_clip {
            eprintln!(
                "[kkc-train] grad_clip override: config={} cli={} (using cli)",
                config.backend.grad_clip, grad_clip
            );
        }
        grad_clip
    } else {
        config.backend.grad_clip
    };
    let mut backend: Box<dyn backend::TrainBackend> =
        new_backend(&config.backend, device, effective_grad_clip)?;
    let ckpt_writer = if async_ckpt_queue > 0 {
        Some(pipeline::AsyncCheckpointWriter::spawn(async_ckpt_queue))
    } else {
        None
    };
    // Wire the async writer's sender into the backend. CPU backends
    // ignore it (their saves are tiny), the tch backend drains its
    // safetensors dump through the writer thread.
    if let Some(writer) = ckpt_writer.as_ref() {
        if let Some(sender) = writer.sender() {
            backend.attach_ckpt_sender(sender);
        }
    }
    if let Some(entry) = last_checkpoint_entry(&state) {
        if entry.kind == backend.kind() {
            let backend_checkpoint =
                checkpoint_sidecar_path(&entry.checkpoint, ".ckpt.json", ".backend.json");
            if backend_checkpoint.exists() {
                backend.load_checkpoint(&backend_checkpoint)?;
            }
        }
    }
    let target_step = state.step.saturating_add(steps);
    let summary = trainer::run_training_loop(
        &mut source,
        backend.as_mut(),
        &mut state,
        run_dir,
        trainer::TrainerLoopConfig {
            target_step,
            checkpoint_every,
            epoch_steps: estimate_epoch_steps(&config)?,
            grad_accum: config.train.grad_accum,
            checkpoint_keep_last: config.train.checkpoint_keep_last,
        },
    )?;
    let eval_loss = if config.eval.batches > 0 {
        Some(eval_backend_dyn(
            &config,
            backend.as_mut(),
            config.eval.batches,
        )?)
    } else {
        None
    };
    if let Some(eval_loss) = eval_loss {
        update_last_checkpoint_metric(&mut state, eval_loss, "minimize");
    }
    // Drop the backend BEFORE finishing the writer. The backend holds a
    // `SyncSender` clone for its async checkpoint path; leaving it alive
    // would keep the writer's channel open and block `writer.finish()`'s
    // thread join indefinitely.
    drop(backend);
    let mut flushed = 0usize;
    if let Some(writer) = ckpt_writer {
        flushed = writer.finish()?;
    }
    write_state(&state_path, &state)?;
    println!("fit_async_ckpt_flushed: {flushed}");
    println!("fit_backend: {}", config.backend.kind);
    println!("fit_steps: {}", summary.final_step);
    println!("fit_epoch: {}", summary.final_epoch);
    println!("fit_elapsed_sec: {:.6}", summary.elapsed_sec);
    println!("fit_steps_per_sec: {:.2}", summary.steps_per_sec);
    println!(
        "fit_last_loss: {}",
        summary
            .last_loss
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "null".to_string())
    );
    println!(
        "fit_last_checkpoint: {}",
        state.last_checkpoint.as_deref().unwrap_or("null")
    );
    println!(
        "fit_eval_loss: {}",
        eval_loss
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "null".to_string())
    );
    Ok(())
}

fn eval_command(
    config_path: &Path,
    run_dir: Option<&Path>,
    batches: Option<usize>,
    device: Device,
) -> Result<()> {
    let config = load_config(config_path)?;
    if device.is_cuda() && !device::backend_supports_cuda(&config.backend.kind) {
        anyhow::bail!(
            "backend kind `{}` cannot run on {device}; use `tch-ctc-nat` or pick a CPU device",
            config.backend.kind,
        );
    }
    // Route through new_backend so the tch backend is reachable.
    // grad_clip=0.0 is safe here: eval_step never calls the optimizer.
    let mut backend: Box<dyn backend::TrainBackend> = new_backend(&config.backend, device, 0.0)?;
    if let Some(run_dir) = run_dir {
        let state = read_state(&run_dir.join("trainer_state.json"))?;
        if let Some(entry) = last_checkpoint_entry(&state) {
            if entry.kind == backend.kind() {
                let backend_checkpoint =
                    checkpoint_sidecar_path(&entry.checkpoint, ".ckpt.json", ".backend.json");
                if backend_checkpoint.exists() {
                    backend.load_checkpoint(&backend_checkpoint)?;
                }
            }
        }
    }
    let batches = batches.unwrap_or(config.eval.batches.max(1));
    let loss = eval_backend_dyn(&config, backend.as_mut(), batches)?;
    println!("eval_backend: {}", config.backend.kind);
    println!("eval_batches: {}", batches);
    println!("eval_loss: {:.6}", loss);
    Ok(())
}

fn load_config(config_path: &Path) -> Result<TrainConfig> {
    let config_text = std::fs::read_to_string(config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let config: TrainConfig = toml::from_str(&config_text).context("parse train config")?;
    validate_config(&config)?;
    Ok(config)
}

/// Eval path that operates on a trait object so the tch backend (which
/// is not part of the `BackendKind` enum) can share the same driver.
/// Calls `eval_step` so the tch backend's `no_grad` forward is honored
/// and the optimizer stays idle — critical for `best_checkpoint`
/// correctness (the post-fit eval must not continue training).
fn eval_backend_dyn(
    config: &TrainConfig,
    backend: &mut dyn backend::TrainBackend,
    batches: usize,
) -> Result<f64> {
    let eval_path = config
        .eval
        .shard
        .as_ref()
        .or(config.dataset.eval_shard.as_ref())
        .unwrap_or(&config.dataset.train_shard);
    let mut source = open_eval_batch_source(config, eval_path)?;
    let mut total_loss = 0.0;
    let mut seen = 0usize;
    for _ in 0..batches.max(1) {
        let Some(batch) = source.next_batch()? else {
            break;
        };
        let step = backend.eval_step(1, &batch)?;
        total_loss += step.loss;
        seen += 1;
    }
    if seen == 0 {
        return Ok(0.0);
    }
    Ok(total_loss / seen as f64)
}

/// Build the training backend for the configured kind and device. CPU kinds
/// stay within [`backend::BackendKind`]; `tch-ctc-nat` routes to the feature-
/// gated tch backend. Other kinds with `device.is_cuda()` are rejected
/// upfront so silent CPU execution on a GPU run never happens.
fn new_backend(
    config: &backend::BackendConfig,
    device: Device,
    grad_clip: f64,
) -> Result<Box<dyn backend::TrainBackend>> {
    #[cfg(feature = "cuda")]
    {
        if config.kind == "tch-ctc-nat" {
            let mut backend = gpu::TchCtcNatBackend::new(config, device)?;
            backend.attach_optimizer(grad_clip)?;
            return Ok(Box::new(backend));
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        if config.kind == "tch-ctc-nat" {
            anyhow::bail!(
                "backend `tch-ctc-nat` requires building kkc-train with `--features cuda`",
            );
        }
    }
    let _ = grad_clip; // honored only by tch-ctc-nat for now
    if device.is_cuda() {
        anyhow::bail!(
            "backend `{}` has no CUDA path; choose `tch-ctc-nat` or run with --device cpu",
            config.kind,
        );
    }
    Ok(Box::new(backend::BackendKind::new(config)?))
}

fn read_manifest(path: &Path) -> Result<RunManifest> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn update_last_checkpoint_metric(state: &mut TrainerState, metric: f64, metric_mode: &str) {
    let Some(last_checkpoint) = state.last_checkpoint.clone() else {
        return;
    };
    if let Some(entry) = state
        .checkpoints
        .iter_mut()
        .rev()
        .find(|entry| entry.checkpoint == last_checkpoint)
    {
        entry.metric = Some(metric);
        entry.metric_mode = metric_mode.to_string();
    }
    let better = match state.best_metric {
        Some(current) => metric < current,
        None => true,
    };
    if better {
        state.best_metric = Some(metric);
        state.best_checkpoint = Some(last_checkpoint);
    }
}

fn read_state(path: &Path) -> Result<TrainerState> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn write_state(path: &Path, state: &TrainerState) -> Result<()> {
    std::fs::write(
        path,
        serde_json::to_vec_pretty(state).context("serialize trainer state")?,
    )
    .with_context(|| format!("write {}", path.display()))
}

fn read_shard_metadata(path: &Path) -> Result<Option<ShardMetadata>> {
    let sidecar = PathBuf::from(format!("{}.meta.json", path.display()));
    if !sidecar.exists() {
        return Ok(None);
    }
    let bytes = std::fs::read(&sidecar).with_context(|| format!("read {}", sidecar.display()))?;
    let metadata =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", sidecar.display()))?;
    Ok(Some(metadata))
}

fn validate_config(config: &TrainConfig) -> Result<()> {
    if config.train.max_input_len > MAX_SEQUENCE_U16 {
        anyhow::bail!(
            "max_input_len {} exceeds supported limit {}",
            config.train.max_input_len,
            MAX_SEQUENCE_U16
        );
    }
    if config.train.max_target_len > MAX_SEQUENCE_U16 {
        anyhow::bail!(
            "max_target_len {} exceeds supported limit {}",
            config.train.max_target_len,
            MAX_SEQUENCE_U16
        );
    }
    if config.backend.learning_rate <= 0.0 {
        anyhow::bail!("backend.learning_rate must be > 0");
    }
    if config.backend.init_scale <= 0.0 {
        anyhow::bail!("backend.init_scale must be > 0");
    }
    if config.backend.parameter_count == 0 {
        anyhow::bail!("backend.parameter_count must be > 0");
    }
    if config.backend.hidden_size == 0 {
        anyhow::bail!("backend.hidden_size must be > 0");
    }
    if config.backend.encoder_layers == 0 {
        anyhow::bail!("backend.encoder_layers must be > 0");
    }
    if config.backend.num_heads == 0 {
        anyhow::bail!("backend.num_heads must be > 0");
    }
    if config.backend.hidden_size % config.backend.num_heads != 0 {
        anyhow::bail!("backend.hidden_size must be divisible by backend.num_heads");
    }
    if config.backend.ffn_size < config.backend.hidden_size {
        anyhow::bail!("backend.ffn_size must be >= backend.hidden_size");
    }
    if config.backend.decoder_layers == 0 {
        anyhow::bail!("backend.decoder_layers must be > 0");
    }
    if config.backend.decoder_heads == 0 {
        anyhow::bail!("backend.decoder_heads must be > 0");
    }
    if config.backend.hidden_size % config.backend.decoder_heads != 0 {
        anyhow::bail!("backend.hidden_size must be divisible by backend.decoder_heads");
    }
    if config.backend.decoder_ffn_size < config.backend.hidden_size {
        anyhow::bail!("backend.decoder_ffn_size must be >= backend.hidden_size");
    }
    if config.backend.output_size < 2 {
        anyhow::bail!("backend.output_size must be >= 2");
    }
    if config.backend.max_positions == 0 {
        anyhow::bail!("backend.max_positions must be > 0");
    }
    if config.backend.blank_id >= config.backend.output_size {
        anyhow::bail!("backend.blank_id must be < backend.output_size");
    }
    if config.backend.kind == "ctc" && config.backend.output_size != config.tokenizer.vocab_size {
        anyhow::bail!(
            "backend.output_size ({}) must match tokenizer.vocab_size ({}) for ctc backend",
            config.backend.output_size,
            config.tokenizer.vocab_size
        );
    }
    if let Some(metadata) = read_shard_metadata(&config.dataset.train_shard)? {
        if metadata.vocab_size != config.tokenizer.vocab_size {
            anyhow::bail!(
                "train shard metadata vocab_size ({}) does not match tokenizer.vocab_size ({}) for {}. Recompile the shard or point the config at the matching tokenizer.",
                metadata.vocab_size,
                config.tokenizer.vocab_size,
                config.dataset.train_shard.display()
            );
        }
        if config.backend.kind == "ctc" && metadata.vocab_size != config.backend.output_size {
            anyhow::bail!(
                "train shard metadata vocab_size ({}) does not match backend.output_size ({}) for {}. Recompile the shard or update the config.",
                metadata.vocab_size,
                config.backend.output_size,
                config.dataset.train_shard.display()
            );
        }
        if metadata.max_reading_tokens > config.train.max_input_len {
            anyhow::bail!(
                "train shard metadata max_reading_tokens ({}) exceeds train.max_input_len ({}) for {}",
                metadata.max_reading_tokens,
                config.train.max_input_len,
                config.dataset.train_shard.display()
            );
        }
        if metadata.max_surface_tokens > config.train.max_target_len {
            anyhow::bail!(
                "train shard metadata max_surface_tokens ({}) exceeds train.max_target_len ({}) for {}",
                metadata.max_surface_tokens,
                config.train.max_target_len,
                config.dataset.train_shard.display()
            );
        }
    }
    let _ = crate::optim::OptimizerState::new(&config.backend)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum MetricMode {
    Maximize,
    Minimize,
}

impl MetricMode {
    fn is_better(self, candidate: f64, current: f64) -> bool {
        match self {
            MetricMode::Maximize => candidate > current,
            MetricMode::Minimize => candidate < current,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            MetricMode::Maximize => "maximize",
            MetricMode::Minimize => "minimize",
        }
    }
}

fn parse_metric_mode(value: &str) -> Result<MetricMode> {
    match value {
        "maximize" => Ok(MetricMode::Maximize),
        "minimize" => Ok(MetricMode::Minimize),
        other => anyhow::bail!("unknown metric_mode: {}", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_config(run_dir: &Path) -> TrainConfig {
        TrainConfig {
            dataset: DatasetConfig {
                train_shard: run_dir.join("train.kkc"),
                eval_shard: None,
            },
            tokenizer: TokenizerConfig {
                path: None,
                max_kanji: 6000,
                vocab_size: 4801,
            },
            model: ModelConfig {
                preset: "phase3_20m".to_string(),
            },
            runtime: RuntimeConfig {
                param_dtype_bytes: 2,
                grad_dtype_bytes: 4,
                adam_state_bytes: 8,
                activation_dtype_bytes: 2,
                prefetch_queue: 0,
            },
            eval: EvalSection::default(),
            backend: backend::BackendConfig {
                kind: "mock".to_string(),
                ..backend::BackendConfig::default()
            },
            train: TrainSection {
                batch_size: 32,
                max_input_len: 128,
                max_target_len: 128,
                grad_accum: 4,
                block_rows: 4096,
                seed: 42,
                checkpoint_keep_last: 2,
            },
        }
    }

    fn sample_manifest(run_dir: &Path) -> RunManifest {
        RunManifest {
            train_shard: run_dir.join("train.kkc").display().to_string(),
            model_preset: "phase3_20m".to_string(),
            vocab_size: 4801,
            backend_kind: "mock".to_string(),
            backend_hidden_size: 32,
            backend_output_size: 512,
            backend_blank_id: 4,
            backend_max_positions: 128,
            batch_size: 32,
            max_input_len: 128,
            max_target_len: 128,
            grad_accum: 4,
            block_rows: 4096,
            seed: 42,
            prefetch_queue: 0,
            parameter_count: 1,
            parameter_bytes: 2,
            gradient_bytes: 4,
            optimizer_bytes: 8,
            activation_bytes: 16,
            logits_bytes: 8,
            total_step_bytes: 30,
            estimated_step_upper_bound: 64,
            optimizer: "adamw".to_string(),
            scheduler: "warmup_cosine".to_string(),
            learning_rate: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            warmup_steps: 0,
            scheduler_total_steps: 0,
            min_lr_scale: 0.1,
            refine_loss_weight: 0.0,
            refine_warmup_steps: 0,
            refine_mask_ratio: 0.3,
            refine_source: "target".to_string(),
            remask_loss_weight: 0.0,
            stop_loss_weight: 0.0,
            grad_clip: 0.0,
        }
    }

    fn sample_state() -> TrainerState {
        TrainerState {
            step: 0,
            epoch: 0,
            data_cursor: 0,
            best_metric: None,
            best_checkpoint: None,
            last_checkpoint: None,
            checkpoints: Vec::new(),
        }
    }

    #[test]
    fn validate_config_rejects_lengths_over_u16() {
        let dir = tempdir().unwrap();
        let mut config = sample_config(dir.path());
        config.train.max_input_len = MAX_SEQUENCE_U16 + 1;
        assert!(validate_config(&config).is_err());

        config.train.max_input_len = 128;
        config.train.max_target_len = MAX_SEQUENCE_U16 + 1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn metric_mode_minimize_picks_lower_value() {
        let mode = parse_metric_mode("minimize").unwrap();
        assert!(mode.is_better(0.1, 0.2));
        assert!(!mode.is_better(0.3, 0.2));
    }

    #[test]
    fn validate_config_rejects_invalid_backend_settings() {
        let dir = tempdir().unwrap();
        let mut config = sample_config(dir.path());
        config.backend.learning_rate = 0.0;
        assert!(validate_config(&config).is_err());

        config.backend.learning_rate = 1e-3;
        config.backend.parameter_count = 0;
        assert!(validate_config(&config).is_err());

        config.backend.parameter_count = 16;
        config.backend.hidden_size = 0;
        assert!(validate_config(&config).is_err());

        config.backend.hidden_size = 32;
        config.backend.output_size = 1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_shard_metadata_vocab_mismatch() {
        let dir = tempdir().unwrap();
        let config = sample_config(dir.path());
        std::fs::write(&config.dataset.train_shard, []).unwrap();
        std::fs::write(
            format!("{}.meta.json", config.dataset.train_shard.display()),
            r#"{
  "shard_version": 1,
  "row_count": 1,
  "max_context_chars": 40,
  "max_reading_tokens": 8,
  "max_surface_tokens": 8,
  "vocab_size": 6653,
  "sources": { "test": 1 }
}"#,
        )
        .unwrap();
        let err = validate_config(&config).unwrap_err().to_string();
        assert!(err.contains("metadata vocab_size"));
    }

    #[test]
    fn write_and_read_state_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trainer_state.json");
        let mut state = sample_state();
        state.last_checkpoint = Some("foo.ckpt".to_string());
        state.checkpoints.push(CheckpointEntry {
            step: 10,
            epoch: 1,
            checkpoint: "foo.ckpt".to_string(),
            metric: Some(0.5),
            kind: "regular".to_string(),
            metric_mode: "maximize".to_string(),
        });
        write_state(&path, &state).unwrap();
        let loaded = read_state(&path).unwrap();
        assert_eq!(loaded.step, state.step);
        assert_eq!(loaded.last_checkpoint, state.last_checkpoint);
        assert_eq!(loaded.checkpoints.len(), 1);
    }

    #[test]
    fn checkpoint_sidecar_path_rewrites_mock_suffix() {
        let path = checkpoint_sidecar_path(
            "D:/tmp/mock_step_00000010.ckpt.json",
            ".ckpt.json",
            ".backend.json",
        );
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("mock_step_00000010.backend.json")
        );
    }

    #[test]
    fn backend_sidecar_requirement_includes_ctc() {
        assert!(backend_kind_requires_sidecar("ctc"));
        assert!(!backend_kind_requires_sidecar("regular"));
    }

    #[test]
    fn resume_requires_existing_checkpoint() {
        let dir = tempdir().unwrap();
        let run_dir = dir.path();
        let manifest_path = run_dir.join("run_manifest.json");
        let state_path = run_dir.join("trainer_state.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&sample_manifest(run_dir)).unwrap(),
        )
        .unwrap();
        write_state(&state_path, &sample_state()).unwrap();

        let state = read_state(&state_path).unwrap();
        assert!(state.last_checkpoint.is_none());
    }

    #[test]
    fn last_checkpoint_entry_finds_latest_matching_entry() {
        let mut state = sample_state();
        state.checkpoints.push(CheckpointEntry {
            step: 10,
            epoch: 0,
            checkpoint: "a.ckpt".to_string(),
            metric: Some(1.0),
            kind: "mock".to_string(),
            metric_mode: "minimize".to_string(),
        });
        state.checkpoints.push(CheckpointEntry {
            step: 20,
            epoch: 0,
            checkpoint: "b.ckpt".to_string(),
            metric: Some(0.5),
            kind: "mock".to_string(),
            metric_mode: "minimize".to_string(),
        });
        state.last_checkpoint = Some("b.ckpt".to_string());
        let entry = last_checkpoint_entry(&state).unwrap();
        assert_eq!(entry.step, 20);
    }

    #[test]
    fn resume_mismatch_detects_untracked_checkpoint() {
        let dir = tempdir().unwrap();
        let config = sample_config(dir.path());
        let manifest = sample_manifest(dir.path());
        let checkpoint = dir.path().join("step_00000001.ckpt.json");
        std::fs::write(&checkpoint, b"{}").unwrap();

        let mut state = sample_state();
        state.last_checkpoint = Some(checkpoint.display().to_string());
        let mismatches = collect_resume_mismatches(&config, &manifest, &state);
        assert!(mismatches
            .iter()
            .any(|m| m.contains("last_checkpoint_untracked")));
    }

    #[test]
    fn resume_mismatch_detects_backend_shape_change() {
        let dir = tempdir().unwrap();
        let mut config = sample_config(dir.path());
        let manifest = sample_manifest(dir.path());
        let checkpoint = dir.path().join("step_00000001.ckpt.json");
        let backend = dir.path().join("step_00000001.backend.json");
        std::fs::write(&checkpoint, b"{}").unwrap();
        std::fs::write(&backend, b"{}").unwrap();

        let mut state = sample_state();
        state.last_checkpoint = Some(checkpoint.display().to_string());
        state.checkpoints.push(CheckpointEntry {
            step: 1,
            epoch: 0,
            checkpoint: checkpoint.display().to_string(),
            metric: Some(1.0),
            kind: "mock".to_string(),
            metric_mode: "minimize".to_string(),
        });

        config.backend.hidden_size += 1;
        let mismatches = collect_resume_mismatches(&config, &manifest, &state);
        assert!(mismatches.iter().any(|m| m.contains("backend_hidden_size")));
    }

    #[test]
    fn resume_mismatch_detects_optim_and_grad_clip_change() {
        let dir = tempdir().unwrap();
        let mut config = sample_config(dir.path());
        let manifest = sample_manifest(dir.path());
        let mut state = sample_state();
        let checkpoint = dir.path().join("step_00000001.ckpt.json");
        std::fs::write(&checkpoint, b"{}").unwrap();
        state.last_checkpoint = Some(checkpoint.display().to_string());
        state.checkpoints.push(CheckpointEntry {
            step: 1,
            epoch: 0,
            checkpoint: checkpoint.display().to_string(),
            metric: Some(1.0),
            kind: "mock".to_string(),
            metric_mode: "minimize".to_string(),
        });
        config.backend.learning_rate = 3e-3;
        config.backend.grad_clip = 0.5;
        config.backend.warmup_steps = 500;
        let mismatches = collect_resume_mismatches(&config, &manifest, &state);
        assert!(
            mismatches.iter().any(|m| m.contains("learning_rate")),
            "missing learning_rate in {mismatches:?}"
        );
        assert!(
            mismatches.iter().any(|m| m.contains("grad_clip")),
            "missing grad_clip in {mismatches:?}"
        );
        assert!(
            mismatches.iter().any(|m| m.contains("warmup_steps")),
            "missing warmup_steps in {mismatches:?}"
        );
    }
}

fn iter_config(config: &TrainConfig) -> BatchIterConfig {
    BatchIterConfig {
        batch_size: config.train.batch_size,
        max_input_len: config.train.max_input_len,
        max_target_len: config.train.max_target_len,
        block_rows: config.train.block_rows,
        seed: config.train.seed,
        drop_last: false,
    }
}

fn sequence_budget(config: &TrainConfig) -> SequenceBudget {
    SequenceBudget {
        batch_size: config.train.batch_size,
        max_input_len: config.train.max_input_len,
        max_target_len: config.train.max_target_len,
    }
}

fn estimate_epoch_steps(config: &TrainConfig) -> Result<usize> {
    let stats = inspect_shard_batches(&config.dataset.train_shard, iter_config(config), 1)?;
    let rows = stats.rows;
    if rows == 0 {
        return Ok(1);
    }
    Ok(rows.div_ceil(
        config
            .train
            .batch_size
            .max(1)
            .saturating_mul(config.train.grad_accum.max(1)),
    ))
}

fn model_estimate(config: &TrainConfig) -> Result<kkc_model::ResourceEstimate> {
    let preset = ctc_nat_preset(&config.model.preset)?;
    Ok(estimate_ctc_nat_resources(
        preset,
        BatchShape {
            batch_size: config.train.batch_size,
            input_len: config.train.max_input_len,
            target_len: config.train.max_target_len,
            vocab_size: config.tokenizer.vocab_size,
        },
        RuntimeAssumptions {
            param_dtype_bytes: config.runtime.param_dtype_bytes,
            grad_dtype_bytes: config.runtime.grad_dtype_bytes,
            adam_state_bytes: config.runtime.adam_state_bytes,
            activation_dtype_bytes: config.runtime.activation_dtype_bytes,
        },
    ))
}

fn collect_resume_mismatches(
    config: &TrainConfig,
    manifest: &RunManifest,
    state: &TrainerState,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    if manifest.train_shard != config.dataset.train_shard.display().to_string() {
        mismatches.push(format!(
            "train_shard: run={} current={}",
            manifest.train_shard,
            config.dataset.train_shard.display()
        ));
    }
    if manifest.model_preset != config.model.preset {
        mismatches.push(format!(
            "model_preset: run={} current={}",
            manifest.model_preset, config.model.preset
        ));
    }
    if manifest.vocab_size != config.tokenizer.vocab_size {
        mismatches.push(format!(
            "vocab_size: run={} current={}",
            manifest.vocab_size, config.tokenizer.vocab_size
        ));
    }
    if manifest.backend_kind != config.backend.kind {
        mismatches.push(format!(
            "backend_kind: run={} current={}",
            manifest.backend_kind, config.backend.kind
        ));
    }
    if manifest.backend_hidden_size != config.backend.hidden_size {
        mismatches.push(format!(
            "backend_hidden_size: run={} current={}",
            manifest.backend_hidden_size, config.backend.hidden_size
        ));
    }
    if manifest.backend_output_size != config.backend.output_size {
        mismatches.push(format!(
            "backend_output_size: run={} current={}",
            manifest.backend_output_size, config.backend.output_size
        ));
    }
    if manifest.backend_blank_id != config.backend.blank_id {
        mismatches.push(format!(
            "backend_blank_id: run={} current={}",
            manifest.backend_blank_id, config.backend.blank_id
        ));
    }
    if manifest.backend_max_positions != config.backend.max_positions {
        mismatches.push(format!(
            "backend_max_positions: run={} current={}",
            manifest.backend_max_positions, config.backend.max_positions
        ));
    }
    if manifest.batch_size != config.train.batch_size {
        mismatches.push(format!(
            "batch_size: run={} current={}",
            manifest.batch_size, config.train.batch_size
        ));
    }
    if manifest.max_input_len != config.train.max_input_len {
        mismatches.push(format!(
            "max_input_len: run={} current={}",
            manifest.max_input_len, config.train.max_input_len
        ));
    }
    if manifest.max_target_len != config.train.max_target_len {
        mismatches.push(format!(
            "max_target_len: run={} current={}",
            manifest.max_target_len, config.train.max_target_len
        ));
    }
    if manifest.grad_accum != config.train.grad_accum {
        mismatches.push(format!(
            "grad_accum: run={} current={}",
            manifest.grad_accum, config.train.grad_accum
        ));
    }
    if manifest.block_rows != config.train.block_rows {
        mismatches.push(format!(
            "block_rows: run={} current={}",
            manifest.block_rows, config.train.block_rows
        ));
    }
    if manifest.seed != config.train.seed {
        mismatches.push(format!(
            "seed: run={} current={}",
            manifest.seed, config.train.seed
        ));
    }
    if manifest.prefetch_queue != config.runtime.prefetch_queue {
        mismatches.push(format!(
            "prefetch_queue: run={} current={}",
            manifest.prefetch_queue, config.runtime.prefetch_queue
        ));
    }
    // Optim + schedule + refine weights. Float comparisons use exact
    // equality because these are configured knobs, not computed values
    // — a silent 1e-12 drift shouldn't happen and, if it does, we want
    // to surface it rather than hide it behind a tolerance.
    if !manifest.optimizer.is_empty() && manifest.optimizer != config.backend.optimizer {
        mismatches.push(format!(
            "optimizer: run={} current={}",
            manifest.optimizer, config.backend.optimizer
        ));
    }
    if !manifest.scheduler.is_empty() && manifest.scheduler != config.backend.scheduler {
        mismatches.push(format!(
            "scheduler: run={} current={}",
            manifest.scheduler, config.backend.scheduler
        ));
    }
    for (label, run, cur) in [
        ("learning_rate", manifest.learning_rate, config.backend.learning_rate),
        ("weight_decay", manifest.weight_decay, config.backend.weight_decay),
        ("beta1", manifest.beta1, config.backend.beta1),
        ("beta2", manifest.beta2, config.backend.beta2),
        ("epsilon", manifest.epsilon, config.backend.epsilon),
        ("min_lr_scale", manifest.min_lr_scale, config.backend.min_lr_scale),
        ("refine_loss_weight", manifest.refine_loss_weight, config.backend.refine_loss_weight),
        ("refine_mask_ratio", manifest.refine_mask_ratio, config.backend.refine_mask_ratio),
        ("remask_loss_weight", manifest.remask_loss_weight, config.backend.remask_loss_weight),
        ("stop_loss_weight", manifest.stop_loss_weight, config.backend.stop_loss_weight),
        ("grad_clip", manifest.grad_clip, config.backend.grad_clip),
    ] {
        // Ignore the "manifest was created before this field existed"
        // case (run==0.0) for keys with a natural default of 0.0. The
        // optimizer / scheduler string checks above carry the real
        // signal for pre-existing runs; this keeps us from spamming
        // mismatches on a fresh upgrade.
        if run != cur && !(run == 0.0 && manifest.optimizer.is_empty()) {
            mismatches.push(format!("{label}: run={run} current={cur}"));
        }
    }
    for (label, run, cur) in [
        (
            "warmup_steps",
            manifest.warmup_steps,
            config.backend.warmup_steps,
        ),
        (
            "scheduler_total_steps",
            manifest.scheduler_total_steps,
            config.backend.scheduler_total_steps,
        ),
        (
            "refine_warmup_steps",
            manifest.refine_warmup_steps,
            config.backend.refine_warmup_steps,
        ),
    ] {
        if run != cur && !(run == 0 && manifest.optimizer.is_empty()) {
            mismatches.push(format!("{label}: run={run} current={cur}"));
        }
    }
    if !manifest.refine_source.is_empty() && manifest.refine_source != config.backend.refine_source
    {
        mismatches.push(format!(
            "refine_source: run={} current={}",
            manifest.refine_source, config.backend.refine_source
        ));
    }
    match state.last_checkpoint.as_deref() {
        Some(last_checkpoint) => {
            if !Path::new(last_checkpoint).exists() {
                mismatches.push(format!("last_checkpoint_missing: {}", last_checkpoint));
            } else {
                match last_checkpoint_entry(state) {
                    Some(entry) => {
                        let needs_backend = backend_kind_requires_sidecar(&entry.kind);
                        if needs_backend {
                            let backend_checkpoint = checkpoint_sidecar_path(
                                last_checkpoint,
                                ".ckpt.json",
                                ".backend.json",
                            );
                            if !backend_checkpoint.exists() {
                                mismatches.push(format!(
                                    "backend_checkpoint_missing: {}",
                                    backend_checkpoint.display()
                                ));
                            }
                        }
                    }
                    None => {
                        mismatches.push(format!("last_checkpoint_untracked: {}", last_checkpoint))
                    }
                }
            }
        }
        None => {
            mismatches.push("last_checkpoint: run has no checkpoint to resume from".to_string())
        }
    }
    mismatches
}

fn backend_kind_requires_sidecar(kind: &str) -> bool {
    matches!(kind, "mock" | "toy" | "surrogate" | "ctc")
}

fn checkpoint_sidecar_path(checkpoint: &str, from_suffix: &str, to_suffix: &str) -> PathBuf {
    let path = PathBuf::from(checkpoint);
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("checkpoint");
    let replaced = file_name
        .strip_suffix(from_suffix)
        .map(|stem| format!("{}{}", stem, to_suffix))
        .unwrap_or_else(|| format!("{}{}", file_name, to_suffix));
    path.with_file_name(replaced)
}

fn last_checkpoint_entry(state: &TrainerState) -> Option<&CheckpointEntry> {
    let last = state.last_checkpoint.as_deref()?;
    state
        .checkpoints
        .iter()
        .rev()
        .find(|entry| entry.checkpoint == last)
}

enum BatchSource {
    Sync(BatchIter),
    Prefetched(PrefetchedBatchIter),
}

impl BatchSource {
    fn next_batch(&mut self) -> Result<Option<kkc_data::PackedBatch>> {
        match self {
            BatchSource::Sync(iter) => iter.next_batch(),
            BatchSource::Prefetched(iter) => iter.next_batch(),
        }
    }
}

impl trainer::BatchStream for BatchSource {
    fn next_batch(&mut self) -> Result<Option<kkc_data::PackedBatch>> {
        BatchSource::next_batch(self)
    }
}

fn open_batch_source(config: &TrainConfig) -> Result<BatchSource> {
    open_batch_source_at_cursor(config, 0)
}

fn open_eval_batch_source(config: &TrainConfig, shard: &Path) -> Result<BatchSource> {
    let iter = BatchIter::open(shard, iter_config(config))?;
    if config.runtime.prefetch_queue > 0 {
        Ok(BatchSource::Prefetched(PrefetchedBatchIter::spawn(
            iter,
            config.runtime.prefetch_queue,
        )))
    } else {
        Ok(BatchSource::Sync(iter))
    }
}

fn open_batch_source_at_cursor(config: &TrainConfig, cursor: usize) -> Result<BatchSource> {
    let iter = BatchIter::open_at_cursor(&config.dataset.train_shard, iter_config(config), cursor)?;
    if config.runtime.prefetch_queue > 0 {
        Ok(BatchSource::Prefetched(PrefetchedBatchIter::spawn(
            iter,
            config.runtime.prefetch_queue,
        )))
    } else {
        Ok(BatchSource::Sync(iter))
    }
}
