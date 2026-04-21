use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kkc_data::{
    compile_jsonl_to_shard, inspect_shard_batches, BatchIter, BatchIterConfig, CompileOptions,
    PrefetchedBatchIter, SequenceBudget,
};
use kkc_model::{ctc_nat_preset, estimate_ctc_nat_resources, BatchShape, RuntimeAssumptions};
use kkc_tokenizer::SharedCharTokenizer;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

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
}

#[derive(Debug, Deserialize)]
struct TrainConfig {
    dataset: DatasetConfig,
    tokenizer: TokenizerConfig,
    model: ModelConfig,
    runtime: RuntimeConfig,
    train: TrainSection,
}

#[derive(Debug, Deserialize)]
struct DatasetConfig {
    train_shard: PathBuf,
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
}

#[derive(Debug, Serialize)]
struct RunManifest {
    train_shard: String,
    model_preset: String,
    vocab_size: usize,
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
    }
}

fn plan(config_path: &Path) -> Result<()> {
    let config = load_config(config_path)?;
    let shard_stats = inspect_shard_batches(
        &config.dataset.train_shard,
        iter_config(&config),
        8,
    )?;
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
    };
    let manifest_path = output.join("run_manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).context("serialize run manifest")?,
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;
    let state_path = output.join("trainer_state.json");
    std::fs::write(
        &state_path,
        br#"{"step":0,"epoch":0,"best_metric":null,"last_checkpoint":null}"#,
    )
    .with_context(|| format!("write {}", state_path.display()))?;
    println!("run dir: {}", output.display());
    println!("manifest: {}", manifest_path.display());
    println!("trainer_state: {}", state_path.display());
    Ok(())
}

fn load_config(config_path: &Path) -> Result<TrainConfig> {
    let config_text = std::fs::read_to_string(config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    toml::from_str(&config_text).context("parse train config")
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

fn open_batch_source(config: &TrainConfig) -> Result<BatchSource> {
    let iter = BatchIter::open(&config.dataset.train_shard, iter_config(config))?;
    if config.runtime.prefetch_queue > 0 {
        Ok(BatchSource::Prefetched(PrefetchedBatchIter::spawn(
            iter,
            config.runtime.prefetch_queue,
        )))
    } else {
        Ok(BatchSource::Sync(iter))
    }
}
