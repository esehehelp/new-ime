//! CLI entrypoint for `rust-data`.
//!
//! Wraps the library API so Python-train and ops scripts can drive shard
//! compilation + inspection from the shell. Phase A of the dev-branch
//! plan (Python-train 主軸 + Rust dataloader 置換).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rust_data::{
    compile_jsonl_to_shard, inspect_shard_batches, BatchIter, BatchIterConfig, CompileOptions,
    ShardMetadata,
};
use rust_tokenizer::SharedCharTokenizer;
use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "rust-data", about = "Shard compile / inspect / stream utilities")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compile a JSONL corpus into a binary shard (+ sidecar meta JSON).
    Compile(CompileArgs),
    /// Print shard statistics as JSON.
    Inspect(InspectArgs),
    /// Stream batches as JSON lines on stdout (stub for future subprocess path).
    Stream(StreamArgs),
}

#[derive(Debug, Parser)]
struct CompileArgs {
    /// JSONL input path.
    #[arg(long)]
    input: PathBuf,
    /// Shard output path (recommended extension: .kkc).
    #[arg(long)]
    output: PathBuf,
    /// Tokenizer JSON (SharedCharTokenizer format).
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value_t = 40)]
    max_context_chars: usize,
    #[arg(long, default_value_t = 128)]
    max_reading_tokens: usize,
    #[arg(long, default_value_t = 128)]
    max_surface_tokens: usize,
    /// Disable sidecar `*.kkc.meta.json` emission.
    #[arg(long)]
    no_meta: bool,
}

#[derive(Debug, Parser)]
struct InspectArgs {
    /// Shard path to inspect.
    shard: PathBuf,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 128)]
    max_input_len: usize,
    #[arg(long, default_value_t = 128)]
    max_target_len: usize,
    #[arg(long, default_value_t = 4096)]
    block_rows: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of batches to materialize for sample_batch_bytes.
    #[arg(long, default_value_t = 4)]
    sample_batches: usize,
}

#[derive(Debug, Parser)]
struct StreamArgs {
    shard: PathBuf,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 128)]
    max_input_len: usize,
    #[arg(long, default_value_t = 128)]
    max_target_len: usize,
    #[arg(long, default_value_t = 4096)]
    block_rows: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    drop_last: bool,
    /// Optional cap; 0 = stream until exhausted.
    #[arg(long, default_value_t = 0)]
    max_batches: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Compile(args) => run_compile(args),
        Command::Inspect(args) => run_inspect(args),
        Command::Stream(args) => run_stream(args),
    }
}

fn run_compile(args: CompileArgs) -> Result<()> {
    let tokenizer = SharedCharTokenizer::load(&args.tokenizer)
        .with_context(|| format!("load tokenizer {}", args.tokenizer.display()))?;
    let options = CompileOptions {
        max_context_chars: args.max_context_chars,
        max_reading_tokens: args.max_reading_tokens,
        max_surface_tokens: args.max_surface_tokens,
    };
    let meta = compile_jsonl_to_shard(&args.input, &args.output, &tokenizer, &options)?;
    if !args.no_meta {
        let sidecar = meta_sidecar_path(&args.output);
        write_meta_sidecar(&sidecar, &meta)
            .with_context(|| format!("write sidecar {}", sidecar.display()))?;
        eprintln!("compile-shard sidecar written: {}", sidecar.display());
    }
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

fn run_inspect(args: InspectArgs) -> Result<()> {
    let config = BatchIterConfig {
        batch_size: args.batch_size,
        max_input_len: args.max_input_len,
        max_target_len: args.max_target_len,
        block_rows: args.block_rows,
        seed: args.seed,
        drop_last: false,
    };
    let stats = inspect_shard_batches(&args.shard, config, args.sample_batches)?;
    let out = InspectOutput {
        shard: args.shard.display().to_string(),
        rows: stats.rows,
        mean_input_tokens: stats.mean_input_tokens,
        mean_target_tokens: stats.mean_target_tokens,
        max_input_tokens: stats.max_input_tokens,
        max_target_tokens: stats.max_target_tokens,
        sample_batch_bytes: stats.sample_batch_bytes,
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn run_stream(args: StreamArgs) -> Result<()> {
    let config = BatchIterConfig {
        batch_size: args.batch_size,
        max_input_len: args.max_input_len,
        max_target_len: args.max_target_len,
        block_rows: args.block_rows,
        seed: args.seed,
        drop_last: args.drop_last,
    };
    let mut iter = BatchIter::open(&args.shard, config)?;
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let cap = args.max_batches;
    let mut emitted = 0usize;
    while let Some(batch) = iter.next_batch()? {
        if cap > 0 && emitted >= cap {
            break;
        }
        let record = StreamBatch {
            batch_size: batch.batch_size,
            max_input_len: batch.max_input_len,
            max_target_len: batch.max_target_len,
            order_cursor: batch.order_cursor,
            input_lengths: &batch.input_lengths,
            target_lengths: &batch.target_lengths,
            writer_ids: &batch.writer_ids,
            domain_ids: &batch.domain_ids,
            source_ids: &batch.source_ids,
            non_padding_input_tokens: batch.non_padding_input_tokens(),
            non_padding_target_tokens: batch.non_padding_target_tokens(),
            bytes: batch.bytes(),
        };
        serde_json::to_writer(&mut out, &record)?;
        out.write_all(b"\n")?;
        emitted += 1;
    }
    out.flush().context("flush stream stdout")?;
    eprintln!("stream done: batches={}", emitted);
    Ok(())
}

fn meta_sidecar_path(shard: &Path) -> PathBuf {
    // Keep any existing extension, then append ".meta.json" so
    // `train.kkc` → `train.kkc.meta.json` and `shard.bin` → `shard.bin.meta.json`.
    let mut s = shard.as_os_str().to_owned();
    s.push(".meta.json");
    PathBuf::from(s)
}

fn write_meta_sidecar(path: &Path, meta: &ShardMetadata) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
    }
    let bytes = serde_json::to_vec_pretty(meta).context("serialize shard meta")?;
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct InspectOutput {
    shard: String,
    rows: usize,
    mean_input_tokens: f64,
    mean_target_tokens: f64,
    max_input_tokens: usize,
    max_target_tokens: usize,
    sample_batch_bytes: usize,
}

#[derive(Debug, Serialize)]
struct StreamBatch<'a> {
    batch_size: usize,
    max_input_len: usize,
    max_target_len: usize,
    order_cursor: usize,
    input_lengths: &'a [u16],
    target_lengths: &'a [u16],
    writer_ids: &'a [u32],
    domain_ids: &'a [u32],
    source_ids: &'a [u32],
    non_padding_input_tokens: usize,
    non_padding_target_tokens: usize,
    bytes: usize,
}
