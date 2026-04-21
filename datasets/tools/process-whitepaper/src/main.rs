//! Bulk ingestion of Japanese government white papers (白書) into training JSONL.
//!
//! Pipeline stages are separate subcommands so each can be retried/resumed:
//!
//!   1. `discover`  — scrape ministry index pages and enumerate PDF URLs
//!   2. `fetch`     — download the PDFs with resume + retry
//!   3. `extract`   — subprocess `pdftotext -layout` to get raw text
//!   4. `ingest`    — clean text, segment into sentences, emit JSONL
//!                    (reading generation via mecab runs in this stage)
//!
//! Outputs land under `datasets/raw/whitepaper/{pdf,text}/` and final JSONL
//! under `datasets/corpus/sentence/whitepaper.jsonl`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod discover;
mod extract;
mod fetch;
mod ingest;

#[derive(Parser, Debug)]
#[command(
    name = "process-whitepaper",
    about = "Download and normalise Japanese government white papers into training JSONL",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Scrape ministry index pages and write a URL manifest.
    Discover {
        /// Output manifest (TSV: ministry\turl\tfilename).
        #[arg(long, default_value = "datasets/raw/whitepaper/manifest.tsv")]
        out: PathBuf,
        /// Restrict to a comma-separated ministry list (default: all).
        #[arg(long)]
        only: Option<String>,
    },
    /// Download PDFs listed in the manifest.
    Fetch {
        #[arg(long, default_value = "datasets/raw/whitepaper/manifest.tsv")]
        manifest: PathBuf,
        #[arg(long, default_value = "datasets/raw/whitepaper/pdf")]
        out_dir: PathBuf,
        /// Concurrent download workers.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Max retries per URL.
        #[arg(long, default_value_t = 3)]
        retries: u32,
        /// Stop after this many MiB downloaded across the run (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        max_mib: u64,
    },
    /// Run `pdftotext -layout` over every downloaded PDF that lacks a text file.
    Extract {
        #[arg(long, default_value = "datasets/raw/whitepaper/pdf")]
        pdf_dir: PathBuf,
        #[arg(long, default_value = "datasets/raw/whitepaper/text")]
        text_dir: PathBuf,
        /// Concurrent pdftotext subprocesses.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
    /// Clean extracted text, segment sentences, generate readings, emit JSONL.
    Ingest {
        #[arg(long, default_value = "datasets/raw/whitepaper/text")]
        text_dir: PathBuf,
        #[arg(long, default_value = "datasets/corpus/sentence/whitepaper.jsonl")]
        out: PathBuf,
        /// Path to the ipadic-neologd dictionary (vibrato format).
        #[arg(long)]
        dict: PathBuf,
        /// Minimum sentence length in characters (after cleaning).
        #[arg(long, default_value_t = 8)]
        min_chars: usize,
        /// Maximum sentence length (sentences longer than this are dropped).
        #[arg(long, default_value_t = 128)]
        max_chars: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Discover { out, only } => discover::run(&out, only.as_deref()),
        Command::Fetch {
            manifest,
            out_dir,
            concurrency,
            retries,
            max_mib,
        } => fetch::run(&manifest, &out_dir, concurrency, retries, max_mib),
        Command::Extract { pdf_dir, text_dir, concurrency } => {
            extract::run(&pdf_dir, &text_dir, concurrency)
        }
        Command::Ingest { text_dir, out, dict, min_chars, max_chars } => {
            ingest::run(&text_dir, &out, &dict, min_chars, max_chars)
        }
    }
    .with_context(|| "process-whitepaper")
}
