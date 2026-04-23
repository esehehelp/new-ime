//! Length-filtered short-line extractor.
//!
//! Reads one or more JSONL sources with {reading, surface, ...} rows and
//! emits only lines whose reading and surface char counts fall inside a
//! configurable window. Dedup is left to the downstream mix builder.
//!
//! Intended use: bulk-harvest short phrases from legacy web/chunks corpora
//! (zenz_llmjp, fineweb2_ja, hplt3_ja, chunks_100m, wiki) to feed the
//! bunsetsu pool when raw bunsetsu sources run out of depth.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Extract short-reading lines from JSONL corpora.")]
struct Args {
    /// Input JSONL path. Repeatable.
    #[arg(long, action = clap::ArgAction::Append)]
    src: Vec<PathBuf>,

    /// Output JSONL path.
    #[arg(long)]
    out: PathBuf,

    /// Reading char count floor (inclusive).
    #[arg(long, default_value_t = 3usize)]
    min_reading: usize,

    /// Reading char count ceiling (inclusive).
    #[arg(long, default_value_t = 25usize)]
    max_reading: usize,

    /// Surface char count floor (inclusive).
    #[arg(long, default_value_t = 2usize)]
    min_surface: usize,

    /// Surface char count ceiling (inclusive).
    #[arg(long, default_value_t = 30usize)]
    max_surface: usize,

    /// Progress report every N lines read.
    #[arg(long, default_value_t = 5_000_000u64)]
    report_every: u64,
}

#[derive(Deserialize)]
struct Row<'a> {
    reading: &'a str,
    surface: &'a str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let out_file =
        File::create(&args.out).with_context(|| format!("create {}", args.out.display()))?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    let t0 = Instant::now();
    let mut total: u64 = 0;
    let mut kept: u64 = 0;
    let mut skip_parse: u64 = 0;
    let mut skip_len: u64 = 0;

    for src in &args.src {
        let file = File::open(src).with_context(|| format!("open {}", src.display()))?;
        let reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut src_total: u64 = 0;
        let mut src_kept: u64 = 0;
        for line in reader.lines() {
            let line = line?;
            total += 1;
            src_total += 1;
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let row: Row = match serde_json::from_str(trimmed) {
                Ok(r) => r,
                Err(_) => {
                    skip_parse += 1;
                    continue;
                }
            };
            let rn = row.reading.chars().count();
            let sn = row.surface.chars().count();
            if rn < args.min_reading
                || rn > args.max_reading
                || sn < args.min_surface
                || sn > args.max_surface
            {
                skip_len += 1;
                continue;
            }
            out.write_all(trimmed.as_bytes())?;
            out.write_all(b"\n")?;
            kept += 1;
            src_kept += 1;
            if total % args.report_every == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                eprintln!(
                    "  [{}] total={}M kept={}M ({:.1}%) rate={:.1}k/s",
                    src.display(),
                    total / 1_000_000,
                    kept / 1_000_000,
                    100.0 * kept as f64 / total as f64,
                    total as f64 / elapsed / 1000.0,
                );
            }
        }
        eprintln!(
            "[{}] done: total={} kept={} ({:.1}%)",
            src.display(),
            src_total,
            src_kept,
            100.0 * src_kept as f64 / src_total.max(1) as f64,
        );
    }
    out.flush()?;
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "\n=== summary ===\n  total read: {}\n  kept: {} ({:.1}%)\n  skip_parse: {}\n  skip_len: {}\n  elapsed: {:.1}s  rate: {:.1}k/s",
        total,
        kept,
        100.0 * kept as f64 / total.max(1) as f64,
        skip_parse,
        skip_len,
        elapsed,
        total as f64 / elapsed / 1000.0,
    );
    Ok(())
}
