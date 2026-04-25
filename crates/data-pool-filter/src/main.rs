//! data-pool-filter: apply rules_v3 to a raw JSONL pool in parallel
//! and write cleaned JSONL (+ optional reject log).
//!
//! Streams (optionally compressed) input once, uses rayon to
//! evaluate rules across cores, preserves original line order in the
//! output so resume / cursor tooling downstream stays stable.
//!
//! Usage:
//!   data-pool-filter \
//!     --input datasets/corpus/bunsetsu/wikibooks.jsonl \
//!     --output datasets/corpus/cleaned/bunsetsu/wikibooks.jsonl \
//!     [--rejects datasets/audits/cleaned/bunsetsu-wikibooks.rejects.jsonl] \
//!     [--report datasets/audits/cleaned/bunsetsu-wikibooks.report.tsv]

mod rules;

use anyhow::{Context, Result};
use clap::Parser;
use data_core::jsonl::{open_output, OutputFormat};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "data-pool-filter",
    about = "Apply rules_v3 to a JSONL pool, producing cleaned output + reject log."
)]
struct Cli {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Optional rejected rows log (with reason). JSONL: {"reason":..,"row":..}.
    #[arg(long)]
    rejects: Option<PathBuf>,
    /// Optional reason-count TSV.
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    compression_level: i32,
    /// Chunk size for parallel evaluation; higher = more mem, fewer sync points.
    #[arg(long, default_value_t = 65536)]
    chunk: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    eprintln!(
        "[filter] input={} output={} rejects={} chunk={}",
        cli.input.display(),
        cli.output.display(),
        cli.rejects
            .as_deref()
            .map(Path::display)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "-".into()),
        cli.chunk
    );

    let reader = open_text_reader(&cli.input)?;
    let mut out_w = open_output(&cli.output, None, cli.compression_level)
        .with_context(|| format!("open output {}", cli.output.display()))?;

    let mut rej_w = match cli.rejects.as_deref() {
        Some(p) => Some(open_output(p, None, cli.compression_level)?),
        None => None,
    };

    let mut reason_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut n_in: u64 = 0;
    let mut n_out: u64 = 0;

    // Read in chunks, evaluate in parallel, then serialize output in original order.
    let mut buf: Vec<String> = Vec::with_capacity(cli.chunk);
    for line in reader.lines() {
        let line = line.context("read line")?;
        if line.trim().is_empty() {
            continue;
        }
        buf.push(line);
        if buf.len() >= cli.chunk {
            process_chunk(
                &mut buf,
                &mut out_w,
                rej_w.as_mut(),
                &mut reason_counts,
                &mut n_in,
                &mut n_out,
            )?;
            if n_in % (cli.chunk as u64 * 10) == 0 {
                eprintln!("  [filter] in={} out={} rejected={}", n_in, n_out, n_in - n_out);
            }
        }
    }
    if !buf.is_empty() {
        process_chunk(
            &mut buf,
            &mut out_w,
            rej_w.as_mut(),
            &mut reason_counts,
            &mut n_in,
            &mut n_out,
        )?;
    }

    drop(out_w);
    if let Some(r) = rej_w {
        drop(r);
    }

    if let Some(report_path) = &cli.report {
        if let Some(parent) = report_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let mut f = File::create(report_path)
            .with_context(|| format!("create report {}", report_path.display()))?;
        writeln!(f, "reason\tcount")?;
        let mut sorted: Vec<_> = reason_counts.iter().collect();
        sorted.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        for (reason, count) in sorted {
            writeln!(f, "{}\t{}", reason, count)?;
        }
    }

    let rate = if n_in > 0 {
        (n_in - n_out) as f64 / n_in as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "[filter] done in={} out={} rejected={} rate={:.2}%",
        n_in,
        n_out,
        n_in - n_out,
        rate
    );
    Ok(())
}

type Evaluation = (String, Option<String>); // (original_line, reject_reason)

fn process_chunk(
    buf: &mut Vec<String>,
    out_w: &mut dyn Write,
    mut rej_w: Option<&mut data_core::jsonl::Writer>,
    reason_counts: &mut BTreeMap<String, u64>,
    n_in: &mut u64,
    n_out: &mut u64,
) -> Result<()> {
    // Parallel evaluate; preserve order by index.
    let evaluated: Vec<Evaluation> = buf
        .par_drain(..)
        .map(|line| {
            let parsed: Result<Value, _> = serde_json::from_str(&line);
            match parsed {
                Ok(row) => {
                    let reason = rules::evaluate(&row);
                    (line, reason)
                }
                Err(_) => (line, Some("lead_punct=parse_error".to_string())),
            }
        })
        .collect();

    for (line, reason) in evaluated {
        *n_in += 1;
        match reason {
            None => {
                out_w.write_all(line.as_bytes())?;
                out_w.write_all(b"\n")?;
                *n_out += 1;
            }
            Some(r) => {
                *reason_counts.entry(r.clone()).or_insert(0) += 1;
                if let Some(ref mut w) = rej_w {
                    // {"reason":..,"row":<original>}
                    w.write_all(b"{\"reason\":")?;
                    let reason_json = serde_json::to_string(&r)?;
                    w.write_all(reason_json.as_bytes())?;
                    w.write_all(b",\"row\":")?;
                    w.write_all(line.as_bytes())?;
                    w.write_all(b"}\n")?;
                }
            }
        }
    }
    Ok(())
}

fn open_text_reader(path: &Path) -> Result<BufReader<Box<dyn Read + Send>>> {
    let file = File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let inner: Box<dyn Read + Send> = match OutputFormat::from_path(path) {
        OutputFormat::Raw => Box::new(file),
        OutputFormat::Zstd => Box::new(zstd::stream::Decoder::new(file)?),
        OutputFormat::Xz => Box::new(xz2::read::XzDecoder::new(file)),
        OutputFormat::Gzip => Box::new(flate2::read::GzDecoder::new(file)),
    };
    Ok(BufReader::with_capacity(16 * 1024 * 1024, inner))
}
