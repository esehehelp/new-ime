//! Rust replacement for `scripts/process_zenz_subset.py`.
//!
//! Reads the zenz-v2.5 llm-jp subset JSONL, renames fields, converts
//! input katakana to hiragana, applies length filters, optional 6-gram
//! contamination audit, and writes rows with `source="zenz_llmjp"`.
//!
//! Matches the Python version's filter semantics bit-for-bit so outputs
//! can be diffed during migration.

use anyhow::{Context, Result};
use clap::Parser;
use data_core::{jsonl, kana, NgramSet, OutputFormat, Row};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
pub struct ZenzRow {
    #[serde(default)]
    pub input: String,
    #[serde(default)]
    pub output: String,
    #[serde(default)]
    pub left_context: Option<String>,
}

const MIN_SURFACE_LEN: usize = 5;
const MAX_SURFACE_LEN: usize = 100;
const MIN_READING_LEN: usize = 2;
const MAX_READING_LEN: usize = 200;

/// Outcome of running one zenz row through the processor's filter chain.
#[derive(Debug, PartialEq)]
pub enum RowDecision {
    /// Row accepted, transformed `Row` is the value to emit.
    Accept(Row),
    /// Input or output empty.
    SkipEmpty,
    /// Reading or surface length outside configured bounds.
    SkipLen,
    /// Surface 6-gram overlaps the contamination set.
    SkipContam,
}

/// Pure transformation used by both the binary entry point and unit tests.
pub fn process_zenz_row(zenz: ZenzRow, contamination: &NgramSet) -> RowDecision {
    if zenz.input.is_empty() || zenz.output.is_empty() {
        return RowDecision::SkipEmpty;
    }
    let reading = kana::kata_to_hira(&zenz.input);
    let surface = zenz.output;
    let context = zenz.left_context.unwrap_or_default();
    let surf_len = surface.chars().count();
    let read_len = reading.chars().count();
    if !(MIN_SURFACE_LEN..=MAX_SURFACE_LEN).contains(&surf_len) {
        return RowDecision::SkipLen;
    }
    if !(MIN_READING_LEN..=MAX_READING_LEN).contains(&read_len) {
        return RowDecision::SkipLen;
    }
    if !contamination.is_empty() && contamination.contains_overlap(&surface) {
        return RowDecision::SkipContam;
    }
    RowDecision::Accept(Row::new(
        reading,
        surface,
        context,
        Some("zenz_llmjp".to_string()),
    ))
}

#[derive(Parser, Debug)]
#[command(about = "Process the zenz-v2.5 llm-jp subset into our training schema.")]
struct Args {
    /// Path to the raw zenz JSONL (llm-jp subset).
    #[arg(long)]
    input: PathBuf,

    /// Output JSONL path (auto-compresses if .zst / .xz / .gz).
    #[arg(long)]
    output: PathBuf,

    /// Optional evaluation JSONL for 6-gram contamination filtering.
    #[arg(long, default_value = "")]
    contamination_ref: String,

    /// N for contamination check.
    #[arg(long, default_value_t = 6usize)]
    contamination_n: usize,

    /// Progress print interval (input rows).
    #[arg(long, default_value_t = 500_000u64)]
    progress_every: u64,

    /// Cap output rows. 0 means unlimited.
    #[arg(long, default_value_t = 0u64)]
    max_rows: u64,

    /// Compression level for the output writer (zstd 1-22, xz 0-9, gzip 1-9).
    #[arg(long, default_value_t = 19)]
    compress_level: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let contamination =
        if !args.contamination_ref.is_empty() && Path::new(&args.contamination_ref).exists() {
            eprintln!(
                "Loading contamination reference from {}",
                args.contamination_ref
            );
            let mut s = NgramSet::new(args.contamination_n);
            s.extend_from_jsonl(Path::new(&args.contamination_ref))?;
            eprintln!("Contamination n-grams: {}", s.len());
            s
        } else {
            NgramSet::new(args.contamination_n)
        };

    let input =
        File::open(&args.input).with_context(|| format!("open {}", args.input.display()))?;
    let reader = BufReader::with_capacity(8 * 1024 * 1024, input);

    let output_format = OutputFormat::from_path(&args.output);
    let mut writer = jsonl::open_output(&args.output, Some(output_format), args.compress_level)?;

    let mut stats = Stats::default();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_no + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        stats.read += 1;
        let zenz: ZenzRow = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        match process_zenz_row(zenz, &contamination) {
            RowDecision::Accept(row) => {
                jsonl::write_row(&mut writer, &row)?;
                stats.written += 1;
                if args.max_rows != 0 && stats.written >= args.max_rows {
                    break;
                }
            }
            RowDecision::SkipEmpty => stats.skipped_empty += 1,
            RowDecision::SkipLen => stats.skipped_len += 1,
            RowDecision::SkipContam => stats.skipped_contam += 1,
        }
        if stats.read % args.progress_every == 0 {
            eprintln!(
                "[{:>12}] written={} skip_empty={} skip_len={} skip_contam={}",
                stats.read,
                stats.written,
                stats.skipped_empty,
                stats.skipped_len,
                stats.skipped_contam
            );
        }
    }

    drop(writer);

    eprintln!(
        "done: read={} written={} skip_empty={} skip_len={} skip_contam={}",
        stats.read, stats.written, stats.skipped_empty, stats.skipped_len, stats.skipped_contam
    );
    eprintln!("output: {}", args.output.display());
    Ok(())
}

#[derive(Default)]
struct Stats {
    read: u64,
    written: u64,
    skipped_empty: u64,
    skipped_len: u64,
    skipped_contam: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zenz(input: &str, output: &str, ctx: Option<&str>) -> ZenzRow {
        ZenzRow {
            input: input.to_string(),
            output: output.to_string(),
            left_context: ctx.map(|s| s.to_string()),
        }
    }

    #[test]
    fn accepts_and_converts_kata_to_hira() {
        let contamination = NgramSet::new(6);
        let row = zenz("ショクイン", "職員の待遇は良好です。", Some("冒頭文"));
        match process_zenz_row(row, &contamination) {
            RowDecision::Accept(r) => {
                assert_eq!(r.reading, "しょくいん");
                assert_eq!(r.surface, "職員の待遇は良好です。");
                assert_eq!(r.context, "冒頭文");
                assert_eq!(r.source.as_deref(), Some("zenz_llmjp"));
            }
            other => panic!("expected Accept, got {:?}", other),
        }
    }

    #[test]
    fn skips_empty_input() {
        let row = zenz("", "空読み", None);
        let contamination = NgramSet::new(6);
        assert_eq!(
            process_zenz_row(row, &contamination),
            RowDecision::SkipEmpty
        );
    }

    #[test]
    fn skips_too_short_surface() {
        // "福澤諭吉" = 4 chars, MIN_SURFACE_LEN=5
        let row = zenz("フクザワユキチ", "福澤諭吉", None);
        let contamination = NgramSet::new(6);
        assert_eq!(process_zenz_row(row, &contamination), RowDecision::SkipLen);
    }

    #[test]
    fn skips_too_short_reading() {
        // reading "ア" = 1 char, MIN_READING_LEN=2
        let row = zenz("ア", "あああああ", None);
        let contamination = NgramSet::new(6);
        assert_eq!(process_zenz_row(row, &contamination), RowDecision::SkipLen);
    }

    #[test]
    fn skips_contamination_match() {
        let mut contamination = NgramSet::new(6);
        contamination.insert_surface("職員の待遇は改善された");
        let row = zenz("ショクイン", "職員の待遇は良好です。", None);
        assert_eq!(
            process_zenz_row(row, &contamination),
            RowDecision::SkipContam
        );
    }

    #[test]
    fn empty_contamination_is_no_op() {
        // Same inputs as contamination test, but set is empty — should accept.
        let contamination = NgramSet::new(6);
        let row = zenz("ショクイン", "職員の待遇は良好です。", None);
        assert!(matches!(
            process_zenz_row(row, &contamination),
            RowDecision::Accept(_)
        ));
    }
}
