//! Rust replacement for `scripts/audit_pools.py`.
//!
//! Produces the same JSON schema the Python version emits: `pools` array
//! with `path`, `lines`, `avg_reading_chars`, `avg_surface_chars`,
//! `kanji_surface_ratio`, `lexical_overlap`, `sixgram_overlap`,
//! `source_histogram`, `source_licenses`, `has_attribution_file` fields.

use anyhow::{Context, Result};
use clap::Parser;
use datacore::{ngram::surface_ngrams, Row};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const KANJI_START: u32 = 0x4E00;
const KANJI_END: u32 = 0x9FFF;

fn source_license(tag: &str) -> &'static str {
    // Keep in sync with Python SOURCE_LICENSE.
    match tag {
        "zenz_llmjp" => "ODC-BY 1.0 (Common Crawl subset of llm-jp-corpus-v3)",
        "hplt3_ja" => "CC0-1.0 (Common Crawl terms of use apply)",
        "fineweb2_ja" => "ODC-BY 1.0 (Common Crawl terms of use apply)",
        "wiki" => "CC-BY-SA 3.0 — derivative of Wikipedia",
        "aozora" => "Public Domain",
        "livedoor" => "CC BY-ND 2.1 JP — evaluation/exploration only",
        "tatoeba" => "CC-BY 2.0 FR",
        "unknown" => "unknown (no source tag in pool)",
        _ => "(not in SOURCE_LICENSE table)",
    }
}

#[derive(Parser, Debug)]
#[command(about = "Audit Phase 3 candidate data pools.")]
struct Args {
    /// JSONL pool files to audit.
    #[arg(long, num_args = 1..)]
    pools: Vec<PathBuf>,

    /// Evaluation JSONL files used for contamination checks.
    #[arg(long, num_args = 0.., default_values_t = [
        "datasets/eval_v3/dev.jsonl".to_string(),
        "datasets/eval_v3/test.jsonl".to_string(),
        "datasets/gold_1k.jsonl".to_string(),
    ])]
    eval_sets: Vec<String>,

    /// Optional per-pool row cap (0 = all).
    #[arg(long, default_value_t = 0usize)]
    limit: usize,

    /// Optional per-eval-set row cap (0 = all).
    #[arg(long, default_value_t = 0usize)]
    eval_limit: usize,

    /// If set, also write JSON to this path.
    #[arg(long, default_value = "")]
    json: String,
}

#[derive(Serialize)]
struct PoolReport {
    path: String,
    lines: usize,
    avg_reading_chars: f64,
    avg_surface_chars: f64,
    kanji_surface_ratio: f64,
    lexical_overlap: usize,
    sixgram_overlap: usize,
    source_histogram: BTreeMap<String, usize>,
    source_licenses: Vec<String>,
    has_attribution_file: Option<bool>,
}

#[derive(Serialize)]
struct Report {
    pools: Vec<PoolReport>,
    eval_sets: Vec<String>,
}

fn read_jsonl(path: &Path, limit: usize) -> Result<Vec<Row>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        if limit > 0 && idx >= limit {
            break;
        }
        let line = line.with_context(|| format!("read line {}", idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Row>(trimmed) {
            Ok(r) => rows.push(r),
            Err(_) => continue,
        }
    }
    Ok(rows)
}

fn contains_kanji(text: &str) -> bool {
    text.chars()
        .any(|c| (c as u32) >= KANJI_START && (c as u32) <= KANJI_END)
}

fn build_eval_indices(
    paths: &[String],
    eval_limit: usize,
) -> Result<(std::collections::HashSet<(String, String, String)>, std::collections::HashSet<String>)>
{
    let mut lexical: std::collections::HashSet<(String, String, String)> =
        std::collections::HashSet::new();
    let mut sixgrams: std::collections::HashSet<String> = std::collections::HashSet::new();
    for path in paths {
        let rows = read_jsonl(Path::new(path), eval_limit)?;
        for row in rows {
            lexical.insert((
                row.reading.clone(),
                row.surface.clone(),
                row.context.clone(),
            ));
            for g in surface_ngrams(&row.surface, 6) {
                sixgrams.insert(g);
            }
        }
    }
    Ok((lexical, sixgrams))
}

fn audit_pool(
    path: &Path,
    eval_lexical: &std::collections::HashSet<(String, String, String)>,
    eval_sixgrams: &std::collections::HashSet<String>,
    limit: usize,
) -> Result<PoolReport> {
    let rows = read_jsonl(path, limit)?;
    if rows.is_empty() {
        return Ok(PoolReport {
            path: path.display().to_string(),
            lines: 0,
            avg_reading_chars: 0.0,
            avg_surface_chars: 0.0,
            kanji_surface_ratio: 0.0,
            lexical_overlap: 0,
            sixgram_overlap: 0,
            source_histogram: BTreeMap::new(),
            source_licenses: vec![],
            has_attribution_file: None,
        });
    }

    let mut reading_chars = 0usize;
    let mut surface_chars = 0usize;
    let mut kanji_count = 0usize;
    let mut lexical_overlap = 0usize;
    let mut sixgram_overlap = 0usize;
    let mut sources: BTreeMap<String, usize> = BTreeMap::new();

    for row in &rows {
        reading_chars += row.reading.chars().count();
        surface_chars += row.surface.chars().count();
        if contains_kanji(&row.surface) {
            kanji_count += 1;
        }
        let key = (
            row.reading.clone(),
            row.surface.clone(),
            row.context.clone(),
        );
        if eval_lexical.contains(&key) {
            lexical_overlap += 1;
        }
        let mut overlapped = false;
        for g in surface_ngrams(&row.surface, 6) {
            if eval_sixgrams.contains(&g) {
                overlapped = true;
                break;
            }
        }
        if overlapped {
            sixgram_overlap += 1;
        }
        let tag = row
            .source
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        *sources.entry(tag).or_insert(0) += 1;
    }

    let lines = rows.len();
    let lines_f = lines as f64;

    // Attribution co-location check.
    let mut attribution_candidates: Vec<PathBuf> = Vec::new();
    if let Some(parent) = path.parent() {
        attribution_candidates.push(parent.join("ATTRIBUTION.md"));
    }
    for tag in sources.keys() {
        if tag == "unknown" {
            continue;
        }
        attribution_candidates.push(PathBuf::from("datasets/src").join(tag).join("ATTRIBUTION.md"));
    }
    let has_attribution_file = attribution_candidates.iter().any(|p| p.exists());

    let mut source_licenses: Vec<String> = sources
        .keys()
        .map(|tag| source_license(tag).to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    source_licenses.sort();

    Ok(PoolReport {
        path: path.display().to_string(),
        lines,
        avg_reading_chars: round2(reading_chars as f64 / lines_f),
        avg_surface_chars: round2(surface_chars as f64 / lines_f),
        kanji_surface_ratio: round4(kanji_count as f64 / lines_f),
        lexical_overlap,
        sixgram_overlap,
        source_histogram: sources,
        source_licenses,
        has_attribution_file: Some(has_attribution_file),
    })
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (eval_lexical, eval_sixgrams) = build_eval_indices(&args.eval_sets, args.eval_limit)?;

    let mut report = Report {
        pools: Vec::new(),
        eval_sets: args.eval_sets.clone(),
    };
    for pool in &args.pools {
        let r = audit_pool(pool, &eval_lexical, &eval_sixgrams, args.limit)?;
        report.pools.push(r);
    }
    let serialized = serde_json::to_string_pretty(&report)?;
    println!("{}", serialized);
    if !args.json.is_empty() {
        std::fs::write(&args.json, &serialized)
            .with_context(|| format!("write {}", args.json))?;
    }
    Ok(())
}
