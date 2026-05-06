//! Build a homophone synth pool from mozc OSS dictionaries.
//!
//! Mozc dictionary format (one entry per line, tab-separated):
//!     <reading>\t<left_id>\t<right_id>\t<cost>\t<surface>
//!
//! For training the model to disambiguate same-reading-different-surface
//! pairs, we group entries by reading, keep readings that have ≥ 2
//! distinct surfaces, and emit one JSONL row per (reading, surface).
//! Output schema matches the existing `synth_homophone.jsonl` so it slots
//! into `data-mix`'s synth pool without further plumbing.
//!
//! Output row shape:
//!     {"reading":..., "surface":..., "left_context_surface":"",
//!      "left_context_reading":"", "span_bunsetsu":1,
//!      "source":"synth_mozc_homophone",
//!      "sentence_id":"mozc:<group_idx>#<surface_idx>"}
//!
//! Cost-weighted oversampling can be added later by using `data-mix`'s
//! ratio knobs; this crate keeps the per-pair output to one row.

use ahash::AHashMap;
use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Mozc OSS dictionary -> homophone synth JSONL")]
struct Args {
    /// Mozc dictionary files (typically dictionary00.txt..dictionary09.txt).
    #[arg(long, num_args = 1.., default_values_t = (0..10).map(|i| format!("datasets/external/mozc/dictionary{i:02}.txt")).collect::<Vec<_>>())]
    inputs: Vec<String>,

    /// Output JSONL path.
    #[arg(long, default_value = "datasets/corpus/synth/mozc_homophone.jsonl")]
    output: PathBuf,

    /// Minimum number of distinct surfaces a reading must have to be
    /// emitted (filter to true homophones).
    #[arg(long, default_value_t = 2)]
    min_surfaces: usize,

    /// Minimum reading length in chars (drop ultra-short single-mora
    /// entries that dominate the long tail without disambiguation value).
    #[arg(long, default_value_t = 2)]
    min_reading_chars: usize,

    /// Maximum reading length in chars (drop multi-bunsetsu compounds
    /// that are noise for single-bunsetsu disambiguation training).
    #[arg(long, default_value_t = 12)]
    max_reading_chars: usize,
}

fn is_kana(c: char) -> bool {
    matches!(c,
        '\u{3041}'..='\u{3096}' | // hira
        '\u{30A1}'..='\u{30FA}' | // kata
        '\u{30FC}'                // chōon ー
    )
}

#[derive(Serialize)]
struct Row<'a> {
    reading: &'a str,
    surface: &'a str,
    left_context_surface: &'a str,
    left_context_reading: &'a str,
    span_bunsetsu: u32,
    source: &'a str,
    sentence_id: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();

    // reading -> {surface -> min_cost}
    let mut groups: AHashMap<String, AHashMap<String, i32>> = AHashMap::new();
    let mut total_lines = 0usize;
    let mut malformed = 0usize;

    for path in &args.inputs {
        let fh = File::open(path).with_context(|| format!("open {path}"))?;
        let reader = BufReader::new(fh);
        for line in reader.lines() {
            let line = line.with_context(|| format!("read {path}"))?;
            total_lines += 1;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 5 {
                malformed += 1;
                continue;
            }
            let reading = parts[0];
            let cost: i32 = parts[3].parse().unwrap_or(i32::MAX);
            let surface = parts[4];
            let chars = reading.chars().count();
            if chars < args.min_reading_chars || chars > args.max_reading_chars {
                continue;
            }
            // Skip identity (reading == surface) — those are kana-only
            // entries with no disambiguation signal for the IME.
            if reading == surface {
                continue;
            }
            // Drop entries whose reading contains anything outside hira /
            // kata / chōon (mozc has some "0さいじ" style mixed entries
            // that are noise for our use case — readings should be pure
            // kana so the trained model sees a clean reading→surface
            // mapping).
            if !reading.chars().all(is_kana) {
                continue;
            }
            let entry = groups
                .entry(reading.to_string())
                .or_default()
                .entry(surface.to_string())
                .or_insert(i32::MAX);
            if cost < *entry {
                *entry = cost;
            }
        }
    }

    eprintln!(
        "[mozc-homophone] parsed lines={total_lines} malformed={malformed} \
         unique_readings={} (after length/identity filter)",
        groups.len()
    );

    let mut sorted_readings: Vec<&String> = groups.keys().collect();
    sorted_readings.sort();

    let out_path = &args.output;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let mut out = BufWriter::new(File::create(out_path).with_context(|| format!("create {}", out_path.display()))?);

    let mut emitted_groups = 0usize;
    let mut emitted_rows = 0usize;

    for (group_idx, reading) in sorted_readings.iter().enumerate() {
        let surfaces = &groups[*reading];
        if surfaces.len() < args.min_surfaces {
            continue;
        }
        emitted_groups += 1;
        let mut surface_list: Vec<(&String, &i32)> = surfaces.iter().collect();
        // Stable order: lower cost first.
        surface_list.sort_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)));
        for (surface_idx, (surface, _cost)) in surface_list.iter().enumerate() {
            let row = Row {
                reading,
                surface,
                left_context_surface: "",
                left_context_reading: "",
                span_bunsetsu: 1,
                source: "synth_mozc_homophone",
                sentence_id: format!("mozc:{group_idx}#{surface_idx}"),
            };
            serde_json::to_writer(&mut out, &row)?;
            out.write_all(b"\n")?;
            emitted_rows += 1;
        }
    }
    out.flush()?;

    let elapsed = started.elapsed().as_secs_f32();
    eprintln!(
        "[mozc-homophone] wrote {emitted_rows} rows over {emitted_groups} \
         readings -> {} ({:.1}s)",
        out_path.display(),
        elapsed
    );
    Ok(())
}
