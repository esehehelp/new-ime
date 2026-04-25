//! data-synth-homophone: mine homophone pairs from bunsetsu-level corpora.
//!
//! Strategy (no external dictionary required):
//!
//!   pass 1: stream every input file, count how many times each
//!           (reading → surface) pair occurs.
//!
//!   filter: a reading is a homophone candidate iff ≥ `min_surfaces`
//!           distinct surfaces each reach ≥ `min_occurrences` count.
//!           This drops typos, furigana noise, and long sentences whose
//!           reading happens to coincide with something common.
//!
//!   pass 2: stream the inputs again; for any row whose reading is a
//!           homophone candidate, emit it unchanged except the `source`
//!           field, which is rewritten to `synth_homophone`. The bunsetsu
//!           `left_context_*` fields carry the disambiguation signal we
//!           want the student to learn on.
//!
//! Output is byte-compatible with `datasets/corpus/bunsetsu/*.jsonl` so the
//! mix tool ingests it with no schema changes.

use ahash::AHashMap;
use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "data-synth-homophone",
    about = "Mine homophone (reading → multiple surfaces) rows out of bunsetsu corpora"
)]
struct Cli {
    /// Bunsetsu-schema input JSONL files (repeat for each corpus file).
    #[arg(long = "input", required = true)]
    inputs: Vec<PathBuf>,
    /// Output JSONL (schema matches bunsetsu corpora).
    #[arg(long)]
    output: PathBuf,
    /// Minimum distinct surfaces a reading must have to be treated as a
    /// homophone. 2 is the natural floor.
    #[arg(long, default_value_t = 2)]
    min_surfaces: u32,
    /// Minimum occurrences per (reading, surface) pair to count as a real
    /// surface (filters typo noise and ultra-rare misparse entries).
    #[arg(long, default_value_t = 5)]
    min_occurrences: u32,
    /// Global cap on emitted rows. 0 = unlimited.
    #[arg(long, default_value_t = 5_000_000)]
    max_output: usize,
    /// Per-(reading, surface) emission cap. Prevents the most common
    /// homophones from dominating the output. 0 = unlimited.
    #[arg(long, default_value_t = 64)]
    per_pair_cap: u32,
}

#[derive(Deserialize, Serialize, Clone)]
struct BunsetsuRow {
    reading: String,
    surface: String,
    #[serde(default)]
    left_context_surface: String,
    #[serde(default)]
    left_context_reading: String,
    #[serde(default)]
    span_bunsetsu: u32,
    // Rewritten on output to "synth_homophone".
    source: String,
    #[serde(default)]
    sentence_id: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.inputs.is_empty() {
        bail!("need at least one --input");
    }

    eprintln!(
        "[pass1] counting (reading,surface) pairs across {} input(s)",
        cli.inputs.len()
    );
    let mut counts: AHashMap<String, AHashMap<String, u32>> = AHashMap::with_capacity(1 << 20);
    let mut pass1_rows: u64 = 0;
    for path in &cli.inputs {
        let reader = open_reader(path)?;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }
            let row: BunsetsuRow = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            counts
                .entry(row.reading)
                .or_default()
                .entry(row.surface)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
            pass1_rows += 1;
            if pass1_rows % 5_000_000 == 0 {
                eprintln!(
                    "  [pass1] {} rows, {} unique readings",
                    pass1_rows,
                    counts.len()
                );
            }
        }
    }
    eprintln!(
        "[pass1] done. rows={}  unique_readings={}",
        pass1_rows,
        counts.len()
    );

    // Filter: keep readings that have ≥ min_surfaces surfaces each at
    // ≥ min_occurrences count. Store the accepted surface set so pass 2
    // can skip readings whose only-common surface is the one seen.
    let mut homophone_lex: AHashMap<String, AHashMap<String, ()>> =
        AHashMap::with_capacity(1 << 17);
    let mut total_kept_pairs: u64 = 0;
    for (reading, surfaces) in counts.drain() {
        let qualifying: Vec<_> = surfaces
            .iter()
            .filter(|(_, c)| **c >= cli.min_occurrences)
            .collect();
        if (qualifying.len() as u32) < cli.min_surfaces {
            continue;
        }
        let mut set = AHashMap::with_capacity(qualifying.len());
        for (s, _) in qualifying {
            set.insert(s.clone(), ());
        }
        total_kept_pairs += set.len() as u64;
        homophone_lex.insert(reading, set);
    }
    eprintln!(
        "[filter] homophone readings={} kept_pairs={}",
        homophone_lex.len(),
        total_kept_pairs
    );
    if homophone_lex.is_empty() {
        bail!(
            "no homophone readings found (min_surfaces={}, min_occurrences={}). Check that inputs are bunsetsu-level.",
            cli.min_surfaces,
            cli.min_occurrences
        );
    }

    // Pass 2: stream inputs again, emit matching rows.
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
    }
    let mut writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(&cli.output).with_context(|| format!("create {}", cli.output.display()))?,
    );
    let mut pair_counts: AHashMap<(String, String), u32> = AHashMap::with_capacity(1 << 18);
    let mut emitted: usize = 0;
    'outer: for path in &cli.inputs {
        let reader = open_reader(path)?;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.is_empty() {
                continue;
            }
            let mut row: BunsetsuRow = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let surfaces = match homophone_lex.get(&row.reading) {
                Some(s) => s,
                None => continue,
            };
            if !surfaces.contains_key(&row.surface) {
                continue;
            }
            if cli.per_pair_cap > 0 {
                let key = (row.reading.clone(), row.surface.clone());
                let count = pair_counts.entry(key).or_insert(0);
                if *count >= cli.per_pair_cap {
                    continue;
                }
                *count += 1;
            }
            row.source = "synth_homophone".to_string();
            // Write using the python-compat formatter so output matches the
            // byte layout of the other synth files (", " key separator).
            let line = serialize_python_compat(&row)?;
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
            emitted += 1;
            if cli.max_output > 0 && emitted >= cli.max_output {
                eprintln!("[pass2] max_output={} reached", cli.max_output);
                break 'outer;
            }
            if emitted % 1_000_000 == 0 {
                eprintln!("  [pass2] emitted {} rows", emitted);
            }
        }
    }
    writer.flush()?;
    eprintln!("[pass2] wrote {} rows to {}", emitted, cli.output.display());
    Ok(())
}

fn open_reader(path: &PathBuf) -> Result<BufReader<File>> {
    Ok(BufReader::with_capacity(
        8 * 1024 * 1024,
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    ))
}

/// serde_json with `", "` + `": "` spacing to match `json.dumps`.
fn serialize_python_compat<T: Serialize>(value: &T) -> Result<String> {
    struct PyFmt;
    impl serde_json::ser::Formatter for PyFmt {
        fn begin_object_key<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
        ) -> std::io::Result<()> {
            w.write_all(b": ")
        }
        fn begin_array_value<W: ?Sized + std::io::Write>(
            &mut self,
            w: &mut W,
            first: bool,
        ) -> std::io::Result<()> {
            if first {
                Ok(())
            } else {
                w.write_all(b", ")
            }
        }
    }
    let mut buf = Vec::with_capacity(256);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFmt);
    value.serialize(&mut ser).context("serialize row")?;
    Ok(String::from_utf8(buf).context("utf-8")?)
}
