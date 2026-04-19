//! Rust port of `datasets/tools/mix/build.py` (Python).
//!
//! v2 mix builder: multi-sentence-src + bunsetsu span filter (span=1 vs 2)
//! + multi-synth-src. Weighted-least-served allocation, buffer-shuffle output.
//!
//! The Python version of this crate lives in `old/build-train-mix-v2-py/build.py`
//! (moved there when this Rust port took over).
//!
//! Field layout expected in bunsetsu source JSONL:
//!   {"reading", "surface", "left_context_surface", "left_context_reading",
//!    "span_bunsetsu": 1 | 2, "source", "sentence_id"}
//!
//! Sentence and synth sources can have any field set — lines are copied
//! verbatim to the output (no re-encoding).

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Build v2 training mix (sentence + bunsetsu span=1/2 + synth).")]
struct Args {
    /// Output JSONL path.
    #[arg(long)]
    output: PathBuf,

    /// Target row count (oversampling to reach).
    #[arg(long, default_value_t = 20_000_000u64)]
    total: u64,

    /// Sentence-level JSONL path. Repeatable for multi-source.
    #[arg(long, action = clap::ArgAction::Append)]
    sentence_src: Vec<PathBuf>,

    /// Directory containing bunsetsu JSONL files.
    /// Every *.jsonl in the dir is read; lines are filtered by
    /// `span_bunsetsu` field for the span=1 / span=2 pools.
    #[arg(long)]
    bunsetsu_src: PathBuf,

    /// Synth pool JSONL path. Repeatable for multi-source.
    #[arg(long, action = clap::ArgAction::Append)]
    synth_src: Vec<PathBuf>,

    /// Pool ratios (unnormalized; builder normalizes so pools sum to 1.0).
    #[arg(long, default_value_t = 0.555)]
    ratio_sentence: f64,
    #[arg(long, default_value_t = 0.278)]
    ratio_bunsetsu2: f64,
    #[arg(long, default_value_t = 0.055)]
    ratio_bunsetsu1: f64,
    #[arg(long, default_value_t = 0.111)]
    ratio_synth: f64,

    /// Rows held in shuffle buffer before flush.
    #[arg(long, default_value_t = 100_000usize)]
    shuffle_buffer: usize,

    /// Report progress every N rows written.
    #[arg(long, default_value_t = 1_000_000u64)]
    report_every: u64,

    #[arg(long, default_value_t = 42u64)]
    seed: u64,
}

/// A pool whose next line is produced by round-robin over a list of files.
/// On EOF of the last file, the file list is reshuffled and reopened.
struct FilePool {
    name: &'static str,
    paths: Vec<PathBuf>,
    /// Optional substring filter on the JSONL line (used for bunsetsu
    /// span filter).
    filter: Option<&'static str>,
    current_reader: Option<BufReader<File>>,
    path_order: Vec<usize>,
    path_cursor: usize,
    rng_seed: u64,
    rng_counter: u64,
}

impl FilePool {
    fn new(
        name: &'static str,
        paths: Vec<PathBuf>,
        filter: Option<&'static str>,
        rng_seed: u64,
    ) -> Self {
        Self {
            name,
            paths,
            filter,
            current_reader: None,
            path_order: Vec::new(),
            path_cursor: 0,
            rng_seed,
            rng_counter: 0,
        }
    }

    fn reshuffle_and_open(&mut self) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(self.rng_seed.wrapping_add(self.rng_counter));
        self.rng_counter = self.rng_counter.wrapping_add(1);
        self.path_order = (0..self.paths.len()).collect();
        self.path_order.shuffle(&mut rng);
        self.path_cursor = 0;
        self.open_next()
    }

    fn open_next(&mut self) -> Result<()> {
        if self.paths.is_empty() {
            return Err(anyhow!("pool {} has no source files", self.name));
        }
        if self.path_cursor >= self.path_order.len() {
            self.reshuffle_and_open()?;
            return Ok(());
        }
        let path = &self.paths[self.path_order[self.path_cursor]];
        let file = File::open(path).with_context(|| format!("open {:?}", path))?;
        self.current_reader = Some(BufReader::new(file));
        self.path_cursor += 1;
        Ok(())
    }

    /// Read the next (filter-passing) line. Loops indefinitely across files.
    fn next_line(&mut self, scratch: &mut String) -> Result<()> {
        if self.current_reader.is_none() {
            self.reshuffle_and_open()?;
        }
        loop {
            scratch.clear();
            let reader = self.current_reader.as_mut().unwrap();
            let n = reader
                .read_line(scratch)
                .with_context(|| format!("read_line in pool {}", self.name))?;
            if n == 0 {
                // EOF on current file, advance.
                self.open_next()?;
                continue;
            }
            if let Some(needle) = self.filter {
                if !scratch.contains(needle) {
                    continue;
                }
            }
            if !scratch.ends_with('\n') {
                scratch.push('\n');
            }
            return Ok(());
        }
    }
}

fn collect_bunsetsu_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read_dir {:?}", dir))? {
        let entry = entry?;
        let p = entry.path();
        if p.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        // Exclude synth_*.jsonl (those go to the synth pool separately).
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if name.starts_with("synth_") {
                continue;
            }
        }
        out.push(p);
    }
    out.sort();
    Ok(out)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Build pool list, filtering out empty ones.
    let sentence_existing: Vec<PathBuf> = args
        .sentence_src
        .iter()
        .filter(|p| p.exists())
        .cloned()
        .collect();
    for p in &args.sentence_src {
        if !p.exists() {
            eprintln!("[warn] sentence src not found: {:?}", p);
        }
    }
    let bunsetsu_files = collect_bunsetsu_files(&args.bunsetsu_src)?;
    let synth_existing: Vec<PathBuf> = args
        .synth_src
        .iter()
        .filter(|p| p.exists())
        .cloned()
        .collect();
    for p in &args.synth_src {
        if !p.exists() {
            eprintln!("[warn] synth src not found: {:?}", p);
        }
    }

    let mut pools: Vec<(FilePool, f64)> = Vec::new();
    if !sentence_existing.is_empty() {
        println!(
            "  sentence pool = {} files:",
            sentence_existing.len()
        );
        for p in &sentence_existing {
            println!("    {}", p.display());
        }
        pools.push((
            FilePool::new("sentence", sentence_existing, None, args.seed ^ 0x1111),
            args.ratio_sentence,
        ));
    }
    if !bunsetsu_files.is_empty() {
        pools.push((
            FilePool::new(
                "bunsetsu2",
                bunsetsu_files.clone(),
                Some("\"span_bunsetsu\": 2"),
                args.seed ^ 0x2222,
            ),
            args.ratio_bunsetsu2,
        ));
        pools.push((
            FilePool::new(
                "bunsetsu1",
                bunsetsu_files,
                Some("\"span_bunsetsu\": 1"),
                args.seed ^ 0x3333,
            ),
            args.ratio_bunsetsu1,
        ));
    }
    if !synth_existing.is_empty() {
        println!(
            "  synth pool = {} files:",
            synth_existing.len()
        );
        for p in &synth_existing {
            println!("    {}", p.display());
        }
        pools.push((
            FilePool::new("synth", synth_existing, None, args.seed ^ 0x4444),
            args.ratio_synth,
        ));
    }

    if pools.is_empty() {
        return Err(anyhow!("no usable pools"));
    }

    // Normalize ratios, compute per-pool targets.
    let sum_ratio: f64 = pools.iter().map(|(_, r)| *r).sum();
    if sum_ratio <= 0.0 {
        return Err(anyhow!("sum of ratios is non-positive"));
    }
    let mut targets: Vec<u64> = pools
        .iter()
        .map(|(_, r)| (args.total as f64 * r / sum_ratio).round() as u64)
        .collect();
    let current_sum: u64 = targets.iter().sum();
    let diff = args.total as i64 - current_sum as i64;
    if diff != 0 && !targets.is_empty() {
        targets[0] = (targets[0] as i64 + diff).max(0) as u64;
    }
    println!("=== Pool mix (normalized) ===");
    for ((pool, _), tgt) in pools.iter().zip(targets.iter()) {
        let r = *tgt as f64 / args.total as f64;
        println!(
            "  {:<12} ratio={:.3}  target_rows={}",
            pool.name, r, tgt
        );
    }

    // Output setup.
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out_file =
        File::create(&args.output).with_context(|| format!("create {:?}", args.output))?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);

    // Buffered shuffled output.
    let mut buffer: Vec<String> = Vec::with_capacity(args.shuffle_buffer);
    let mut served: Vec<u64> = vec![0; pools.len()];
    let mut written: u64 = 0;
    let t0 = Instant::now();
    let mut scratch = String::with_capacity(1024);

    while written < args.total {
        // Weighted-least-served selection.
        let weights: Vec<f64> = (0..pools.len())
            .map(|i| (targets[i] as i64 - served[i] as i64).max(0) as f64)
            .collect();
        let total_w: f64 = weights.iter().sum();
        if total_w <= 0.0 {
            break;
        }
        let mut x = rng.gen::<f64>() * total_w;
        let mut idx = pools.len() - 1;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                idx = i;
                break;
            }
            x -= *w;
        }

        let (pool, _ratio) = &mut pools[idx];
        pool.next_line(&mut scratch)?;
        buffer.push(scratch.clone());
        served[idx] += 1;

        if buffer.len() >= args.shuffle_buffer {
            buffer.shuffle(&mut rng);
            for line in buffer.drain(..) {
                writer.write_all(line.as_bytes())?;
                written += 1;
            }
            if written / args.report_every
                > (written - args.shuffle_buffer as u64) / args.report_every
            {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = written as f64 / elapsed.max(1e-6);
                let eta = (args.total - written) as f64 / rate.max(1e-6);
                println!(
                    "  written={}/{} rate={:.1}k rows/s  eta={:.1}min",
                    written,
                    args.total,
                    rate / 1000.0,
                    eta / 60.0
                );
            }
        }
    }

    if !buffer.is_empty() {
        buffer.shuffle(&mut rng);
        for line in buffer.drain(..) {
            writer.write_all(line.as_bytes())?;
            written += 1;
        }
    }
    writer.flush()?;

    let elapsed = t0.elapsed().as_secs_f64();
    println!("\n=== Summary ===");
    println!(
        "  total written: {} rows -> {}",
        written,
        args.output.display()
    );
    println!(
        "  elapsed: {:.0}s  ({:.1}k rows/s)",
        elapsed,
        written as f64 / elapsed / 1000.0
    );
    let size = std::fs::metadata(&args.output)?.len() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("  file size: {:.2} GiB", size);
    println!("  served per pool:");
    for ((pool, _), (tgt, got)) in pools.iter().zip(targets.iter().zip(served.iter())) {
        let pct = *got as f64 / *tgt as f64 * 100.0;
        println!(
            "    {:<12} served={}  target={}  ({:.1}%)",
            pool.name, got, tgt, pct
        );
    }

    Ok(())
}
