//! data-pool-sampler: reservoir-sample N lines per round from a JSONL pool.
//!
//! Purpose: produce a per-pool audit directory for iterative rule-based
//! postprocess QA. For each round R in 1..=rounds, the tool seeds an RNG
//! with `seed + R`, streams the (optionally-compressed) input once, and
//! writes a reservoir sample of `samples` rows to
//! `<out-dir>/<pool-name>/round_<N>/samples.jsonl` with an index sidecar.
//!
//! Deterministic: same (seed, rounds, samples, input) → same sampled rows.
//!
//! Usage:
//!   data-pool-sampler \
//!     --input datasets/corpus/bunsetsu/wikibooks.jsonl \
//!     --pool-name wikibooks \
//!     --out-dir datasets/audits/pool-qa \
//!     --rounds 10 --samples 100 --seed 42

use anyhow::{Context, Result};
use clap::Parser;
use data_core::jsonl::OutputFormat;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "data-pool-sampler",
    about = "Reservoir-sample N rounds × K lines from a JSONL pool for iterative QA"
)]
struct Cli {
    /// Input JSONL (.zst / .xz / .gz accepted).
    #[arg(long)]
    input: PathBuf,
    /// Short pool name used in the audit directory layout.
    #[arg(long)]
    pool_name: String,
    /// Parent audit directory (one subdir per pool).
    #[arg(long, default_value = "datasets/audits/pool-qa")]
    out_dir: PathBuf,
    /// Total rounds to sample.
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    /// Samples per round.
    #[arg(long, default_value_t = 100)]
    samples: usize,
    /// Base PRNG seed. Round R uses seed + R.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Overwrite existing round outputs.
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let pool_dir = cli.out_dir.join(&cli.pool_name);
    fs::create_dir_all(&pool_dir)
        .with_context(|| format!("mkdir {}", pool_dir.display()))?;

    // Pre-check: if round_1 exists and --force isn't set, bail early.
    let round1 = pool_dir.join("round_1").join("samples.jsonl");
    if round1.exists() && !cli.force {
        anyhow::bail!(
            "{} already exists (use --force to overwrite)",
            round1.display()
        );
    }

    eprintln!(
        "[sampler] pool={} input={} rounds={} samples={}",
        cli.pool_name,
        cli.input.display(),
        cli.rounds,
        cli.samples,
    );

    // Allocate reservoirs (one per round) and a shared RNG per round so the
    // same streaming pass feeds all rounds simultaneously. We seed round R
    // deterministically with `seed + R` and run per-round Algorithm R.
    let mut reservoirs: Vec<Vec<(u64, String)>> = (0..cli.rounds)
        .map(|_| Vec::with_capacity(cli.samples))
        .collect();
    let mut rngs: Vec<StdRng> = (0..cli.rounds)
        .map(|r| StdRng::seed_from_u64(cli.seed.wrapping_add(r as u64 + 1)))
        .collect();

    let mut total_read: u64 = 0;
    let reader = open_text_reader(&cli.input)?;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("read line {}", line_no))?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        total_read += 1;
        for r in 0..cli.rounds {
            let reservoir = &mut reservoirs[r];
            if reservoir.len() < cli.samples {
                reservoir.push((total_read, line.clone()));
            } else {
                // Algorithm R: replace at random index in [0, total_read) with prob k/i.
                // line_no 0-indexed; use total_read as 1-indexed position.
                let j = rngs[r].gen_range(0..total_read);
                if (j as usize) < cli.samples {
                    reservoir[j as usize] = (total_read, line.clone());
                }
            }
        }
        if total_read % 1_000_000 == 0 {
            eprintln!("  [sampler] read {}", total_read);
        }
    }
    eprintln!("[sampler] total_read={}", total_read);

    for r in 0..cli.rounds {
        let round_dir = pool_dir.join(format!("round_{}", r + 1));
        fs::create_dir_all(&round_dir)
            .with_context(|| format!("mkdir {}", round_dir.display()))?;
        let samples_path = round_dir.join("samples.jsonl");
        let index_path = round_dir.join("index.tsv");
        let mut w = BufWriter::new(
            File::create(&samples_path)
                .with_context(|| format!("create {}", samples_path.display()))?,
        );
        let mut idx_w = BufWriter::new(
            File::create(&index_path)
                .with_context(|| format!("create {}", index_path.display()))?,
        );
        writeln!(idx_w, "rank\tline_1indexed")?;
        let mut reservoir = std::mem::take(&mut reservoirs[r]);
        // Sort by line order for deterministic output (stable wrt source).
        reservoir.sort_by_key(|(pos, _)| *pos);
        for (rank, (pos, line)) in reservoir.iter().enumerate() {
            writeln!(idx_w, "{}\t{}", rank + 1, pos)?;
            w.write_all(line.as_bytes())?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        idx_w.flush()?;
        eprintln!(
            "  [sampler] round_{}: {} samples -> {}",
            r + 1,
            reservoir.len(),
            samples_path.display()
        );
    }

    // Manifest for the pool (top-level, overwritten every run).
    let manifest_path = pool_dir.join("manifest.tsv");
    let mut m = BufWriter::new(
        File::create(&manifest_path)
            .with_context(|| format!("create {}", manifest_path.display()))?,
    );
    writeln!(m, "key\tvalue")?;
    writeln!(m, "input\t{}", cli.input.display())?;
    writeln!(m, "pool_name\t{}", cli.pool_name)?;
    writeln!(m, "total_read\t{}", total_read)?;
    writeln!(m, "rounds\t{}", cli.rounds)?;
    writeln!(m, "samples_per_round\t{}", cli.samples)?;
    writeln!(m, "base_seed\t{}", cli.seed)?;
    m.flush()?;

    eprintln!("[sampler] done  manifest={}", manifest_path.display());
    Ok(())
}

fn open_text_reader(path: &Path) -> Result<BufReader<Box<dyn Read + Send>>> {
    let file = File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let inner: Box<dyn Read + Send> = match OutputFormat::from_path(path) {
        OutputFormat::Raw => Box::new(file),
        OutputFormat::Zstd => Box::new(
            zstd::stream::Decoder::new(file).context("init zstd decoder")?,
        ),
        OutputFormat::Xz => Box::new(xz2::read::XzDecoder::new(file)),
        OutputFormat::Gzip => Box::new(flate2::read::GzDecoder::new(file)),
    };
    Ok(BufReader::with_capacity(8 * 1024 * 1024, inner))
}
