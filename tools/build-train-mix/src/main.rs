//! Rust replacement for `scripts/build_phase3_train.py`.
//!
//! Identical output semantics, byte-compatible JSONL. Roughly 5-10x faster
//! than the Python version on the 200M-row Phase 3 mix thanks to native
//! JSON parse, AHash contamination set, and a tight weighted-least-served
//! loop.

use anyhow::{bail, Result};
use clap::Parser;
use datacore::{jsonl, NgramSet, OutputFormat, Row};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Build the Phase 3 CTC-NAT train.jsonl from pool-mix proportions.")]
struct Args {
    /// Output JSONL path. Add .zst/.xz/.gz to enable compression, or use
    /// --compress to explicitly append a suffix.
    #[arg(long)]
    output: PathBuf,

    /// Total target rows. Default 200M.
    #[arg(long, default_value_t = 200_000_000u64)]
    total: u64,

    /// Surface length cutoff separating super-short (<) from chunks_main (>=).
    #[arg(long, default_value_t = 8usize)]
    super_cutoff: usize,

    /// Reject rows whose surface is shorter than this (absolute floor).
    #[arg(long, default_value_t = 1usize)]
    min_surface_len: usize,

    /// Reject rows whose surface exceeds this.
    #[arg(long, default_value_t = 128usize)]
    max_surface_len: usize,

    /// (UNUSED) Reserved for a future shuffled writer. Current implementation
    /// emits pool rows in weighted-least-served order, so output is fully
    /// deterministic given fixed input files and ratios.
    #[arg(long, default_value_t = 42u64)]
    seed: u64,

    // Pool paths.
    #[arg(long, default_value = "datasets/chunks_v3_100m.jsonl")]
    chunks_path: PathBuf,
    #[arg(long, default_value = "datasets/zenz_llmjp_clean.jsonl")]
    zenz_path: PathBuf,
    #[arg(long, default_value = "datasets/wiki_clean_v3.jsonl")]
    wiki_path: PathBuf,
    #[arg(long, default_value = "datasets/aozora_clean.jsonl")]
    aozora_path: PathBuf,
    #[arg(long, default_value = "datasets/fineweb2_ja_clean.jsonl")]
    fineweb2_path: PathBuf,
    #[arg(long, default_value = "datasets/hplt3_ja_clean.jsonl")]
    hplt_path: PathBuf,

    // Ratios (must sum to 1.0).
    #[arg(long, default_value_t = 0.50)]
    ratio_chunks: f64,
    #[arg(long, default_value_t = 0.10)]
    ratio_super: f64,
    #[arg(long, default_value_t = 0.15)]
    ratio_zenz: f64,
    #[arg(long, default_value_t = 0.10)]
    ratio_wiki: f64,
    #[arg(long, default_value_t = 0.10)]
    ratio_fineweb2: f64,
    #[arg(long, default_value_t = 0.05)]
    ratio_hplt: f64,

    /// Evaluation JSONL(s) for 6-gram contamination filter on un-filtered pools
    /// (chunks and legacy wiki/aozora). Empty list disables contamination
    /// filtering.
    #[arg(long, num_args = 0.., default_values_t = ["datasets/eval/general/test.jsonl".to_string()])]
    contamination_ref: Vec<String>,

    /// N for contamination n-gram check.
    #[arg(long, default_value_t = 6usize)]
    contamination_n: usize,

    /// Emit a progress line every N rows written.
    #[arg(long, default_value_t = 5_000_000u64)]
    progress_every: u64,

    /// Explicit compression override. Defaults to auto-detect from --output suffix.
    #[arg(long, value_enum)]
    compress: Option<CompressMode>,

    /// Compression level (zstd 1-22, xz 0-9, gzip 1-9). Default 19.
    #[arg(long, default_value_t = 19)]
    compress_level: i32,
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum CompressMode {
    None,
    Zstd,
    Xz,
    Gzip,
}

impl CompressMode {
    fn to_output_format(self) -> OutputFormat {
        match self {
            CompressMode::None => OutputFormat::Raw,
            CompressMode::Zstd => OutputFormat::Zstd,
            CompressMode::Xz => OutputFormat::Xz,
            CompressMode::Gzip => OutputFormat::Gzip,
        }
    }
    fn suffix(self) -> Option<&'static str> {
        match self {
            CompressMode::None => None,
            CompressMode::Zstd => Some(".zst"),
            CompressMode::Xz => Some(".xz"),
            CompressMode::Gzip => Some(".gz"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LengthPredicate {
    AtLeast(usize),
    LessThan(usize),
    None,
}

impl LengthPredicate {
    fn passes(self, len: usize) -> bool {
        match self {
            LengthPredicate::AtLeast(v) => len >= v,
            LengthPredicate::LessThan(v) => len < v,
            LengthPredicate::None => true,
        }
    }
}

struct PoolSpec {
    name: &'static str,
    paths: Vec<PathBuf>,
    source_tag: &'static str,
    ratio: f64,
    length_pred: LengthPredicate,
    filter_contamination: bool,
    budget: u64,
    served: u64,
    exhausted: bool,
    cycler: Option<jsonl::CyclingJsonl>,
}

impl PoolSpec {
    fn share(&self) -> f64 {
        if self.budget == 0 {
            1.0
        } else {
            self.served as f64 / self.budget as f64
        }
    }
}

fn build_specs(args: &Args) -> Vec<PoolSpec> {
    let super_cutoff = args.super_cutoff;
    vec![
        PoolSpec {
            name: "chunks_main",
            paths: vec![args.chunks_path.clone()],
            source_tag: "chunks_main",
            ratio: args.ratio_chunks,
            length_pred: LengthPredicate::AtLeast(super_cutoff),
            filter_contamination: true,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
        PoolSpec {
            name: "super_short",
            paths: vec![args.chunks_path.clone()],
            source_tag: "chunks_super",
            ratio: args.ratio_super,
            length_pred: LengthPredicate::LessThan(super_cutoff),
            filter_contamination: true,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
        PoolSpec {
            name: "zenz",
            paths: vec![args.zenz_path.clone()],
            source_tag: "zenz_llmjp",
            ratio: args.ratio_zenz,
            length_pred: LengthPredicate::None,
            filter_contamination: false,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
        PoolSpec {
            name: "wiki_aozora",
            paths: vec![args.wiki_path.clone(), args.aozora_path.clone()],
            source_tag: "wiki_aozora",
            ratio: args.ratio_wiki,
            length_pred: LengthPredicate::None,
            filter_contamination: true,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
        PoolSpec {
            name: "fineweb2",
            paths: vec![args.fineweb2_path.clone()],
            source_tag: "fineweb2_ja",
            ratio: args.ratio_fineweb2,
            length_pred: LengthPredicate::None,
            filter_contamination: false,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
        PoolSpec {
            name: "hplt",
            paths: vec![args.hplt_path.clone()],
            source_tag: "hplt3_ja",
            ratio: args.ratio_hplt,
            length_pred: LengthPredicate::None,
            filter_contamination: false,
            budget: 0,
            served: 0,
            exhausted: false,
            cycler: None,
        },
    ]
}

/// Allocate `total` integer units across pools proportionally to `ratios`,
/// guaranteeing `sum(allocated) == total`. Uses the largest-remainder method
/// (aka Hamilton apportionment).
fn allocate_budgets(total: u64, ratios: &[f64]) -> Vec<u64> {
    if ratios.is_empty() {
        return vec![];
    }
    let total_f = total as f64;
    let exact: Vec<f64> = ratios.iter().map(|r| r * total_f).collect();
    let mut floors: Vec<u64> = exact.iter().map(|v| v.floor() as u64).collect();
    let allocated: u64 = floors.iter().sum();
    let remainder = total.saturating_sub(allocated);

    // Order indices by descending fractional residual; ties broken by ratio order.
    let mut order: Vec<usize> = (0..ratios.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for idx in order.into_iter().take(remainder as usize) {
        floors[idx] += 1;
    }
    floors
}

#[cfg(test)]
mod budget_tests {
    use super::allocate_budgets;

    #[test]
    fn sum_always_equals_total() {
        // The Phase 3 default mix + a mid-sized total that round-each would
        // under-fill by a few rows.
        let ratios = [0.50, 0.10, 0.15, 0.10, 0.10, 0.05];
        for total in [1_001u64, 1_000_001, 200_000_000, 199_999_997] {
            let b = allocate_budgets(total, &ratios);
            assert_eq!(b.iter().sum::<u64>(), total, "sum mismatch for total={}", total);
            // Each budget is close to the exact proportion (within 1).
            for (bi, r) in b.iter().zip(ratios.iter()) {
                let exact = (total as f64) * r;
                let diff = (*bi as f64 - exact).abs();
                assert!(diff <= 1.0, "budget {} vs exact {}", bi, exact);
            }
        }
    }

    #[test]
    fn handles_zero_ratio() {
        let b = allocate_budgets(100, &[0.7, 0.3, 0.0]);
        assert_eq!(b.iter().sum::<u64>(), 100);
        assert_eq!(b[2], 0);
    }

    #[test]
    fn empty_ratios() {
        let b = allocate_budgets(100, &[]);
        assert!(b.is_empty());
    }
}

fn pick_pool_to_serve(specs: &[PoolSpec]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, s) in specs.iter().enumerate() {
        if s.exhausted || s.served >= s.budget {
            continue;
        }
        let share = s.share();
        match best {
            None => best = Some((i, share)),
            Some((_, cur)) if share < cur => best = Some((i, share)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

fn next_accepted_row(
    spec: &mut PoolSpec,
    contamination: &NgramSet,
    contam_n: usize,
    max_surface_len: usize,
    min_surface_len: usize,
) -> Result<Option<Row>> {
    // Each call may need to chew through many rejected rows before finding
    // one that passes the filter, so loop until accepted or exhausted.
    loop {
        if spec.cycler.is_none() {
            spec.cycler = Some(jsonl::CyclingJsonl::new(spec.paths.clone()));
        }
        let cycler = spec.cycler.as_mut().unwrap();
        let row = match cycler.next_row()? {
            Some(r) => r,
            None => {
                spec.exhausted = true;
                return Ok(None);
            }
        };
        if row.reading.is_empty() || row.surface.is_empty() {
            continue;
        }
        let surf_len = row.surface.chars().count();
        if surf_len < min_surface_len || surf_len > max_surface_len {
            continue;
        }
        if !spec.length_pred.passes(surf_len) {
            continue;
        }
        if spec.filter_contamination
            && !contamination.is_empty()
            && contamination.contains_overlap(&row.surface)
        {
            let _ = contam_n; // contamination_n is carried on the set
            continue;
        }
        return Ok(Some(row));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let _ = args.seed; // explicitly unused; see field doc.

    let ratio_sum = args.ratio_chunks
        + args.ratio_super
        + args.ratio_zenz
        + args.ratio_wiki
        + args.ratio_fineweb2
        + args.ratio_hplt;
    if (ratio_sum - 1.0).abs() > 1e-6 {
        bail!("ratios must sum to 1.0, got {:.6}", ratio_sum);
    }

    // Normalise output path + compression choice so we behave like the Python
    // script's auto-suffix logic.
    let explicit_compress = args.compress;
    let mut output_path = args.output.clone();
    if let Some(mode) = explicit_compress {
        if let Some(suffix) = mode.suffix() {
            let as_str = output_path.to_string_lossy().to_string();
            if !as_str.ends_with(suffix) {
                output_path = PathBuf::from(format!("{}{}", as_str, suffix));
            }
        }
    }
    let output_format = explicit_compress
        .map(|m| m.to_output_format())
        .unwrap_or_else(|| OutputFormat::from_path(&output_path));

    // Build combined contamination n-gram set across all refs.
    let contamination_paths: Vec<PathBuf> = args
        .contamination_ref
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
    let mut contamination = NgramSet::new(args.contamination_n);
    for p in &contamination_paths {
        eprintln!("Loading contamination reference: {}", p.display());
        contamination.extend_from_jsonl(p)?;
    }
    eprintln!("  combined n-grams: {}", contamination.len());

    // Set up pool specs and allocate integer budgets via largest-remainder so
    // that sum(budgets) == --total exactly (the naive round-each approach can
    // leave the emission loop short of the target by up to N-1 rows).
    let mut specs = build_specs(&args);
    let budgets = allocate_budgets(
        args.total,
        &specs.iter().map(|s| s.ratio).collect::<Vec<_>>(),
    );
    for (spec, budget) in specs.iter_mut().zip(budgets.iter()) {
        spec.budget = *budget;
    }
    eprintln!(
        "Target total: {} rows → {}",
        args.total,
        output_path.display()
    );
    for spec in specs.iter() {
        eprintln!(
            "  {:<12} ratio={:.2} budget={:>12} src={} filter_contam={} paths={}",
            spec.name,
            spec.ratio,
            spec.budget,
            spec.source_tag,
            spec.filter_contamination,
            spec.paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    let budget_total: u64 = specs.iter().map(|s| s.budget).sum();
    assert_eq!(budget_total, args.total, "allocate_budgets must sum to total");

    // Main emission loop. `picks` tracks written rows; each pool's `served`
    // increments independently so the fraction stays accurate.
    let mut writer = jsonl::open_output(&output_path, Some(output_format), args.compress_level)?;
    let mut written: u64 = 0;
    let progress_every = args.progress_every.max(1);
    let next_progress = progress_every;
    let mut next_progress = next_progress;

    loop {
        let Some(idx) = pick_pool_to_serve(&specs) else {
            break;
        };
        let (contamination_ref, n_for_log) = (&contamination, args.contamination_n);
        let spec = &mut specs[idx];
        let row = match next_accepted_row(
            spec,
            contamination_ref,
            n_for_log,
            args.max_surface_len,
            args.min_surface_len,
        )? {
            Some(r) => r,
            None => {
                eprintln!(
                    "  [{:>12}] pool {} exhausted at {}/{} — continuing without it",
                    written, spec.name, spec.served, spec.budget
                );
                continue;
            }
        };
        let out_row = Row::new(
            row.reading,
            row.surface,
            row.context,
            Some(spec.source_tag.to_string()),
        );
        jsonl::write_row(&mut writer, &out_row)?;
        spec.served += 1;
        written += 1;
        if written >= next_progress {
            let shares: Vec<String> = specs
                .iter()
                .map(|s| format!("{}={:.2}", s.name, s.share().min(9.99)))
                .collect();
            eprintln!(
                "  [{:>12}] served_fraction: {}",
                written,
                shares.join(" ")
            );
            next_progress += progress_every;
        }
    }

    drop(writer); // flush + finalise compressor
    eprintln!("\ndone: {} rows → {}", written, output_path.display());
    for spec in &specs {
        eprintln!(
            "  {:<12} served={}  budget={}  {}",
            spec.name,
            spec.served,
            spec.budget,
            if spec.exhausted { "EXHAUSTED" } else { "OK" }
        );
    }
    let size = std::fs::metadata(&output_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("  size: {:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0);
    Ok(())
}
