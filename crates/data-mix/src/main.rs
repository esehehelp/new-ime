//! v3 mix builder: chunks / zenz / wiki+aozora / bunsetsu / fineweb2 / hplt.
//!
//! Weighted-least-served allocation with buffer-shuffle output. 6-gram
//! contamination filter supports both JSONL refs (via data_core::NgramSet)
//! and probe_v3 / AJIMEE JSON-array refs.
//!
//! v3 default ratios (user-specified 2026-04-20):
//!   chunks    0.10
//!   zenz      0.15
//!   wiki      0.15
//!   bunsetsu  0.30
//!   fineweb2  0.20
//!   hplt      0.10
//!
//! Contamination defaults include probe_v3 + AJIMEE + general/{dev,test}.jsonl.

use ahash::AHashMap;
use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use data_core::NgramSet;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    about = "v3 training mix builder (chunks/zenz/wiki/bunsetsu/fineweb/hplt + contamination)."
)]
struct Args {
    /// Output JSONL path.
    #[arg(long)]
    output: PathBuf,

    /// Target row count (oversampling to reach).
    #[arg(long, default_value_t = 200_000_000u64)]
    total: u64,

    // Pool source paths.
    #[arg(long, default_value = "datasets/corpus/legacy/chunks_100m.jsonl")]
    chunks_path: PathBuf,
    #[arg(long, default_value = "datasets/corpus/legacy/zenz_llmjp.jsonl")]
    zenz_path: PathBuf,
    #[arg(long, default_value = "datasets/corpus/legacy/wiki.jsonl")]
    wiki_path: PathBuf,
    #[arg(long, default_value = "datasets/corpus/legacy/aozora.jsonl")]
    aozora_path: PathBuf,
    #[arg(long, default_value = "datasets/corpus/legacy/fineweb2_ja.jsonl")]
    fineweb2_path: PathBuf,
    #[arg(long, default_value = "datasets/corpus/legacy/hplt3_ja.jsonl")]
    hplt_path: PathBuf,
    /// Bunsetsu pool: multiple .jsonl files (wikibooks / wiktionary / wikinews /
    /// aozora_dialogue / tatoeba).
    #[arg(long, num_args = 0.., default_values_t = [
        "datasets/corpus/bunsetsu/wikibooks.jsonl".to_string(),
        "datasets/corpus/bunsetsu/wiktionary.jsonl".to_string(),
        "datasets/corpus/bunsetsu/wikinews.jsonl".to_string(),
        "datasets/corpus/bunsetsu/aozora_dialogue.jsonl".to_string(),
        "datasets/corpus/bunsetsu/tatoeba.jsonl".to_string(),
    ])]
    bunsetsu_paths: Vec<String>,

    /// Synth pool: python-generated synth (numeric_ext) + new Rust-generated
    /// synth (homophone, numeric_units, name). All bunsetsu-schema. Pure
    /// synthesis, no contamination filter needed.
    #[arg(long, num_args = 0.., default_values_t = [
        "datasets/corpus/synth/numeric.jsonl".to_string(),
        "datasets/corpus/synth/numeric_ext.jsonl".to_string(),
        "datasets/corpus/synth/homophone.jsonl".to_string(),
        "datasets/corpus/synth/numeric_units.jsonl".to_string(),
        "datasets/corpus/synth/name.jsonl".to_string(),
    ])]
    synth_paths: Vec<String>,

    // Ratios (must sum to 1.0). v3.1 defaults (synth added, bunsetsu
    // reduced to make room — synth now carries the short-phrase coverage
    // previously provided by bunsetsu span-1 rows).
    #[arg(long, default_value_t = 0.10)]
    ratio_chunks: f64,
    #[arg(long, default_value_t = 0.15)]
    ratio_zenz: f64,
    #[arg(long, default_value_t = 0.15)]
    ratio_wiki: f64,
    #[arg(long, default_value_t = 0.20)]
    ratio_bunsetsu: f64,
    #[arg(long, default_value_t = 0.20)]
    ratio_fineweb2: f64,
    #[arg(long, default_value_t = 0.10)]
    ratio_hplt: f64,
    /// Proportion of the output that comes from the synth pool (numeric,
    /// homophone, unit, name). Replaces what the Python 20M mix used to
    /// allocate to synth (~11%). 0.0 disables the pool entirely.
    #[arg(long, default_value_t = 0.10)]
    ratio_synth: f64,

    /// Surface length floor (chars). Lines below are skipped. v2.x default
    /// bumped 1→5 to drop ultra-short lexical fragments (bunsetsu pools
    /// contributed lots of 1-3 char entries that dominate the long tail
    /// and hurt homophone recall).
    #[arg(long, default_value_t = 5usize)]
    min_surface_len: usize,
    /// Surface length ceiling (chars). Lines above are skipped.
    #[arg(long, default_value_t = 128usize)]
    max_surface_len: usize,
    /// Reading length floor (chars). Filters near-empty readings that slip
    /// past the surface-len check (e.g. hiragana-only fragments where
    /// `reading == surface` and both are 1-2 chars).
    #[arg(long, default_value_t = 2usize)]
    min_reading_len: usize,
    /// Global default dedup cap. A (reading, surface) pair may appear at
    /// most this many times within a single pool's output. 0 disables. The
    /// default cap=3 lets genuinely-common collocations through but kills
    /// wiktionary/wikibooks-style lexical repetition. Individual pools can
    /// override this via `--dedup-cap-<pool>`.
    #[arg(long, default_value_t = 3u32)]
    dedup_cap: u32,
    /// Override dedup cap for the synth pool. Synth is intentionally
    /// template-based repetition, so the default 3 would drop 99% of it.
    /// 0 disables dedup for synth entirely.
    #[arg(long, default_value_t = 0u32)]
    dedup_cap_synth: u32,
    /// Override dedup cap for the bunsetsu pool. The bunsetsu sources
    /// (wikibooks/wiktionary/wikinews/aozora_dialogue/tatoeba) share heavy
    /// phrase repetition, so the global cap=3 exhausts the pool early. When
    /// unset, falls back to `--dedup-cap`. 0 disables dedup for bunsetsu.
    #[arg(long)]
    dedup_cap_bunsetsu: Option<u32>,

    /// 6-gram contamination JSONL refs (rows with a `surface` field).
    #[arg(long, num_args = 0.., default_values_t = [
        "datasets/eval/general/test.jsonl".to_string(),
        "datasets/eval/general/dev.jsonl".to_string(),
    ])]
    contamination_ref: Vec<String>,

    /// 6-gram contamination JSON-array refs (probe_v3 / AJIMEE).
    /// Extracts `expected_output` list + `original_text` per item.
    #[arg(long, num_args = 0.., default_values_t = [
        "datasets/eval/probe/probe.json".to_string(),
        "references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json".to_string(),
    ])]
    contamination_probe_ref: Vec<String>,

    /// Length of contamination n-grams.
    #[arg(long, default_value_t = 6usize)]
    contamination_n: usize,

    /// Pools whose contamination_filter flag is true will reject lines whose
    /// surface shares any n-gram with the contamination set.
    /// zenz / fineweb2 / hplt are assumed pre-filtered; chunks / wiki+aozora /
    /// bunsetsu are filtered on the fly.
    #[arg(long, default_value_t = true)]
    filter_chunks: bool,
    #[arg(long, default_value_t = false)]
    filter_zenz: bool,
    #[arg(long, default_value_t = true)]
    filter_wiki: bool,
    #[arg(long, default_value_t = true)]
    filter_bunsetsu: bool,
    #[arg(long, default_value_t = false)]
    filter_fineweb2: bool,
    #[arg(long, default_value_t = false)]
    filter_hplt: bool,
    /// Pure synth is never contaminated with eval sets by construction.
    #[arg(long, default_value_t = false)]
    filter_synth: bool,

    /// Rows held in shuffle buffer before flush.
    #[arg(long, default_value_t = 100_000usize)]
    shuffle_buffer: usize,

    /// Report progress every N rows written.
    #[arg(long, default_value_t = 1_000_000u64)]
    report_every: u64,

    #[arg(long, default_value_t = 42u64)]
    seed: u64,
}

#[derive(Deserialize)]
struct ProbeItem {
    expected_output: Vec<String>,
    #[serde(default)]
    original_text: String,
}

fn extend_contamination_from_probe_json(set: &mut NgramSet, path: &Path) -> Result<usize> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let items: Vec<ProbeItem> =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let mut count = 0usize;
    for it in &items {
        for s in &it.expected_output {
            if !s.is_empty() {
                set.insert_surface(s);
                count += 1;
            }
        }
        if !it.original_text.is_empty() {
            set.insert_surface(&it.original_text);
            count += 1;
        }
    }
    Ok(count)
}

/// Minimal row view: we only need `surface` / `reading` for filtering +
/// length checks. The raw JSONL line is emitted verbatim (no re-encoding).
#[derive(Deserialize)]
struct RowSurface {
    #[serde(default)]
    surface: String,
    #[serde(default)]
    reading: String,
}

struct FilePool {
    name: &'static str,
    paths: Vec<PathBuf>,
    source_tag: &'static str,
    filter_contamination: bool,
    /// Pool-specific dedup cap override. When set, takes precedence over the
    /// CLI's global `--dedup-cap` for this pool only.
    dedup_cap_override: Option<u32>,
    current_reader: Option<BufReader<File>>,
    path_order: Vec<usize>,
    path_cursor: usize,
    rng_seed: u64,
    rng_counter: u64,
    skipped_contam: u64,
    skipped_length: u64,
    skipped_parse: u64,
    skipped_dedup: u64,
    /// (reading, surface) → emit count, capped by `dedup_cap` in the
    /// calling CLI. Uses AHashMap for fast hashing on small keys.
    seen: AHashMap<u64, u32>,
}

impl FilePool {
    fn new(
        name: &'static str,
        paths: Vec<PathBuf>,
        source_tag: &'static str,
        filter_contamination: bool,
        rng_seed: u64,
    ) -> Self {
        Self {
            name,
            paths,
            source_tag,
            filter_contamination,
            dedup_cap_override: None,
            current_reader: None,
            path_order: Vec::new(),
            path_cursor: 0,
            rng_seed,
            rng_counter: 0,
            skipped_contam: 0,
            skipped_length: 0,
            skipped_parse: 0,
            skipped_dedup: 0,
            seen: AHashMap::with_capacity(1 << 20),
        }
    }

    fn with_dedup_override(mut self, cap: u32) -> Self {
        self.dedup_cap_override = Some(cap);
        self
    }

    fn with_optional_dedup_override(mut self, cap: Option<u32>) -> Self {
        if cap.is_some() {
            self.dedup_cap_override = cap;
        }
        self
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
        self.current_reader = Some(BufReader::with_capacity(4 * 1024 * 1024, file));
        self.path_cursor += 1;
        Ok(())
    }

    fn next_line(
        &mut self,
        scratch: &mut String,
        contamination: &NgramSet,
        min_len: usize,
        max_len: usize,
        min_reading_len: usize,
        dedup_cap: u32,
    ) -> Result<NextLine> {
        if self.current_reader.is_none() {
            self.reshuffle_and_open()?;
        }
        // A pool is "dead" if we scan this many consecutive lines without any
        // emit. That happens when every (reading, surface) pair has hit the
        // dedup cap. Without this guard, next_line spins forever and the
        // whole mix hangs (observed at 73M/200M with bunsetsu+cap=3).
        const DEAD_THRESHOLD: u64 = 20_000_000;
        let mut consecutive_skip: u64 = 0;
        loop {
            if consecutive_skip > DEAD_THRESHOLD {
                return Ok(NextLine::Exhausted);
            }
            scratch.clear();
            let reader = self.current_reader.as_mut().unwrap();
            let n = reader
                .read_line(scratch)
                .with_context(|| format!("read_line in pool {}", self.name))?;
            if n == 0 {
                self.open_next()?;
                continue;
            }
            let trimmed = scratch.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let row: RowSurface = match serde_json::from_str(trimmed) {
                Ok(r) => r,
                Err(_) => {
                    self.skipped_parse += 1;
                    consecutive_skip += 1;
                    continue;
                }
            };
            if row.surface.is_empty() {
                self.skipped_parse += 1;
                continue;
            }
            let surf_len = row.surface.chars().count();
            if surf_len < min_len || surf_len > max_len {
                self.skipped_length += 1;
                consecutive_skip += 1;
                continue;
            }
            if min_reading_len > 0 && row.reading.chars().count() < min_reading_len {
                self.skipped_length += 1;
                consecutive_skip += 1;
                continue;
            }
            if self.filter_contamination
                && !contamination.is_empty()
                && contamination.contains_overlap(&row.surface)
            {
                self.skipped_contam += 1;
                consecutive_skip += 1;
                continue;
            }
            let effective_cap = self.dedup_cap_override.unwrap_or(dedup_cap);
            if effective_cap > 0 {
                let mut hasher = ahash::AHasher::default();
                row.reading.hash(&mut hasher);
                0u8.hash(&mut hasher);
                row.surface.hash(&mut hasher);
                let key = hasher.finish();
                let slot = self.seen.entry(key).or_insert(0);
                if *slot >= effective_cap {
                    self.skipped_dedup += 1;
                    consecutive_skip += 1;
                    continue;
                }
                *slot += 1;
            }
            if !scratch.ends_with('\n') {
                scratch.push('\n');
            }
            return Ok(NextLine::Yielded);
        }
    }
}

/// Result of a single `next_line` call. `Exhausted` means the pool can no
/// longer produce new rows (every line hits dedup cap / length / contam) and
/// the main loop should stop asking this pool for more rows.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum NextLine {
    Yielded,
    Exhausted,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    let ratio_sum = args.ratio_chunks
        + args.ratio_zenz
        + args.ratio_wiki
        + args.ratio_bunsetsu
        + args.ratio_fineweb2
        + args.ratio_hplt
        + args.ratio_synth;
    if (ratio_sum - 1.0).abs() > 1e-6 {
        bail!("ratios must sum to 1.0, got {:.6}", ratio_sum);
    }

    // Build contamination set.
    let mut contamination = NgramSet::new(args.contamination_n);
    for p_str in &args.contamination_ref {
        let p = PathBuf::from(p_str);
        if !p.exists() {
            eprintln!("[warn] contamination JSONL ref not found: {}", p_str);
            continue;
        }
        eprintln!("Loading contamination JSONL: {}", p.display());
        contamination.extend_from_jsonl(&p)?;
    }
    for p_str in &args.contamination_probe_ref {
        let p = PathBuf::from(p_str);
        if !p.exists() {
            eprintln!("[warn] contamination probe JSON ref not found: {}", p_str);
            continue;
        }
        let n = extend_contamination_from_probe_json(&mut contamination, &p)?;
        eprintln!(
            "Loading contamination probe JSON: {} (+{} strings)",
            p.display(),
            n
        );
    }
    eprintln!(
        "  combined contamination n-grams (n={}): {}",
        args.contamination_n,
        contamination.len()
    );

    // Assemble pools in fixed order. Same order as ratios / targets below.
    let bunsetsu_paths: Vec<PathBuf> = args.bunsetsu_paths.iter().map(PathBuf::from).collect();
    let specs: Vec<(FilePool, f64)> = vec![
        (
            FilePool::new(
                "chunks",
                vec![args.chunks_path.clone()],
                "chunks",
                args.filter_chunks,
                args.seed ^ 0x1111,
            ),
            args.ratio_chunks,
        ),
        (
            FilePool::new(
                "zenz",
                vec![args.zenz_path.clone()],
                "zenz_llmjp",
                args.filter_zenz,
                args.seed ^ 0x2222,
            ),
            args.ratio_zenz,
        ),
        (
            FilePool::new(
                "wiki_aozora",
                vec![args.wiki_path.clone(), args.aozora_path.clone()],
                "wiki_aozora",
                args.filter_wiki,
                args.seed ^ 0x3333,
            ),
            args.ratio_wiki,
        ),
        (
            FilePool::new(
                "bunsetsu",
                bunsetsu_paths,
                "bunsetsu",
                args.filter_bunsetsu,
                args.seed ^ 0x4444,
            )
            .with_optional_dedup_override(args.dedup_cap_bunsetsu),
            args.ratio_bunsetsu,
        ),
        (
            FilePool::new(
                "fineweb2",
                vec![args.fineweb2_path.clone()],
                "fineweb2_ja",
                args.filter_fineweb2,
                args.seed ^ 0x5555,
            ),
            args.ratio_fineweb2,
        ),
        (
            FilePool::new(
                "hplt",
                vec![args.hplt_path.clone()],
                "hplt3_ja",
                args.filter_hplt,
                args.seed ^ 0x6666,
            ),
            args.ratio_hplt,
        ),
        (
            FilePool::new(
                "synth",
                args.synth_paths.iter().map(PathBuf::from).collect(),
                "synth",
                args.filter_synth,
                args.seed ^ 0x7777,
            )
            .with_dedup_override(args.dedup_cap_synth),
            args.ratio_synth,
        ),
    ];
    let mut pools: Vec<(FilePool, f64)> = Vec::new();
    for (pool, ratio) in specs {
        let existing: Vec<PathBuf> = pool.paths.iter().filter(|p| p.exists()).cloned().collect();
        if existing.is_empty() || ratio <= 0.0 {
            eprintln!(
                "[warn] dropping pool {} (ratio={:.3}, paths_exist={}/{})",
                pool.name,
                ratio,
                existing.len(),
                pool.paths.len()
            );
            continue;
        }
        let mut p = pool;
        p.paths = existing;
        pools.push((p, ratio));
    }
    if pools.is_empty() {
        return Err(anyhow!("no usable pools"));
    }

    // Normalize ratios → integer targets summing exactly to args.total.
    let sum_ratio: f64 = pools.iter().map(|(_, r)| *r).sum();
    let exact: Vec<f64> = pools
        .iter()
        .map(|(_, r)| args.total as f64 * r / sum_ratio)
        .collect();
    let mut targets: Vec<u64> = exact.iter().map(|v| v.floor() as u64).collect();
    let mut alloc: u64 = targets.iter().sum();
    // Distribute remainder via largest-fractional order.
    let mut order: Vec<usize> = (0..pools.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for idx in order {
        if alloc >= args.total {
            break;
        }
        targets[idx] += 1;
        alloc += 1;
    }

    println!("=== v3 mix (target={}, normalized) ===", args.total);
    for ((pool, _), tgt) in pools.iter().zip(targets.iter()) {
        let r = *tgt as f64 / args.total as f64;
        println!(
            "  {:<12} ratio={:.3}  target_rows={:>12}  filter_contam={}  src={} ({} paths)",
            pool.name,
            r,
            tgt,
            pool.filter_contamination,
            pool.source_tag,
            pool.paths.len()
        );
    }

    // Output writer.
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out_file =
        File::create(&args.output).with_context(|| format!("create {:?}", args.output))?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, out_file);

    // Buffered shuffled output.
    let mut buffer: Vec<String> = Vec::with_capacity(args.shuffle_buffer);
    let mut served: Vec<u64> = vec![0; pools.len()];
    let mut written: u64 = 0;
    let t0 = Instant::now();
    let mut scratch = String::with_capacity(4096);
    let mut last_report: u64 = 0;

    while written < args.total {
        // Weighted-least-served via remaining-target sampling.
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
        match pool.next_line(
            &mut scratch,
            &contamination,
            args.min_surface_len,
            args.max_surface_len,
            args.min_reading_len,
            args.dedup_cap,
        )? {
            NextLine::Yielded => {
                buffer.push(scratch.clone());
                served[idx] += 1;
            }
            NextLine::Exhausted => {
                // Pool can't produce new rows (dedup cap saturated, or all
                // remaining lines fail filters). Mark its remaining target
                // as served so weighted-choice stops picking it. The total
                // row count at the end will be lower than args.total.
                eprintln!(
                    "[warn] pool {} exhausted at served={}/target={} (remaining={} rows won't emit)",
                    pool.name,
                    served[idx],
                    targets[idx],
                    targets[idx] as i64 - served[idx] as i64
                );
                targets[idx] = served[idx];
            }
        }

        if buffer.len() >= args.shuffle_buffer {
            buffer.shuffle(&mut rng);
            for line in buffer.drain(..) {
                writer.write_all(line.as_bytes())?;
                written += 1;
            }
            if written - last_report >= args.report_every {
                last_report = written;
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = written as f64 / elapsed.max(1e-6);
                let eta = (args.total - written) as f64 / rate.max(1e-6);
                println!(
                    "  written={}/{}  rate={:.1}k rows/s  eta={:.1}min",
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
        let pct = *got as f64 / (*tgt).max(1) as f64 * 100.0;
        println!(
            "    {:<12} served={}  target={}  ({:.1}%)  skip_contam={}  skip_length={}  skip_parse={}  skip_dedup={}",
            pool.name,
            got,
            tgt,
            pct,
            pool.skipped_contam,
            pool.skipped_length,
            pool.skipped_parse,
            pool.skipped_dedup
        );
    }

    Ok(())
}
