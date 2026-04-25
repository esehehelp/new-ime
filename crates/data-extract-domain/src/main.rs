//! Extract domain-filtered, char-separated surface text for KenLM training.
//!
//! Reads a JSONL file whose rows have a `surface` field, writes one sentence
//! per line to the output file with a single ASCII space between every
//! character — the format expected by `KenLMCharScorer`. Multiple `--domain`
//! filters can be chained; the tool runs a single sequential pass and emits
//! every matching row to the corresponding output.
//!
//! Supported heuristics (character-class ratios computed on surface):
//!   * `tech`   : >=2 consecutive ASCII letters OR >=4 consecutive katakana
//!   * `entity` : contains at least one named-entity surface marker — any
//!                uppercase-leading ASCII run of length >= 2 (brand names
//!                like "Apple"), or >=2 consecutive CJK kanji with specific
//!                pattern (proper-noun ender: 「氏/市/区/町/駅」...)
//!   * `general`: always keep (no filter)
//!
//! Rust because: scanning 46 GiB JSONL + char-class categorisation is CPU
//! bound, and memory `feedback_script_language_preference` says data pipes
//! default to Rust. Python version would take ~10 min, Rust ~60 s.

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Extract domain-filtered surface text (char-separated) for KenLM training.")]
struct Args {
    /// Input JSONL (surface field required).
    #[arg(long)]
    input: PathBuf,

    /// `--domain <name>=<path>` pairs; any of {tech, entity, general}.
    /// Example: `--domain tech=data/tech.txt --domain general=data/general.txt`
    #[arg(long, num_args = 1.., required = true)]
    domain: Vec<String>,

    /// Cap per-domain output lines (0 = unlimited).
    #[arg(long, default_value_t = 0usize)]
    max_lines: usize,

    /// Progress report interval in input lines.
    #[arg(long, default_value_t = 5_000_000u64)]
    progress_every: u64,

    /// Surface length lower bound (chars).
    #[arg(long, default_value_t = 2usize)]
    min_chars: usize,
    /// Surface length upper bound (chars).
    #[arg(long, default_value_t = 128usize)]
    max_chars: usize,
}

#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    surface: String,
}

fn is_ascii_alpha(c: char) -> bool {
    c.is_ascii_alphabetic()
}

fn is_katakana(c: char) -> bool {
    let cp = c as u32;
    (0x30A1..=0x30FA).contains(&cp) || cp == 0x30FC /* ー */
}

/// Tech: >=2 consecutive ASCII letters, OR >=4 consecutive katakana.
fn is_tech(surface: &str) -> bool {
    let mut ascii_run = 0usize;
    let mut kata_run = 0usize;
    for c in surface.chars() {
        if is_ascii_alpha(c) {
            ascii_run += 1;
            kata_run = 0;
            if ascii_run >= 2 {
                return true;
            }
        } else if is_katakana(c) {
            kata_run += 1;
            ascii_run = 0;
            if kata_run >= 4 {
                return true;
            }
        } else {
            ascii_run = 0;
            kata_run = 0;
        }
    }
    false
}

const ENTITY_END_MARKERS: &[char] = &[
    '氏', '市', '区', '町', '村', '県', '府', '都', '駅', '線', '社', '会', '部', '院', '党', '神',
    '寺', '藩', '郡', '省', '庁', '所', '殿', '様', '君', '展',
];

/// Entity: uppercase-leading ASCII run length >= 2 (brands like "Apple", "AI"),
/// or kanji sequence ending in a named-entity marker.
fn is_entity(surface: &str) -> bool {
    // Heuristic 1: ASCII run starting with uppercase of length >= 2.
    let chars: Vec<char> = surface.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i].is_ascii_uppercase() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j - i >= 2 {
                return true;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    // Heuristic 2: any entity-end marker preceded by a kanji.
    let mut prev_is_kanji = false;
    for c in surface.chars() {
        let cp = c as u32;
        let is_kanji = (0x4E00..=0x9FFF).contains(&cp);
        if prev_is_kanji && ENTITY_END_MARKERS.contains(&c) {
            return true;
        }
        prev_is_kanji = is_kanji;
    }
    false
}

/// Parse `--domain name=path` form. Unknown names are rejected.
fn parse_domains(specs: &[String]) -> Result<Vec<(String, PathBuf)>> {
    let allowed = ["tech", "entity", "general"];
    let mut out = Vec::with_capacity(specs.len());
    for spec in specs {
        let (name, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow!("bad --domain spec (want name=path): {}", spec))?;
        if !allowed.contains(&name) {
            return Err(anyhow!("unknown domain {:?}; allowed: {:?}", name, allowed));
        }
        out.push((name.to_string(), PathBuf::from(path)));
    }
    Ok(out)
}

/// Char-separate a surface: "今日" → "今 日".
fn char_separate(surface: &str, dst: &mut String) {
    dst.clear();
    let mut first = true;
    for c in surface.chars() {
        if !first {
            dst.push(' ');
        }
        dst.push(c);
        first = false;
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let domains = parse_domains(&args.domain)?;
    eprintln!(
        "data-extract-domain: input={} domains=[{}]",
        args.input.display(),
        domains
            .iter()
            .map(|(n, p)| format!("{}→{}", n, p.display()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let input =
        File::open(&args.input).with_context(|| format!("open {}", args.input.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, input);

    let mut writers: HashMap<String, BufWriter<File>> = HashMap::new();
    for (name, path) in &domains {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let f = File::create(path).with_context(|| format!("create {}", path.display()))?;
        writers.insert(name.clone(), BufWriter::with_capacity(4 * 1024 * 1024, f));
    }

    let mut counts: HashMap<String, usize> = domains.iter().map(|(n, _)| (n.clone(), 0)).collect();
    let mut scanned: u64 = 0;
    let mut parsed_ok: u64 = 0;
    let mut len_skipped: u64 = 0;
    let t0 = Instant::now();
    let mut line_buf = String::with_capacity(4096);
    let mut spaced = String::with_capacity(1024);

    loop {
        line_buf.clear();
        let n = reader.read_line(&mut line_buf)?;
        if n == 0 {
            break;
        }
        scanned += 1;
        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        let row: Row = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if row.surface.is_empty() {
            continue;
        }
        parsed_ok += 1;
        let len = row.surface.chars().count();
        if len < args.min_chars || len > args.max_chars {
            len_skipped += 1;
            continue;
        }

        // Emit to every matching domain.
        let want_tech = is_tech(&row.surface);
        let want_entity = is_entity(&row.surface);

        for (name, _) in &domains {
            let emit = match name.as_str() {
                "tech" => want_tech,
                "entity" => want_entity,
                "general" => true,
                _ => false,
            };
            if !emit {
                continue;
            }
            let c = counts.get_mut(name).unwrap();
            if args.max_lines > 0 && *c >= args.max_lines {
                continue;
            }
            char_separate(&row.surface, &mut spaced);
            let w = writers.get_mut(name).unwrap();
            w.write_all(spaced.as_bytes())?;
            w.write_all(b"\n")?;
            *c += 1;
        }

        if scanned % args.progress_every == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = scanned as f64 / elapsed.max(1e-6);
            let counts_str = counts
                .iter()
                .map(|(n, c)| format!("{}={}", n, c))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "  scanned={:>12} parsed={:>12} rate={:.0} rows/s   {}",
                scanned, parsed_ok, rate, counts_str
            );
        }
    }

    for w in writers.values_mut() {
        w.flush()?;
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "\ndone: scanned={} parsed_ok={} len_skipped={} elapsed={:.1}s",
        scanned, parsed_ok, len_skipped, elapsed
    );
    for (name, c) in &counts {
        eprintln!("  {:<8} rows: {}", name, c);
    }
    Ok(())
}
