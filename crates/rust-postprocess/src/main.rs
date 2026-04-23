use ahash::AHashSet;
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Parser)]
#[command(about = "Post-process JSONL sentence pairs for quality filtering")]
struct Args {
    /// Input JSONL files
    #[arg(short, long, num_args = 1..)]
    input: Vec<String>,

    /// Output JSONL file
    #[arg(short, long)]
    output: String,

    /// Minimum surface length
    #[arg(long, default_value = "8")]
    min_len: usize,

    /// Maximum surface length
    #[arg(long, default_value = "100")]
    max_len: usize,

    /// Disable deduplication
    #[arg(long)]
    no_dedup: bool,

    /// Print statistics
    #[arg(long)]
    stats: bool,
}

#[derive(Deserialize)]
struct InputPair {
    reading: String,
    surface: String,
    #[serde(default)]
    context: String,
}

#[derive(Serialize)]
struct OutputPair {
    reading: String,
    surface: String,
    context: String,
}

struct Filters {
    re_valid_reading: Regex,
    re_pos_leak: Regex,
    re_old_kana: Regex,
    re_old_iteration: Regex,
    re_old_sokuon: Regex,
    re_old_fu: Regex,
    re_old_syou: Regex,
    re_starts_number: Regex,
    re_stage_direction: Regex,
    re_chapter: Regex,
    re_bare_title: Regex,
    re_leading_trailing_ws: Regex,
    re_multi_ws: Regex,
}

impl Filters {
    fn new() -> Self {
        Self {
            re_valid_reading: Regex::new(
                r"^[\u{3040}-\u{309f}\u{30fc}、。！？「」『』（）・…―─\u{3000}\s]+$"
            ).unwrap(),
            re_pos_leak: Regex::new(r"-[a-zA-Z\u{4e00}-\u{9fff}\u{30a0}-\u{30ff}]+").unwrap(),
            re_old_kana: Regex::new(r"[ゐゑヰヱ]").unwrap(),
            re_old_iteration: Regex::new(r"[ゝゞヽヾ]").unwrap(),
            re_old_sokuon: Regex::new(r"[^っ]つ[たてだで]").unwrap(),
            re_old_fu: Regex::new(r"[はかさたなまらわがざだば]ふ[。、）」\s]|ふ[。」]$").unwrap(),
            re_old_syou: Regex::new(r"[でせ]せう|ませう").unwrap(),
            re_starts_number: Regex::new(r"^\d+[\u{3000}\s]|^[０-９]+[\u{3000}\s]").unwrap(),
            re_stage_direction: Regex::new(
                r"^[\u{3040}-\u{9fff}\u{30a0}-\u{30ff}A-Za-z]{1,10}[\u{3000}\s]{2,}|^[\u{3040}-\u{9fff}\u{30a0}-\u{30ff}]{1,10}\u{3000}[（(]"
            ).unwrap(),
            re_chapter: Regex::new(
                r"^[\u{3000}\s]*(その|第)[一二三四五六七八九十百千\d]+[章節回話]?[\u{3000}\s]*$"
            ).unwrap(),
            re_bare_title: Regex::new(r"^[「『](.{1,20})[」』]$").unwrap(),
            re_leading_trailing_ws: Regex::new(r"^[\u{3000}\s]+|[\u{3000}\s]+$").unwrap(),
            re_multi_ws: Regex::new(r"\u{3000}{2,}").unwrap(),
        }
    }
}

fn has_kanji(s: &str) -> bool {
    s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

fn has_katakana(s: &str) -> bool {
    s.chars().any(|c| ('\u{30a1}'..='\u{30fa}').contains(&c))
}

fn has_ascii_alpha(s: &str) -> bool {
    s.chars().any(|c| c.is_ascii_alphabetic())
}

fn has_repeated_chars(s: &str, min_repeat: usize) -> bool {
    let mut prev = '\0';
    let mut count = 1;
    for c in s.chars() {
        if c == prev {
            count += 1;
            if count >= min_repeat {
                return true;
            }
        } else {
            prev = c;
            count = 1;
        }
    }
    false
}

fn has_non_hiragana(s: &str) -> bool {
    s.chars().any(|c| {
        !(('\u{3040}'..='\u{309f}').contains(&c)
            || "ー、。！？「」『』（）・…―─\u{3000} \n".contains(c))
    })
}

fn check(
    pair: &InputPair,
    filters: &Filters,
    min_len: usize,
    max_len: usize,
    reject_counts: &mut HashMap<&'static str, usize>,
) -> Option<OutputPair> {
    let mut reading = filters.re_leading_trailing_ws.replace_all(&pair.reading, "").to_string();
    let mut surface = filters.re_leading_trailing_ws.replace_all(&pair.surface, "").to_string();
    let context = filters.re_leading_trailing_ws.replace_all(&pair.context, "").to_string();

    surface = filters.re_multi_ws.replace_all(&surface, "\u{3000}").to_string();
    reading = filters.re_multi_ws.replace_all(&reading, "\u{3000}").to_string();

    // POS leak cleanup
    if filters.re_pos_leak.is_match(&reading) {
        let cleaned = filters.re_pos_leak.replace_all(&reading, "").to_string();
        if !cleaned.is_empty() && filters.re_valid_reading.is_match(&cleaned) {
            reading = cleaned;
        } else {
            *reject_counts.entry("pos_leak").or_insert(0) += 1;
            return None;
        }
    }

    let surface_chars: usize = surface.chars().count();
    if surface_chars < min_len {
        *reject_counts.entry("surface_too_short").or_insert(0) += 1;
        return None;
    }
    if surface_chars > max_len {
        *reject_counts.entry("surface_too_long").or_insert(0) += 1;
        return None;
    }

    let reading_chars: usize = reading.chars().count();
    if reading_chars < 5 {
        *reject_counts.entry("reading_too_short").or_insert(0) += 1;
        return None;
    }

    if !filters.re_valid_reading.is_match(&reading) {
        *reject_counts.entry("invalid_reading_chars").or_insert(0) += 1;
        return None;
    }
    if has_kanji(&reading) {
        *reject_counts.entry("kanji_in_reading").or_insert(0) += 1;
        return None;
    }
    if has_katakana(&reading) {
        *reject_counts.entry("katakana_in_reading").or_insert(0) += 1;
        return None;
    }
    if has_ascii_alpha(&reading) {
        *reject_counts.entry("ascii_in_reading").or_insert(0) += 1;
        return None;
    }

    // Old orthography
    if filters.re_old_kana.is_match(&surface) {
        *reject_counts.entry("old_kana").or_insert(0) += 1;
        return None;
    }
    if filters.re_old_iteration.is_match(&surface) {
        *reject_counts.entry("old_iteration_mark").or_insert(0) += 1;
        return None;
    }
    if filters.re_old_sokuon.is_match(&surface) {
        *reject_counts.entry("old_sokuon").or_insert(0) += 1;
        return None;
    }
    if filters.re_old_fu.is_match(&surface) {
        *reject_counts.entry("old_fu").or_insert(0) += 1;
        return None;
    }
    if filters.re_old_syou.is_match(&surface) {
        *reject_counts.entry("old_syou").or_insert(0) += 1;
        return None;
    }
    if filters.re_starts_number.is_match(&surface) {
        *reject_counts.entry("starts_with_number").or_insert(0) += 1;
        return None;
    }
    if filters.re_stage_direction.is_match(&surface) {
        *reject_counts.entry("stage_direction").or_insert(0) += 1;
        return None;
    }
    if filters.re_chapter.is_match(&surface) {
        *reject_counts.entry("chapter_number").or_insert(0) += 1;
        return None;
    }
    if let Some(caps) = filters.re_bare_title.captures(&surface) {
        if caps.get(1).map_or(0, |m| m.as_str().chars().count()) < 15 {
            *reject_counts.entry("bare_title").or_insert(0) += 1;
            return None;
        }
    }

    // Ratio check
    if reading_chars > 0 {
        let ratio = surface_chars as f64 / reading_chars as f64;
        if ratio < 0.3 || ratio > 1.5 {
            *reject_counts.entry("bad_ratio").or_insert(0) += 1;
            return None;
        }
    }

    if has_repeated_chars(&reading, 5) || has_repeated_chars(&surface, 5) {
        *reject_counts.entry("repeated_chars").or_insert(0) += 1;
        return None;
    }

    // Identity check
    if !has_non_hiragana(&surface) && reading == surface {
        // All-hiragana, reading == surface: valid
    } else if reading == surface && has_non_hiragana(&surface) {
        *reject_counts.entry("reading_equals_surface").or_insert(0) += 1;
        return None;
    }

    Some(OutputPair {
        reading,
        surface,
        context,
    })
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let filters = Filters::new();

    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);

    let mut seen: Option<AHashSet<u64>> = if args.no_dedup {
        None
    } else {
        Some(AHashSet::new())
    };

    let mut total_in = 0usize;
    let mut total_out = 0usize;
    let mut dedup_count = 0usize;
    let mut reject_counts: HashMap<&str, usize> = HashMap::new();

    for input_path in &args.input {
        eprintln!("Processing {}...", input_path);
        let reader = BufReader::new(File::open(input_path)?);

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            total_in += 1;

            let pair: InputPair = match serde_json::from_str(&line) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let cleaned = match check(&pair, &filters, args.min_len, args.max_len, &mut reject_counts) {
                Some(p) => p,
                None => continue,
            };

            // Dedup by surface
            if let Some(ref mut set) = seen {
                use std::hash::{Hash, Hasher};
                let mut hasher = ahash::AHasher::default();
                cleaned.surface.hash(&mut hasher);
                let hash = hasher.finish();
                if !set.insert(hash) {
                    dedup_count += 1;
                    continue;
                }
            }

            serde_json::to_writer(&mut writer, &cleaned)?;
            writer.write_all(b"\n")?;
            total_out += 1;

            if total_in % 500_000 == 0 {
                eprintln!("  {:>12} in, {:>12} out...", total_in, total_out);
            }
        }
    }

    writer.flush()?;

    let total_rejected: usize = reject_counts.values().sum();
    let pct = if total_in > 0 {
        total_out as f64 / total_in as f64 * 100.0
    } else {
        0.0
    };
    eprintln!(
        "\nTotal: {} in -> {} out ({:.1}% kept)",
        total_in, total_out, pct
    );
    if !args.no_dedup {
        eprintln!("Deduplicated: {}", dedup_count);
    }

    if args.stats {
        eprintln!("\nRejection reasons:");
        let mut sorted: Vec<_> = reject_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (reason, count) in sorted {
            eprintln!("  {}: {}", reason, count);
        }
        eprintln!("  TOTAL rejected: {}", total_rejected);
    }

    Ok(())
}
