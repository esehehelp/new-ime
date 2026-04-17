use ahash::AHashMap;
use clap::Parser;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Parser)]
#[command(about = "Build output tokenizer vocabulary from character frequencies")]
struct Args {
    /// Input JSONL files (reads 'surface' field)
    #[arg(short, long, num_args = 1..)]
    input: Vec<String>,

    /// Output vocab JSON file
    #[arg(short, long)]
    output: String,

    /// Maximum number of kanji in vocabulary
    #[arg(long, default_value = "4000")]
    max_kanji: usize,

    /// Output frequency stats JSON
    #[arg(long)]
    stats: Option<String>,
}

#[derive(Deserialize)]
struct Pair {
    surface: String,
}

const SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[BLANK]", "[MASK]"];

fn is_hiragana(c: char) -> bool {
    ('\u{3041}'..='\u{3096}').contains(&c)
}

fn is_katakana(c: char) -> bool {
    ('\u{30A1}'..='\u{30FA}').contains(&c)
}

fn is_cjk(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

fn is_ascii_printable(c: char) -> bool {
    ('\u{0020}'..='\u{007E}').contains(&c)
}

fn is_fullwidth_ascii(c: char) -> bool {
    ('\u{FF01}'..='\u{FF5E}').contains(&c)
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Count character frequencies
    let mut freq: AHashMap<char, u64> = AHashMap::new();
    let mut total_chars = 0u64;

    for path in &args.input {
        eprintln!("Counting characters in {}...", path);
        let reader = BufReader::new(File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let pair: Pair = match serde_json::from_str(&line) {
                Ok(p) => p,
                Err(_) => continue,
            };
            for c in pair.surface.chars() {
                *freq.entry(c).or_insert(0) += 1;
                total_chars += 1;
            }
        }
    }

    eprintln!(
        "Total: {} characters, {} unique",
        total_chars,
        freq.len()
    );

    // Categorize and sort by frequency
    let mut hiragana: Vec<(char, u64)> = Vec::new();
    let mut katakana: Vec<(char, u64)> = Vec::new();
    let mut ascii: Vec<(char, u64)> = Vec::new();
    let mut kanji: Vec<(char, u64)> = Vec::new();
    let mut symbols: Vec<(char, u64)> = Vec::new();

    for (&c, &count) in &freq {
        if is_hiragana(c) {
            hiragana.push((c, count));
        } else if is_katakana(c) {
            katakana.push((c, count));
        } else if is_ascii_printable(c) {
            ascii.push((c, count));
        } else if is_cjk(c) {
            kanji.push((c, count));
        } else {
            symbols.push((c, count));
        }
    }

    // Sort each by frequency descending
    for list in [&mut hiragana, &mut katakana, &mut ascii, &mut kanji, &mut symbols] {
        list.sort_by(|a, b| b.1.cmp(&a.1));
    }

    // Build vocabulary
    let mut vocab: Vec<(String, usize)> = Vec::new();
    let mut idx = 0usize;

    // Special tokens
    for tok in SPECIAL_TOKENS {
        vocab.push((tok.to_string(), idx));
        idx += 1;
    }

    // Byte fallback tokens
    for b in 0..256u16 {
        vocab.push((format!("<0x{:02X}>", b), idx));
        idx += 1;
    }

    // Hiragana
    for (c, _) in &hiragana {
        vocab.push((c.to_string(), idx));
        idx += 1;
    }

    // Katakana
    for (c, _) in &katakana {
        vocab.push((c.to_string(), idx));
        idx += 1;
    }

    // ASCII
    for (c, _) in &ascii {
        vocab.push((c.to_string(), idx));
        idx += 1;
    }

    // Fullwidth ASCII
    for cp in 0xFF01u32..=0xFF5Eu32 {
        if let Some(c) = char::from_u32(cp) {
            if !freq.contains_key(&c) {
                // Add even if not seen (for completeness)
            }
            vocab.push((c.to_string(), idx));
            idx += 1;
        }
    }

    // Symbols (by frequency)
    for (c, _) in &symbols {
        vocab.push((c.to_string(), idx));
        idx += 1;
    }

    // Kanji (limited to max_kanji, by frequency)
    let kanji_count = kanji.len().min(args.max_kanji);
    for (c, _) in kanji.iter().take(kanji_count) {
        vocab.push((c.to_string(), idx));
        idx += 1;
    }

    eprintln!("\nVocabulary summary:");
    eprintln!("  Special tokens: {}", SPECIAL_TOKENS.len());
    eprintln!("  Byte fallback: 256");
    eprintln!("  Hiragana: {}", hiragana.len());
    eprintln!("  Katakana: {}", katakana.len());
    eprintln!("  ASCII: {}", ascii.len());
    eprintln!("  Symbols: {}", symbols.len());
    eprintln!(
        "  Kanji: {} (of {} total, capped at {})",
        kanji_count,
        kanji.len(),
        args.max_kanji
    );
    eprintln!("  Total vocab size: {}", idx);

    // Write vocab JSON: {"type": "output", "token_to_id": {...}}
    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);

    write!(writer, "{{\"type\":\"output\",\"token_to_id\":{{")?;
    for (i, (token, id)) in vocab.iter().enumerate() {
        if i > 0 {
            write!(writer, ",")?;
        }
        // Escape token for JSON
        let escaped = serde_json::to_string(token).unwrap();
        write!(writer, "{}:{}", escaped, id)?;
    }
    write!(writer, "}}}}")?;
    writer.flush()?;

    eprintln!("\nVocabulary saved to {}", args.output);

    // Optional stats
    if let Some(stats_path) = &args.stats {
        let stats_file = File::create(stats_path)?;
        let mut sw = BufWriter::new(stats_file);
        write!(sw, "{{\"total_characters\":{},\"unique_characters\":{},\"vocab_size\":{},\"top_100_chars\":[",
            total_chars, freq.len(), idx)?;
        let mut all_chars: Vec<(char, u64)> = freq.into_iter().collect();
        all_chars.sort_by(|a, b| b.1.cmp(&a.1));
        for (i, (c, count)) in all_chars.iter().take(100).enumerate() {
            if i > 0 {
                write!(sw, ",")?;
            }
            let cs = serde_json::to_string(&c.to_string()).unwrap();
            write!(sw, "{{\"char\":{},\"count\":{}}}", cs, count)?;
        }
        write!(sw, "]}}")?;
        sw.flush()?;
        eprintln!("Stats saved to {}", stats_path);
    }

    Ok(())
}
