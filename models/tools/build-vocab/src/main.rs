use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(about = "Build shared tokenizer vocabulary from JSONL character frequencies")]
struct Args {
    /// Input JSONL files
    #[arg(short, long, num_args = 1..)]
    input: Vec<String>,

    /// Output vocab JSON file
    #[arg(short, long)]
    output: String,

    /// Maximum number of kanji in vocabulary
    #[arg(long, default_value = "4000")]
    max_kanji: usize,

    /// JSON fields to count
    #[arg(long, num_args = 1.., default_values_t = vec![
        String::from("reading"),
        String::from("surface"),
        String::from("context"),
    ])]
    fields: Vec<String>,

    /// Output frequency stats JSON
    #[arg(long)]
    stats: Option<String>,
}

#[derive(Deserialize)]
struct Pair {
    #[serde(default)]
    reading: String,
    #[serde(default)]
    surface: String,
    #[serde(default)]
    context: String,
}

#[derive(Serialize)]
struct TokenizerJson {
    #[serde(rename = "type")]
    tokenizer_type: String,
    max_kanji: usize,
    token_to_id: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct StatsEntry {
    char: String,
    count: u64,
}

#[derive(Serialize)]
struct StatsJson {
    fields: Vec<String>,
    total_characters: u64,
    unique_characters: usize,
    vocab_size: usize,
    top_100_chars: Vec<StatsEntry>,
}

const SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[BLANK]", "[MASK]"];
const JP_SYMBOLS: &[char] = &[
    '\u{3000}', '、', '。', '！', '？', '「', '」', '『', '』', '（', '）', '【', '】', '〔', '〕',
    '｛', '｝', '〈', '〉', '《', '》', '・', 'ー', '～', '…', '‥', '々', '〇', '〻', 'ヶ', 'ヵ',
];

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

fn selected_fields<'a>(pair: &'a Pair, fields: &'a [String]) -> impl Iterator<Item = &'a str> {
    fields.iter().filter_map(|field| match field.as_str() {
        "reading" => Some(pair.reading.as_str()),
        "surface" => Some(pair.surface.as_str()),
        "context" => Some(pair.context.as_str()),
        _ => None,
    })
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();

    // Count character frequencies
    let mut freq: HashMap<char, u64> = HashMap::new();
    let mut total_chars = 0u64;
    let mut total_rows = 0u64;

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
            total_rows += 1;
            for text in selected_fields(&pair, &args.fields) {
                for c in text.chars() {
                    *freq.entry(c).or_insert(0) += 1;
                    total_chars += 1;
                }
            }
        }
    }

    eprintln!(
        "Total: {} rows, {} characters, {} unique",
        total_rows, total_chars, freq.len()
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
    let mut seen = HashMap::<String, usize>::new();
    let mut idx = 0usize;

    // Special tokens
    for tok in SPECIAL_TOKENS {
        vocab.push((tok.to_string(), idx));
        seen.insert(tok.to_string(), idx);
        idx += 1;
    }

    // Byte fallback tokens
    for b in 0..256u16 {
        let token = format!("<0x{:02X}>", b);
        vocab.push((token.clone(), idx));
        seen.insert(token, idx);
        idx += 1;
    }

    // Hiragana
    for (c, _) in &hiragana {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
    }

    // Katakana
    for (c, _) in &katakana {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
    }

    // ASCII
    for (c, _) in &ascii {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
    }

    // Fullwidth ASCII
    for cp in 0xFF01u32..=0xFF5Eu32 {
        if let Some(c) = char::from_u32(cp) {
            let token = c.to_string();
            if !seen.contains_key(&token) {
                vocab.push((token.clone(), idx));
                seen.insert(token, idx);
                idx += 1;
            }
        }
    }

    // Always-available Japanese symbols, including fullwidth space.
    for c in JP_SYMBOLS {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
    }

    // Symbols (by frequency)
    for (c, _) in &symbols {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
    }

    // Kanji (limited to max_kanji, by frequency)
    let kanji_count = kanji.len().min(args.max_kanji);
    for (c, _) in kanji.iter().take(kanji_count) {
        let token = c.to_string();
        if !seen.contains_key(&token) {
            vocab.push((token.clone(), idx));
            seen.insert(token, idx);
            idx += 1;
        }
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

    let token_to_id = vocab.into_iter().collect::<BTreeMap<_, _>>();
    let tokenizer_json = TokenizerJson {
        tokenizer_type: "shared".to_string(),
        max_kanji: args.max_kanji,
        token_to_id,
    };
    if let Some(parent) = Path::new(&args.output).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let out_file = File::create(&args.output)?;
    let writer = BufWriter::new(out_file);
    serde_json::to_writer_pretty(writer, &tokenizer_json)?;

    eprintln!("\nVocabulary saved to {}", args.output);

    // Optional stats
    if let Some(stats_path) = &args.stats {
        if let Some(parent) = Path::new(stats_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let stats_file = File::create(stats_path)?;
        let mut all_chars: Vec<(char, u64)> = freq.iter().map(|(c, count)| (*c, *count)).collect();
        all_chars.sort_by(|a, b| b.1.cmp(&a.1));
        let top_100_chars = all_chars
            .into_iter()
            .take(100)
            .map(|(c, count)| StatsEntry {
                char: c.to_string(),
                count,
            })
            .collect::<Vec<_>>();
        let stats = StatsJson {
            fields: args.fields.clone(),
            total_characters: total_chars,
            unique_characters: freq.len(),
            vocab_size: idx,
            top_100_chars,
        };
        let sw = BufWriter::new(stats_file);
        serde_json::to_writer_pretty(sw, &stats)?;
        eprintln!("Stats saved to {}", stats_path);
    }

    Ok(())
}
