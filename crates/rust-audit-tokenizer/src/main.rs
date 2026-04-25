use clap::Parser;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};

const INVALID_BYTE_TOKEN: &str = "〓";

#[derive(Parser, Debug)]
#[command(about = "Audit SharedCharTokenizer byte fallback ratios over JSONL data")]
struct Args {
    /// Input JSONL file
    #[arg(short, long)]
    input: String,

    /// Shared tokenizer JSON file
    #[arg(short, long)]
    tokenizer: String,

    /// JSON fields to inspect
    #[arg(long, num_args = 1.., default_values_t = vec![
        String::from("reading"),
        String::from("surface"),
        String::from("context"),
    ])]
    fields: Vec<String>,

    /// Maximum number of rows to inspect
    #[arg(long, default_value_t = 0)]
    limit: usize,
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

#[derive(Deserialize)]
struct TokenizerJson {
    token_to_id: BTreeMap<String, usize>,
}

#[derive(Default)]
struct FieldStats {
    fallback_chars: u64,
    total_chars: u64,
    examples: Vec<String>,
}

fn selected_fields<'a>(
    pair: &'a Pair,
    fields: &'a [String],
) -> impl Iterator<Item = (&'a str, &'a str)> {
    fields.iter().filter_map(|field| match field.as_str() {
        "reading" => Some(("reading", pair.reading.as_str())),
        "surface" => Some(("surface", pair.surface.as_str())),
        "context" => Some(("context", pair.context.as_str())),
        _ => None,
    })
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let tokenizer_reader = BufReader::new(File::open(&args.tokenizer)?);
    let tokenizer: TokenizerJson = serde_json::from_reader(tokenizer_reader)?;

    let mut per_field = HashMap::<String, FieldStats>::new();
    for field in &args.fields {
        per_field.insert(field.clone(), FieldStats::default());
    }

    let reader = BufReader::new(File::open(&args.input)?);
    let mut rows = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let pair: Pair = match serde_json::from_str(&line) {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        rows += 1;
        for (field_name, text) in selected_fields(&pair, &args.fields) {
            let stats = per_field.get_mut(field_name).unwrap();
            let mut fallback = 0u64;
            let mut total = 0u64;
            for ch in text.chars() {
                total += 1;
                if !tokenizer.token_to_id.contains_key(&ch.to_string()) {
                    fallback += 1;
                }
            }
            stats.fallback_chars += fallback;
            stats.total_chars += total;
            if fallback > 0 && stats.examples.len() < 5 {
                stats.examples.push(text.chars().take(120).collect());
            }
        }
        if args.limit > 0 && rows >= args.limit {
            break;
        }
    }

    println!("rows={rows}");
    println!("invalid_decode_sentinel={INVALID_BYTE_TOKEN}");
    for field in &args.fields {
        if let Some(stats) = per_field.get(field) {
            let ratio = if stats.total_chars == 0 {
                0.0
            } else {
                stats.fallback_chars as f64 / stats.total_chars as f64
            };
            println!(
                "{field}: fallback_chars={} total_chars={} ratio={ratio:.6}",
                stats.fallback_chars, stats.total_chars
            );
            for (idx, example) in stats.examples.iter().enumerate() {
                println!("  example{}: {}", idx + 1, example);
            }
        }
    }

    Ok(())
}
