use ahash::AHashSet;
use clap::Parser;
use serde::Serialize;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};

#[derive(Parser)]
#[command(about = "Generate bunsetsu-level chunk training data from MeCab TSV")]
struct Args {
    /// Input TSV file (from mecab_to_tsv.py)
    #[arg(short, long)]
    input: String,

    /// Output JSONL file
    #[arg(short, long)]
    output: String,

    /// Maximum number of output chunks
    #[arg(short, long, default_value = "10000000")]
    max_samples: usize,

    /// Maximum bunsetsu window size
    #[arg(short = 'w', long, default_value = "4")]
    max_window: usize,

    /// Skip deduplication (faster, uses less memory)
    #[arg(long, default_value = "false")]
    no_dedup: bool,
}

#[derive(Serialize)]
struct Pair {
    reading: String,
    surface: String,
    context: String,
}

struct Morpheme {
    surface: String,
    reading: String,
    pos: String,
}

/// Content POS that start a new bunsetsu
fn is_content_pos(pos: &str) -> bool {
    matches!(
        pos,
        "名詞" | "動詞" | "形容詞" | "副詞" | "連体詞" | "接続詞" | "感動詞" | "接頭辞"
    )
}

/// Function POS that end a bunsetsu
fn is_boundary_pos(pos: &str) -> bool {
    matches!(pos, "助詞" | "助動詞")
}

/// Check if reading is valid hiragana (+ prolonged sound mark, punctuation)
fn is_valid_reading(reading: &str) -> bool {
    if reading.is_empty() {
        return false;
    }
    for c in reading.chars() {
        if ('\u{3040}'..='\u{309f}').contains(&c) // hiragana
            || c == 'ー' // prolonged
            || "、。！？「」『』（）・…―─　 ".contains(c)
        {
            continue;
        }
        return false;
    }
    true
}

struct Bunsetsu {
    surface: String,
    reading: String,
}

fn split_bunsetsu(morphemes: &[Morpheme]) -> Vec<Bunsetsu> {
    let mut result = Vec::new();
    let mut cur_surface = String::new();
    let mut cur_reading = String::new();
    let mut has_content = false;
    let mut prev_pos = "";

    for m in morphemes {
        if m.reading.is_empty() {
            // Unknown reading: flush and skip
            if !cur_surface.is_empty() && has_content {
                result.push(Bunsetsu {
                    surface: std::mem::take(&mut cur_surface),
                    reading: std::mem::take(&mut cur_reading),
                });
            }
            cur_surface.clear();
            cur_reading.clear();
            has_content = false;
            prev_pos = &m.pos;
            continue;
        }

        // Bunsetsu boundary: after function word, before content word
        if is_boundary_pos(prev_pos)
            && is_content_pos(&m.pos)
            && !cur_surface.is_empty()
            && has_content
        {
            result.push(Bunsetsu {
                surface: std::mem::take(&mut cur_surface),
                reading: std::mem::take(&mut cur_reading),
            });
            has_content = false;
        }

        cur_surface.push_str(&m.surface);
        cur_reading.push_str(&m.reading);
        if is_content_pos(&m.pos) {
            has_content = true;
        }
        prev_pos = &m.pos;
    }

    // Flush last
    if !cur_surface.is_empty() && has_content {
        result.push(Bunsetsu {
            surface: cur_surface,
            reading: cur_reading,
        });
    }

    result
}

fn generate_chunks(
    bunsetsu: &[Bunsetsu],
    max_window: usize,
    seen: &mut Option<AHashSet<u64>>,
    writer: &mut BufWriter<File>,
    count: &mut usize,
    max_samples: usize,
) -> io::Result<()> {
    use std::hash::{Hash, Hasher};

    for window in 1..=max_window.min(bunsetsu.len()) {
        for i in 0..=bunsetsu.len() - window {
            let chunk = &bunsetsu[i..i + window];

            let surface: String = chunk.iter().map(|b| b.surface.as_str()).collect();
            let reading: String = chunk.iter().map(|b| b.reading.as_str()).collect();

            // Skip too short/long
            let slen = surface.chars().count();
            if slen < 2 || slen > 60 {
                continue;
            }
            if !is_valid_reading(&reading) {
                continue;
            }
            // Skip identity (no kanji)
            if reading == surface {
                continue;
            }

            // Dedup via hash
            if let Some(ref mut set) = seen {
                let mut hasher = ahash::AHasher::default();
                reading.hash(&mut hasher);
                surface.hash(&mut hasher);
                let hash = hasher.finish();
                if !set.insert(hash) {
                    continue;
                }
            }

            let pair = Pair {
                reading,
                surface,
                context: String::new(),
            };
            serde_json::to_writer(&mut *writer, &pair)?;
            writer.write_all(b"\n")?;
            *count += 1;

            if *count >= max_samples {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let reader = BufReader::new(File::open(&args.input)?);
    let out_file = File::create(&args.output)?;
    let mut writer = BufWriter::new(out_file);

    let mut seen: Option<AHashSet<u64>> = if args.no_dedup {
        None
    } else {
        Some(AHashSet::new())
    };

    let mut count = 0usize;
    let mut sentences = 0usize;
    let mut morphemes: Vec<Morpheme> = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if line.is_empty() {
            // End of sentence block
            if !morphemes.is_empty() {
                let bunsetsu = split_bunsetsu(&morphemes);
                generate_chunks(
                    &bunsetsu,
                    args.max_window,
                    &mut seen,
                    &mut writer,
                    &mut count,
                    args.max_samples,
                )?;
                morphemes.clear();
                sentences += 1;

                if sentences % 500_000 == 0 {
                    eprintln!("  {} sentences, {} chunks...", sentences, count);
                }
                if count >= args.max_samples {
                    break;
                }
            }
            continue;
        }

        // Parse TSV: surface\treading\tpos
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() >= 3 {
            morphemes.push(Morpheme {
                surface: parts[0].to_string(),
                reading: parts[1].to_string(),
                pos: parts[2].to_string(),
            });
        }
    }

    // Flush last sentence
    if !morphemes.is_empty() {
        let bunsetsu = split_bunsetsu(&morphemes);
        generate_chunks(
            &bunsetsu,
            args.max_window,
            &mut seen,
            &mut writer,
            &mut count,
            args.max_samples,
        )?;
    }

    writer.flush()?;
    eprintln!(
        "Done: {} sentences -> {} chunks -> {}",
        sentences, count, args.output
    );

    Ok(())
}
