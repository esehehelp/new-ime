//! Turn the extracted white-paper text into bunsetsu-schema-compatible JSONL
//! with reading + surface fields.
//!
//! Per-file pipeline:
//!
//!   1. Read the whole file. Skip if it looks garbled (too few Japanese
//!      chars per kilobyte — cao-gender h13-h15 PDFs render as PUA glyphs
//!      because pdftotext lacked Adobe-Japan1 CMap data).
//!
//!   2. Normalise whitespace and strip control chars.
//!
//!   3. Split into sentences on 。?! (plus ASCII fallbacks) and drop ones
//!      outside the [min_chars, max_chars] window.
//!
//!   4. For each surviving sentence, tokenise with vibrato, concatenate
//!      each token's reading field (katakana), then kata→hira. That
//!      hiragana string is the `reading`; the original sentence is the
//!      `surface`.
//!
//!   5. Emit row `{"reading","surface","context":"","source":"whitepaper",
//!       "sentence_id":"whitepaper:<ministry>:<file>:<idx>"}`.
//!
//! Single-threaded: vibrato's Worker is !Send, and 77MB of text finishes in
//! a few minutes anyway.

use anyhow::{bail, Context, Result};
use data_core::kana::kata_to_hira;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use vibrato::{Dictionary, Tokenizer};

#[derive(Serialize)]
struct Row<'a> {
    reading: String,
    surface: &'a str,
    context: &'static str,
    source: &'static str,
    sentence_id: String,
}

pub fn run(
    text_dir: &Path,
    out: &Path,
    dict_path: &Path,
    min_chars: usize,
    max_chars: usize,
) -> Result<()> {
    eprintln!("[ingest] loading vibrato dict: {}", dict_path.display());
    let dict_file =
        File::open(dict_path).with_context(|| format!("open dict {}", dict_path.display()))?;
    let decoder = zstd::stream::Decoder::new(dict_file).context("open zstd stream on dict")?;
    let dict = Dictionary::read(decoder).context("parse vibrato dict")?;
    let tokenizer = Tokenizer::new(dict);
    let mut worker = tokenizer.new_worker();

    // Collect all text files.
    let mut jobs: Vec<(String, PathBuf)> = Vec::new();
    for ministry_entry in
        fs::read_dir(text_dir).with_context(|| format!("read_dir {}", text_dir.display()))?
    {
        let ministry_entry = ministry_entry?;
        if !ministry_entry.file_type()?.is_dir() {
            continue;
        }
        let ministry = ministry_entry.file_name().to_string_lossy().to_string();
        for txt_entry in fs::read_dir(ministry_entry.path())? {
            let txt_entry = txt_entry?;
            let path = txt_entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("txt") {
                jobs.push((ministry.clone(), path));
            }
        }
    }
    eprintln!("[ingest] {} text files to process", jobs.len());
    if jobs.is_empty() {
        bail!("no text files under {}", text_dir.display());
    }

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut writer = BufWriter::with_capacity(
        8 * 1024 * 1024,
        File::create(out).with_context(|| format!("create {}", out.display()))?,
    );

    let mut emitted: u64 = 0;
    let mut skipped_file_garbled: u64 = 0;
    let mut skipped_len: u64 = 0;
    let mut skipped_reading: u64 = 0;
    let mut line_buf: Vec<u8> = Vec::with_capacity(512);
    let mut reading_buf = String::with_capacity(512);

    let files_total = jobs.len();
    for (file_idx, (ministry, path)) in jobs.into_iter().enumerate() {
        if file_idx % 50 == 0 {
            eprintln!(
                "[ingest] {}/{} files, emitted={}",
                file_idx, files_total, emitted
            );
        }
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[ingest] skip {}: {}", path.display(), e);
                continue;
            }
        };
        if is_garbled(&text) {
            skipped_file_garbled += 1;
            continue;
        }
        let normalised = normalise(&text);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut idx_in_file = 0usize;
        for sentence in iter_sentences(&normalised) {
            let n = sentence.chars().count();
            if n < min_chars || n > max_chars {
                skipped_len += 1;
                continue;
            }
            if !looks_japanese(sentence) {
                skipped_len += 1;
                continue;
            }
            worker.reset_sentence(sentence);
            worker.tokenize();
            reading_buf.clear();
            let mut ok = true;
            for token in worker.token_iter() {
                let feature = token.feature();
                if let Some(r) = parse_reading(feature) {
                    reading_buf.push_str(r);
                    continue;
                }
                // Fallback: token surface itself if already kana.
                let surf = token.surface();
                if surf.chars().all(is_kana) {
                    reading_buf.push_str(&surf);
                    continue;
                }
                ok = false;
                break;
            }
            if !ok {
                skipped_reading += 1;
                continue;
            }
            let reading = kata_to_hira(&reading_buf);
            let row = Row {
                reading,
                surface: sentence,
                context: "",
                source: "whitepaper",
                sentence_id: format!("whitepaper:{}:{}:{}", ministry, stem, idx_in_file),
            };
            line_buf.clear();
            serde_json::to_writer(&mut line_buf, &row).context("serialize row")?;
            line_buf.push(b'\n');
            writer.write_all(&line_buf)?;
            emitted += 1;
            idx_in_file += 1;
        }
    }
    writer.flush()?;
    eprintln!(
        "[ingest] done. emitted={} skipped_len={} skipped_reading={} skipped_file_garbled={}",
        emitted, skipped_len, skipped_reading, skipped_file_garbled
    );
    Ok(())
}

/// Extract the `reading` field from an IPADIC feature string.
/// Format: `品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音`
fn parse_reading(feature: &str) -> Option<&str> {
    feature
        .split(',')
        .nth(7)
        .filter(|s| !s.is_empty() && *s != "*")
}

fn is_kana(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x309F |
        0x30A0..=0x30FF |
        0x31F0..=0x31FF
    )
}

fn is_japanese_char(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x309F |
        0x30A0..=0x30FF |
        0x4E00..=0x9FFF |
        0x3400..=0x4DBF |
        0xFF61..=0xFF9F
    )
}

fn is_garbled(text: &str) -> bool {
    let mut jp = 0usize;
    let mut total = 0usize;
    for c in text.chars().take(4000) {
        if c.is_whitespace() || c.is_ascii_punctuation() || c.is_ascii_digit() {
            continue;
        }
        total += 1;
        if is_japanese_char(c) {
            jp += 1;
        }
    }
    if total < 100 {
        return false;
    }
    (jp as f64) / (total as f64) < 0.05
}

fn looks_japanese(s: &str) -> bool {
    let mut jp = 0usize;
    let mut total = 0usize;
    for c in s.chars() {
        if c.is_whitespace() {
            continue;
        }
        total += 1;
        if is_japanese_char(c) {
            jp += 1;
        }
    }
    if total == 0 {
        return false;
    }
    (jp as f64) / (total as f64) >= 0.5
}

fn normalise(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_was_space = false;
    for c in text.chars() {
        if c == '\u{3000}' {
            if !prev_was_space {
                out.push(' ');
                prev_was_space = true;
            }
            continue;
        }
        if c.is_whitespace() && c != '\n' {
            if !prev_was_space {
                out.push(' ');
                prev_was_space = true;
            }
            continue;
        }
        if c == '\n' {
            out.push('\n');
            prev_was_space = false;
            continue;
        }
        if (c as u32) < 0x20 {
            continue;
        }
        out.push(c);
        prev_was_space = false;
    }
    out
}

fn iter_sentences(text: &str) -> SentenceIter<'_> {
    SentenceIter { rest: text }
}

struct SentenceIter<'a> {
    rest: &'a str,
}

impl<'a> Iterator for SentenceIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let s = self.rest;
            if s.is_empty() {
                return None;
            }
            let mut end_byte = s.len();
            for (i, c) in s.char_indices() {
                if matches!(c, '。' | '！' | '？' | '?' | '!' | '\n') {
                    end_byte = i + c.len_utf8();
                    // Consume any immediate closing brackets.
                    let mut j = end_byte;
                    while j < s.len() {
                        let tail = &s[j..];
                        if let Some(cc) = tail.chars().next() {
                            if matches!(cc, '」' | '』' | ')' | '）' | ']' | '］' | '》') {
                                j += cc.len_utf8();
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    end_byte = j;
                    break;
                }
            }
            let sentence = s[..end_byte].trim();
            self.rest = &s[end_byte..];
            if sentence.is_empty() {
                continue;
            }
            return Some(sentence);
        }
    }
}
