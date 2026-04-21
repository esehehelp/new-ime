use crate::shard::{MAGIC, VERSION};
use anyhow::{Context, Result};
use datacore::JsonlLines;
use kkc_tokenizer::SharedCharTokenizer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub max_context_chars: usize,
    pub max_reading_tokens: usize,
    pub max_surface_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetadata {
    pub shard_version: u32,
    pub row_count: usize,
    pub max_context_chars: usize,
    pub max_reading_tokens: usize,
    pub max_surface_tokens: usize,
    pub vocab_size: usize,
    pub sources: BTreeMap<String, u32>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            max_context_chars: 40,
            max_reading_tokens: 128,
            max_surface_tokens: 128,
        }
    }
}

pub fn compile_jsonl_to_shard(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    tokenizer: &SharedCharTokenizer,
    options: &CompileOptions,
) -> Result<ShardMetadata> {
    let mut payload = Vec::<u8>::new();
    let mut offsets = Vec::<u64>::new();
    let mut source_table = BTreeMap::<String, u32>::new();
    for row in JsonlLines::open(input.as_ref()).context("open jsonl")? {
        let row = row?;
        let context: String = row
            .context
            .chars()
            .rev()
            .take(options.max_context_chars)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let mut reading = tokenizer.encode(&row.reading);
        let mut surface = tokenizer.encode(&row.surface);
        let mut context_ids = tokenizer.encode(&context);
        reading.truncate(options.max_reading_tokens);
        surface.truncate(options.max_surface_tokens);
        context_ids.truncate(options.max_context_chars.max(1));
        let source_id = match row.source {
            Some(source) => {
                let next = source_table.len() as u32 + 1;
                *source_table.entry(source).or_insert(next)
            }
            None => 0,
        };
        offsets.push(payload.len() as u64);
        write_vec(&mut payload, &reading, &surface, &context_ids, source_id)?;
    }
    let payload_offset = 36u64;
    let index_offset = payload_offset + payload.len() as u64;
    let mut out = Vec::<u8>::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(offsets.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload_offset.to_le_bytes());
    out.extend_from_slice(&index_offset.to_le_bytes());
    out.extend_from_slice(&payload);
    for offset in &offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    if let Some(parent) = output.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
    }
    std::fs::write(output.as_ref(), out)
        .with_context(|| format!("write shard {}", output.as_ref().display()))?;
    Ok(ShardMetadata {
        shard_version: VERSION,
        row_count: offsets.len(),
        max_context_chars: options.max_context_chars,
        max_reading_tokens: options.max_reading_tokens,
        max_surface_tokens: options.max_surface_tokens,
        vocab_size: tokenizer.vocab_size(),
        sources: source_table,
    })
}

fn write_vec(
    out: &mut Vec<u8>,
    reading: &[u32],
    surface: &[u32],
    context: &[u32],
    source_id: u32,
) -> Result<()> {
    out.write_all(&(reading.len() as u32).to_le_bytes())?;
    out.write_all(&(surface.len() as u32).to_le_bytes())?;
    out.write_all(&(context.len() as u32).to_le_bytes())?;
    out.write_all(&source_id.to_le_bytes())?;
    for token in reading.iter().chain(surface).chain(context) {
        out.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}
