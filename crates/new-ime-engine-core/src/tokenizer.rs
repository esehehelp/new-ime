//! SharedCharTokenizer with `.vocab.hex.tsv` sidecar loader.
//!
//! Matches the Python SharedCharTokenizer used during training. Vocab file
//! format: one token per line, `id\thex(utf8_bytes)`.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SharedCharTokenizer {
    pub id_to_token: Vec<String>,
    pub token_to_id: HashMap<String, u32>,
    pub pad_id: u32,
    pub unk_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub blank_id: u32,
    pub mask_id: u32,
}

impl SharedCharTokenizer {
    pub fn load_vocab_hex_tsv(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let reader = BufReader::new(f);

        let mut token_to_id: HashMap<String, u32> = HashMap::new();
        let mut max_id: i64 = -1;
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let (id_str, hex_str) = match line.split_once('\t') {
                Some(pair) => pair,
                None => continue,
            };
            let id: u32 = id_str.parse()?;
            let token = decode_hex_bytes(hex_str)?;
            token_to_id.insert(token, id);
            if id as i64 > max_id {
                max_id = id as i64;
            }
        }
        anyhow::ensure!(max_id >= 0, "no tokens parsed from {}", path.display());

        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (tok, id) in &token_to_id {
            id_to_token[*id as usize] = tok.clone();
        }

        let lookup =
            |name: &str, default: u32| -> u32 { token_to_id.get(name).copied().unwrap_or(default) };
        Ok(Self {
            pad_id: lookup("[PAD]", 0),
            unk_id: lookup("[UNK]", 1),
            cls_id: lookup("[CLS]", 2),
            sep_id: lookup("[SEP]", 3),
            blank_id: lookup("[BLANK]", 4),
            mask_id: lookup("[MASK]", 5),
            id_to_token,
            token_to_id,
        })
    }

    pub fn encode_char(&self, ch: &str) -> u32 {
        self.token_to_id.get(ch).copied().unwrap_or(self.unk_id)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if matches!(
                id,
                x if x == self.pad_id
                    || x == self.cls_id
                    || x == self.sep_id
                    || x == self.blank_id
                    || x == self.mask_id
            ) {
                continue;
            }
            if let Some(tok) = self.id_to_token.get(id as usize) {
                out.push_str(tok);
            }
        }
        out
    }
}

fn decode_hex_bytes(hex: &str) -> Result<String> {
    anyhow::ensure!(hex.len() % 2 == 0, "odd-length hex");
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    let b = hex.as_bytes();
    for i in (0..hex.len()).step_by(2) {
        let hi = hex_nibble(b[i])?;
        let lo = hex_nibble(b[i + 1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(String::from_utf8(bytes)?)
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => anyhow::bail!("bad hex byte {}", c),
    }
}
