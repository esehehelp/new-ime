use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const SEP_TOKEN: &str = "[SEP]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const BLANK_TOKEN: &str = "[BLANK]";
pub const MASK_TOKEN: &str = "[MASK]";

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const SEP_ID: u32 = 2;
pub const CLS_ID: u32 = 3;
pub const BLANK_ID: u32 = 4;
pub const MASK_ID: u32 = 5;

const SPECIAL_TOKENS: [&str; 6] = [
    PAD_TOKEN,
    UNK_TOKEN,
    SEP_TOKEN,
    CLS_TOKEN,
    BLANK_TOKEN,
    MASK_TOKEN,
];

const KANA_MARKS: &str = "ー・ヽヾゝゞ";
const JP_SYMBOLS: &str = "　、。！？「」『』（）【】〔〕｛｝〈〉《》・ー～…‥々〇〻ヶヵ";
const HIRAGANA_START: u32 = 0x3041;
const HIRAGANA_END: u32 = 0x3096;
const KATAKANA_START: u32 = 0x30A1;
const KATAKANA_END: u32 = 0x30FA;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenizerDisk {
    #[serde(default)]
    token_to_id: BTreeMap<String, u32>,
}

#[derive(Debug, Clone)]
pub struct SharedCharTokenizer {
    token_to_id: BTreeMap<String, u32>,
    id_to_token: Vec<String>,
}

impl SharedCharTokenizer {
    pub fn new_default(max_kanji: u32) -> Self {
        let mut token_to_id = BTreeMap::new();
        let mut id_to_token = Vec::new();
        for token in SPECIAL_TOKENS {
            push_token(&mut token_to_id, &mut id_to_token, token.to_string());
        }
        for byte in 0u32..=255 {
            push_token(
                &mut token_to_id,
                &mut id_to_token,
                format!("<0x{byte:02X}>"),
            );
        }
        push_range(
            &mut token_to_id,
            &mut id_to_token,
            HIRAGANA_START,
            HIRAGANA_END,
        );
        push_range(
            &mut token_to_id,
            &mut id_to_token,
            KATAKANA_START,
            KATAKANA_END,
        );
        push_chars(&mut token_to_id, &mut id_to_token, KANA_MARKS);
        push_range(&mut token_to_id, &mut id_to_token, 0x20, 0x7E);
        push_range(&mut token_to_id, &mut id_to_token, 0xFF01, 0xFF5E);
        push_chars(&mut token_to_id, &mut id_to_token, JP_SYMBOLS);
        push_range(
            &mut token_to_id,
            &mut id_to_token,
            0x4E00,
            0x4E00 + max_kanji.saturating_sub(1),
        );
        Self {
            token_to_id,
            id_to_token,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("read tokenizer {}", path.as_ref().display()))?;
        let disk: TokenizerDisk = serde_json::from_str(&text).context("parse tokenizer json")?;
        let max_id = disk.token_to_id.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, id) in &disk.token_to_id {
            let slot = id_to_token
                .get_mut(*id as usize)
                .context("tokenizer id out of range")?;
            *slot = token.clone();
        }
        Ok(Self {
            token_to_id: disk.token_to_id,
            id_to_token,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let disk = TokenizerDisk {
            token_to_id: self.token_to_id.clone(),
        };
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("mkdir {}", parent.display()))?;
            }
        }
        let text = serde_json::to_string_pretty(&disk).context("serialize tokenizer")?;
        std::fs::write(path.as_ref(), text)
            .with_context(|| format!("write tokenizer {}", path.as_ref().display()))?;
        Ok(())
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let key = ch.to_string();
            if let Some(id) = self.token_to_id.get(&key) {
                ids.push(*id);
                continue;
            }
            for byte in ch.to_string().into_bytes() {
                let token = format!("<0x{byte:02X}>");
                ids.push(*self.token_to_id.get(&token).unwrap_or(&UNK_ID));
            }
        }
        ids
    }

    pub fn encode_with_special(&self, context: &str, reading: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        ids.push(CLS_ID);
        ids.extend(self.encode(context));
        ids.push(SEP_ID);
        ids.extend(self.encode(reading));
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut bytes = Vec::new();
        for id in ids {
            if matches!(*id, PAD_ID | CLS_ID | SEP_ID | BLANK_ID | MASK_ID) {
                continue;
            }
            let token = self
                .id_to_token
                .get(*id as usize)
                .map(String::as_str)
                .unwrap_or(UNK_TOKEN);
            if let Some(byte) = parse_byte_token(token) {
                bytes.push(byte);
                continue;
            }
            if !bytes.is_empty() {
                flush_bytes(&mut out, &mut bytes);
            }
            out.push_str(token);
        }
        if !bytes.is_empty() {
            flush_bytes(&mut out, &mut bytes);
        }
        out
    }
}

fn push_token(
    token_to_id: &mut BTreeMap<String, u32>,
    id_to_token: &mut Vec<String>,
    token: String,
) {
    if token_to_id.contains_key(&token) {
        return;
    }
    let id = id_to_token.len() as u32;
    token_to_id.insert(token.clone(), id);
    id_to_token.push(token);
}

fn push_range(
    token_to_id: &mut BTreeMap<String, u32>,
    id_to_token: &mut Vec<String>,
    start: u32,
    end: u32,
) {
    for cp in start..=end {
        if let Some(ch) = char::from_u32(cp) {
            push_token(token_to_id, id_to_token, ch.to_string());
        }
    }
}

fn push_chars(token_to_id: &mut BTreeMap<String, u32>, id_to_token: &mut Vec<String>, text: &str) {
    for ch in text.chars() {
        push_token(token_to_id, id_to_token, ch.to_string());
    }
}

fn parse_byte_token(token: &str) -> Option<u8> {
    token
        .strip_prefix("<0x")
        .and_then(|tail| tail.strip_suffix('>'))
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
}

fn flush_bytes(out: &mut String, bytes: &mut Vec<u8>) {
    match String::from_utf8(std::mem::take(bytes)) {
        Ok(text) => out.push_str(&text),
        Err(_) => out.push('〓'),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ascii_and_japanese() {
        let tok = SharedCharTokenizer::new_default(64);
        let text = "abcかな漢字🙂";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn adds_special_prefix() {
        let tok = SharedCharTokenizer::new_default(32);
        let ids = tok.encode_with_special("前", "かな");
        assert_eq!(ids[0], CLS_ID);
        assert!(ids.contains(&SEP_ID));
    }
}
