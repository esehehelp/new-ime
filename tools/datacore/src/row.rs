//! Minimal per-sample record produced by processors and consumed by the
//! mixer / training loop.
//!
//! We deliberately preserve the exact field set and ordering used by the
//! Python scripts so that old consumers (KanaKanjiDataset, audit_pools, etc.)
//! remain byte-compatible during the migration.

use serde::{Deserialize, Serialize};

/// One kana-kanji pair as written to `*.jsonl` training files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Row {
    pub reading: String,
    pub surface: String,
    #[serde(default)]
    pub context: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl Row {
    pub fn new(reading: String, surface: String, context: String, source: Option<String>) -> Self {
        Self {
            reading,
            surface,
            context,
            source,
        }
    }
}
