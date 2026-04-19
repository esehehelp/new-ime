//! Shared primitives for the Rust replacement of the Python data pipeline.
//!
//! - `jsonl`: streaming line-oriented JSONL reader/writer with optional
//!   zstd/xz/gzip compression on the output side.
//! - `ngram`: build-once 6-gram set over an evaluation JSONL, then O(n·k)
//!   substring membership test used for contamination filtering.
//! - `kana`: katakana → hiragana conversion keeping ASCII/symbols intact.
//! - `row`: the minimal `(reading, surface, context, source)` record that
//!   flows through every pool script.

pub mod jsonl;
pub mod kana;
pub mod ngram;
pub mod row;

pub use jsonl::{open_output, write_row, JsonlLines, OutputFormat};
pub use kana::kata_to_hira;
pub use ngram::NgramSet;
pub use row::Row;
