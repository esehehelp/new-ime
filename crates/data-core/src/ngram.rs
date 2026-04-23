//! Character-level n-gram contamination set.
//!
//! The Python pipeline filters training pools against evaluation JSONL files
//! by collecting every length-n substring of every surface in the eval set
//! and rejecting training rows that share at least one of those substrings.
//! This replicates that exact behaviour.
//!
//! For n=6, a single eval file (2-10k rows) produces ~300k n-grams; the
//! process-time membership check is an O(|surface|) AHash lookup per row.

use ahash::AHashSet;
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;

use crate::jsonl::JsonlLines;

/// A set of character-level n-grams used to detect cross-set contamination.
#[derive(Default)]
pub struct NgramSet {
    n: usize,
    set: AHashSet<String>,
}

impl NgramSet {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            set: AHashSet::new(),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Insert every length-n substring of `surface`. Surfaces shorter than
    /// `n` are inserted as-is (matches the Python ``surface_sixgrams``
    /// fallback so short targets still register).
    pub fn insert_surface(&mut self, surface: &str) {
        insert_ngrams_into(&mut self.set, surface, self.n);
    }

    /// Ingest every row's `surface` from the JSON Lines file at `path`.
    ///
    /// Goes through `JsonlLines`, which transparently handles `.zst`, `.xz`,
    /// and `.gz` suffixes — so pre-compressed evaluation artifacts work
    /// without decompression first.
    pub fn extend_from_jsonl(&mut self, path: &Path) -> Result<()> {
        let lines = JsonlLines::open(path)?;
        for row_result in lines {
            let row = row_result?;
            self.insert_surface(&row.surface);
        }
        Ok(())
    }

    /// Build a fresh set from multiple evaluation files.
    pub fn build(n: usize, paths: &[impl AsRef<Path>]) -> Result<Self> {
        let mut set = Self::new(n);
        for p in paths {
            set.extend_from_jsonl(p.as_ref())?;
        }
        Ok(set)
    }

    /// True if any length-n substring of `surface` is in the contamination
    /// set. Mirrors ``contains_contaminated_ngram``.
    pub fn contains_overlap(&self, surface: &str) -> bool {
        if self.set.is_empty() {
            return false;
        }
        let chars: Vec<char> = surface.chars().collect();
        if chars.len() < self.n {
            return self.set.contains(surface);
        }
        let n = self.n;
        // Slide a window of `n` characters.
        let mut buf = String::with_capacity(n * 4);
        for start in 0..=chars.len() - n {
            buf.clear();
            for ch in &chars[start..start + n] {
                buf.push(*ch);
            }
            if self.set.contains(&buf) {
                return true;
            }
        }
        false
    }
}

fn insert_ngrams_into(set: &mut AHashSet<String>, surface: &str, n: usize) {
    let chars: Vec<char> = surface.chars().collect();
    if chars.len() < n {
        if !surface.is_empty() {
            set.insert(surface.to_string());
        }
        return;
    }
    for start in 0..=chars.len() - n {
        let slice: String = chars[start..start + n].iter().collect();
        set.insert(slice);
    }
}

/// Convenience: collect every length-n substring of `surface` into an ordered
/// BTreeSet. Used by tests and audit tooling, where deterministic ordering is
/// helpful.
pub fn surface_ngrams(surface: &str, n: usize) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = BTreeSet::new();
    let chars: Vec<char> = surface.chars().collect();
    if chars.len() < n {
        if !surface.is_empty() {
            out.insert(surface.to_string());
        }
        return out;
    }
    for start in 0..=chars.len() - n {
        out.insert(chars[start..start + n].iter().collect());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut s = NgramSet::new(6);
        s.insert_surface("今日はいい天気ですね");
        assert!(s.contains_overlap("今日はいい天気"));
        assert!(s.contains_overlap("今日はいい天気です"));
        // Overlaps share at least one 6-gram.
        assert!(s.contains_overlap("雨です。今日はいい天気ですね明日も"));
        // No 6-char overlap.
        assert!(!s.contains_overlap("全然別の文章"));
    }

    #[test]
    fn short_surface_exact_match() {
        let mut s = NgramSet::new(6);
        s.insert_surface("短い");
        assert!(s.contains_overlap("短い"));
        assert!(!s.contains_overlap("別の短い文"));
    }

    #[test]
    fn empty_set_never_matches() {
        let s = NgramSet::new(6);
        assert!(!s.contains_overlap("anything"));
    }

    #[test]
    fn surface_ngrams_order() {
        let ngrams = surface_ngrams("あいうえおかき", 3);
        let collected: Vec<String> = ngrams.into_iter().collect();
        assert_eq!(
            collected,
            vec!["あいう", "いうえ", "うえお", "えおか", "おかき"]
        );
    }
}
