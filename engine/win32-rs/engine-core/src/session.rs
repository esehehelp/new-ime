//! Engine session: ONNX model + tokenizer holder + greedy decode entry point.
//!
//! KenLM integration and prefix beam search are left as TODO — for the first
//! port we ship greedy CTC so the TSF DLL can light up with baseline quality.
//! That keeps the initial migration narrow: once the DLL is stable we swap in
//! beam + KenLM without touching TSF glue.

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};

use std::collections::HashMap;

use crate::beam::{prefix_beam_search, BeamHypothesis};
use crate::kenlm::{CategoryEstimator, KenLMCharScorer, KenLMMixture, LmScorer};
use crate::tokenizer::SharedCharTokenizer;

fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow!("ort: {}", e)
}

pub struct EngineSession {
    pub tokenizer: SharedCharTokenizer,
    session: Session,
    pub seq_len: usize,
    pub max_context: usize,
    /// Internal beam search width. Larger = better homophone recall at the
    /// cost of linear beam-search time. At `30` a typical reading runs
    /// beam in ~8-12 ms on CPU, still dominated by the ~25 ms ONNX forward.
    pub beam_width: usize,
    /// Per-step expansion breadth. `32` keeps rare kanji tokens alive long
    /// enough for the fused (CTC + KenLM) score to rescue them — at 16 the
    /// beam would often prune e.g. `至` before it could combine with `高`.
    pub top_k_per_step: usize,
    pub lm_alpha: f32,
    pub lm_beta: f32,
    /// Single-domain scorer. Mutually exclusive with `lm_moe`; whichever
    /// is Some() is used for shallow fusion.
    pub lm: Option<KenLMCharScorer>,
    /// MoE scorer — loaded via `attach_kenlm_moe`. Takes precedence over
    /// `lm` when both are Some.
    pub lm_moe: Option<KenLMMixture>,
    /// Estimator used to re-weight the MoE per input reading. Only
    /// meaningful when `lm_moe` is active.
    pub category_estimator: Option<CategoryEstimator>,
    /// Max candidates returned to the UI. Decoupled from `beam_width` so we
    /// can search wide but display a compact list.
    pub max_candidates: usize,
}

impl EngineSession {
    pub fn load(onnx_path: &Path) -> Result<Self> {
        let tsv_path = derive_vocab_hex_tsv_path(onnx_path);
        let tokenizer = SharedCharTokenizer::load_vocab_hex_tsv(&tsv_path)
            .with_context(|| format!("load tokenizer vocab {}", tsv_path.display()))?;

        let session = Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(4)
            .map_err(ort_err)?
            .commit_from_file(onnx_path)
            .map_err(ort_err)?;

        Ok(Self {
            tokenizer,
            session,
            seq_len: 128,
            max_context: 40,
            beam_width: env_usize("NEWIME_BEAM_WIDTH", 50),
            top_k_per_step: env_usize("NEWIME_TOP_K", 64),
            lm_alpha: env_f32("NEWIME_LM_ALPHA", 0.3),
            lm_beta: env_f32("NEWIME_LM_BETA", 0.7),
            lm: None,
            lm_moe: None,
            category_estimator: None,
            // Default matches `beam_width` so the UI can page through the
            // full beam instead of silently truncating the tail.
            max_candidates: env_usize("NEWIME_MAX_CANDIDATES", 50),
        })
    }

    /// Attach a single-domain KenLM scorer. Clears any MoE.
    pub fn attach_kenlm(&mut self, lm_path: &Path) -> Result<()> {
        let model = crate::kenlm::KenLM::load(lm_path)
            .with_context(|| format!("load kenlm {}", lm_path.display()))?;
        self.lm = Some(KenLMCharScorer::new(model, self.tokenizer.clone()));
        self.lm_moe = None;
        self.category_estimator = None;
        Ok(())
    }

    /// Attach an MoE of character-level KenLMs (e.g. general / tech /
    /// entity). `category_estimator` becomes the default
    /// `{general, tech, entity}` estimator — the caller can overwrite
    /// `self.category_estimator` afterwards to customise the profile.
    pub fn attach_kenlm_moe(
        &mut self,
        paths: &HashMap<String, std::path::PathBuf>,
    ) -> Result<()> {
        let moe = KenLMMixture::load(paths, self.tokenizer.clone())
            .with_context(|| "load kenlm mixture")?;
        self.category_estimator = Some(CategoryEstimator::new(moe.domains().to_vec()));
        self.lm_moe = Some(moe);
        self.lm = None;
        Ok(())
    }

    /// Top-N kanji candidates for a hiragana `reading` given prior `context`.
    /// Runs ONNX once + CTC prefix beam + optional KenLM shallow fusion.
    pub fn convert(&mut self, context: &str, reading: &str) -> Result<Vec<String>> {
        if reading.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, mask, actual) = self.encode(context, reading);
        let ids_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), ids)?;
        let mask_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), mask)?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(ort_err)?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(ort_err)?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(ort_err)?;
        let (shape, flat) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let dims = shape.as_ref();
        anyhow::ensure!(dims.len() == 3, "unexpected logits rank {}", dims.len());
        let vocab = dims[2] as usize;

        // Extract logits for the valid range and convert to log-softmax.
        let mut log_probs = Array2::<f32>::from_elem((actual, vocab), 0.0);
        for t in 0..actual {
            let row_start = t * vocab;
            // log_softmax: x - logsumexp
            let row = &flat[row_start..row_start + vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &v in row {
                sum += (v - max).exp();
            }
            let lse = max + sum.ln();
            for v in 0..vocab {
                log_probs[[t, v]] = row[v] - lse;
            }
        }

        // Pick the active scorer. MoE wins if both are attached; the
        // category estimator sets per-reading weights before scoring.
        let scorer: Option<&dyn LmScorer> = if let Some(moe) = &self.lm_moe {
            if let Some(est) = &self.category_estimator {
                let w = est.estimate(reading, context);
                moe.set_weights(&w);
            }
            Some(moe as &dyn LmScorer)
        } else {
            self.lm.as_ref().map(|s| s as &dyn LmScorer)
        };

        let hyps: Vec<BeamHypothesis> = prefix_beam_search(
            &log_probs,
            self.tokenizer.blank_id,
            self.beam_width,
            self.top_k_per_step,
            scorer,
            self.lm_alpha,
            self.lm_beta,
        );

        // Dedup decoded strings (beam can end with the same text via
        // different token paths) and cap at `max_candidates` — the wider
        // internal beam is for recall; the UI still wants a compact list.
        let mut out: Vec<String> = Vec::with_capacity(self.max_candidates);
        let mut seen = std::collections::HashSet::new();
        for h in hyps {
            if out.len() >= self.max_candidates {
                break;
            }
            let text = normalize_candidate(&self.tokenizer.decode(&h.tokens));
            if !text.is_empty() && seen.insert(text.clone()) {
                out.push(text);
            }
        }
        Ok(out)
    }

    pub fn encode(&self, context: &str, reading: &str) -> (Vec<i64>, Vec<i64>, usize) {
        let ctx_chars: Vec<String> = chars(context);
        let take_ctx = ctx_chars.len().saturating_sub(self.max_context);
        let reading_chars = chars(reading);

        let mut ids: Vec<i64> = Vec::with_capacity(self.seq_len);
        ids.push(self.tokenizer.cls_id as i64);
        for ch in &ctx_chars[take_ctx..] {
            ids.push(self.tokenizer.encode_char(ch) as i64);
        }
        ids.push(self.tokenizer.sep_id as i64);
        for ch in &reading_chars {
            ids.push(self.tokenizer.encode_char(ch) as i64);
        }
        if ids.len() > self.seq_len {
            ids.truncate(self.seq_len);
        }
        let actual = ids.len();
        let mut mask: Vec<i64> = vec![0; self.seq_len];
        for i in 0..actual {
            mask[i] = 1;
        }
        while ids.len() < self.seq_len {
            ids.push(self.tokenizer.pad_id as i64);
        }
        (ids, mask, actual)
    }

    pub fn greedy_decode(&mut self, context: &str, reading: &str) -> Result<String> {
        if reading.is_empty() {
            return Ok(String::new());
        }
        let (ids, mask, actual) = self.encode(context, reading);
        let ids_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), ids)?;
        let mask_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), mask)?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(ort_err)?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(ort_err)?;
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(ort_err)?;
        let (shape, flat) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let dims = shape.as_ref();
        anyhow::ensure!(dims.len() == 3, "unexpected logits rank {}", dims.len());
        let vocab = dims[2] as usize;

        let mut decoded: Vec<u32> = Vec::with_capacity(actual);
        let mut last: i64 = -1;
        for t in 0..actual {
            let base = t * vocab;
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for v in 0..vocab {
                let val = flat[base + v];
                if val > best_v {
                    best_v = val;
                    best = v;
                }
            }
            let bid = best as i64;
            if bid != last && (best as u32) != self.tokenizer.blank_id {
                decoded.push(best as u32);
            }
            last = bid;
        }
        Ok(self.tokenizer.decode(&decoded))
    }
}

fn chars(s: &str) -> Vec<String> {
    s.chars().map(|c| c.to_string()).collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Normalise a decoded candidate string before it reaches the IME UI:
///   * Full-width ASCII (U+FF01..U+FF5E) → half-width equivalent. Model
///     output often contains full-width latin letters (e.g. `ＡＩ` for
///     `えーあい`); acronyms / English terms should look like plain ASCII
///     (`AI`), preserving case — so the previous aggressive lowercase rule
///     has been dropped.
///   * Comma-family characters (`,` / `，`) → `、` (ideographic comma).
///   * Period-family characters (`.` / `．`) → `。` (ideographic period).
///     These are unified because the keyboard key that produces them maps
///     to `、` / `。` during IME input, so the conversion output should
///     never disagree with what the user sees while composing.
///   * Whitespace-only or empty results collapse to empty string so the
///     caller can filter them out.
fn normalize_candidate(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| match c {
            ',' | '\u{FF0C}' => '\u{3001}',
            '.' | '\u{FF0E}' => '\u{3002}',
            '\u{FF01}'..='\u{FF5E}' => {
                char::from_u32(c as u32 - 0xFEE0).unwrap_or(c)
            }
            other => other,
        })
        .collect();
    if mapped.chars().all(|c| c.is_whitespace()) {
        return String::new();
    }
    mapped
}

#[cfg(test)]
mod tests {
    use super::normalize_candidate;

    #[test]
    fn fullwidth_ascii_to_halfwidth_preserves_case() {
        assert_eq!(normalize_candidate("ＡＩ"), "AI");
        assert_eq!(normalize_candidate("ａｂｃ"), "abc");
        assert_eq!(normalize_candidate("Ａpple"), "Apple");
        assert_eq!(normalize_candidate("PC は"), "PC は");
    }

    #[test]
    fn whitespace_only_collapses() {
        assert_eq!(normalize_candidate("  "), "");
    }

    #[test]
    fn unifies_punctuation_family() {
        assert_eq!(normalize_candidate("a,b.c"), "a、b。c");
        assert_eq!(normalize_candidate("あ，い．う"), "あ、い。う");
        assert_eq!(normalize_candidate("今日は、"), "今日は、");
    }
}

fn derive_vocab_hex_tsv_path(onnx_path: &Path) -> PathBuf {
    let stem = onnx_path.with_extension("");
    let direct = PathBuf::from(format!("{}.tokenizer.json.vocab.hex.tsv", stem.display()));
    if direct.exists() {
        return direct;
    }
    let stem_str = stem.display().to_string();
    for suffix in [".int8", ".fp32"] {
        if let Some(base) = stem_str.strip_suffix(suffix) {
            let alt = PathBuf::from(format!("{}.fp32.tokenizer.json.vocab.hex.tsv", base));
            if alt.exists() {
                return alt;
            }
        }
    }
    direct
}
