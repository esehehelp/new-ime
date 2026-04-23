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

#[derive(Debug, Clone, Copy)]
pub struct ProposalFrame {
    pub top1_id: u32,
    pub top1_log_prob: f32,
    pub top2_log_prob: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollapsedToken {
    pub token_id: u32,
    pub start_frame: usize,
    pub end_frame: usize,
    pub min_log_prob: f32,
    pub mean_log_prob: f32,
    pub min_margin: f32,
    pub mean_margin: f32,
}

impl CollapsedToken {
    pub fn frame_count(&self) -> usize {
        self.end_frame + 1 - self.start_frame
    }

    pub fn confidence(&self) -> f32 {
        self.min_log_prob.exp()
    }
}

#[derive(Debug)]
pub struct RefinerOutput {
    pub logits: Array2<f32>,
    /// Per-position logit for the learned re-mask head. `None` for older
    /// (v1) exports that don't include it.
    pub remask_logits: Option<Vec<f32>>,
    /// Scalar logit from the learned stop head. `None` for older exports.
    pub stop_logit: Option<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug)]
pub struct ProposalOutput {
    pub log_probs: Array2<f32>,
    pub frames: Vec<ProposalFrame>,
    pub input_len: usize,
}

impl ProposalOutput {
    pub fn collapsed_tokens(&self, blank_id: u32) -> Vec<u32> {
        collapse_frame_ids(
            &self.frames.iter().map(|f| f.top1_id).collect::<Vec<_>>(),
            blank_id,
        )
    }

    pub fn collapsed_with_alignment(&self, blank_id: u32) -> Vec<CollapsedToken> {
        collapse_frames_with_alignment(&self.frames, blank_id)
    }

    pub fn select_mask_positions(
        &self,
        blank_id: u32,
        confidence_threshold: f32,
        max_masks: usize,
    ) -> Vec<usize> {
        select_mask_positions(
            &self.collapsed_with_alignment(blank_id),
            confidence_threshold,
            max_masks,
        )
    }
}

pub struct EngineSession {
    pub tokenizer: SharedCharTokenizer,
    proposal_session: Session,
    refiner_session: Option<Session>,
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
    /// Off by default until a trained refiner exists. When enabled and the
    /// sibling `*.refiner.onnx` artifact is present, the engine will run one
    /// masked refinement pass after the CTC proposal path.
    pub enable_refiner: bool,
    pub refiner_confidence_threshold: f32,
    pub refiner_max_masks: usize,
    /// Number of mask-refinement passes to apply at inference time. Default
    /// 1 matches the legacy one-shot behaviour; 2-3 exercises the iterative
    /// loop (learned remask/stop heads) when the model was trained with
    /// `--refine-iterations > 1`.
    pub refiner_iterations: usize,
    /// Sigmoid threshold on the learned remask head. Positions with
    /// P(remask) >= threshold get masked for the next iteration.
    pub refiner_remask_threshold: f32,
    /// Sigmoid threshold on the learned stop head. When P(done) >=
    /// threshold we stop iterating early for the current row.
    pub refiner_stop_threshold: f32,
}

impl EngineSession {
    pub fn load(onnx_path: &Path) -> Result<Self> {
        let tsv_path = derive_vocab_hex_tsv_path(onnx_path);
        let tokenizer = SharedCharTokenizer::load_vocab_hex_tsv(&tsv_path)
            .with_context(|| format!("load tokenizer vocab {}", tsv_path.display()))?;

        let proposal_session = Session::builder()
            .map_err(ort_err)?
            .with_intra_threads(4)
            .map_err(ort_err)?
            .commit_from_file(onnx_path)
            .map_err(ort_err)?;
        let refiner_path = derive_refiner_onnx_path(onnx_path);
        let refiner_session = if refiner_path.exists() {
            Some(
                Session::builder()
                    .map_err(ort_err)?
                    .with_intra_threads(4)
                    .map_err(ort_err)?
                    .commit_from_file(&refiner_path)
                    .map_err(ort_err)?,
            )
        } else {
            None
        };

        Ok(Self {
            tokenizer,
            proposal_session,
            refiner_session,
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
            enable_refiner: std::env::var("NEWIME_ENABLE_REFINER")
                .ok()
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
                .unwrap_or(false),
            refiner_confidence_threshold: env_f32("NEWIME_REFINER_CONF_THRESHOLD", 0.45),
            refiner_max_masks: env_usize("NEWIME_REFINER_MAX_MASKS", 2),
            refiner_iterations: env_usize("NEWIME_REFINER_ITERATIONS", 1),
            refiner_remask_threshold: env_f32("NEWIME_REFINER_REMASK_THRESHOLD", 0.5),
            refiner_stop_threshold: env_f32("NEWIME_REFINER_STOP_THRESHOLD", 0.5),
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

    pub fn has_refiner(&self) -> bool {
        self.refiner_session.is_some()
    }

    pub fn configure_refiner(
        &mut self,
        enabled: bool,
        confidence_threshold: f32,
        max_masks: usize,
    ) {
        self.enable_refiner = enabled;
        self.refiner_confidence_threshold = confidence_threshold;
        self.refiner_max_masks = max_masks;
    }

    pub fn configure_refiner_iter(
        &mut self,
        iterations: usize,
        remask_threshold: f32,
        stop_threshold: f32,
    ) {
        self.refiner_iterations = iterations.max(1);
        self.refiner_remask_threshold = remask_threshold;
        self.refiner_stop_threshold = stop_threshold;
    }

    /// Attach an MoE of character-level KenLMs (e.g. general / tech /
    /// entity). `category_estimator` becomes the default
    /// `{general, tech, entity}` estimator — the caller can overwrite
    /// `self.category_estimator` afterwards to customise the profile.
    pub fn attach_kenlm_moe(&mut self, paths: &HashMap<String, std::path::PathBuf>) -> Result<()> {
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
        let proposal = self.run_proposal(context, reading)?;
        if self.enable_refiner && self.refiner_session.is_some() && self.beam_width <= 1 {
            let refined = if self.refiner_iterations > 1 {
                self.refine_iterative(context, reading, &proposal, self.refiner_iterations)?
            } else {
                self.refine_greedy(context, reading, &proposal)?
            };
            if !refined.is_empty() {
                return Ok(vec![refined]);
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
            &proposal.log_probs,
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

    pub fn run_proposal(&mut self, context: &str, reading: &str) -> Result<ProposalOutput> {
        let (ids, mask, actual) = self.encode(context, reading);
        let ids_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), ids)?;
        let mask_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), mask)?;

        let ids_tensor = Tensor::from_array(ids_arr).map_err(ort_err)?;
        let mask_tensor = Tensor::from_array(mask_arr).map_err(ort_err)?;
        let outputs = self
            .proposal_session
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

        let mut log_probs = Array2::<f32>::from_elem((actual, vocab), 0.0);
        let mut frames = Vec::with_capacity(actual);
        for t in 0..actual {
            let row_start = t * vocab;
            let row = &flat[row_start..row_start + vocab];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &v in row {
                sum += (v - max).exp();
            }
            let lse = max + sum.ln();
            let mut top1_idx = 0usize;
            let mut top1_lp = f32::NEG_INFINITY;
            let mut top2_lp = f32::NEG_INFINITY;
            for v in 0..vocab {
                let lp = row[v] - lse;
                log_probs[[t, v]] = lp;
                if lp > top1_lp {
                    top2_lp = top1_lp;
                    top1_lp = lp;
                    top1_idx = v;
                } else if lp > top2_lp {
                    top2_lp = lp;
                }
            }
            if vocab == 1 {
                top2_lp = top1_lp;
            }
            frames.push(ProposalFrame {
                top1_id: top1_idx as u32,
                top1_log_prob: top1_lp,
                top2_log_prob: top2_lp,
            });
        }

        Ok(ProposalOutput {
            log_probs,
            frames,
            input_len: actual,
        })
    }

    pub fn run_refiner(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
        hypothesis_ids: &[i64],
        hypothesis_attention_mask: &[i64],
        hyp_len: usize,
    ) -> Result<RefinerOutput> {
        let Some(refiner_session) = &mut self.refiner_session else {
            anyhow::bail!("refiner session not loaded");
        };
        let ids_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), input_ids.to_vec())?;
        let mask_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), attention_mask.to_vec())?;
        let hyp_arr = Array2::<i64>::from_shape_vec((1, self.seq_len), hypothesis_ids.to_vec())?;
        let hyp_mask_arr =
            Array2::<i64>::from_shape_vec((1, self.seq_len), hypothesis_attention_mask.to_vec())?;
        let outputs = refiner_session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(ids_arr).map_err(ort_err)?,
                "attention_mask" => Tensor::from_array(mask_arr).map_err(ort_err)?,
                "hypothesis_ids" => Tensor::from_array(hyp_arr).map_err(ort_err)?,
                "hypothesis_attention_mask" => Tensor::from_array(hyp_mask_arr).map_err(ort_err)?,
            ])
            .map_err(ort_err)?;
        let (shape, flat) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_err)?;
        let dims = shape.as_ref();
        anyhow::ensure!(
            dims.len() == 3,
            "unexpected refiner logits rank {}",
            dims.len()
        );
        let vocab = dims[2] as usize;
        let mut logits = Array2::<f32>::zeros((hyp_len, vocab));
        for t in 0..hyp_len {
            let row_start = t * vocab;
            for v in 0..vocab {
                logits[[t, v]] = flat[row_start + v];
            }
        }
        // Optional: learned remask + stop heads (present only for models
        // exported with artifact_version >= 2). Older graphs return just
        // "logits"; in that case we fall back to confidence heuristics
        // on the Rust side.
        let remask_logits = outputs
            .get("remask_logits")
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_shape, flat)| flat[..hyp_len].to_vec());
        let stop_logit = outputs
            .get("stop_logit")
            .and_then(|v| v.try_extract_tensor::<f32>().ok())
            .map(|(_shape, flat)| flat[0]);
        Ok(RefinerOutput {
            logits,
            remask_logits,
            stop_logit,
        })
    }

    /// Run up to `max_iterations` rounds of masked refinement using the
    /// learned re-mask + stop heads when present, falling back to the
    /// confidence heuristic otherwise. Returns the decoded text after the
    /// final iteration.
    pub fn refine_iterative(
        &mut self,
        context: &str,
        reading: &str,
        proposal: &ProposalOutput,
        max_iterations: usize,
    ) -> Result<String> {
        let collapsed = proposal.collapsed_with_alignment(self.tokenizer.blank_id);
        if collapsed.is_empty() {
            return Ok(String::new());
        }
        let hyp_len = collapsed.len();
        let mut hypothesis_ids: Vec<i64> = vec![self.tokenizer.pad_id as i64; self.seq_len];
        let mut hypothesis_mask: Vec<i64> = vec![0i64; self.seq_len];
        for (idx, token) in collapsed.iter().enumerate() {
            hypothesis_ids[idx] = token.token_id as i64;
            hypothesis_mask[idx] = 1;
        }
        // Initial mask: pick low-confidence positions from the collapsed
        // proposal. Subsequent iterations overwrite this based on head
        // output (learned) or logit confidence (fallback).
        let mut mask_positions = select_mask_positions(
            &collapsed,
            self.refiner_confidence_threshold,
            self.refiner_max_masks,
        );
        if mask_positions.is_empty() {
            // Nothing to do. Return the proposal decoded text.
            let decoded: Vec<u32> = collapsed.iter().map(|t| t.token_id).collect();
            return Ok(normalize_candidate(&self.tokenizer.decode(&decoded)));
        }
        for pos in &mask_positions {
            hypothesis_ids[*pos] = self.tokenizer.mask_id as i64;
        }

        let (input_ids, input_mask, _) = self.encode(context, reading);
        let iters = max_iterations.max(1);
        for it in 0..iters {
            let refine = self.run_refiner(
                &input_ids,
                &input_mask,
                &hypothesis_ids,
                &hypothesis_mask,
                hyp_len,
            )?;
            // Fill the current mask positions with argmax.
            for &pos in &mask_positions {
                let row = refine.logits.row(pos);
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for (idx, &val) in row.iter().enumerate() {
                    if val > best_val {
                        best_idx = idx;
                        best_val = val;
                    }
                }
                hypothesis_ids[pos] = sanitize_refiner_token(
                    best_idx as u32,
                    collapsed[pos].token_id,
                    &self.tokenizer,
                ) as i64;
            }
            // Stop head (learned) first — cheap short-circuit.
            if let Some(stop) = refine.stop_logit {
                if sigmoid(stop) >= self.refiner_stop_threshold {
                    break;
                }
            }
            if it + 1 >= iters {
                break;
            }
            // Re-mask selection: learned head if available, else confidence.
            mask_positions = if let Some(remask) = refine.remask_logits.as_ref() {
                let mut picks: Vec<(usize, f32)> = (0..hyp_len)
                    .map(|i| (i, sigmoid(remask[i])))
                    .filter(|(_, p)| *p >= self.refiner_remask_threshold)
                    .collect();
                picks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if self.refiner_max_masks > 0 {
                    picks.truncate(self.refiner_max_masks);
                }
                picks.into_iter().map(|(i, _)| i).collect()
            } else {
                let mut picks: Vec<(usize, f32)> = (0..hyp_len)
                    .map(|i| {
                        let row = refine.logits.row(i);
                        let max_logit = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let sum_exp: f32 = row.iter().map(|v| (v - max_logit).exp()).sum();
                        // softmax max prob in log-sum-exp-stable form.
                        let max_prob = 1.0 / sum_exp;
                        (i, max_prob)
                    })
                    .filter(|(_, p)| *p < self.refiner_confidence_threshold)
                    .collect();
                picks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                if self.refiner_max_masks > 0 {
                    picks.truncate(self.refiner_max_masks);
                }
                picks.into_iter().map(|(i, _)| i).collect()
            };
            if mask_positions.is_empty() {
                break;
            }
            for &pos in &mask_positions {
                hypothesis_ids[pos] = self.tokenizer.mask_id as i64;
            }
        }

        let final_ids: Vec<u32> = hypothesis_ids[..hyp_len]
            .iter()
            .map(|&v| v as u32)
            .collect();
        Ok(normalize_candidate(&self.tokenizer.decode(&final_ids)))
    }

    pub fn refine_greedy(
        &mut self,
        context: &str,
        reading: &str,
        proposal: &ProposalOutput,
    ) -> Result<String> {
        let collapsed = proposal.collapsed_with_alignment(self.tokenizer.blank_id);
        let mask_positions = select_mask_positions(
            &collapsed,
            self.refiner_confidence_threshold,
            self.refiner_max_masks,
        );
        if mask_positions.is_empty() {
            return Ok(String::new());
        }
        let (input_ids, input_mask, _) = self.encode(context, reading);
        let mut hypothesis_ids = vec![self.tokenizer.pad_id as i64; self.seq_len];
        let mut hypothesis_mask = vec![0i64; self.seq_len];
        for (idx, token) in collapsed.iter().enumerate() {
            let is_masked = mask_positions.contains(&idx);
            hypothesis_ids[idx] = if is_masked {
                self.tokenizer.mask_id as i64
            } else {
                token.token_id as i64
            };
            hypothesis_mask[idx] = 1;
        }
        let hyp_len = collapsed.len();
        let refine = self.run_refiner(
            &input_ids,
            &input_mask,
            &hypothesis_ids,
            &hypothesis_mask,
            hyp_len,
        )?;
        for &pos in &mask_positions {
            let row = refine.logits.row(pos);
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &val) in row.iter().enumerate() {
                if val > best_val {
                    best_idx = idx;
                    best_val = val;
                }
            }
            hypothesis_ids[pos] =
                sanitize_refiner_token(best_idx as u32, collapsed[pos].token_id, &self.tokenizer)
                    as i64;
        }
        let final_ids: Vec<u32> = hypothesis_ids[..hyp_len]
            .iter()
            .map(|&v| v as u32)
            .collect();
        Ok(normalize_candidate(&self.tokenizer.decode(&final_ids)))
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
        let proposal = self.run_proposal(context, reading)?;
        let decoded = proposal.collapsed_tokens(self.tokenizer.blank_id);
        Ok(self.tokenizer.decode(&decoded))
    }
}

pub fn collapse_frame_ids(frame_ids: &[u32], blank_id: u32) -> Vec<u32> {
    let mut decoded: Vec<u32> = Vec::with_capacity(frame_ids.len());
    let mut last: Option<u32> = None;
    for &token in frame_ids {
        if token != blank_id && Some(token) != last {
            decoded.push(token);
        }
        last = Some(token);
    }
    decoded
}

pub fn collapse_frames_with_alignment(
    frames: &[ProposalFrame],
    blank_id: u32,
) -> Vec<CollapsedToken> {
    let mut out: Vec<CollapsedToken> = Vec::new();
    let mut current_id: Option<u32> = None;
    let mut start_frame = 0usize;
    let mut log_prob_sum = 0.0f32;
    let mut margin_sum = 0.0f32;
    let mut min_log_prob = f32::INFINITY;
    let mut min_margin = f32::INFINITY;
    let mut count = 0usize;
    let mut prev_token: Option<u32> = None;

    let flush = |out: &mut Vec<CollapsedToken>,
                 current_id: &mut Option<u32>,
                 start_frame: usize,
                 log_prob_sum: &mut f32,
                 margin_sum: &mut f32,
                 min_log_prob: &mut f32,
                 min_margin: &mut f32,
                 count: &mut usize| {
        if let Some(token_id) = *current_id {
            if *count > 0 {
                out.push(CollapsedToken {
                    token_id,
                    start_frame,
                    end_frame: start_frame + *count - 1,
                    min_log_prob: *min_log_prob,
                    mean_log_prob: *log_prob_sum / *count as f32,
                    min_margin: *min_margin,
                    mean_margin: *margin_sum / *count as f32,
                });
            }
        }
        *current_id = None;
        *log_prob_sum = 0.0;
        *margin_sum = 0.0;
        *min_log_prob = f32::INFINITY;
        *min_margin = f32::INFINITY;
        *count = 0;
    };

    for (t, frame) in frames.iter().enumerate() {
        let token = frame.top1_id;
        let margin = frame.top1_log_prob - frame.top2_log_prob;
        if token == blank_id {
            flush(
                &mut out,
                &mut current_id,
                start_frame,
                &mut log_prob_sum,
                &mut margin_sum,
                &mut min_log_prob,
                &mut min_margin,
                &mut count,
            );
            prev_token = Some(token);
            continue;
        }

        if prev_token == Some(token) {
            if current_id.is_none() {
                current_id = Some(token);
                start_frame = t;
            }
            log_prob_sum += frame.top1_log_prob;
            margin_sum += margin;
            min_log_prob = min_log_prob.min(frame.top1_log_prob);
            min_margin = min_margin.min(margin);
            count += 1;
        } else {
            flush(
                &mut out,
                &mut current_id,
                start_frame,
                &mut log_prob_sum,
                &mut margin_sum,
                &mut min_log_prob,
                &mut min_margin,
                &mut count,
            );
            current_id = Some(token);
            start_frame = t;
            log_prob_sum = frame.top1_log_prob;
            margin_sum = margin;
            min_log_prob = frame.top1_log_prob;
            min_margin = margin;
            count = 1;
        }
        prev_token = Some(token);
    }

    flush(
        &mut out,
        &mut current_id,
        start_frame,
        &mut log_prob_sum,
        &mut margin_sum,
        &mut min_log_prob,
        &mut min_margin,
        &mut count,
    );
    out
}

pub fn select_mask_positions(
    tokens: &[CollapsedToken],
    confidence_threshold: f32,
    max_masks: usize,
) -> Vec<usize> {
    let mut ranked: Vec<(usize, f32, f32)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(idx, tok)| {
            let confidence = tok.confidence();
            (confidence < confidence_threshold).then_some((idx, confidence, tok.min_margin))
        })
        .collect();
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.0.cmp(&b.0))
    });
    if max_masks > 0 && ranked.len() > max_masks {
        ranked.truncate(max_masks);
    }
    ranked.into_iter().map(|(idx, _, _)| idx).collect()
}

pub fn sanitize_refiner_token(
    predicted: u32,
    fallback: u32,
    tokenizer: &SharedCharTokenizer,
) -> u32 {
    if predicted == tokenizer.pad_id
        || predicted == tokenizer.cls_id
        || predicted == tokenizer.sep_id
        || predicted == tokenizer.blank_id
        || predicted == tokenizer.mask_id
    {
        return fallback;
    }
    predicted
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
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFEE0).unwrap_or(c),
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
    use super::{
        collapse_frame_ids, collapse_frames_with_alignment, derive_refiner_onnx_path,
        normalize_candidate, sanitize_refiner_token, select_mask_positions, CollapsedToken,
        EngineSession, ProposalFrame, ProposalOutput,
    };
    use crate::tokenizer::SharedCharTokenizer;
    use ndarray::Array2;
    use std::path::{Path, PathBuf};

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

    #[test]
    fn collapse_frame_ids_drops_blanks_and_repeats() {
        let frames = vec![4, 7, 7, 9, 4, 9, 11, 11];
        assert_eq!(collapse_frame_ids(&frames, 4), vec![7, 9, 9, 11]);
    }

    #[test]
    fn proposal_output_collapse_uses_top1_ids() {
        let proposal = ProposalOutput {
            log_probs: Array2::<f32>::zeros((4, 3)),
            frames: vec![
                ProposalFrame {
                    top1_id: 4,
                    top1_log_prob: -0.1,
                    top2_log_prob: -0.4,
                },
                ProposalFrame {
                    top1_id: 8,
                    top1_log_prob: -0.2,
                    top2_log_prob: -0.7,
                },
                ProposalFrame {
                    top1_id: 8,
                    top1_log_prob: -0.3,
                    top2_log_prob: -0.9,
                },
                ProposalFrame {
                    top1_id: 9,
                    top1_log_prob: -0.2,
                    top2_log_prob: -1.1,
                },
            ],
            input_len: 4,
        };
        assert_eq!(proposal.collapsed_tokens(4), vec![8, 9]);
    }

    #[test]
    fn collapse_frames_with_alignment_tracks_spans_and_confidence() {
        let frames = vec![
            ProposalFrame {
                top1_id: 4,
                top1_log_prob: -0.1,
                top2_log_prob: -1.0,
            },
            ProposalFrame {
                top1_id: 7,
                top1_log_prob: -0.4,
                top2_log_prob: -0.5,
            },
            ProposalFrame {
                top1_id: 7,
                top1_log_prob: -0.2,
                top2_log_prob: -0.8,
            },
            ProposalFrame {
                top1_id: 4,
                top1_log_prob: -0.1,
                top2_log_prob: -1.1,
            },
            ProposalFrame {
                top1_id: 7,
                top1_log_prob: -0.9,
                top2_log_prob: -0.92,
            },
            ProposalFrame {
                top1_id: 9,
                top1_log_prob: -0.3,
                top2_log_prob: -1.2,
            },
            ProposalFrame {
                top1_id: 9,
                top1_log_prob: -0.25,
                top2_log_prob: -0.6,
            },
        ];
        let collapsed = collapse_frames_with_alignment(&frames, 4);
        assert_eq!(collapsed.len(), 3);
        assert_eq!(collapsed[0].token_id, 7);
        assert_eq!((collapsed[0].start_frame, collapsed[0].end_frame), (1, 2));
        assert_eq!(collapsed[1].token_id, 7);
        assert_eq!((collapsed[1].start_frame, collapsed[1].end_frame), (4, 4));
        assert_eq!(collapsed[2].token_id, 9);
        assert_eq!((collapsed[2].start_frame, collapsed[2].end_frame), (5, 6));
        assert!(collapsed[1].confidence() < collapsed[0].confidence());
    }

    #[test]
    fn select_mask_positions_prefers_lowest_confidence() {
        let tokens = vec![
            CollapsedToken {
                token_id: 7,
                start_frame: 0,
                end_frame: 1,
                min_log_prob: 0.85f32.ln(),
                mean_log_prob: 0.9f32.ln(),
                min_margin: 1.2,
                mean_margin: 1.4,
            },
            CollapsedToken {
                token_id: 8,
                start_frame: 2,
                end_frame: 2,
                min_log_prob: 0.31f32.ln(),
                mean_log_prob: 0.31f32.ln(),
                min_margin: 0.03,
                mean_margin: 0.03,
            },
            CollapsedToken {
                token_id: 9,
                start_frame: 3,
                end_frame: 4,
                min_log_prob: 0.44f32.ln(),
                mean_log_prob: 0.5f32.ln(),
                min_margin: 0.12,
                mean_margin: 0.18,
            },
        ];
        assert_eq!(select_mask_positions(&tokens, 0.5, 1), vec![1]);
    }

    #[test]
    fn derive_refiner_onnx_path_appends_refiner_suffix() {
        let path = Path::new("D:\\Dev\\new-ime\\models\\onnx\\ctc-nat-30m.int8.onnx");
        let derived = derive_refiner_onnx_path(path);
        assert_eq!(
            derived.to_string_lossy(),
            "D:\\Dev\\new-ime\\models\\onnx\\ctc-nat-30m.int8.refiner.onnx"
        );
    }

    #[test]
    fn sanitize_refiner_token_rejects_specials() {
        let tokenizer = SharedCharTokenizer {
            id_to_token: vec![
                "[PAD]".into(),
                "[UNK]".into(),
                "[CLS]".into(),
                "[SEP]".into(),
                "[BLANK]".into(),
                "[MASK]".into(),
                "東".into(),
            ],
            token_to_id: std::collections::HashMap::new(),
            pad_id: 0,
            unk_id: 1,
            cls_id: 2,
            sep_id: 3,
            blank_id: 4,
            mask_id: 5,
        };
        assert_eq!(sanitize_refiner_token(0, 6, &tokenizer), 6);
        assert_eq!(sanitize_refiner_token(4, 6, &tokenizer), 6);
        assert_eq!(sanitize_refiner_token(5, 6, &tokenizer), 6);
        assert_eq!(sanitize_refiner_token(6, 1, &tokenizer), 6);
    }

    #[test]
    fn real_onnx_session_loads_and_runs_refiner_when_artifacts_exist() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..");
        let onnx = repo_root
            .join("models")
            .join("onnx")
            .join("ctc-nat-30m-student-step160000.fp32.onnx");
        let refiner = derive_refiner_onnx_path(&onnx);
        if !onnx.exists() || !refiner.exists() {
            return;
        }

        let mut session = EngineSession::load(&onnx).expect("load engine session");
        assert!(session.has_refiner());
        session.configure_refiner(true, 0.999, 2);

        let greedy = session
            .greedy_decode("", "きょうは")
            .expect("greedy decode");
        assert!(!greedy.is_empty());

        let refined = session.convert("", "きょうは").expect("refined convert");
        assert!(!refined.is_empty());
        assert!(!refined[0].is_empty());

        // Iterative path: K>1 exercises the learned remask/stop heads if
        // the ONNX graph exposes them, otherwise falls back cleanly to
        // confidence selection.
        session.configure_refiner_iter(3, 0.5, 0.5);
        let refined_iter = session
            .convert("", "きょうは")
            .expect("iterative refined convert");
        assert!(!refined_iter.is_empty());
        assert!(!refined_iter[0].is_empty());
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

fn derive_refiner_onnx_path(onnx_path: &Path) -> PathBuf {
    onnx_path.with_file_name(format!(
        "{}.refiner{}",
        onnx_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
        onnx_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| format!(".{s}"))
            .unwrap_or_default()
    ))
}
