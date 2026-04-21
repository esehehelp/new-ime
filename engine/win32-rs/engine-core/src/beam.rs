//! CTC prefix beam search, optional KenLM shallow fusion.
//!
//! Direct port of `models/src/eval/ctc_beam.py::prefix_beam_search` so scores
//! are comparable to the evaluation harness:
//!   * pb / pnb accumulation with logsumexp merging
//!   * repeat collapse only across blank (CTC rule)
//!   * `logp_ctc(prefix) + alpha * logp_lm(prefix) + beta * len(prefix)` ranking
//!   * top-K per timestep bounds the inner loop

use std::collections::HashMap;

use ndarray::Array2;

use crate::kenlm::LmScorer;

const NEG_INF: f32 = f32::NEG_INFINITY;

fn logsumexp(a: f32, b: f32) -> f32 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp().ln_1p()
}

#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
}

/// Run prefix beam search.
///
/// `log_probs` is (T, V). Returns up to `beam_width` ranked hypotheses.
pub fn prefix_beam_search(
    log_probs: &Array2<f32>,
    blank_id: u32,
    beam_width: usize,
    top_k_per_step: usize,
    lm: Option<&dyn LmScorer>,
    lm_alpha: f32,
    lm_beta: f32,
) -> Vec<BeamHypothesis> {
    assert!(beam_width > 0, "beam_width must be positive");
    let shape = log_probs.shape();
    let (t_len, vocab) = (shape[0], shape[1]);
    let top_k = top_k_per_step.min(vocab);

    // Precompute per-step top-K token ids + their log probs.
    let mut step_topk: Vec<Vec<(u32, f32)>> = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let row = log_probs.row(t);
        let mut idx: Vec<usize> = (0..vocab).collect();
        // Partial sort by log prob descending.
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
            row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut top: Vec<(u32, f32)> = idx[..top_k]
            .iter()
            .map(|&i| (i as u32, row[i]))
            .collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        step_topk.push(top);
    }

    // Beam entries: prefix tokens → (pb, pnb).
    let mut beam: HashMap<Vec<u32>, (f32, f32)> = HashMap::new();
    beam.insert(Vec::new(), (0.0, NEG_INF));

    for t in 0..t_len {
        let mut next_beam: HashMap<Vec<u32>, (f32, f32)> = HashMap::new();
        let blank_logp = log_probs[[t, blank_id as usize]];

        for (prefix, (pb, pnb)) in &beam {
            // Emit blank.
            let new_blank = logsumexp(*pb, *pnb) + blank_logp;
            merge(&mut next_beam, prefix.clone(), new_blank, NEG_INF);

            // Emit each non-blank top-K.
            for &(c, c_logp) in &step_topk[t] {
                if c == blank_id {
                    continue;
                }
                if prefix.last().copied() == Some(c) {
                    // Repeat — only the blank-ending path can emit a new token.
                    let mut extended = prefix.clone();
                    extended.push(c);
                    merge(&mut next_beam, extended, NEG_INF, pb + c_logp);
                    merge(&mut next_beam, prefix.clone(), NEG_INF, pnb + c_logp);
                } else {
                    let new_pnb = logsumexp(*pb, *pnb) + c_logp;
                    let mut extended = prefix.clone();
                    extended.push(c);
                    merge(&mut next_beam, extended, NEG_INF, new_pnb);
                }
            }
        }

        // Prune to top `beam_width` by fused score.
        let mut scored: Vec<(Vec<u32>, f32)> = next_beam
            .iter()
            .map(|(p, (pb, pnb))| {
                let s = rank(p, *pb, *pnb, lm, lm_alpha, lm_beta);
                (p.clone(), s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(beam_width);

        beam.clear();
        for (p, _) in scored {
            if let Some(v) = next_beam.remove(&p) {
                beam.insert(p, v);
            }
        }
    }

    let mut final_list: Vec<BeamHypothesis> = beam
        .into_iter()
        .map(|(p, (pb, pnb))| {
            let s = rank(&p, pb, pnb, lm, lm_alpha, lm_beta);
            BeamHypothesis { tokens: p, score: s }
        })
        .collect();
    final_list.sort_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    final_list
}

fn merge(
    beam: &mut HashMap<Vec<u32>, (f32, f32)>,
    prefix: Vec<u32>,
    pb: f32,
    pnb: f32,
) {
    match beam.get_mut(&prefix) {
        Some(entry) => {
            entry.0 = logsumexp(entry.0, pb);
            entry.1 = logsumexp(entry.1, pnb);
        }
        None => {
            beam.insert(prefix, (pb, pnb));
        }
    }
}

fn rank(
    prefix: &[u32],
    pb: f32,
    pnb: f32,
    lm: Option<&dyn LmScorer>,
    lm_alpha: f32,
    lm_beta: f32,
) -> f32 {
    let ctc = logsumexp(pb, pnb);
    let lm_part = if lm_alpha != 0.0 && !prefix.is_empty() {
        if let Some(scorer) = lm {
            lm_alpha * scorer.score_ids(prefix)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let length_part = lm_beta * prefix.len() as f32;
    ctc + lm_part + length_part
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn greedy_matches_beam_without_lm() {
        // Very simple logits: T=3, V=4 (blank=0, tokens 1/2/3).
        // Step 0: peak at 1
        // Step 1: peak at 1 (repeat, needs blank in between)
        // Step 2: peak at 2
        // Greedy collapse would yield [1, 2] since no blank between the two 1s.
        let logits = arr2(&[
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -0.1, -3.0, -3.0],
            [-3.0, -3.0, -0.1, -3.0],
        ]);
        let out = prefix_beam_search(&logits, 0, 4, 4, None, 0.0, 0.0);
        assert!(!out.is_empty());
        assert_eq!(out[0].tokens, vec![1, 2]);
    }

    #[test]
    fn blank_separated_repeats_expand() {
        // Step 0: token 1, Step 1: blank, Step 2: token 1. Expected: [1,1].
        let logits = arr2(&[
            [-3.0, -0.1, -3.0],
            [-0.1, -3.0, -3.0],
            [-3.0, -0.1, -3.0],
        ]);
        let out = prefix_beam_search(&logits, 0, 4, 3, None, 0.0, 0.0);
        assert_eq!(out[0].tokens, vec![1, 1]);
    }
}
