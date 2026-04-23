const LOG_ZERO: f64 = f64::NEG_INFINITY;
const MIN_PROB: f64 = 1e-12;

#[derive(Debug, Clone)]
pub struct CtcLoss {
    pub loss: f64,
    pub grad_logits: Vec<f64>,
}

pub fn ctc_loss_and_grad(
    probs: &[f64],
    input_len: usize,
    targets: &[usize],
    blank_id: usize,
    vocab_size: usize,
) -> CtcLoss {
    let mut grad_logits = vec![0.0; probs.len()];
    if input_len == 0 || vocab_size == 0 || probs.len() != input_len * vocab_size {
        return CtcLoss {
            loss: 0.0,
            grad_logits,
        };
    }

    let mut ext = Vec::with_capacity(targets.len() * 2 + 1);
    ext.push(blank_id);
    for &target in targets {
        if target >= vocab_size {
            return CtcLoss {
                loss: 0.0,
                grad_logits,
            };
        }
        ext.push(target);
        ext.push(blank_id);
    }
    let state_count = ext.len();
    let mut log_probs = vec![0.0; probs.len()];
    for (dst, src) in log_probs.iter_mut().zip(probs.iter().copied()) {
        *dst = src.max(MIN_PROB).ln();
    }

    let mut alpha = vec![LOG_ZERO; input_len * state_count];
    alpha[idx(0, 0, state_count)] = emit(&log_probs, 0, ext[0], vocab_size);
    if state_count > 1 {
        alpha[idx(0, 1, state_count)] = emit(&log_probs, 0, ext[1], vocab_size);
    }
    for t in 1..input_len {
        for s in 0..state_count {
            let label = ext[s];
            let mut score = alpha[idx(t - 1, s, state_count)];
            if s > 0 {
                score = log_sum_exp(score, alpha[idx(t - 1, s - 1, state_count)]);
            }
            if s > 1 && label != blank_id && label != ext[s - 2] {
                score = log_sum_exp(score, alpha[idx(t - 1, s - 2, state_count)]);
            }
            alpha[idx(t, s, state_count)] = score + emit(&log_probs, t, label, vocab_size);
        }
    }

    let mut beta = vec![LOG_ZERO; input_len * state_count];
    beta[idx(input_len - 1, state_count - 1, state_count)] =
        emit(&log_probs, input_len - 1, ext[state_count - 1], vocab_size);
    if state_count > 1 {
        beta[idx(input_len - 1, state_count - 2, state_count)] =
            emit(&log_probs, input_len - 1, ext[state_count - 2], vocab_size);
    }
    if input_len > 1 {
        for t in (0..input_len - 1).rev() {
            for s in 0..state_count {
                let label = ext[s];
                let mut score = beta[idx(t + 1, s, state_count)];
                if s + 1 < state_count {
                    score = log_sum_exp(score, beta[idx(t + 1, s + 1, state_count)]);
                }
                if s + 2 < state_count && label != blank_id && label != ext[s + 2] {
                    score = log_sum_exp(score, beta[idx(t + 1, s + 2, state_count)]);
                }
                beta[idx(t, s, state_count)] = score + emit(&log_probs, t, label, vocab_size);
            }
        }
    }

    let mut log_z = alpha[idx(input_len - 1, state_count - 1, state_count)];
    if state_count > 1 {
        log_z = log_sum_exp(
            log_z,
            alpha[idx(input_len - 1, state_count - 2, state_count)],
        );
    }
    if !log_z.is_finite() {
        return CtcLoss {
            loss: 0.0,
            grad_logits,
        };
    }

    let mut posterior = vec![0.0; probs.len()];
    for t in 0..input_len {
        for s in 0..state_count {
            let label = ext[s];
            let occ = alpha[idx(t, s, state_count)] + beta[idx(t, s, state_count)]
                - emit(&log_probs, t, label, vocab_size)
                - log_z;
            if occ.is_finite() {
                posterior[t * vocab_size + label] += occ.exp();
            }
        }
    }

    let scale = 1.0 / targets.len().max(1) as f64;
    for t in 0..input_len {
        let row = &probs[t * vocab_size..(t + 1) * vocab_size];
        let row_post = &posterior[t * vocab_size..(t + 1) * vocab_size];
        for v in 0..vocab_size {
            grad_logits[t * vocab_size + v] = (row[v] - row_post[v]) * scale;
        }
    }
    CtcLoss {
        loss: -log_z * scale,
        grad_logits,
    }
}

fn idx(t: usize, s: usize, state_count: usize) -> usize {
    t * state_count + s
}

fn emit(log_probs: &[f64], t: usize, label: usize, vocab_size: usize) -> f64 {
    log_probs[t * vocab_size + label]
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => LOG_ZERO,
        (true, false) => a,
        (false, true) => b,
        (true, true) => {
            let hi = a.max(b);
            let lo = a.min(b);
            hi + (lo - hi).exp().ln_1p()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_loss_is_finite_for_simple_alignment() {
        let probs = vec![
            0.1, 0.8, 0.1, //
            0.7, 0.2, 0.1, //
            0.1, 0.1, 0.8, //
        ];
        let out = ctc_loss_and_grad(&probs, 3, &[1, 2], 0, 3);
        assert!(out.loss.is_finite());
        assert!(out.loss > 0.0);
        assert_eq!(out.grad_logits.len(), probs.len());
    }

    #[test]
    fn ctc_loss_returns_zero_for_impossible_alignment() {
        let probs = vec![0.9, 0.1, 0.0, 0.8, 0.2, 0.0];
        let out = ctc_loss_and_grad(&probs, 2, &[1, 1, 1], 0, 3);
        assert_eq!(out.loss, 0.0);
        assert!(out.grad_logits.iter().all(|v| *v == 0.0));
    }
}
