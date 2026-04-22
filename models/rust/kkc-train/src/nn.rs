use crate::backend::{DecoderBlock, EncoderBlock};

#[derive(Debug, Clone)]
pub struct AttentionCache {
    pub query_src: Vec<f64>,
    pub key_src: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub attn_weights: Vec<f64>,
    pub attended: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AttentionGrads {
    pub q_proj: Vec<f64>,
    pub k_proj: Vec<f64>,
    pub v_proj: Vec<f64>,
    pub o_proj: Vec<f64>,
    pub grad_query_src: Vec<f64>,
    pub grad_key_src: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EncoderBlockCache {
    pub input: Vec<f64>,
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    pub attn_weights: Vec<f64>,
    pub attended: Vec<f64>,
    pub mixed: Vec<f64>,
    pub ff1_pre: Vec<f64>,
    pub ff1_act: Vec<f64>,
    pub output: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct EncoderBlockGrads {
    pub q_proj: Vec<f64>,
    pub k_proj: Vec<f64>,
    pub v_proj: Vec<f64>,
    pub o_proj: Vec<f64>,
    pub ff_in: Vec<f64>,
    pub ff_in_bias: Vec<f64>,
    pub ff_out: Vec<f64>,
    pub ff_out_bias: Vec<f64>,
    pub grad_input: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecoderBlockCache {
    pub input: Vec<f64>,
    pub memory: Vec<f64>,
    pub self_attn: AttentionCache,
    pub mixed: Vec<f64>,
    pub cross_attn: AttentionCache,
    pub crossed: Vec<f64>,
    pub ff1_pre: Vec<f64>,
    pub ff1_act: Vec<f64>,
    pub output: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DecoderBlockGrads {
    pub self_q_proj: Vec<f64>,
    pub self_k_proj: Vec<f64>,
    pub self_v_proj: Vec<f64>,
    pub self_o_proj: Vec<f64>,
    pub cross_q_proj: Vec<f64>,
    pub cross_k_proj: Vec<f64>,
    pub cross_v_proj: Vec<f64>,
    pub cross_o_proj: Vec<f64>,
    pub ff_in: Vec<f64>,
    pub ff_in_bias: Vec<f64>,
    pub ff_out: Vec<f64>,
    pub ff_out_bias: Vec<f64>,
    pub grad_input: Vec<f64>,
    pub grad_memory: Vec<f64>,
}

pub fn masked_mean_over_time(input: &[f32], mask: &[f32], hidden: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; hidden];
    let mut count = 0.0f64;
    for (time_idx, &active) in mask.iter().enumerate() {
        if active <= 0.0 {
            continue;
        }
        count += 1.0;
        let row = &input[time_idx * hidden..(time_idx + 1) * hidden];
        for (dst, src) in out.iter_mut().zip(row.iter().copied()) {
            *dst += src as f64;
        }
    }
    if count > 0.0 {
        for value in &mut out {
            *value /= count;
        }
    }
    out
}

pub fn linear(features: &[f64], weight: &[f64], bias: &[f64], out_dim: usize) -> Vec<f64> {
    let in_dim = features.len();
    let mut out = vec![0.0f64; out_dim];
    for out_idx in 0..out_dim {
        let mut acc = bias[out_idx];
        let base = out_idx * in_dim;
        for in_idx in 0..in_dim {
            acc += features[in_idx] * weight[base + in_idx];
        }
        out[out_idx] = acc;
    }
    out
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for logit in logits {
        let value = (*logit - max_logit).exp();
        out.push(value);
        sum += value;
    }
    if sum > 0.0 {
        for value in &mut out {
            *value /= sum;
        }
    }
    out
}

pub fn cross_entropy_grad(probs: &[f64], target_id: usize) -> (f64, Vec<f64>) {
    let prob = probs.get(target_id).copied().unwrap_or(1e-12).max(1e-12);
    let mut grad = Vec::with_capacity(probs.len());
    for (idx, value) in probs.iter().copied().enumerate() {
        grad.push(value - if idx == target_id { 1.0 } else { 0.0 });
    }
    (-prob.ln(), grad)
}

pub fn mean_over_rows(values: &[f64], rows: usize, width: usize) -> Vec<f64> {
    let mut out = vec![0.0; width];
    if rows == 0 {
        return out;
    }
    for row_idx in 0..rows {
        let row = &values[row_idx * width..(row_idx + 1) * width];
        for (dst, src) in out.iter_mut().zip(row.iter().copied()) {
            *dst += src;
        }
    }
    for value in &mut out {
        *value /= rows as f64;
    }
    out
}

pub fn encoder_block_forward(
    hidden: &[f64],
    seq_len: usize,
    width: usize,
    num_heads: usize,
    block: &EncoderBlock,
) -> Vec<f64> {
    encoder_block_forward_cached(hidden, seq_len, width, num_heads, block).0
}

pub fn decoder_block_forward(
    hidden: &[f64],
    memory: &[f64],
    seq_len: usize,
    memory_len: usize,
    width: usize,
    num_heads: usize,
    block: &DecoderBlock,
) -> Vec<f64> {
    decoder_block_forward_cached(hidden, memory, seq_len, memory_len, width, num_heads, block).0
}

fn attention_forward(
    query_src: &[f64],
    key_src: &[f64],
    value_src: &[f64],
    query_len: usize,
    key_len: usize,
    width: usize,
    num_heads: usize,
    q_proj: &[f64],
    k_proj: &[f64],
    v_proj: &[f64],
    o_proj: &[f64],
) -> Vec<f64> {
    let (attended, _cache) = attention_forward_cached(
        query_src, key_src, value_src, query_len, key_len, width, num_heads, q_proj, k_proj, v_proj,
    );
    let mut out = vec![0.0; query_src.len()];
    for t in 0..query_len {
        let proj_row = linear(
            &attended[t * width..(t + 1) * width],
            o_proj,
            &vec![0.0; width],
            width,
        );
        out[t * width..(t + 1) * width].copy_from_slice(&proj_row);
    }
    out
}

fn attention_forward_cached(
    query_src: &[f64],
    key_src: &[f64],
    value_src: &[f64],
    query_len: usize,
    key_len: usize,
    width: usize,
    num_heads: usize,
    q_proj: &[f64],
    k_proj: &[f64],
    v_proj: &[f64],
) -> (Vec<f64>, AttentionCache) {
    let mut q = vec![0.0; query_src.len()];
    let mut k = vec![0.0; key_src.len()];
    let mut v = vec![0.0; value_src.len()];
    for t in 0..query_len {
        let q_row = linear(
            &query_src[t * width..(t + 1) * width],
            q_proj,
            &vec![0.0; width],
            width,
        );
        q[t * width..(t + 1) * width].copy_from_slice(&q_row);
    }
    for t in 0..key_len {
        let k_row = linear(
            &key_src[t * width..(t + 1) * width],
            k_proj,
            &vec![0.0; width],
            width,
        );
        let v_row = linear(
            &value_src[t * width..(t + 1) * width],
            v_proj,
            &vec![0.0; width],
            width,
        );
        k[t * width..(t + 1) * width].copy_from_slice(&k_row);
        v[t * width..(t + 1) * width].copy_from_slice(&v_row);
    }
    let head_count = num_heads.max(1);
    let head_width = (width / head_count).max(1);
    let mut attended = vec![0.0; query_src.len()];
    let mut attn_weights = vec![0.0; head_count * query_len * key_len];
    for t in 0..query_len {
        for head in 0..head_count {
            let start = head * head_width;
            let end = ((head + 1) * head_width).min(width);
            if start >= end {
                continue;
            }
            let q_row = &q[t * width + start..t * width + end];
            let mut scores = vec![0.0; key_len];
            for s in 0..key_len {
                let k_row = &k[s * width + start..s * width + end];
                scores[s] = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / ((end - start) as f64).sqrt().max(1.0);
            }
            let weights = softmax(&scores);
            let weight_base = (head * query_len + t) * key_len;
            attn_weights[weight_base..weight_base + key_len].copy_from_slice(&weights);
            for (s, weight) in weights.iter().copied().enumerate() {
                let v_row = &v[s * width + start..s * width + end];
                let dst = &mut attended[t * width + start..t * width + end];
                for (dst_h, src_h) in dst.iter_mut().zip(v_row.iter()) {
                    *dst_h += weight * *src_h;
                }
            }
        }
    }
    (
        attended.clone(),
        AttentionCache {
            query_src: query_src.to_vec(),
            key_src: key_src.to_vec(),
            q,
            k,
            v,
            attn_weights,
            attended,
        },
    )
}

fn attention_backward(
    cache: &AttentionCache,
    grad_after_proj: &[f64],
    query_len: usize,
    key_len: usize,
    width: usize,
    o_proj: &[f64],
    q_proj: &[f64],
    k_proj: &[f64],
    v_proj: &[f64],
) -> AttentionGrads {
    let head_count = (cache.attn_weights.len() / (query_len * key_len).max(1)).max(1);
    let head_width = (width / head_count).max(1);
    let mut grad_o_proj = vec![0.0; o_proj.len()];
    let mut grad_attended = vec![0.0; cache.attended.len()];
    for t in 0..query_len {
        let grad_row = &grad_after_proj[t * width..(t + 1) * width];
        let attended_row = &cache.attended[t * width..(t + 1) * width];
        for out_idx in 0..width {
            let base = out_idx * width;
            for in_idx in 0..width {
                grad_o_proj[base + in_idx] += grad_row[out_idx] * attended_row[in_idx];
                grad_attended[t * width + in_idx] += grad_row[out_idx] * o_proj[base + in_idx];
            }
        }
    }

    let mut grad_q = vec![0.0; cache.q.len()];
    let mut grad_k = vec![0.0; cache.k.len()];
    let mut grad_v = vec![0.0; cache.v.len()];
    for t in 0..query_len {
        for head in 0..head_count {
            let start = head * head_width;
            let end = ((head + 1) * head_width).min(width);
            if start >= end {
                continue;
            }
            let weight_base = (head * query_len + t) * key_len;
            let weights = &cache.attn_weights[weight_base..weight_base + key_len];
            let grad_attn_row = &grad_attended[t * width + start..t * width + end];
            let q_row = &cache.q[t * width + start..t * width + end];
            let mut grad_weights = vec![0.0; key_len];
            for s in 0..key_len {
                let v_row = &cache.v[s * width + start..s * width + end];
                for dim in 0..(end - start) {
                    grad_weights[s] += grad_attn_row[dim] * v_row[dim];
                    grad_v[s * width + start + dim] += grad_attn_row[dim] * weights[s];
                }
            }
            let dot: f64 = grad_weights
                .iter()
                .zip(weights.iter())
                .map(|(g, w)| g * w)
                .sum();
            let mut grad_scores = vec![0.0; key_len];
            for s in 0..key_len {
                grad_scores[s] = weights[s] * (grad_weights[s] - dot);
            }
            let scale = ((end - start) as f64).sqrt().max(1.0);
            for s in 0..key_len {
                let k_row = &cache.k[s * width + start..s * width + end];
                for dim in 0..(end - start) {
                    grad_q[t * width + start + dim] += grad_scores[s] * k_row[dim] / scale;
                    grad_k[s * width + start + dim] += grad_scores[s] * q_row[dim] / scale;
                }
            }
        }
    }

    let mut grad_q_proj = vec![0.0; q_proj.len()];
    let mut grad_k_proj = vec![0.0; k_proj.len()];
    let mut grad_v_proj = vec![0.0; v_proj.len()];
    let mut grad_query_src = vec![0.0; cache.query_src.len()];
    let mut grad_key_src = vec![0.0; cache.key_src.len()];
    for t in 0..query_len {
        let input_q = &cache.query_src[t * width..(t + 1) * width];
        let grad_q_row = &grad_q[t * width..(t + 1) * width];
        for out_idx in 0..width {
            let base = out_idx * width;
            for in_idx in 0..width {
                grad_q_proj[base + in_idx] += grad_q_row[out_idx] * input_q[in_idx];
                grad_query_src[t * width + in_idx] += grad_q_row[out_idx] * q_proj[base + in_idx];
            }
        }
    }
    for s in 0..key_len {
        let input_k = &cache.key_src[s * width..(s + 1) * width];
        let grad_k_row = &grad_k[s * width..(s + 1) * width];
        let grad_v_row = &grad_v[s * width..(s + 1) * width];
        for out_idx in 0..width {
            let base = out_idx * width;
            for in_idx in 0..width {
                grad_k_proj[base + in_idx] += grad_k_row[out_idx] * input_k[in_idx];
                grad_v_proj[base + in_idx] += grad_v_row[out_idx] * input_k[in_idx];
                grad_key_src[s * width + in_idx] += grad_k_row[out_idx] * k_proj[base + in_idx];
                grad_key_src[s * width + in_idx] += grad_v_row[out_idx] * v_proj[base + in_idx];
            }
        }
    }

    AttentionGrads {
        q_proj: grad_q_proj,
        k_proj: grad_k_proj,
        v_proj: grad_v_proj,
        o_proj: grad_o_proj,
        grad_query_src,
        grad_key_src,
    }
}

pub fn encoder_block_forward_cached(
    hidden: &[f64],
    seq_len: usize,
    width: usize,
    num_heads: usize,
    block: &EncoderBlock,
) -> (Vec<f64>, EncoderBlockCache) {
    let mut q = vec![0.0; hidden.len()];
    let mut k = vec![0.0; hidden.len()];
    let mut v = vec![0.0; hidden.len()];
    for t in 0..seq_len {
        let row = &hidden[t * width..(t + 1) * width];
        let q_row = linear(row, &block.q_proj, &vec![0.0; width], width);
        let k_row = linear(row, &block.k_proj, &vec![0.0; width], width);
        let v_row = linear(row, &block.v_proj, &vec![0.0; width], width);
        q[t * width..(t + 1) * width].copy_from_slice(&q_row);
        k[t * width..(t + 1) * width].copy_from_slice(&k_row);
        v[t * width..(t + 1) * width].copy_from_slice(&v_row);
    }

    let head_width = (width / num_heads.max(1)).max(1);
    let mut attended = vec![0.0; hidden.len()];
    let mut attn_weights = vec![0.0; num_heads.max(1) * seq_len * seq_len];
    for t in 0..seq_len {
        for head in 0..num_heads.max(1) {
            let start = head * head_width;
            let end = ((head + 1) * head_width).min(width);
            if start >= end {
                continue;
            }
            let q_row = &q[t * width + start..t * width + end];
            let mut scores = vec![0.0; seq_len];
            for s in 0..seq_len {
                let k_row = &k[s * width + start..s * width + end];
                let score = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    / ((end - start) as f64).sqrt().max(1.0);
                scores[s] = score;
            }
            let weights = softmax(&scores);
            let weight_base = (head * seq_len + t) * seq_len;
            attn_weights[weight_base..weight_base + seq_len].copy_from_slice(&weights);
            for (s, weight) in weights.iter().copied().enumerate() {
                let v_row = &v[s * width + start..s * width + end];
                let dst = &mut attended[t * width + start..t * width + end];
                for (dst_h, src_h) in dst.iter_mut().zip(v_row.iter()) {
                    *dst_h += weight * *src_h;
                }
            }
        }
    }

    let mut mixed = vec![0.0; hidden.len()];
    for t in 0..seq_len {
        let attn_row = &attended[t * width..(t + 1) * width];
        let proj_row = linear(attn_row, &block.o_proj, &vec![0.0; width], width);
        let out_row = &mut mixed[t * width..(t + 1) * width];
        for i in 0..width {
            out_row[i] = (hidden[t * width + i] + proj_row[i]).tanh();
        }
    }

    let ffn_hidden = block.ff_in_bias.len();
    let mut out = vec![0.0; hidden.len()];
    let mut ff1_pre_all = vec![0.0; seq_len * ffn_hidden];
    let mut ff1_act_all = vec![0.0; seq_len * ffn_hidden];
    for t in 0..seq_len {
        let row = &mixed[t * width..(t + 1) * width];
        let ff1 = linear(row, &block.ff_in, &block.ff_in_bias, ffn_hidden);
        ff1_pre_all[t * ffn_hidden..(t + 1) * ffn_hidden].copy_from_slice(&ff1);
        let ff1: Vec<f64> = ff1.into_iter().map(gelu).collect();
        ff1_act_all[t * ffn_hidden..(t + 1) * ffn_hidden].copy_from_slice(&ff1);
        let ff2 = linear(&ff1, &block.ff_out, &block.ff_out_bias, width);
        let dst = &mut out[t * width..(t + 1) * width];
        for i in 0..width {
            dst[i] = (mixed[t * width + i] + ff2[i]).tanh();
        }
    }
    (
        out.clone(),
        EncoderBlockCache {
            input: hidden.to_vec(),
            q,
            k,
            v,
            attn_weights,
            attended,
            mixed,
            ff1_pre: ff1_pre_all,
            ff1_act: ff1_act_all,
            output: out,
        },
    )
}

pub fn decoder_block_forward_cached(
    hidden: &[f64],
    memory: &[f64],
    seq_len: usize,
    memory_len: usize,
    width: usize,
    num_heads: usize,
    block: &DecoderBlock,
) -> (Vec<f64>, DecoderBlockCache) {
    let (self_attended, self_cache) = attention_forward_cached(
        hidden,
        hidden,
        hidden,
        seq_len,
        seq_len,
        width,
        num_heads,
        &block.self_q_proj,
        &block.self_k_proj,
        &block.self_v_proj,
    );
    let mut mixed = vec![0.0; hidden.len()];
    for i in 0..hidden.len() {
        mixed[i] = (hidden[i]
            + linear(
                &self_attended[(i / width) * width..(i / width + 1) * width],
                &block.self_o_proj,
                &vec![0.0; width],
                width,
            )[i % width])
            .tanh();
    }
    let (cross_attended, cross_cache) = attention_forward_cached(
        &mixed,
        memory,
        memory,
        seq_len,
        memory_len,
        width,
        num_heads,
        &block.cross_q_proj,
        &block.cross_k_proj,
        &block.cross_v_proj,
    );
    let mut crossed = vec![0.0; hidden.len()];
    for i in 0..hidden.len() {
        crossed[i] = (mixed[i]
            + linear(
                &cross_attended[(i / width) * width..(i / width + 1) * width],
                &block.cross_o_proj,
                &vec![0.0; width],
                width,
            )[i % width])
            .tanh();
    }
    let ffn_hidden = block.ff_in_bias.len();
    let mut out = vec![0.0; hidden.len()];
    let mut ff1_pre_all = vec![0.0; seq_len * ffn_hidden];
    let mut ff1_act_all = vec![0.0; seq_len * ffn_hidden];
    for t in 0..seq_len {
        let row = &crossed[t * width..(t + 1) * width];
        let ff1 = linear(row, &block.ff_in, &block.ff_in_bias, ffn_hidden);
        ff1_pre_all[t * ffn_hidden..(t + 1) * ffn_hidden].copy_from_slice(&ff1);
        let ff1_act: Vec<f64> = ff1.into_iter().map(gelu).collect();
        ff1_act_all[t * ffn_hidden..(t + 1) * ffn_hidden].copy_from_slice(&ff1_act);
        let ff2 = linear(&ff1_act, &block.ff_out, &block.ff_out_bias, width);
        let dst = &mut out[t * width..(t + 1) * width];
        for i in 0..width {
            dst[i] = (crossed[t * width + i] + ff2[i]).tanh();
        }
    }
    (
        out.clone(),
        DecoderBlockCache {
            input: hidden.to_vec(),
            memory: memory.to_vec(),
            self_attn: self_cache,
            mixed,
            cross_attn: cross_cache,
            crossed,
            ff1_pre: ff1_pre_all,
            ff1_act: ff1_act_all,
            output: out,
        },
    )
}

pub fn decoder_block_backward(
    cache: &DecoderBlockCache,
    block: &DecoderBlock,
    grad_output: &[f64],
    seq_len: usize,
    width: usize,
) -> DecoderBlockGrads {
    let ffn_hidden = block.ff_in_bias.len();
    let mut grad_ff_in = vec![0.0; block.ff_in.len()];
    let mut grad_ff_in_bias = vec![0.0; block.ff_in_bias.len()];
    let mut grad_ff_out = vec![0.0; block.ff_out.len()];
    let mut grad_ff_out_bias = vec![0.0; block.ff_out_bias.len()];
    let mut grad_crossed = vec![0.0; cache.crossed.len()];

    for t in 0..seq_len {
        let out_row = &cache.output[t * width..(t + 1) * width];
        let crossed_row = &cache.crossed[t * width..(t + 1) * width];
        let grad_out_row = &grad_output[t * width..(t + 1) * width];
        let ff1_pre = &cache.ff1_pre[t * ffn_hidden..(t + 1) * ffn_hidden];
        let ff1_act = &cache.ff1_act[t * ffn_hidden..(t + 1) * ffn_hidden];
        let mut grad_after_tanh = vec![0.0; width];
        for i in 0..width {
            grad_after_tanh[i] = grad_out_row[i] * (1.0 - out_row[i] * out_row[i]);
        }
        for out_idx in 0..width {
            grad_ff_out_bias[out_idx] += grad_after_tanh[out_idx];
            let base = out_idx * ffn_hidden;
            for hid_idx in 0..ffn_hidden {
                grad_ff_out[base + hid_idx] += grad_after_tanh[out_idx] * ff1_act[hid_idx];
            }
        }
        let mut grad_ff1 = vec![0.0; ffn_hidden];
        for hid_idx in 0..ffn_hidden {
            let mut acc = 0.0;
            for out_idx in 0..width {
                acc += grad_after_tanh[out_idx] * block.ff_out[out_idx * ffn_hidden + hid_idx];
            }
            grad_ff1[hid_idx] = acc * gelu_grad(ff1_pre[hid_idx]);
            grad_ff_in_bias[hid_idx] += grad_ff1[hid_idx];
        }
        for hid_idx in 0..ffn_hidden {
            let base = hid_idx * width;
            for in_idx in 0..width {
                grad_ff_in[base + in_idx] += grad_ff1[hid_idx] * crossed_row[in_idx];
            }
        }
        for in_idx in 0..width {
            let mut grad = grad_after_tanh[in_idx];
            for hid_idx in 0..ffn_hidden {
                grad += grad_ff1[hid_idx] * block.ff_in[hid_idx * width + in_idx];
            }
            grad_crossed[t * width + in_idx] =
                grad * (1.0 - crossed_row[in_idx] * crossed_row[in_idx]);
        }
    }

    let cross_attn_grads = attention_backward(
        &cache.cross_attn,
        &grad_crossed,
        seq_len,
        cache.cross_attn.k.len() / width,
        width,
        &block.cross_o_proj,
        &block.cross_q_proj,
        &block.cross_k_proj,
        &block.cross_v_proj,
    );
    let mut grad_mixed = cross_attn_grads.grad_query_src.clone();
    for i in 0..grad_mixed.len() {
        grad_mixed[i] += grad_crossed[i];
        grad_mixed[i] *= 1.0 - cache.mixed[i] * cache.mixed[i];
    }
    let self_attn_grads = attention_backward(
        &cache.self_attn,
        &grad_mixed,
        seq_len,
        seq_len,
        width,
        &block.self_o_proj,
        &block.self_q_proj,
        &block.self_k_proj,
        &block.self_v_proj,
    );
    let mut grad_input = self_attn_grads.grad_query_src.clone();
    for i in 0..grad_input.len() {
        grad_input[i] += grad_mixed[i];
    }
    DecoderBlockGrads {
        self_q_proj: self_attn_grads.q_proj,
        self_k_proj: self_attn_grads.k_proj,
        self_v_proj: self_attn_grads.v_proj,
        self_o_proj: self_attn_grads.o_proj,
        cross_q_proj: cross_attn_grads.q_proj,
        cross_k_proj: cross_attn_grads.k_proj,
        cross_v_proj: cross_attn_grads.v_proj,
        cross_o_proj: cross_attn_grads.o_proj,
        ff_in: grad_ff_in,
        ff_in_bias: grad_ff_in_bias,
        ff_out: grad_ff_out,
        ff_out_bias: grad_ff_out_bias,
        grad_input,
        grad_memory: cross_attn_grads.grad_key_src,
    }
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_grad(x: f64) -> f64 {
    let x3 = x * x * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    let tanh_inner = inner.tanh();
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x)
}

pub fn encoder_block_backward(
    cache: &EncoderBlockCache,
    block: &EncoderBlock,
    grad_output: &[f64],
    seq_len: usize,
    width: usize,
) -> EncoderBlockGrads {
    let ffn_hidden = block.ff_in_bias.len();
    let num_heads = (width / (cache.attn_weights.len() / (seq_len * seq_len)).max(1)).max(1);
    let head_count = (cache.attn_weights.len() / (seq_len * seq_len)).max(1);
    let head_width = (width / head_count).max(1);
    let mut grad_q_proj = vec![0.0; block.q_proj.len()];
    let mut grad_k_proj = vec![0.0; block.k_proj.len()];
    let mut grad_v_proj = vec![0.0; block.v_proj.len()];
    let mut grad_o_proj = vec![0.0; block.o_proj.len()];
    let mut grad_ff_in = vec![0.0; block.ff_in.len()];
    let mut grad_ff_in_bias = vec![0.0; block.ff_in_bias.len()];
    let mut grad_ff_out = vec![0.0; block.ff_out.len()];
    let mut grad_ff_out_bias = vec![0.0; block.ff_out_bias.len()];
    let mut grad_input = vec![0.0; cache.input.len()];
    let mut grad_q = vec![0.0; cache.q.len()];
    let mut grad_k = vec![0.0; cache.k.len()];
    let mut grad_v = vec![0.0; cache.v.len()];
    let mut grad_attended = vec![0.0; cache.attended.len()];

    for t in 0..seq_len {
        let out_row = &cache.output[t * width..(t + 1) * width];
        let mixed_row = &cache.mixed[t * width..(t + 1) * width];
        let attended_row = &cache.attended[t * width..(t + 1) * width];
        let grad_out_row = &grad_output[t * width..(t + 1) * width];
        let ff1_pre = &cache.ff1_pre[t * ffn_hidden..(t + 1) * ffn_hidden];
        let ff1_act = &cache.ff1_act[t * ffn_hidden..(t + 1) * ffn_hidden];

        let mut grad_after_tanh = vec![0.0; width];
        for i in 0..width {
            grad_after_tanh[i] = grad_out_row[i] * (1.0 - out_row[i] * out_row[i]);
        }

        for out_idx in 0..width {
            grad_ff_out_bias[out_idx] += grad_after_tanh[out_idx];
            let base = out_idx * ffn_hidden;
            for hid_idx in 0..ffn_hidden {
                grad_ff_out[base + hid_idx] += grad_after_tanh[out_idx] * ff1_act[hid_idx];
            }
        }

        let mut grad_ff1 = vec![0.0; ffn_hidden];
        for hid_idx in 0..ffn_hidden {
            let mut acc = 0.0;
            for out_idx in 0..width {
                acc += grad_after_tanh[out_idx] * block.ff_out[out_idx * ffn_hidden + hid_idx];
            }
            grad_ff1[hid_idx] = acc * gelu_grad(ff1_pre[hid_idx]);
            grad_ff_in_bias[hid_idx] += grad_ff1[hid_idx];
        }
        for hid_idx in 0..ffn_hidden {
            let base = hid_idx * width;
            for in_idx in 0..width {
                grad_ff_in[base + in_idx] += grad_ff1[hid_idx] * mixed_row[in_idx];
            }
        }

        let mut grad_mixed = grad_after_tanh.clone();
        for in_idx in 0..width {
            for hid_idx in 0..ffn_hidden {
                grad_mixed[in_idx] += grad_ff1[hid_idx] * block.ff_in[hid_idx * width + in_idx];
            }
        }

        let mut grad_pre_mixed = vec![0.0; width];
        for i in 0..width {
            grad_pre_mixed[i] = grad_mixed[i] * (1.0 - mixed_row[i] * mixed_row[i]);
            grad_input[t * width + i] += grad_pre_mixed[i];
        }

        for out_idx in 0..width {
            let base = out_idx * width;
            for in_idx in 0..width {
                grad_o_proj[base + in_idx] += grad_pre_mixed[out_idx] * attended_row[in_idx];
                grad_attended[t * width + in_idx] +=
                    grad_pre_mixed[out_idx] * block.o_proj[base + in_idx];
            }
        }
    }

    for t in 0..seq_len {
        for head in 0..head_count {
            let start = head * head_width;
            let end = ((head + 1) * head_width).min(width);
            if start >= end {
                continue;
            }
            let weight_base = (head * seq_len + t) * seq_len;
            let weights = &cache.attn_weights[weight_base..weight_base + seq_len];
            let grad_attn_row = &grad_attended[t * width + start..t * width + end];
            let q_row = &cache.q[t * width + start..t * width + end];

            let mut grad_weights = vec![0.0; seq_len];
            for s in 0..seq_len {
                let v_row = &cache.v[s * width + start..s * width + end];
                for dim in 0..(end - start) {
                    grad_weights[s] += grad_attn_row[dim] * v_row[dim];
                    grad_v[s * width + start + dim] += grad_attn_row[dim] * weights[s];
                }
            }

            let dot: f64 = grad_weights
                .iter()
                .zip(weights.iter())
                .map(|(g, w)| g * w)
                .sum();
            let mut grad_scores = vec![0.0; seq_len];
            for s in 0..seq_len {
                grad_scores[s] = weights[s] * (grad_weights[s] - dot);
            }

            let scale = ((end - start) as f64).sqrt().max(1.0);
            for s in 0..seq_len {
                let k_row = &cache.k[s * width + start..s * width + end];
                for dim in 0..(end - start) {
                    grad_q[t * width + start + dim] += grad_scores[s] * k_row[dim] / scale;
                    grad_k[s * width + start + dim] += grad_scores[s] * q_row[dim] / scale;
                }
            }
        }
    }

    for t in 0..seq_len {
        let input_row = &cache.input[t * width..(t + 1) * width];
        let q_row = &cache.q[t * width..(t + 1) * width];
        let k_row = &cache.k[t * width..(t + 1) * width];
        let v_row = &cache.v[t * width..(t + 1) * width];
        let grad_q_row = &grad_q[t * width..(t + 1) * width];
        let grad_k_row = &grad_k[t * width..(t + 1) * width];
        let grad_v_row = &grad_v[t * width..(t + 1) * width];
        for out_idx in 0..width {
            let base = out_idx * width;
            for in_idx in 0..width {
                grad_q_proj[base + in_idx] += grad_q_row[out_idx] * input_row[in_idx];
                grad_k_proj[base + in_idx] += grad_k_row[out_idx] * input_row[in_idx];
                grad_v_proj[base + in_idx] += grad_v_row[out_idx] * input_row[in_idx];
                grad_input[t * width + in_idx] += grad_q_row[out_idx] * block.q_proj[base + in_idx];
                grad_input[t * width + in_idx] += grad_k_row[out_idx] * block.k_proj[base + in_idx];
                grad_input[t * width + in_idx] += grad_v_row[out_idx] * block.v_proj[base + in_idx];
            }
        }
        let _ = (q_row, k_row, v_row, num_heads);
    }

    EncoderBlockGrads {
        q_proj: grad_q_proj,
        k_proj: grad_k_proj,
        v_proj: grad_v_proj,
        o_proj: grad_o_proj,
        ff_in: grad_ff_in,
        ff_in_bias: grad_ff_in_bias,
        ff_out: grad_ff_out,
        ff_out_bias: grad_ff_out_bias,
        grad_input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DecoderBlock, EncoderBlock};

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cross_entropy_grad_matches_target_shape() {
        let (loss, grad) = cross_entropy_grad(&[0.1, 0.7, 0.2], 1);
        assert!(loss > 0.0);
        assert_eq!(grad.len(), 3);
        assert!(grad[1] < 0.0);
    }

    #[test]
    fn masked_mean_over_time_respects_time_mask() {
        let input = vec![
            1.0, 3.0, //
            10.0, 30.0, //
            100.0, 300.0,
        ];
        let mean = masked_mean_over_time(&input, &[1.0, 0.0, 1.0], 2);
        assert_eq!(mean, vec![50.5, 151.5]);
    }

    #[test]
    fn encoder_block_forward_preserves_shape() {
        let block = EncoderBlock {
            q_proj: vec![0.1; 16],
            k_proj: vec![0.1; 16],
            v_proj: vec![0.1; 16],
            o_proj: vec![0.1; 16],
            ff_in: vec![0.1; 32],
            ff_in_bias: vec![0.0; 8],
            ff_out: vec![0.1; 32],
            ff_out_bias: vec![0.0; 4],
        };
        let hidden = vec![0.1; 12];
        let out = encoder_block_forward(&hidden, 3, 4, 2, &block);
        assert_eq!(out.len(), hidden.len());
    }

    #[test]
    fn decoder_block_forward_preserves_shape() {
        let block = DecoderBlock {
            self_q_proj: vec![0.1; 16],
            self_k_proj: vec![0.1; 16],
            self_v_proj: vec![0.1; 16],
            self_o_proj: vec![0.1; 16],
            cross_q_proj: vec![0.1; 16],
            cross_k_proj: vec![0.1; 16],
            cross_v_proj: vec![0.1; 16],
            cross_o_proj: vec![0.1; 16],
            ff_in: vec![0.1; 32],
            ff_in_bias: vec![0.0; 8],
            ff_out: vec![0.1; 32],
            ff_out_bias: vec![0.0; 4],
        };
        let hidden = vec![0.1; 12];
        let memory = vec![0.2; 12];
        let out = decoder_block_forward(&hidden, &memory, 3, 3, 4, 2, &block);
        assert_eq!(out.len(), hidden.len());
    }

    #[test]
    fn decoder_block_backward_handles_mismatched_memory_len() {
        let block = DecoderBlock {
            self_q_proj: vec![0.1; 16],
            self_k_proj: vec![0.1; 16],
            self_v_proj: vec![0.1; 16],
            self_o_proj: vec![0.1; 16],
            cross_q_proj: vec![0.1; 16],
            cross_k_proj: vec![0.1; 16],
            cross_v_proj: vec![0.1; 16],
            cross_o_proj: vec![0.1; 16],
            ff_in: vec![0.1; 32],
            ff_in_bias: vec![0.0; 8],
            ff_out: vec![0.1; 32],
            ff_out_bias: vec![0.0; 4],
        };
        let hidden = vec![0.1; 8];
        let memory = vec![0.2; 12];
        let (out, cache) = decoder_block_forward_cached(&hidden, &memory, 2, 3, 4, 2, &block);
        let grad_output = vec![0.01; out.len()];
        let grads = decoder_block_backward(&cache, &block, &grad_output, 2, 4);
        assert_eq!(grads.grad_input.len(), hidden.len());
        assert_eq!(grads.grad_memory.len(), memory.len());
    }

    #[test]
    fn mean_over_rows_computes_average() {
        let values = vec![1.0, 3.0, 5.0, 7.0];
        let mean = mean_over_rows(&values, 2, 2);
        assert_eq!(mean, vec![3.0, 5.0]);
    }
}
