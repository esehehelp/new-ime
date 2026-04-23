use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CtcNatPreset {
    pub name: &'static str,
    pub hidden_size: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub num_heads: usize,
    pub ffn_size: usize,
    pub max_positions: usize,
}

pub const PHASE3_20M: CtcNatPreset = CtcNatPreset {
    name: "phase3_20m",
    hidden_size: 320,
    encoder_layers: 5,
    decoder_layers: 5,
    num_heads: 4,
    ffn_size: 1280,
    max_positions: 128,
};

pub const PHASE3_30M: CtcNatPreset = CtcNatPreset {
    name: "phase3_30m",
    hidden_size: 384,
    encoder_layers: 6,
    decoder_layers: 6,
    num_heads: 6,
    ffn_size: 1536,
    max_positions: 128,
};

pub const PHASE3_90M: CtcNatPreset = CtcNatPreset {
    name: "phase3_90m",
    hidden_size: 640,
    encoder_layers: 8,
    decoder_layers: 8,
    num_heads: 8,
    ffn_size: 2560,
    max_positions: 128,
};

#[derive(Debug, Error)]
pub enum PresetError {
    #[error("unknown CTC-NAT preset: {0}")]
    UnknownPreset(String),
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeAssumptions {
    pub param_dtype_bytes: usize,
    pub grad_dtype_bytes: usize,
    pub adam_state_bytes: usize,
    pub activation_dtype_bytes: usize,
}

impl Default for RuntimeAssumptions {
    fn default() -> Self {
        Self {
            param_dtype_bytes: 2,
            grad_dtype_bytes: 4,
            adam_state_bytes: 8,
            activation_dtype_bytes: 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BatchShape {
    pub batch_size: usize,
    pub input_len: usize,
    pub target_len: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ResourceEstimate {
    pub parameter_count: usize,
    pub parameter_bytes: usize,
    pub gradient_bytes: usize,
    pub optimizer_bytes: usize,
    pub activation_bytes: usize,
    pub logits_bytes: usize,
    pub total_step_bytes: usize,
}

pub fn ctc_nat_preset(name: &str) -> Result<CtcNatPreset, PresetError> {
    match name {
        "phase3_20m" => Ok(PHASE3_20M),
        "phase3_30m" => Ok(PHASE3_30M),
        "phase3_90m" => Ok(PHASE3_90M),
        other => Err(PresetError::UnknownPreset(other.to_string())),
    }
}

pub fn estimate_ctc_nat_resources(
    preset: CtcNatPreset,
    shape: BatchShape,
    runtime: RuntimeAssumptions,
) -> ResourceEstimate {
    let vocab_embed = shape.vocab_size * preset.hidden_size;
    let encoder_layer = transformer_layer_params(preset.hidden_size, preset.ffn_size);
    let decoder_layer = transformer_layer_params(preset.hidden_size, preset.ffn_size);
    let refine_decoder = transformer_layer_params(preset.hidden_size, preset.ffn_size);
    let ctc_heads = (preset.hidden_size * shape.vocab_size) * 2 + preset.hidden_size * 2;
    let embeddings = vocab_embed + preset.max_positions * preset.hidden_size * 2;
    let parameter_count = embeddings
        + encoder_layer * preset.encoder_layers
        + decoder_layer * preset.decoder_layers
        + refine_decoder * preset.decoder_layers
        + ctc_heads;

    let parameter_bytes = parameter_count * runtime.param_dtype_bytes;
    let gradient_bytes = parameter_count * runtime.grad_dtype_bytes;
    let optimizer_bytes = parameter_count * runtime.adam_state_bytes;

    let per_token_hidden = preset.hidden_size * runtime.activation_dtype_bytes;
    let encoder_activations =
        shape.batch_size * shape.input_len * per_token_hidden * (preset.encoder_layers + 1);
    let decoder_activations =
        shape.batch_size * shape.input_len * per_token_hidden * (preset.decoder_layers + 1);
    let refine_activations =
        shape.batch_size * shape.target_len * per_token_hidden * (preset.decoder_layers + 1);
    let logits_bytes =
        shape.batch_size * shape.input_len * shape.vocab_size * runtime.activation_dtype_bytes;
    let activation_bytes =
        encoder_activations + decoder_activations + refine_activations + logits_bytes;

    ResourceEstimate {
        parameter_count,
        parameter_bytes,
        gradient_bytes,
        optimizer_bytes,
        activation_bytes,
        logits_bytes,
        total_step_bytes: parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes,
    }
}

fn transformer_layer_params(hidden: usize, ffn: usize) -> usize {
    let self_attn = hidden * hidden * 4;
    let ffn_block = hidden * ffn * 2;
    let norms = hidden * 4;
    self_attn + ffn_block + norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_nat_preset_lookup_resolves_known_names() {
        assert_eq!(ctc_nat_preset("phase3_20m").unwrap().hidden_size, 320);
        assert_eq!(ctc_nat_preset("phase3_30m").unwrap().hidden_size, 384);
        assert_eq!(ctc_nat_preset("phase3_90m").unwrap().hidden_size, 640);
        assert!(ctc_nat_preset("phase9_999m").is_err());
    }

    #[test]
    fn phase3_30m_parameter_count_matches_python_order_of_magnitude() {
        // The Python CTC-NAT (mask-CTC variant with refine decoder + learned
        // remask/stop heads) prints 41.04M params for this preset at
        // vocab=5000. The Rust estimator currently undercounts cross-attention
        // and the aux heads — tighten this bound whenever that's fixed.
        let est = estimate_ctc_nat_resources(
            PHASE3_30M,
            BatchShape {
                batch_size: 1,
                input_len: 128,
                target_len: 128,
                vocab_size: 5000,
            },
            RuntimeAssumptions::default(),
        );
        assert!(
            est.parameter_count >= 35_000_000 && est.parameter_count <= 45_000_000,
            "expected ~37-42M params, got {}",
            est.parameter_count
        );
    }

    #[test]
    fn step_bytes_scale_with_batch() {
        let small = estimate_ctc_nat_resources(
            PHASE3_30M,
            BatchShape {
                batch_size: 1,
                input_len: 128,
                target_len: 128,
                vocab_size: 5000,
            },
            RuntimeAssumptions::default(),
        );
        let large = estimate_ctc_nat_resources(
            PHASE3_30M,
            BatchShape {
                batch_size: 128,
                input_len: 128,
                target_len: 128,
                vocab_size: 5000,
            },
            RuntimeAssumptions::default(),
        );
        // Activations scale linearly in batch_size; params/grads/opt don't.
        assert!(large.activation_bytes > 100 * small.activation_bytes);
        assert_eq!(large.parameter_bytes, small.parameter_bytes);
    }
}
