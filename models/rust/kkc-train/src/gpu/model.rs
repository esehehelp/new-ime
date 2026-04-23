//! The CTC-NAT student model, built on `tch::nn::VarStore`.
//!
//! Matches the Python `CTCNAT` (models/src/model/ctc_nat.py) in shape:
//!
//! ```text
//!   input_ids [B, T]
//!        │
//!        ▼
//!   SmallEncoder:
//!     token_embed[V, H] + pos_embed[T_max, H]
//!     → N × EncoderLayer (post-norm, GELU) → LayerNorm
//!        │
//!        ▼  encoder_out [B, T, H]
//!        │
//!        ├──► NATDecoder (proposal, no token embed):
//!        │      pos_embed[T_max, H]
//!        │      → N × DecoderLayer (pre-norm, self-attn + cross-attn)
//!        │      → LayerNorm → ctc_head: Linear(H, V)
//!        │      => proposal_logits [B, T, V]
//!        │
//!        └──► MaskCTCRefinementDecoder (refine):
//!               token_embed[V, H] + pos_embed[T_max, H]
//!               → N × DecoderLayer → LayerNorm
//!               → refine_head, remask_head(Linear H→1), stop_head(Linear H→1)
//!               => refined_logits [B, T, V], remask [B, T], stop [B]
//! ```
//!
//! Step 1 only exposes forward. Losses and backward land in step 2.

use super::layers::{DecoderLayer, EncoderLayer};
use crate::backend::BackendConfig;
use anyhow::Result;
use tch::nn::{self, Embedding, Init, LinearConfig, Module, Path};
use tch::{Kind, Tensor};

/// All tensors the model produces per batch. Kept as one struct so the
/// loss module in step 2 can consume a single value.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub struct CtcNatForward {
    pub encoder_out: Tensor,
    /// `[B, T, V]` — CTC logits from the proposal head.
    pub proposal_logits: Tensor,
    /// `[B, T, V]` — refined logits. Only populated when `refine` was run.
    pub refined_logits: Option<Tensor>,
    /// `[B, T]` — per-position logit for the learned remask head.
    pub remask_logits: Option<Tensor>,
    /// `[B]` — per-sample logit for the learned stop head.
    pub stop_logit: Option<Tensor>,
}

#[derive(Debug)]
pub struct CtcNatModel {
    /// Shared source-side token embedding. Tied to the CTC head projection
    /// and the refinement decoder's token embedding, matching Python
    /// CTCNAT (ctc_nat.py:85-143). Sharing cuts ~5.5M params vs. the naive
    /// layout and is load-bearing for parity with the `phase3_30m` preset
    /// (41.04M reference).
    token_embed: Embedding,
    pos_embed: Embedding,
    encoder_layers: Vec<EncoderLayer>,
    encoder_final_norm: nn::LayerNorm,

    proposal_pos_embed: Embedding,
    proposal_layers: Vec<DecoderLayer>,
    proposal_final_norm: nn::LayerNorm,
    /// Bias for the tied CTC head (`logits = x @ token_embed.ws.T + bias`).
    ctc_head_bias: Tensor,

    refine_pos_embed: Embedding,
    refine_layers: Vec<DecoderLayer>,
    refine_final_norm: nn::LayerNorm,
    /// Bias for the tied refine head (shares weights with `token_embed`).
    refine_head_bias: Tensor,
    remask_head: nn::Linear,
    stop_head: nn::Linear,

    pub blank_id: i64,
    pub mask_token_id: i64,
}

impl CtcNatModel {
    pub fn new(p: &Path, config: &BackendConfig) -> Result<Self> {
        let hidden = config.hidden_size as i64;
        let vocab = config.output_size as i64;
        let max_positions = config.max_positions as i64;
        let embed_cfg = nn::EmbeddingConfig::default();
        let linear_cfg = LinearConfig::default();

        let token_embed = nn::embedding(p / "token_embed", vocab, hidden, embed_cfg);
        let pos_embed = nn::embedding(p / "pos_embed", max_positions, hidden, embed_cfg);

        let encoder_layers = (0..config.encoder_layers as i64)
            .map(|i| {
                EncoderLayer::new(
                    &(p / format!("enc_{i}")),
                    hidden,
                    config.num_heads as i64,
                    config.ffn_size as i64,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let encoder_final_norm =
            nn::layer_norm(p / "enc_final_norm", vec![hidden], Default::default());

        let proposal_pos_embed =
            nn::embedding(p / "proposal_pos_embed", max_positions, hidden, embed_cfg);
        let proposal_layers = (0..config.decoder_layers as i64)
            .map(|i| {
                DecoderLayer::new(
                    &(p / format!("prop_{i}")),
                    hidden,
                    config.decoder_heads as i64,
                    config.decoder_ffn_size as i64,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let proposal_final_norm =
            nn::layer_norm(p / "prop_final_norm", vec![hidden], Default::default());
        let ctc_head_bias = (p / "ctc_head_bias").var("bias", &[vocab], Init::Const(0.0));

        let refine_pos_embed =
            nn::embedding(p / "refine_pos_embed", max_positions, hidden, embed_cfg);
        let refine_layers = (0..config.decoder_layers as i64)
            .map(|i| {
                DecoderLayer::new(
                    &(p / format!("refine_{i}")),
                    hidden,
                    config.decoder_heads as i64,
                    config.decoder_ffn_size as i64,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let refine_final_norm =
            nn::layer_norm(p / "refine_final_norm", vec![hidden], Default::default());
        let refine_head_bias = (p / "refine_head_bias").var("bias", &[vocab], Init::Const(0.0));
        let remask_head = nn::linear(p / "remask_head", hidden, 1, linear_cfg);
        let stop_head = nn::linear(p / "stop_head", hidden, 1, linear_cfg);

        Ok(Self {
            token_embed,
            pos_embed,
            encoder_layers,
            encoder_final_norm,
            proposal_pos_embed,
            proposal_layers,
            proposal_final_norm,
            ctc_head_bias,
            refine_pos_embed,
            refine_layers,
            refine_final_norm,
            refine_head_bias,
            remask_head,
            stop_head,
            blank_id: config.blank_id as i64,
            mask_token_id: config.mask_token_id as i64,
        })
    }

    /// Apply the tied projection `x @ token_embed.ws^T + bias`. Used by
    /// both the CTC head and the refine head so they share the weight
    /// matrix with `token_embed`.
    fn tied_projection(&self, x: &Tensor, bias: &Tensor) -> Tensor {
        // token_embed.ws has shape [V, H]; we want [B, T, V] = x @ W^T + b
        let w = &self.token_embed.ws; // [V, H]
        let w_t = w.transpose(0, 1); // [H, V]
        x.matmul(&w_t) + bias
    }

    /// Encode `input_ids [B, T]` → `encoder_out [B, T, H]`.
    /// `attention_mask [B, T]`: true = valid.
    pub fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Tensor {
        let (b, t) = input_ids.size2().expect("input_ids must be 2-D");
        let device = input_ids.device();
        let positions = Tensor::arange(t, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([b, t], false);
        let mut x = self.token_embed.forward(input_ids) + self.pos_embed.forward(&positions);
        for layer in &self.encoder_layers {
            x = layer.forward(&x, Some(attention_mask));
        }
        self.encoder_final_norm.forward(&x)
    }

    /// Run the proposal (CTC) head on encoder output.
    pub fn proposal(&self, encoder_out: &Tensor, attention_mask: &Tensor) -> Tensor {
        let (b, t, _h) = encoder_out.size3().expect("encoder_out must be 3-D");
        let device = encoder_out.device();
        let positions = Tensor::arange(t, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([b, t], false);
        let mut x = encoder_out + self.proposal_pos_embed.forward(&positions);
        for layer in &self.proposal_layers {
            x = layer.forward(&x, encoder_out, Some(attention_mask), Some(attention_mask));
        }
        let x = self.proposal_final_norm.forward(&x);
        self.tied_projection(&x, &self.ctc_head_bias)
    }

    /// Run the refinement decoder on a masked hypothesis.
    /// Returns (refined_logits [B, T, V], remask_logits [B, T], stop_logit [B]).
    pub fn refine(
        &self,
        hypothesis_ids: &Tensor,
        hypothesis_mask: &Tensor,
        encoder_out: &Tensor,
        encoder_mask: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let (b, t) = hypothesis_ids.size2().expect("hypothesis_ids must be 2-D");
        let device = hypothesis_ids.device();
        let positions = Tensor::arange(t, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([b, t], false);
        // refine decoder shares the encoder's token embedding (see Python
        // ctc_nat.py:127-140 — `tied_embedding = encoder.get_input_embedding()`).
        let mut x =
            self.token_embed.forward(hypothesis_ids) + self.refine_pos_embed.forward(&positions);
        for layer in &self.refine_layers {
            x = layer.forward(&x, encoder_out, Some(hypothesis_mask), Some(encoder_mask));
        }
        let x = self.refine_final_norm.forward(&x);
        let refined_logits = self.tied_projection(&x, &self.refine_head_bias); // [B, T, V]
        let remask_logits = self.remask_head.forward(&x).squeeze_dim(-1); // [B, T]

        // Stop head: mean-pool valid positions then project to scalar.
        // Python stop_head takes the mean over attention_mask-valid tokens.
        let mask_f = hypothesis_mask.to_kind(Kind::Float).unsqueeze(-1); // [B, T, 1]
        let summed =
            (&x * &mask_f).sum_dim_intlist([1i64].as_ref(), /*keepdim=*/ false, Kind::Float); // [B, H]
        let counts = mask_f
            .sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
            .clamp_min(1.0); // [B, 1]
        let pooled = summed / counts;
        let stop_logit = self.stop_head.forward(&pooled).squeeze_dim(-1); // [B]

        (refined_logits, remask_logits, stop_logit)
    }

    /// Forward without refinement — cheap smoke path used by unit tests.
    #[cfg(test)]
    pub fn forward_proposal_only(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> CtcNatForward {
        let encoder_out = self.encode(input_ids, attention_mask);
        let proposal_logits = self.proposal(&encoder_out, attention_mask);
        CtcNatForward {
            encoder_out,
            proposal_logits,
            refined_logits: None,
            remask_logits: None,
            stop_logit: None,
        }
    }

    /// Forward with both proposal and refinement. Caller provides a
    /// pre-built masked hypothesis (typically the gold target with
    /// `mask_token_id` substituted at sampled positions — see
    /// `backend::CtcBackend::build_target_refinement_batch` for the CPU
    /// reference).
    #[cfg(test)]
    pub fn forward_with_refine(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        hypothesis_ids: &Tensor,
        hypothesis_mask: &Tensor,
    ) -> CtcNatForward {
        let encoder_out = self.encode(input_ids, attention_mask);
        let proposal_logits = self.proposal(&encoder_out, attention_mask);
        let (refined_logits, remask_logits, stop_logit) = self.refine(
            hypothesis_ids,
            hypothesis_mask,
            &encoder_out,
            attention_mask,
        );
        CtcNatForward {
            encoder_out,
            proposal_logits,
            refined_logits: Some(refined_logits),
            remask_logits: Some(remask_logits),
            stop_logit: Some(stop_logit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device};

    fn tiny_config() -> BackendConfig {
        BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            hidden_size: 16,
            encoder_layers: 2,
            num_heads: 4,
            ffn_size: 32,
            decoder_layers: 2,
            decoder_heads: 4,
            decoder_ffn_size: 32,
            output_size: 12,
            blank_id: 4,
            max_positions: 16,
            mask_token_id: 5,
            ..BackendConfig::default()
        }
    }

    #[test]
    fn forward_proposal_only_produces_expected_shapes() {
        let vs = VarStore::new(Device::Cpu);
        let cfg = tiny_config();
        let model = CtcNatModel::new(&vs.root(), &cfg).unwrap();
        let input_ids = Tensor::randint(cfg.output_size as i64, [2, 8], (Kind::Int64, Device::Cpu));
        let mask = Tensor::ones([2, 8], (Kind::Bool, Device::Cpu));
        let out = model.forward_proposal_only(&input_ids, &mask);
        assert_eq!(out.encoder_out.size(), vec![2, 8, 16]);
        assert_eq!(out.proposal_logits.size(), vec![2, 8, 12]);
        assert!(out.refined_logits.is_none());
    }

    #[test]
    fn forward_with_refine_produces_expected_shapes() {
        let vs = VarStore::new(Device::Cpu);
        let cfg = tiny_config();
        let model = CtcNatModel::new(&vs.root(), &cfg).unwrap();
        let input_ids = Tensor::randint(cfg.output_size as i64, [2, 8], (Kind::Int64, Device::Cpu));
        let input_mask = Tensor::ones([2, 8], (Kind::Bool, Device::Cpu));
        let hyp_ids = Tensor::randint(cfg.output_size as i64, [2, 6], (Kind::Int64, Device::Cpu));
        let hyp_mask = Tensor::ones([2, 6], (Kind::Bool, Device::Cpu));
        let out = model.forward_with_refine(&input_ids, &input_mask, &hyp_ids, &hyp_mask);
        assert_eq!(out.proposal_logits.size(), vec![2, 8, 12]);
        assert_eq!(out.refined_logits.as_ref().unwrap().size(), vec![2, 6, 12]);
        assert_eq!(out.remask_logits.as_ref().unwrap().size(), vec![2, 6]);
        assert_eq!(out.stop_logit.as_ref().unwrap().size(), vec![2]);
    }

    /// The `phase3_30m` preset is Suiko-v1-small's architecture. We want
    /// the tch model's trainable parameter count to land within ±3% of
    /// the Python reference (~41.04M reported by kkc-model's estimator).
    #[test]
    fn phase3_30m_parameter_count_is_within_three_percent_of_reference() {
        let cfg = BackendConfig {
            kind: "tch-ctc-nat".to_string(),
            hidden_size: 384,
            encoder_layers: 6,
            num_heads: 6,
            ffn_size: 1536,
            decoder_layers: 6,
            decoder_heads: 6,
            decoder_ffn_size: 1536,
            // `output_size` here stands in for the vocab used to estimate
            // param count. Char-5k tokenizer = 4801; pad to 4801 for the
            // sanity check. A production run uses the real vocab.
            output_size: 4801,
            blank_id: 4,
            max_positions: 128,
            mask_token_id: 5,
            ..BackendConfig::default()
        };
        let vs = VarStore::new(Device::Cpu);
        let _model = CtcNatModel::new(&vs.root(), &cfg).unwrap();
        let total: i64 = vs
            .trainable_variables()
            .iter()
            .map(|t| t.numel() as i64)
            .sum();
        let reference = 41_040_000i64;
        let diff = (total - reference).abs();
        let pct = (diff as f64) / (reference as f64);
        assert!(
            pct < 0.03,
            "param count {total} differs from reference {reference} by {pct:.3}"
        );
    }
}
