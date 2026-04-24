//! The CTC-NAT student model, built on `tch::nn::VarStore`.
//!
//! Matches the Python `CTCNAT` (legacy Python reference in shape:
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
use crate::backend::{BackendConfig, CvaeConfig};
use anyhow::Result;
use tch::nn::{self, Embedding, Init, Linear, LinearConfig, Module, Path, RNN};
use tch::{Kind, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct CvaeLabelSpaces {
    pub writer_labels: i64,
    pub domain_labels: i64,
    pub source_labels: i64,
}

impl CvaeLabelSpaces {
    pub fn new(writer_labels: usize, domain_labels: usize, source_labels: usize) -> Self {
        Self {
            writer_labels: writer_labels.max(1) as i64,
            domain_labels: domain_labels.max(1) as i64,
            source_labels: source_labels.max(1) as i64,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct CvaeOutput {
    latent: Tensor,
    mean: Tensor,
    logvar: Tensor,
    kl: Tensor,
    film_conditioning: Vec<(Tensor, Tensor)>,
}

#[derive(Debug)]
pub(crate) struct ProposalOutput {
    pub(crate) encoder_out: Tensor,
    pub(crate) proposal_logits: Tensor,
    pub(crate) film_conditioning: Option<Vec<(Tensor, Tensor)>>,
    pub(crate) kl: Option<Tensor>,
}

#[derive(Debug)]
struct LabelPriorEncoder {
    writer_embedding: Embedding,
    domain_embedding: Embedding,
    source_embedding: Embedding,
    mlp_in: Linear,
    mlp_out: Linear,
}

impl LabelPriorEncoder {
    fn new(p: &Path, label_hidden: i64, latent: i64, label_spaces: CvaeLabelSpaces) -> Self {
        let embed_cfg = nn::EmbeddingConfig::default();
        Self {
            writer_embedding: nn::embedding(
                p / "writer_embedding",
                label_spaces.writer_labels,
                label_hidden,
                embed_cfg,
            ),
            domain_embedding: nn::embedding(
                p / "domain_embedding",
                label_spaces.domain_labels,
                label_hidden,
                embed_cfg,
            ),
            source_embedding: nn::embedding(
                p / "source_embedding",
                label_spaces.source_labels,
                label_hidden,
                embed_cfg,
            ),
            mlp_in: nn::linear(
                p / "mlp_in",
                label_hidden * 3,
                label_hidden * 2,
                Default::default(),
            ),
            mlp_out: nn::linear(
                p / "mlp_out",
                label_hidden * 2,
                latent * 2,
                Default::default(),
            ),
        }
    }

    fn forward(
        &self,
        writer_ids: Option<&Tensor>,
        domain_ids: Option<&Tensor>,
        source_ids: Option<&Tensor>,
        batch_size: i64,
        device: tch::Device,
    ) -> (Tensor, Tensor) {
        let zeros = Tensor::zeros([batch_size], (Kind::Int64, device));
        let writer_ids = writer_ids.unwrap_or(&zeros);
        let domain_ids = domain_ids.unwrap_or(&zeros);
        let source_ids = source_ids.unwrap_or(&zeros);
        let x = Tensor::cat(
            &[
                self.writer_embedding.forward(writer_ids),
                self.domain_embedding.forward(domain_ids),
                self.source_embedding.forward(source_ids),
            ],
            -1,
        );
        let stats = self.mlp_out.forward(&self.mlp_in.forward(&x).gelu("none"));
        let parts = stats.split(stats.size()[1] / 2, -1);
        (parts[0].shallow_clone(), parts[1].shallow_clone())
    }
}

#[derive(Debug)]
struct FiLMProjector {
    to_gamma: Vec<Linear>,
    to_beta: Vec<Linear>,
}

impl FiLMProjector {
    fn new(p: &Path, latent: i64, hidden: i64, num_layers: usize) -> Self {
        let to_gamma = (0..num_layers)
            .map(|idx| {
                nn::linear(
                    p / format!("gamma_{idx}"),
                    latent,
                    hidden,
                    Default::default(),
                )
            })
            .collect();
        let to_beta = (0..num_layers)
            .map(|idx| {
                nn::linear(
                    p / format!("beta_{idx}"),
                    latent,
                    hidden,
                    Default::default(),
                )
            })
            .collect();
        Self { to_gamma, to_beta }
    }

    fn forward(&self, latent: &Tensor) -> Vec<(Tensor, Tensor)> {
        self.to_gamma
            .iter()
            .zip(self.to_beta.iter())
            .map(|(gamma_layer, beta_layer)| {
                let gamma = gamma_layer.forward(latent).unsqueeze(1) + 1.0;
                let beta = beta_layer.forward(latent).unsqueeze(1);
                (gamma, beta)
            })
            .collect()
    }
}

#[derive(Debug)]
struct CvaeConditioner {
    posterior: nn::GRU,
    posterior_mean: Linear,
    posterior_logvar: Linear,
    prior: LabelPriorEncoder,
    film: FiLMProjector,
}

impl CvaeConditioner {
    fn new(
        p: &Path,
        hidden_size: i64,
        num_decoder_layers: usize,
        config: &CvaeConfig,
        label_spaces: CvaeLabelSpaces,
    ) -> Self {
        let posterior_cfg = nn::RNNConfig {
            bidirectional: true,
            batch_first: true,
            ..Default::default()
        };
        let posterior_hidden = config.posterior_hidden_size.max(1) as i64;
        let latent_size = config.latent_size.max(1) as i64;
        let label_hidden = config.label_hidden_size.max(1) as i64;
        Self {
            posterior: nn::gru(
                p / "posterior_gru",
                hidden_size,
                posterior_hidden,
                posterior_cfg,
            ),
            posterior_mean: nn::linear(
                p / "posterior_mean",
                posterior_hidden * 2,
                latent_size,
                Default::default(),
            ),
            posterior_logvar: nn::linear(
                p / "posterior_logvar",
                posterior_hidden * 2,
                latent_size,
                Default::default(),
            ),
            prior: LabelPriorEncoder::new(&(p / "prior"), label_hidden, latent_size, label_spaces),
            film: FiLMProjector::new(&(p / "film"), latent_size, hidden_size, num_decoder_layers),
        }
    }

    fn reparameterize(mean: &Tensor, logvar: &Tensor) -> Tensor {
        let std = (logvar * 0.5).exp();
        let eps = Tensor::randn_like(&std);
        mean + eps * std
    }

    fn kl_divergence(
        q_mean: &Tensor,
        q_logvar: &Tensor,
        p_mean: &Tensor,
        p_logvar: &Tensor,
    ) -> Tensor {
        let q_var = q_logvar.exp();
        let p_var = p_logvar.exp();
        let mean_diff = q_mean - p_mean;
        let kl: Tensor = 0.5
            * (p_logvar - q_logvar + (q_var + &mean_diff * &mean_diff) / p_var.clamp_min(1e-6)
                - 1.0);
        kl.sum_dim_intlist([-1i64].as_ref(), false, Kind::Float)
            .mean(Kind::Float)
    }

    fn forward(
        &self,
        target_embeddings: Option<&Tensor>,
        target_valid: Option<&Tensor>,
        writer_ids: Option<&Tensor>,
        domain_ids: Option<&Tensor>,
        source_ids: Option<&Tensor>,
        batch_size: i64,
        device: tch::Device,
        sample_posterior: bool,
    ) -> CvaeOutput {
        let (prior_mean, prior_logvar) = self
            .prior
            .forward(writer_ids, domain_ids, source_ids, batch_size, device);
        match target_embeddings {
            None => {
                let latent = prior_mean.shallow_clone();
                let kl = Tensor::zeros([], (Kind::Float, device));
                CvaeOutput {
                    latent: latent.shallow_clone(),
                    mean: prior_mean,
                    logvar: prior_logvar,
                    kl,
                    film_conditioning: self.film.forward(&latent),
                }
            }
            Some(target_embeddings) => {
                let (outputs, _) = self.posterior.seq(target_embeddings);
                let pooled = match target_valid {
                    Some(valid) => {
                        let valid_f = valid.to_kind(Kind::Float).unsqueeze(-1);
                        (&outputs * &valid_f).sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                            / valid_f
                                .sum_dim_intlist([1i64].as_ref(), false, Kind::Float)
                                .clamp_min(1.0)
                    }
                    None => outputs.mean_dim([1i64].as_ref(), false, Kind::Float),
                };
                let post_mean = self.posterior_mean.forward(&pooled);
                let post_logvar = self.posterior_logvar.forward(&pooled);
                let latent = if sample_posterior {
                    Self::reparameterize(&post_mean, &post_logvar)
                } else {
                    post_mean.shallow_clone()
                };
                let kl = Self::kl_divergence(&post_mean, &post_logvar, &prior_mean, &prior_logvar);
                CvaeOutput {
                    latent: latent.shallow_clone(),
                    mean: post_mean,
                    logvar: post_logvar,
                    kl,
                    film_conditioning: self.film.forward(&latent),
                }
            }
        }
    }
}

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
    /// Scalar KL term from the optional CVAE path.
    pub kl: Option<Tensor>,
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
    cvae: Option<CvaeConditioner>,

    pub blank_id: i64,
    pub mask_token_id: i64,
}

impl CtcNatModel {
    pub fn new(
        p: &Path,
        config: &BackendConfig,
        cvae_config: &CvaeConfig,
        cvae_labels: CvaeLabelSpaces,
    ) -> Result<Self> {
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
        let cvae = if cvae_config.enabled {
            Some(CvaeConditioner::new(
                &(p / "cvae"),
                hidden,
                config.decoder_layers,
                cvae_config,
                cvae_labels,
            ))
        } else {
            None
        };

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
            cvae,
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
    pub fn proposal(
        &self,
        encoder_out: &Tensor,
        attention_mask: &Tensor,
        film_conditioning: Option<&[(Tensor, Tensor)]>,
    ) -> Tensor {
        let (b, t, _h) = encoder_out.size3().expect("encoder_out must be 3-D");
        let device = encoder_out.device();
        let positions = Tensor::arange(t, (Kind::Int64, device))
            .unsqueeze(0)
            .expand([b, t], false);
        let mut x = encoder_out + self.proposal_pos_embed.forward(&positions);
        for (layer_idx, layer) in self.proposal_layers.iter().enumerate() {
            let film_condition = film_conditioning
                .and_then(|all| all.get(layer_idx))
                .map(|(gamma, beta)| (gamma, beta));
            x = layer.forward(
                &x,
                encoder_out,
                Some(attention_mask),
                Some(attention_mask),
                film_condition,
            );
        }
        let x = self.proposal_final_norm.forward(&x);
        self.tied_projection(&x, &self.ctc_head_bias)
    }

    fn build_cvae_output(
        &self,
        target_ids: Option<&Tensor>,
        target_lengths: Option<&Tensor>,
        writer_ids: Option<&Tensor>,
        domain_ids: Option<&Tensor>,
        source_ids: Option<&Tensor>,
        batch_size: i64,
        device: tch::Device,
        sample_posterior: bool,
    ) -> Option<CvaeOutput> {
        let cvae = self.cvae.as_ref()?;
        let target_embeddings = target_ids.map(|ids| self.token_embed.forward(ids));
        let target_valid = match (target_ids, target_lengths) {
            (Some(ids), Some(lengths)) => {
                let seq_len = ids.size()[1];
                let positions = Tensor::arange(seq_len, (Kind::Int64, device)).unsqueeze(0);
                Some(positions.lt_tensor(&lengths.unsqueeze(-1)))
            }
            _ => None,
        };
        Some(cvae.forward(
            target_embeddings.as_ref(),
            target_valid.as_ref(),
            writer_ids,
            domain_ids,
            source_ids,
            batch_size,
            device,
            sample_posterior,
        ))
    }

    pub fn proposal_output(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        target_ids: Option<&Tensor>,
        target_lengths: Option<&Tensor>,
        writer_ids: Option<&Tensor>,
        domain_ids: Option<&Tensor>,
        source_ids: Option<&Tensor>,
        sample_posterior: bool,
    ) -> ProposalOutput {
        let encoder_out = self.encode(input_ids, attention_mask);
        let batch_size = input_ids.size()[0];
        let device = input_ids.device();
        let cvae_output = self.build_cvae_output(
            target_ids,
            target_lengths,
            writer_ids,
            domain_ids,
            source_ids,
            batch_size,
            device,
            sample_posterior,
        );
        let proposal_logits = self.proposal(
            &encoder_out,
            attention_mask,
            cvae_output
                .as_ref()
                .map(|output| output.film_conditioning.as_slice()),
        );
        match cvae_output {
            Some(output) => ProposalOutput {
                encoder_out,
                proposal_logits,
                film_conditioning: Some(output.film_conditioning),
                kl: Some(output.kl),
            },
            None => ProposalOutput {
                encoder_out,
                proposal_logits,
                film_conditioning: None,
                kl: None,
            },
        }
    }

    /// Run the refinement decoder on a masked hypothesis.
    /// Returns (refined_logits [B, T, V], remask_logits [B, T], stop_logit [B]).
    pub fn refine(
        &self,
        hypothesis_ids: &Tensor,
        hypothesis_mask: &Tensor,
        encoder_out: &Tensor,
        encoder_mask: &Tensor,
        film_conditioning: Option<&[(Tensor, Tensor)]>,
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
        for (layer_idx, layer) in self.refine_layers.iter().enumerate() {
            let film_condition = film_conditioning
                .and_then(|all| all.get(layer_idx))
                .map(|(gamma, beta)| (gamma, beta));
            x = layer.forward(
                &x,
                encoder_out,
                Some(hypothesis_mask),
                Some(encoder_mask),
                film_condition,
            );
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
        let proposal_logits = self.proposal(&encoder_out, attention_mask, None);
        CtcNatForward {
            encoder_out,
            proposal_logits,
            refined_logits: None,
            remask_logits: None,
            stop_logit: None,
            kl: None,
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
        let proposal = self.proposal_output(
            input_ids,
            attention_mask,
            None,
            None,
            None,
            None,
            None,
            false,
        );
        let (refined_logits, remask_logits, stop_logit) = self.refine(
            hypothesis_ids,
            hypothesis_mask,
            &proposal.encoder_out,
            attention_mask,
            proposal.film_conditioning.as_deref(),
        );
        CtcNatForward {
            encoder_out: proposal.encoder_out,
            proposal_logits: proposal.proposal_logits,
            refined_logits: Some(refined_logits),
            remask_logits: Some(remask_logits),
            stop_logit: Some(stop_logit),
            kl: proposal.kl,
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
        let model = CtcNatModel::new(
            &vs.root(),
            &cfg,
            &CvaeConfig::default(),
            CvaeLabelSpaces::new(1, 1, 1),
        )
        .unwrap();
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
        let model = CtcNatModel::new(
            &vs.root(),
            &cfg,
            &CvaeConfig::default(),
            CvaeLabelSpaces::new(1, 1, 1),
        )
        .unwrap();
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
    /// the Python reference (~41.04M reported by rust-model's estimator).
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
        let _model = CtcNatModel::new(
            &vs.root(),
            &cfg,
            &CvaeConfig::default(),
            CvaeLabelSpaces::new(1, 1, 1),
        )
        .unwrap();
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

    #[test]
    fn cvae_proposal_output_produces_kl_and_shapes() {
        let vs = VarStore::new(Device::Cpu);
        let cfg = tiny_config();
        let cvae = CvaeConfig {
            enabled: true,
            ..CvaeConfig::default()
        };
        let model =
            CtcNatModel::new(&vs.root(), &cfg, &cvae, CvaeLabelSpaces::new(4, 3, 2)).unwrap();
        let input_ids = Tensor::randint(cfg.output_size as i64, [2, 8], (Kind::Int64, Device::Cpu));
        let input_mask = Tensor::ones([2, 8], (Kind::Bool, Device::Cpu));
        let target_ids =
            Tensor::randint(cfg.output_size as i64, [2, 6], (Kind::Int64, Device::Cpu));
        let target_lengths = Tensor::from_slice(&[6i64, 4]);
        let writer_ids = Tensor::from_slice(&[1i64, 2]);
        let domain_ids = Tensor::from_slice(&[1i64, 0]);
        let source_ids = Tensor::from_slice(&[1i64, 1]);
        let out = model.proposal_output(
            &input_ids,
            &input_mask,
            Some(&target_ids),
            Some(&target_lengths),
            Some(&writer_ids),
            Some(&domain_ids),
            Some(&source_ids),
            true,
        );
        assert_eq!(out.encoder_out.size(), vec![2, 8, 16]);
        assert_eq!(out.proposal_logits.size(), vec![2, 8, 12]);
        assert!(out.kl.is_some());
        assert_eq!(
            out.film_conditioning.as_ref().map(|layers| layers.len()),
            Some(cfg.decoder_layers)
        );
    }
}
