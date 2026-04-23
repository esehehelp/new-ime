use super::layers::{DecoderLayer, EncoderLayer};
use super::BenchBackendConfig;
use anyhow::Result;
use tch::nn::{self, Embedding, Init, Module, Path};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct CtcNatModel {
    token_embed: Embedding,
    pos_embed: Embedding,
    encoder_layers: Vec<EncoderLayer>,
    encoder_final_norm: nn::LayerNorm,
    proposal_pos_embed: Embedding,
    proposal_layers: Vec<DecoderLayer>,
    proposal_final_norm: nn::LayerNorm,
    ctc_head_bias: Tensor,
    pub blank_id: i64,
}

impl CtcNatModel {
    pub fn new(p: &Path, config: &BenchBackendConfig) -> Result<Self> {
        let hidden = config.hidden_size as i64;
        let vocab = config.output_size as i64;
        let max_positions = config.max_positions as i64;
        let embed_cfg = nn::EmbeddingConfig::default();

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

        Ok(Self {
            token_embed,
            pos_embed,
            encoder_layers,
            encoder_final_norm,
            proposal_pos_embed,
            proposal_layers,
            proposal_final_norm,
            ctc_head_bias,
            blank_id: config.blank_id as i64,
        })
    }

    fn tied_projection(&self, x: &Tensor, bias: &Tensor) -> Tensor {
        let w_t = self.token_embed.ws.transpose(0, 1);
        x.matmul(&w_t) + bias
    }

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
}
