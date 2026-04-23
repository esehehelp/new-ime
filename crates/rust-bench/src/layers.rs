use anyhow::{bail, Result};
use tch::nn::{self, LinearConfig, Module, Path};
use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct MultiHeadAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: i64,
    head_dim: i64,
    hidden: i64,
}

impl MultiHeadAttention {
    pub fn new(p: &Path, hidden: i64, num_heads: i64) -> Result<Self> {
        if hidden % num_heads != 0 {
            bail!("MHA hidden={hidden} is not divisible by num_heads={num_heads}");
        }
        let head_dim = hidden / num_heads;
        let cfg = LinearConfig::default();
        Ok(Self {
            q_proj: nn::linear(p / "q_proj", hidden, hidden, cfg),
            k_proj: nn::linear(p / "k_proj", hidden, hidden, cfg),
            v_proj: nn::linear(p / "v_proj", hidden, hidden, cfg),
            out_proj: nn::linear(p / "out_proj", hidden, hidden, cfg),
            num_heads,
            head_dim,
            hidden,
        })
    }

    pub fn forward(&self, query: &Tensor, kv: &Tensor, kv_padding_mask: Option<&Tensor>) -> Tensor {
        let (b, tq, _h) = query.size3().expect("query must be 3-D");
        let tk = kv.size()[1];
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self
            .q_proj
            .forward(query)
            .view([b, tq, nh, hd])
            .transpose(1, 2);
        let k = self
            .k_proj
            .forward(kv)
            .view([b, tk, nh, hd])
            .transpose(1, 2);
        let v = self
            .v_proj
            .forward(kv)
            .view([b, tk, nh, hd])
            .transpose(1, 2);

        let scale = (hd as f64).sqrt();
        let mut scores = q.matmul(&k.transpose(-2, -1)) / scale;

        if let Some(pad_mask) = kv_padding_mask {
            let invalid = pad_mask.logical_not().unsqueeze(1).unsqueeze(1);
            scores = scores.masked_fill(&invalid, f64::NEG_INFINITY);
        }

        let probs = scores.softmax(-1, Kind::Float);
        let out = probs.matmul(&v);
        let out = out.transpose(1, 2).contiguous().view([b, tq, self.hidden]);
        self.out_proj.forward(&out)
    }
}

#[derive(Debug)]
pub struct EncoderLayer {
    self_attn: MultiHeadAttention,
    ffn_in: nn::Linear,
    ffn_out: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl EncoderLayer {
    pub fn new(p: &Path, hidden: i64, num_heads: i64, ffn_size: i64) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(&(p / "self_attn"), hidden, num_heads)?,
            ffn_in: nn::linear(p / "ffn_in", hidden, ffn_size, Default::default()),
            ffn_out: nn::linear(p / "ffn_out", ffn_size, hidden, Default::default()),
            norm1: nn::layer_norm(p / "norm1", vec![hidden], Default::default()),
            norm2: nn::layer_norm(p / "norm2", vec![hidden], Default::default()),
        })
    }

    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Tensor {
        let attn = self.self_attn.forward(x, x, padding_mask);
        let x = self.norm1.forward(&(x + attn));
        let ffn = self.ffn_out.forward(&self.ffn_in.forward(&x).gelu("none"));
        self.norm2.forward(&(x + ffn))
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ffn_in: nn::Linear,
    ffn_out: nn::Linear,
    self_attn_norm: nn::LayerNorm,
    cross_attn_norm: nn::LayerNorm,
    ffn_norm: nn::LayerNorm,
}

impl DecoderLayer {
    pub fn new(p: &Path, hidden: i64, num_heads: i64, ffn_size: i64) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(&(p / "self_attn"), hidden, num_heads)?,
            cross_attn: MultiHeadAttention::new(&(p / "cross_attn"), hidden, num_heads)?,
            ffn_in: nn::linear(p / "ffn_in", hidden, ffn_size, Default::default()),
            ffn_out: nn::linear(p / "ffn_out", ffn_size, hidden, Default::default()),
            self_attn_norm: nn::layer_norm(p / "self_attn_norm", vec![hidden], Default::default()),
            cross_attn_norm: nn::layer_norm(
                p / "cross_attn_norm",
                vec![hidden],
                Default::default(),
            ),
            ffn_norm: nn::layer_norm(p / "ffn_norm", vec![hidden], Default::default()),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        encoder_out: &Tensor,
        self_padding_mask: Option<&Tensor>,
        enc_padding_mask: Option<&Tensor>,
    ) -> Tensor {
        let pre = self.self_attn_norm.forward(x);
        let x = x + self.self_attn.forward(&pre, &pre, self_padding_mask);
        let pre = self.cross_attn_norm.forward(&x);
        let x = &x + self.cross_attn.forward(&pre, encoder_out, enc_padding_mask);
        let pre = self.ffn_norm.forward(&x);
        let ffn = self
            .ffn_out
            .forward(&self.ffn_in.forward(&pre).gelu("none"));
        &x + ffn
    }
}
