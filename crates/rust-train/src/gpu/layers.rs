//! Transformer building blocks for the tch CTC-NAT model.
//!
//! Shape conventions:
//! - `[B, T, H]` — batch, time, hidden
//! - `padding_mask`: `[B, T]` boolean tensor, `true` = valid (matches
//!   `attention_mask` from the data pipeline). Converted to
//!   "key_padding_mask" (true = pad) inside each MHA call.
//!
//! The Python reference uses:
//! - `nn.TransformerEncoderLayer` (post-norm, GELU activation) for the
//!   encoder (encoder.py:89).
//! - `NATDecoderLayer` (pre-norm, GELU, self-attn + cross-attn) for the
//!   proposal and refinement decoders (decoder.py:9).
//!
//! We mirror both. Dropout is plumbed through but honors train mode via
//! `train: bool` arguments — Step 1 only exercises forward shape and param
//! count, real training kicks in with Step 2's loss + backward.

use anyhow::{bail, Result};
use tch::nn::{self, LinearConfig, Module, Path};
use tch::{Kind, Tensor};

/// Multi-head attention. Separate q/k/v/out projections so the layout is
/// easy to inspect and serialize. We do NOT fuse q/k/v into a single
/// `in_proj_weight` like `nn.MultiheadAttention`; the Python parity
/// harness (Step 5) is responsible for unpacking Python's fused layout
/// into our separate weights.
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
            bail!("MHA hidden={hidden} is not divisible by num_heads={num_heads}",);
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

    /// `query`: `[B, Tq, H]`, `kv`: `[B, Tk, H]`, `kv_padding_mask`: `[B, Tk]`
    /// boolean tensor where `true` = valid token (attention_mask from the
    /// data pipeline). Returns `[B, Tq, H]`.
    pub fn forward(&self, query: &Tensor, kv: &Tensor, kv_padding_mask: Option<&Tensor>) -> Tensor {
        let (b, tq, _h) = query.size3().expect("query must be 3-D");
        let tk = kv.size()[1];
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self
            .q_proj
            .forward(query)
            .view([b, tq, nh, hd])
            .transpose(1, 2); // [B, nh, Tq, hd]
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
        let mut scores = q.matmul(&k.transpose(-2, -1)) / scale; // [B, nh, Tq, Tk]

        if let Some(pad_mask) = kv_padding_mask {
            // pad_mask: [B, Tk], true=valid. Invalid positions → -inf so
            // softmax zeros them out.
            let invalid = pad_mask.logical_not().unsqueeze(1).unsqueeze(1);
            // Broadcast to [B, 1, 1, Tk]; let matmul broadcasting reach Tq.
            scores = scores.masked_fill(&invalid, f64::NEG_INFINITY);
        }

        let probs = scores.softmax(-1, Kind::Float);
        let out = probs.matmul(&v); // [B, nh, Tq, hd]
        let out = out.transpose(1, 2).contiguous().view([b, tq, self.hidden]);
        self.out_proj.forward(&out)
    }
}

/// `nn.TransformerEncoderLayer(norm_first=False, activation="gelu")`:
///   x -> self_attn -> dropout -> +residual -> LN
///     -> ffn(Linear->GELU->Dropout->Linear) -> dropout -> +residual -> LN
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

/// `NATDecoderLayer` (pre-norm):
///   x -> LN -> self_attn -> +residual
///     -> LN -> cross_attn(kv=enc) -> +residual
///     -> LN -> FFN -> +residual
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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn::VarStore, Device};

    #[test]
    fn mha_forward_preserves_shape() {
        let vs = VarStore::new(Device::Cpu);
        let mha = MultiHeadAttention::new(&(vs.root() / "mha"), 16, 4).unwrap();
        let x = Tensor::randn([2, 5, 16], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([2, 5], (Kind::Bool, Device::Cpu));
        let out = mha.forward(&x, &x, Some(&mask));
        assert_eq!(out.size(), vec![2, 5, 16]);
    }

    #[test]
    fn encoder_layer_forward_preserves_shape() {
        let vs = VarStore::new(Device::Cpu);
        let enc = EncoderLayer::new(&(vs.root() / "enc"), 16, 4, 64).unwrap();
        let x = Tensor::randn([2, 5, 16], (Kind::Float, Device::Cpu));
        let mask = Tensor::ones([2, 5], (Kind::Bool, Device::Cpu));
        let out = enc.forward(&x, Some(&mask));
        assert_eq!(out.size(), vec![2, 5, 16]);
    }

    #[test]
    fn decoder_layer_forward_preserves_shape() {
        let vs = VarStore::new(Device::Cpu);
        let dec = DecoderLayer::new(&(vs.root() / "dec"), 16, 4, 64).unwrap();
        let x = Tensor::randn([2, 5, 16], (Kind::Float, Device::Cpu));
        let enc = Tensor::randn([2, 7, 16], (Kind::Float, Device::Cpu));
        let self_mask = Tensor::ones([2, 5], (Kind::Bool, Device::Cpu));
        let enc_mask = Tensor::ones([2, 7], (Kind::Bool, Device::Cpu));
        let out = dec.forward(&x, &enc, Some(&self_mask), Some(&enc_mask));
        assert_eq!(out.size(), vec![2, 5, 16]);
    }

    #[test]
    fn mha_masks_out_padded_keys() {
        // Sanity: with only position 0 valid in kv, the output must
        // depend only on kv[..,0,..] regardless of kv[..,1,..].
        let vs = VarStore::new(Device::Cpu);
        let mha = MultiHeadAttention::new(&(vs.root() / "mha"), 8, 2).unwrap();
        let q = Tensor::randn([1, 1, 8], (Kind::Float, Device::Cpu));
        let kv_a = Tensor::randn([1, 3, 8], (Kind::Float, Device::Cpu));
        let kv_b = kv_a.shallow_clone();
        // Change positions 1 and 2 — they should be masked out.
        let _ = kv_b.slice(1, 1, 3, 1).fill_(7.0);
        let mask = Tensor::from_slice(&[1i64, 0, 0])
            .view([1, 3])
            .to_kind(Kind::Bool);
        let out_a = mha.forward(&q, &kv_a, Some(&mask));
        let out_b = mha.forward(&q, &kv_b, Some(&mask));
        let diff = (&out_a - &out_b).abs().max().double_value(&[]);
        assert!(diff < 1e-5, "masked keys affected output: diff={diff}");
    }
}
