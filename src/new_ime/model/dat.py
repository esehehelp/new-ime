"""DA-Transformer (DAT) for kana→kanji conversion.

A non-autoregressive seq2seq model: the encoder produces source-side
hidden states, which are upsampled by a fixed scale and fed into a
self-attention decoder. Each decoder vertex emits one token; transitions
between vertices form a DAG, and each path corresponds to one possible
output. Training marginalizes over all paths via the DP loss in
`training/loss/dat_dp.py`.

This v1.0 deliberately omits the cross-attention to encoder and the length
predictor from the reference (`thu-coai/DA-Transformer`):

    * source info enters the decoder through the upsampled encoder
      hidden (`repeat_interleave`) plus learnable positional embedding,
      not through a separate cross-attn block — kana→kanji has weak
      reordering, so the upsample carries enough source signal for MVP.
    * upsample_base is fixed to "source" (T_up = T_in × scale), so no
      length prediction loss is needed.

Forward returns the standard new-ime training contract:
    {"encoder_out", "encoder_padding_mask", "decoder_out", "logits",
     "links", "loss" (when target_ids/target_lengths supplied)}

The DAT decoder accepts an optional `glance_ratio` for GLAT (Stage 4);
in v1.0 / Stage 2 the default 0.0 disables glancing and the model runs
a single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from new_ime.model.dat_links import DagLinkExtractor
from new_ime.model.encoder import SmallEncoder
from new_ime.training.loss.dat_dp import (
    torch_dag_logsoftmax_gather,
    torch_dag_loss,
)


@dataclass
class DATPreset:
    name: str
    hidden_size: int
    encoder_layers: int
    decoder_layers: int
    num_heads: int
    ffn_size: int
    max_positions: int
    num_link_heads: int = 4


# Keep param-budget alignment with the CTC-NAT presets so DAT and CTC-NAT
# can be compared like-for-like at the same hidden_size / layer counts.
DAT_PRESETS: dict[str, DATPreset] = {
    "phase3_20m": DATPreset(
        name="phase3_20m",
        hidden_size=320,
        encoder_layers=5,
        decoder_layers=5,
        num_heads=4,
        ffn_size=1280,
        max_positions=128,
        num_link_heads=4,
    ),
    "phase3_30m": DATPreset(
        name="phase3_30m",
        hidden_size=384,
        encoder_layers=6,
        decoder_layers=6,
        num_heads=6,
        ffn_size=1536,
        max_positions=128,
        num_link_heads=4,
    ),
    "phase3_90m": DATPreset(
        name="phase3_90m",
        hidden_size=640,
        encoder_layers=8,
        decoder_layers=8,
        num_heads=8,
        ffn_size=2560,
        max_positions=128,
        num_link_heads=8,
    ),
}


class DATDecoder(nn.Module):
    """Self-attention decoder over the upsampled encoder hidden.

    The decoder input at position p is `encoder_out[..., p // scale, :]`
    plus a learnable position embedding indexed by p in `[0, T_up)`. When
    `hint_ids` is provided (GLAT leak), the corresponding token embedding
    is added too — this is how GLAT supplies oracle tokens at a subset of
    vertices during the second forward pass.

    Cross-attention is omitted (see module docstring).
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        max_upsampled_positions: int,
        hint_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.pos_embedding = nn.Embedding(max_upsampled_positions, hidden_size)
        # Hint embedding is tied to the encoder's token embedding by default;
        # when `hint_ids` is None or all-pad, no token signal is added.
        self.hint_embedding = hint_embedding
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        upsampled_features: Tensor,  # (B, T_up, H)
        upsampled_mask: Tensor,      # (B, T_up) long, 1=valid
        hint_ids: Tensor | None = None,  # (B, T_up) — oracle tokens at leaked positions, 0 elsewhere
    ) -> Tensor:
        batch_size, t_up, _ = upsampled_features.shape
        positions = torch.arange(t_up, device=upsampled_features.device).unsqueeze(0).expand(batch_size, -1)
        x = upsampled_features + self.pos_embedding(positions)
        if hint_ids is not None and self.hint_embedding is not None:
            x = x + self.hint_embedding(hint_ids)
        pad_mask = ~upsampled_mask.bool()
        x = self.layers(x, src_key_padding_mask=pad_mask)
        return self.final_norm(x)


class DAT(nn.Module):
    """Directed Acyclic Transformer for kana→kanji.

    See `references/dat-survey.md` for background. The training contract
    matches CTC-NAT: forward returns a dict with at least `loss` (when
    targets are supplied) and `logits`.
    """

    def __init__(
        self,
        encoder: SmallEncoder,
        output_vocab_size: int,
        decoder_layers: int,
        decoder_heads: int,
        decoder_ffn_size: int,
        upsample_scale: int = 4,
        num_link_heads: int = 4,
        dropout: float = 0.1,
        max_positions: int = 128,
        blank_id: int = 4,
    ) -> None:
        super().__init__()
        if upsample_scale < 1:
            raise ValueError(f"upsample_scale must be >= 1, got {upsample_scale}")
        self.encoder = encoder
        self.output_vocab_size = output_vocab_size
        self.upsample_scale = upsample_scale
        self.blank_id = blank_id  # kept for API parity with CTC-NAT (evaluate.py reads it)
        self._preset_name: str | None = None
        self._max_seq_len: int = max_positions

        hidden_size = encoder.hidden_size
        max_upsampled = max_positions * upsample_scale

        # Tie the GLAT hint embedding to the encoder's token embedding so the
        # decoder receives oracle tokens in the same space the encoder
        # operates in, with no extra params.
        encoder_token_embedding = encoder.get_input_embedding()
        self.decoder = DATDecoder(
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ffn_size=decoder_ffn_size,
            dropout=dropout,
            max_upsampled_positions=max_upsampled,
            hint_embedding=encoder_token_embedding,
        )

        # Link-extraction position embedding: separate from decoder pos_embed
        # so the link head can learn its own sense of position without
        # competing with the decoder body. Reference also uses two separate
        # embeddings for these roles.
        self.link_pos_embedding = nn.Embedding(max_upsampled, hidden_size)
        self.link_extractor = DagLinkExtractor(
            hidden_size=hidden_size,
            num_heads=num_link_heads,
            link_pos_embedding=self.link_pos_embedding,
        )

        self.output_projection = nn.Linear(hidden_size, output_vocab_size, bias=False)
        # Tie output projection to encoder embeddings if the dimensions match
        # (matches CTC-NAT's tie pattern; halves the param count of the head).
        emb = encoder.get_input_embedding()
        if emb.weight.shape == self.output_projection.weight.shape:
            self.output_projection.weight = emb.weight

        # GLAT state (Stage 4 will write this from training/run.py:_on_step_start).
        self._glance_ratio: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        *,
        vocab_size: int,
        upsample_scale: int = 4,
        num_link_heads: int | None = None,
        dropout: float = 0.1,
        blank_id: int = 4,
        max_positions: int | None = None,
    ) -> "DAT":
        if preset_name not in DAT_PRESETS:
            raise ValueError(
                f"Unknown DAT preset {preset_name!r}; choose from {sorted(DAT_PRESETS)}"
            )
        preset = DAT_PRESETS[preset_name]
        positions = max_positions if max_positions is not None else preset.max_positions
        encoder = SmallEncoder(
            vocab_size=vocab_size,
            hidden_size=preset.hidden_size,
            num_layers=preset.encoder_layers,
            num_heads=preset.num_heads,
            ffn_size=preset.ffn_size,
            max_positions=positions,
            dropout=dropout,
        )
        model = cls(
            encoder=encoder,
            output_vocab_size=vocab_size,
            decoder_layers=preset.decoder_layers,
            decoder_heads=preset.num_heads,
            decoder_ffn_size=preset.ffn_size,
            upsample_scale=upsample_scale,
            num_link_heads=num_link_heads if num_link_heads is not None else preset.num_link_heads,
            dropout=dropout,
            max_positions=positions,
            blank_id=blank_id,
        )
        model._preset_name = preset_name
        model._max_seq_len = positions
        return model

    # ------------------------------------------------------------------
    # Loop / evaluate / checkpoint contract
    # ------------------------------------------------------------------

    def checkpoint_metadata(self) -> dict:
        return {
            "arch_tag": "dat",
            "preset": self._preset_name,
            "vocab_size": self.output_vocab_size,
            "use_cvae": False,
            "max_seq_len": self._max_seq_len,
            "upsample_scale": self.upsample_scale,
        }

    def compute_aux_losses(self, batch: dict, outputs: dict) -> dict[str, Tensor]:
        """DAT carries its main objective in `outputs['loss']`; no aux losses."""
        return {}

    def set_glance_ratio(self, ratio: float) -> None:
        """GLAT schedule hook (Stage 4). Stored on the module; `forward`
        consumes it when targets are supplied."""
        self._glance_ratio = float(ratio)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _upsample(
        self,
        encoder_out: Tensor,        # (B, T_in, H)
        attention_mask: Tensor,     # (B, T_in)
    ) -> tuple[Tensor, Tensor]:
        """Repeat each encoder timestep `upsample_scale` times. Returns
        (upsampled_features, upsampled_mask) on the same device/dtype."""
        scale = self.upsample_scale
        upsampled_features = encoder_out.repeat_interleave(scale, dim=1)
        upsampled_mask = attention_mask.repeat_interleave(scale, dim=1)
        return upsampled_features, upsampled_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        target_ids: Tensor | None = None,
        target_lengths: Tensor | None = None,
        writer_ids: Tensor | None = None,
        domain_ids: Tensor | None = None,
        source_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Encoder.
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_padding_mask = ~attention_mask.bool()

        # Upsample to T_up = T_in * scale.
        up_features, up_mask = self._upsample(encoder_out, attention_mask)

        # When GLAT is active and we have targets, run a no_grad pass to
        # compute oracle vertex assignment, then re-forward with the leaked
        # oracle tokens as decoder hints.
        glat_active = (
            self.training
            and self._glance_ratio > 0.0
            and target_ids is not None
            and target_lengths is not None
        )
        hint_ids: Tensor | None = None
        if glat_active:
            hint_ids = self._glance_hint_ids(
                up_features=up_features,
                up_mask=up_mask,
                target_ids=target_ids,
                target_lengths=target_lengths,
            )

        # Decoder over upsampled positions (optionally with GLAT hints).
        decoder_out = self.decoder(up_features, up_mask, hint_ids=hint_ids)

        # Per-vertex token logits.
        logits = self.output_projection(decoder_out)  # (B, T_up, V)

        # DAG transition matrix (log space, row-normalized over successors).
        links = self.link_extractor(decoder_out, valid_mask=up_mask.bool())  # (B, T_up, T_up)

        result: dict[str, Tensor] = {
            "encoder_out": encoder_out,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_out": decoder_out,
            "logits": logits,
            "links": links,
        }
        if hint_ids is not None:
            # Surface so callers / tests can inspect the leak pattern.
            result["glance_hint_ids"] = hint_ids

        if target_ids is not None and target_lengths is not None:
            result["loss"] = self._dag_loss(
                logits=logits,
                links=links,
                up_mask=up_mask,
                target_ids=target_ids,
                target_lengths=target_lengths,
            )

        return result

    @torch.no_grad()
    def _glance_hint_ids(
        self,
        *,
        up_features: Tensor,    # (B, T_up, H)
        up_mask: Tensor,        # (B, T_up) long
        target_ids: Tensor,     # (B, T_tgt)
        target_lengths: Tensor, # (B,)
    ) -> Tensor:
        """Run a no-grad first pass and pick which oracle tokens to leak.

        Returns `(B, T_up)` long tensor: oracle token id at sampled positions,
        0 elsewhere. The decoder embedding at id 0 (typically <pad>) acts as
        the "no hint" baseline.
        """
        # First-pass forward (no grad, no hints) to obtain proposal logits and
        # the DAG link matrix.
        decoder_out_first = self.decoder(up_features, up_mask, hint_ids=None)
        logits_first = self.output_projection(decoder_out_first)
        links_first = self.link_extractor(decoder_out_first, valid_mask=up_mask.bool())

        prelen = logits_first.shape[1]
        select_idx = target_ids.unsqueeze(1).expand(-1, prelen, -1)
        _, match = torch_dag_logsoftmax_gather(logits_first, select_idx)
        match_t = match.transpose(1, 2)  # (B, T_tgt, prelen)

        from new_ime.training.loss.dat_dp import torch_dag_best_alignment

        output_length = up_mask.sum(dim=1).long().clamp(min=1)
        target_length = target_lengths.long().clamp(min=1)
        # path[b, j] = target token index assigned to vertex j on the best
        # path, or -1 if vertex j is off-path.
        path = torch_dag_best_alignment(match_t, links_first, output_length, target_length)
        assigned_mask = path >= 0  # (B, prelen)

        oracle = target_ids.gather(-1, path.clamp(min=0))  # (B, prelen) — undefined where path < 0

        pred_tokens = logits_first.argmax(dim=-1)  # (B, prelen)
        same_count = ((pred_tokens == oracle) & assigned_mask).sum(dim=1)  # (B,)

        # number-random sampling (the only strategy supported in v1.0).
        miss = (target_length - same_count).clamp(min=0).to(oracle.dtype)
        glance_nums = ((miss * self._glance_ratio) + 0.5).long()  # (B,)
        rand = torch.randn(oracle.shape, device=oracle.device, dtype=torch.float)
        rand = rand.masked_fill(~assigned_mask, -float("inf"))
        sorted_rand, _ = rand.sort(dim=-1, descending=True)
        # threshold[b] = sorted_rand[b, glance_nums[b] - 1]
        idx = (glance_nums - 1).clamp(min=0).unsqueeze(-1)  # (B, 1)
        thresh = sorted_rand.gather(-1, idx).squeeze(-1)
        thresh = torch.where(
            glance_nums == 0,
            torch.full_like(thresh, float("inf")),
            thresh,
        )
        leak_mask = (rand >= thresh.unsqueeze(-1)) & assigned_mask  # (B, prelen)

        hint_ids = torch.where(
            leak_mask,
            oracle,
            torch.zeros_like(oracle),
        )
        return hint_ids

    def _dag_loss(
        self,
        *,
        logits: Tensor,        # (B, T_up, V)
        links: Tensor,         # (B, T_up, T_up)
        up_mask: Tensor,       # (B, T_up) long
        target_ids: Tensor,    # (B, T_tgt)
        target_lengths: Tensor,  # (B,)
    ) -> Tensor:
        prelen = logits.shape[1]
        # match_all[b, j, i] = log P(target[b, i] | vertex j)
        select_idx = target_ids.unsqueeze(1).expand(-1, prelen, -1)
        _, match_all = torch_dag_logsoftmax_gather(logits, select_idx)
        match_all_t = match_all.transpose(1, 2)  # (B, T_tgt, prelen) for the DP API

        output_length = up_mask.sum(dim=1).long().clamp(min=1)
        target_length = target_lengths.long().clamp(min=1)

        # log P(target | DAG); clamp paths that can never reach target_length
        # tokens within output_length vertices (sanitizes pathological batches).
        log_p = torch_dag_loss(match_all_t, links, output_length, target_length)
        # Per-sample loss = -log_p / target_length, batch-averaged outside is
        # handled by the loop (it averages by step). We average inside so the
        # loop sees a stable scalar regardless of batch padding.
        loss = -(log_p / target_length.to(log_p.dtype)).mean()
        return loss

    # ------------------------------------------------------------------
    # Decode (Stage 3 will replace this with greedy/lookahead/viterbi).
    # ------------------------------------------------------------------

    def _decode_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        links = outputs["links"]
        up_mask = attention_mask.repeat_interleave(self.upsample_scale, dim=1)
        output_length = up_mask.sum(dim=1).long().clamp(min=1)
        return logits, links, output_length

    @torch.no_grad()
    def greedy_decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        writer_ids: Tensor | None = None,
        domain_ids: Tensor | None = None,
        source_ids: Tensor | None = None,
    ) -> list[list[int]]:
        from new_ime.model.dat_decoding import greedy_decode as _greedy

        was_training = self.training
        self.eval()
        try:
            logits, links, output_length = self._decode_inputs(input_ids, attention_mask)
            return _greedy(logits, links, output_length, blank_id=self.blank_id)
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def lookahead_decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        beta: float = 1.0,
    ) -> list[list[int]]:
        from new_ime.model.dat_decoding import lookahead_decode as _lookahead

        was_training = self.training
        self.eval()
        try:
            logits, links, output_length = self._decode_inputs(input_ids, attention_mask)
            return _lookahead(
                logits, links, output_length,
                blank_id=self.blank_id, beta=beta,
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def viterbi_decode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        length_penalty: float = 1.0,
        max_length: int | None = None,
    ) -> list[list[int]]:
        from new_ime.model.dat_decoding import viterbi_decode as _viterbi

        was_training = self.training
        self.eval()
        try:
            logits, links, output_length = self._decode_inputs(input_ids, attention_mask)
            return _viterbi(
                logits, links, output_length,
                blank_id=self.blank_id,
                length_penalty=length_penalty,
                max_length=max_length,
            )
        finally:
            if was_training:
                self.train()
