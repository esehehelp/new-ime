"""ConversionBackend for TeacherSeq2Seq checkpoints.

Transformer encoder + AR decoder teacher (`models.src.model.teacher_seq2seq`).
Checkpoints store the class name in `ckpt["preset"]` instead of the preset
key, so the backend infers the architecture from the state dict shapes.
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.src.data.tokenizer import SharedCharTokenizer
from models.src.eval.run_eval import ConversionBackend
from models.src.model.teacher_seq2seq import TEACHER_PRESETS, TeacherSeq2Seq


def _infer_preset(state: dict) -> str:
    hidden = state["token_embedding.weight"].shape[1]
    enc_layers = sum(
        1 for k in state if k.startswith("encoder.layers.") and k.endswith(".norm1.weight")
    )
    dec_layers = sum(
        1 for k in state if k.startswith("decoder.layers.") and k.endswith(".norm1.weight")
    )
    for name, p in TEACHER_PRESETS.items():
        if p.hidden_size == hidden and p.encoder_layers == enc_layers and p.decoder_layers == dec_layers:
            return name
    raise ValueError(
        f"no matching TEACHER_PRESET for hidden={hidden} enc={enc_layers} dec={dec_layers}"
    )


class TeacherBackend(ConversionBackend):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        max_context: int = 40,
        max_new_tokens: int = 64,
        name: str | None = None,
    ) -> None:
        ckpt_path = Path(checkpoint_path)
        self.ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        tokenizer_path = Path(str(ckpt_path).replace(".pt", "_tokenizer.json"))
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"missing tokenizer sidecar: {tokenizer_path}")
        self.tokenizer = SharedCharTokenizer.load(str(tokenizer_path))

        state = self.ckpt["model_state_dict"]
        preset_name = _infer_preset(state)
        self.model = TeacherSeq2Seq.from_preset(preset_name, vocab_size=self.tokenizer.vocab_size)
        self.model.load_state_dict(state)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        self.max_context = max_context
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = self.model.max_positions
        self.step = self.ckpt.get("step", "?")

        ckpt_id = f"{ckpt_path.parent.name}/{ckpt_path.name}"
        self._name = name or f"teacher({preset_name},{ckpt_id}@step{self.step},greedy)"

    @property
    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def convert(self, reading: str, context: str) -> list[str]:
        ctx = context[-self.max_context :] if context else ""
        ids = self.tokenizer.encode_with_special(ctx, reading)
        ids = ids[: self.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        tokens, _ = self.model.generate(
            input_ids, attention_mask, max_new_tokens=self.max_new_tokens
        )
        # Strip BOS and everything from first EOS onward.
        out_ids = tokens[0].tolist()[1:]
        eos = self.model.eos_id
        if eos in out_ids:
            out_ids = out_ids[: out_ids.index(eos)]
        text = self.tokenizer.decode(out_ids)
        return [text] if text else []
