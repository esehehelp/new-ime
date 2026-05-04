"""ConversionBackend wrapper for zenz-v2 / zenz-v2.5 (HuggingFace transformers).

Uses the prompt format from AzooKey's ZenzPromptBuilder:
    INPUT(\\u{EE00}) + reading + [CTX(\\u{EE02}) + context] + OUTPUT(\\u{EE01})

zenz models expect KATAKANA reading; hiragana inputs are auto-converted.
"""

from __future__ import annotations

import jaconv
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from new_ime.eval.runner import ConversionBackend


class ZenzV2Backend(ConversionBackend):
    INPUT_TAG = "\ueE00"
    OUTPUT_TAG = "\ueE01"
    CTX_TAG = "\ueE02"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
        num_beams: int = 1,
        num_return: int = 1,
        max_context_chars: int = 40,
        name_suffix: str = "",
    ) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.num_return = max(1, min(num_return, num_beams))
        self.max_ctx = max_context_chars
        self._name = f"zenz-v2.5({model_path.split('/')[-1]}){name_suffix}"

    @property
    def name(self) -> str:
        return self._name

    def _build_prompt(self, reading: str, context: str) -> str:
        kata = jaconv.hira2kata(reading)
        ctx = context[-self.max_ctx :] if context else ""
        if ctx:
            return self.INPUT_TAG + kata + self.CTX_TAG + ctx + self.OUTPUT_TAG
        return self.INPUT_TAG + kata + self.OUTPUT_TAG

    @torch.no_grad()
    def convert(self, reading: str, context: str) -> list[str]:
        prompt = self._build_prompt(reading, context)
        ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        gen_kwargs = dict(
            max_new_tokens=min(self.max_new_tokens, len(reading) * 3 + 20),
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.num_beams > 1:
            gen_kwargs.update(
                num_beams=self.num_beams,
                num_return_sequences=self.num_return,
                early_stopping=True,
            )
        else:
            gen_kwargs.update(num_beams=1)

        out = self.model.generate(input_ids, **gen_kwargs)
        prefix_len = len(ids)
        results: list[str] = []
        for seq in out:
            tokens = seq[prefix_len:].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            # Strip terminator markers
            for marker in ("</s>", self.INPUT_TAG, self.OUTPUT_TAG, self.CTX_TAG, "[PAD]"):
                if marker in text:
                    text = text.split(marker)[0]
            results.append(text)
        return results
