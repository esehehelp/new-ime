from __future__ import annotations

import jaconv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.src.eval.run_eval import ConversionBackend


class JinenV1Backend(ConversionBackend):
    INPUT_TAG = "\uee00"
    OUTPUT_TAG = "\uee01"
    CTX_TAG = "\uee02"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_new_tokens: int = 80,
        num_beams: int = 1,
        num_return: int = 1,
        max_context_chars: int = 40,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype or torch.float32,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.num_return = max(1, min(num_return, num_beams))
        self.max_ctx = max_context_chars
        self._name = f"jinen-v1({model_path.split('/')[-1]})"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self) -> str:
        return self._name

    def _build_prompt(self, reading: str, context: str) -> str:
        kata = jaconv.hira2kata(reading)
        ctx = context[-self.max_ctx:] if context else ""
        return f"{self.CTX_TAG}{ctx}{self.INPUT_TAG}{kata}{self.OUTPUT_TAG}"

    @torch.no_grad()
    def convert(self, reading: str, context: str) -> list[str]:
        prompt = self._build_prompt(reading, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=min(self.max_new_tokens, len(reading) * 3 + 20),
            do_sample=False,
            num_beams=max(1, self.num_beams),
            num_return_sequences=self.num_return if self.num_beams > 1 else 1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if self.num_beams > 1:
            gen_kwargs["early_stopping"] = True

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
        prefix_len = input_ids.shape[1]
        results: list[str] = []
        for seq in out:
            text = self.tokenizer.decode(seq[prefix_len:], skip_special_tokens=False)
            if self.OUTPUT_TAG in text:
                text = text.split(self.OUTPUT_TAG, 1)[-1]
            for marker in ("</s>", self.INPUT_TAG, self.OUTPUT_TAG, self.CTX_TAG, "[PAD]"):
                if marker in text:
                    text = text.split(marker, 1)[0]
            results.append(text.strip())
        return results
