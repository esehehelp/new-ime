"""Benchmark の TOML schema。protocol は docs/benchmark.md。

Backend 種別は `[model] type` で分岐:
    type = "ctc-nat"   : Suiko 系 (ckpt + tokenizer + preset)
    type = "zenz-v2.5" : zenz GPT2 ファミリ (HF directory or Hub ID)
    type = "zenz-v3.1" : 同上 (異なる weight、backend 同じ)
    type = "jinen-v1"  : jinen Causal LM (HF AutoModel)
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSection(_Strict):
    name: str
    out_dir: Path


class CtcNatModelSection(_Strict):
    """CTC-NAT (Suiko 系) モデル設定。preset は ckpt の preset field と合わせる。"""

    type: Literal["ctc-nat"] = "ctc-nat"
    checkpoint: Path
    tokenizer: Path
    preset: Literal["phase3_20m", "phase3_30m", "phase3_90m"]


class HfModelSection(_Strict):
    """HuggingFace transformers backend (zenz / jinen)。

    `path` はローカル dir か HF Hub ID (例: "togatogah/jinen-v1-xsmall")。
    transformers の `from_pretrained` がそのまま解決する。
    """

    type: Literal["zenz-v2.5", "zenz-v3.1", "jinen-v1"]
    path: str
    max_new_tokens: int = 80
    max_context_chars: int = 40


# 識別 field 'type' で discriminated union。`type` 省略時は 'ctc-nat' に解決。
ModelSection = Annotated[
    Union[CtcNatModelSection, HfModelSection],
    Field(discriminator="type"),
]


class DecodeSection(_Strict):
    mode: Literal["greedy", "beam"] = "greedy"
    num_beams: int = 1
    num_return: int = 1
    top_k: int = 5


class DeviceSection(_Strict):
    backend: Literal["cpu", "cuda"] = "cpu"


class LmSection(_Strict):
    """KenLM shallow fusion (single LM or MoE)。CTC-NAT backend のみ有効。

    mode="single" + path                 : 単一 4-gram KenLM
    mode="moe"    + paths_by_domain (>=2): domain 別混合 (general / tech / entity 等)

    α/β は CTC-LM 融合の標準重み:
        score = ctc_logp + alpha * lm_logp + beta * len(prefix)
    """

    mode: Literal["single", "moe"]
    alpha: float
    beta: float
    path: Optional[Path] = None
    paths_by_domain: Optional[Dict[str, Path]] = None
    gate_min_conf: float = 0.0  # 負値で条件付き融合

    @model_validator(mode="after")
    def _check_mode(self) -> "LmSection":
        if self.mode == "single":
            if self.path is None:
                raise ValueError("[lm] mode='single' は path 必須")
            if self.paths_by_domain:
                raise ValueError("[lm] mode='single' は paths_by_domain 不可")
        else:  # moe
            if not self.paths_by_domain or len(self.paths_by_domain) < 2:
                raise ValueError(
                    "[lm] mode='moe' は paths_by_domain (>=2 entries) が必須"
                )
            if self.path is not None:
                raise ValueError("[lm] mode='moe' は path 指定不可")
        return self


class BenchConfig(_Strict):
    run: RunSection
    model: ModelSection
    decode: DecodeSection
    benches: Dict[str, Path]  # bench name -> dataset path
    device: DeviceSection = DeviceSection()
    lm: Optional[LmSection] = None
