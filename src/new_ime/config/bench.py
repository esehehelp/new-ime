"""Benchmark config schema. See `docs/benchmark.md` for the protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSection(_Strict):
    name: str
    out_dir: Path


class ModelSection(_Strict):
    checkpoint: Path
    tokenizer: Path
    # Preset names match the value stored in the checkpoint's `preset`
    # field. phase3_30m (= ~46M with refine head, marketed as 41M) is
    # what Suiko-v1-small uses.
    preset: Literal["phase3_20m", "phase3_30m", "phase3_90m"]


class DecodeSection(_Strict):
    mode: Literal["greedy", "beam"] = "greedy"
    num_beams: int = 1
    num_return: int = 1
    top_k: int = 5


class DeviceSection(_Strict):
    backend: Literal["cpu", "cuda"] = "cpu"


class LmSection(_Strict):
    """KenLM shallow fusion (single LM or domain MoE).

    Two modes (mutually exclusive):
        mode="single"  + path                       single 4-gram KenLM
        mode="moe"     + paths_by_domain (>=2)      domain mixture estimator

    `alpha`/`beta` are the standard CTC-LM fusion weights:
        score = ctc_logp + alpha * lm_logp + beta * len(prefix)
    """

    mode: Literal["single", "moe"]
    alpha: float
    beta: float
    path: Optional[Path] = None
    paths_by_domain: Optional[Dict[str, Path]] = None
    gate_min_conf: float = 0.0  # negative → conditional fusion

    @model_validator(mode="after")
    def _check_mode(self) -> "LmSection":
        if self.mode == "single":
            if self.path is None:
                raise ValueError("[lm] mode='single' requires `path`")
            if self.paths_by_domain:
                raise ValueError("[lm] mode='single' must not set paths_by_domain")
        else:  # moe
            if not self.paths_by_domain or len(self.paths_by_domain) < 2:
                raise ValueError(
                    "[lm] mode='moe' requires `paths_by_domain` with >=2 entries"
                )
            if self.path is not None:
                raise ValueError("[lm] mode='moe' must not set `path`")
        return self


class BenchConfig(_Strict):
    run: RunSection
    model: ModelSection
    decode: DecodeSection
    benches: Dict[str, Path]  # bench_name -> dataset path
    device: DeviceSection = DeviceSection()
    lm: Optional[LmSection] = None
