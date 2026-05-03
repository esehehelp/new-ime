"""Benchmark config schema. See `docs/benchmark.md` for the protocol."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict


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


class BenchConfig(_Strict):
    run: RunSection
    model: ModelSection
    decode: DecodeSection
    benches: Dict[str, Path]  # bench_name -> dataset path
    device: DeviceSection = DeviceSection()
