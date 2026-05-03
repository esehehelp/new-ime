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
    preset: Literal["ctc-nat-30m", "ctc-nat-41m", "ctc-nat-90m"]


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
