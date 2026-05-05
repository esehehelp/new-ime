"""Backend factory。

cfg.model.type で dispatch する:
  ctc-nat   -> RustEngineBackend (Suiko: ONNX artifact + Rust daemon)
  zenz-v2.5 -> ZenzV2Backend (HF GPT2)
  zenz-v3.1 -> ZenzV2Backend (同 backend、異なる weight)
  jinen-v1  -> JinenV1Backend (HF AutoModelForCausalLM)

Suiko 推論は Rust runtime (new-ime-engine-cli) に閉じる。Python 側に
推論ロジックは持たない。外部 baseline は HF Transformers のまま。
"""
from __future__ import annotations

from typing import List

from new_ime.config.bench import BenchConfig
from new_ime.eval.rust_engine_backend import RustEngineBackend


class CTCNATBackend:
    """BenchConfig -> RustEngineBackend の薄いアダプタ。

    runtime / artifact_format は RustEngineBackend から伝播。
    """

    runtime: str = "rust-engine-daemon"

    def __init__(self, cfg: BenchConfig) -> None:
        assert cfg.model.type == "ctc-nat"
        self.name = cfg.run.name
        self._inner = RustEngineBackend(cfg)
        self.artifact_format: str = self._inner.artifact_format

    def convert(self, reading: str, context: str) -> List[str]:
        return self._inner.convert(reading, context)

    @property
    def last_engine_ms(self) -> float:
        return self._inner.last_engine_ms

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> "CTCNATBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _HfBackendAdapter:
    """HF backend (zenz / jinen) のアダプタ。BenchConfig 経由で生成し、
    cfg.run.name を ConversionBackend.name にする。convert() は
    legacy 実装 (verbatim port) にそのまま委譲。
    """

    runtime: str = "python-hf"
    artifact_format: str = "hf-pytorch"

    def __init__(self, inner, name: str) -> None:
        self._inner = inner
        self.name = name

    def convert(self, reading: str, context: str) -> List[str]:
        return self._inner.convert(reading, context)


def _build_zenz(cfg: BenchConfig):
    from new_ime.eval.zenz_backend import ZenzV2Backend

    assert cfg.model.type in ("zenz-v2.5", "zenz-v3.1")
    inner = ZenzV2Backend(
        model_path=cfg.model.path,
        device=cfg.device.backend,
        max_new_tokens=cfg.model.max_new_tokens,
        num_beams=cfg.decode.num_beams,
        num_return=cfg.decode.num_return,
        max_context_chars=cfg.model.max_context_chars,
    )
    return _HfBackendAdapter(inner, cfg.run.name)


def _build_jinen(cfg: BenchConfig):
    from new_ime.eval.jinen_backend import JinenV1Backend

    assert cfg.model.type == "jinen-v1"
    inner = JinenV1Backend(
        model_path=cfg.model.path,
        device=cfg.device.backend,
        max_new_tokens=cfg.model.max_new_tokens,
        num_beams=cfg.decode.num_beams,
        num_return=cfg.decode.num_return,
        max_context_chars=cfg.model.max_context_chars,
    )
    return _HfBackendAdapter(inner, cfg.run.name)


def build_backend(cfg: BenchConfig):
    """cfg.model.type で backend を dispatch。"""
    t = cfg.model.type
    if t == "ctc-nat":
        return CTCNATBackend(cfg)
    if t in ("zenz-v2.5", "zenz-v3.1"):
        return _build_zenz(cfg)
    if t == "jinen-v1":
        return _build_jinen(cfg)
    raise ValueError(f"unknown model.type: {t!r}")
