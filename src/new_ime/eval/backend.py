"""Backend factory。

cfg.model.type で dispatch する:
  ctc-nat   -> CTCNATBackend (Suiko-v1-small 系、ckpt + tokenizer)
  zenz-v2.5 -> ZenzV2Backend (HF GPT2)
  zenz-v3.1 -> ZenzV2Backend (同 backend、異なる weight)
  jinen-v1  -> JinenV1Backend (HF AutoModelForCausalLM)

具体的な model / tokenizer / decode は archive/pre-v2 から verbatim
port した module を呼び出す。
"""
from __future__ import annotations

from typing import List

from new_ime.config.bench import BenchConfig
from new_ime.eval._ctc_nat_backend_legacy import CTCNATBackend as _LegacyCTCNATBackend


class CTCNATBackend:
    """BenchConfig -> 既存 CTCNATBackend のアダプタ。

    decode mode マッピング:
        greedy -> beam_width=1
        beam   -> beam_width=cfg.decode.num_beams
    KenLM 設定は cfg.lm から渡す (single / MoE)。
    """

    def __init__(self, cfg: BenchConfig) -> None:
        assert cfg.model.type == "ctc-nat"
        self.name = cfg.run.name
        beam_width = (
            1 if cfg.decode.mode == "greedy" else int(cfg.decode.num_beams)
        )

        # KenLM shallow fusion を legacy backend の lm_* kwargs に流し込む
        lm_kwargs: dict = {}
        if cfg.lm is not None:
            if cfg.lm.mode == "single":
                lm_kwargs.update(
                    lm_path=str(cfg.lm.path),
                    lm_alpha=float(cfg.lm.alpha),
                    lm_beta=float(cfg.lm.beta),
                    lm_gate_min_conf=float(cfg.lm.gate_min_conf),
                )
            else:  # moe
                lm_kwargs.update(
                    lm_paths_by_domain={
                        k: str(v) for k, v in (cfg.lm.paths_by_domain or {}).items()
                    },
                    lm_alpha=float(cfg.lm.alpha),
                    lm_beta=float(cfg.lm.beta),
                    lm_gate_min_conf=float(cfg.lm.gate_min_conf),
                )

        self._inner = _LegacyCTCNATBackend(
            checkpoint_path=str(cfg.model.checkpoint),
            device=cfg.device.backend,
            beam_width=beam_width,
            beam_top_k=int(cfg.decode.top_k),
            name=cfg.run.name,
            **lm_kwargs,
        )

    def convert(self, reading: str, context: str) -> List[str]:
        return self._inner.convert(reading, context)


class _HfBackendAdapter:
    """HF backend (zenz / jinen) のアダプタ。BenchConfig 経由で生成し、
    cfg.run.name を ConversionBackend.name にする。convert() は legacy
    実装 (verbatim port) にそのまま委譲。
    """

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
