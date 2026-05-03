"""Backend factory.

Wraps the ported `_ctc_nat_backend_legacy.CTCNATBackend` with a TOML
config-driven constructor. The underlying model / tokenizer / decode
implementation is the verbatim port from
`archive/pre-v2:legacy/python/models/src/eval/ctc_nat_backend.py` so
output is bit-identical to the pre-v2 results when given the same
checkpoint.
"""
from __future__ import annotations

from typing import List

from new_ime.config.bench import BenchConfig
from new_ime.eval._ctc_nat_backend_legacy import CTCNATBackend as _LegacyCTCNATBackend


class CTCNATBackend:
    """Adapter from BenchConfig -> the legacy CTCNATBackend.

    Decode mapping:
        cfg.decode.mode == "greedy" -> beam_width=1
        cfg.decode.mode == "beam"   -> beam_width=cfg.decode.num_beams
    """

    def __init__(self, cfg: BenchConfig) -> None:
        self.name = cfg.run.name
        beam_width = (
            1 if cfg.decode.mode == "greedy" else int(cfg.decode.num_beams)
        )

        # KenLM shallow fusion: pass through single-LM or MoE settings to
        # the legacy backend. The legacy backend interprets `lm_path` for
        # single mode and `lm_paths_by_domain` for MoE.
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


def build_backend(cfg: BenchConfig) -> CTCNATBackend:
    return CTCNATBackend(cfg)
