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
        self._inner = _LegacyCTCNATBackend(
            checkpoint_path=str(cfg.model.checkpoint),
            device=cfg.device.backend,
            beam_width=beam_width,
            beam_top_k=int(cfg.decode.top_k),
            name=cfg.run.name,
        )

    def convert(self, reading: str, context: str) -> List[str]:
        return self._inner.convert(reading, context)


def build_backend(cfg: BenchConfig) -> CTCNATBackend:
    return CTCNATBackend(cfg)
