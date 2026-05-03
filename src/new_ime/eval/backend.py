"""Backend factory + CTCNATBackend wrapper.

The backend is the dependency that produces ranked candidates from a
(reading, context) pair. Currently a stub: model + tokenizer + decode
porting from `archive/pre-v2:legacy/python/models/src/{model/ctc_nat,
data/tokenizer,eval/{ctc_beam,ctc_nat_backend}}.py` lands in the next
commit.
"""
from __future__ import annotations

from typing import List

from new_ime.config.bench import BenchConfig


class CTCNATBackend:
    """Loads a Suiko-v1.x .pt checkpoint + sidecar tokenizer.json and
    serves `convert(reading, context) -> list[str]`.

    Decoding modes:
        greedy    — beam_width=1, returns the single argmax-decoded path
        beam      — prefix beam search over CTC emissions, top-K by score
    """

    def __init__(self, cfg: BenchConfig) -> None:
        self.name = cfg.run.name
        self._cfg = cfg
        # Lazy: actual torch model loading happens in `_load()` so an
        # unused backend (e.g. during dry-run validation) never pays the
        # import cost.
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        raise NotImplementedError(
            "CTCNATBackend._load: model/tokenizer/decode port pending. "
            "See archive/pre-v2:legacy/python/models/src/model/ctc_nat.py "
            "and eval/ctc_nat_backend.py for the reference implementation."
        )

    def convert(self, reading: str, context: str) -> List[str]:
        self._load()
        raise NotImplementedError("convert: pending model port")


def build_backend(cfg: BenchConfig) -> CTCNATBackend:
    return CTCNATBackend(cfg)
