"""Training loop. To be implemented.

The pre-v2 training loop lived in legacy/python/models/src/training/
train_ctc_nat.py (a 2000+ line monolith). It has been retired; the v2
re-implementation will be split across:

    new_ime.train.run        — top-level dispatch (this module)
    new_ime.train.loop       — main step / eval / checkpoint loop
    new_ime.train.kd         — KD teacher loading + alpha schedule
    new_ime.train.refine     — MaskCTC refine head + iterative decode

Reference: archive/pre-v2 branch in git history if specific behavior
needs to be re-derived.
"""
from __future__ import annotations

from pathlib import Path

from new_ime.config.train import TrainConfig


def run(cfg: TrainConfig, config_path: Path) -> int:
    raise NotImplementedError(
        "v2 training loop not yet ported. See archive/pre-v2 for the "
        "pre-restructure implementation."
    )
