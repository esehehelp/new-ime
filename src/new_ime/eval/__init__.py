"""Evaluation / benchmarking. To be implemented.

Will host:
    new_ime.eval.run             — top-level bench dispatch
    new_ime.eval.metrics         — EM1 / EM5 / CharAcc
    new_ime.eval.loaders         — probe.json / AJIMEE-Bench loaders
    new_ime.eval.ctc_nat_backend — checkpoint -> candidates pipeline
"""
from __future__ import annotations

from pathlib import Path

from new_ime.config.bench import BenchConfig


def run(cfg: BenchConfig, config_path: Path) -> int:
    raise NotImplementedError(
        "v2 bench runner not yet ported. See archive/pre-v2 for the "
        "pre-restructure implementation."
    )
