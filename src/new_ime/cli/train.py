"""Training entry point. Single positional argument: a TOML config path.

Usage:
    python -m new_ime.cli.train configs/train/<experiment>.toml
    ime-train configs/train/<experiment>.toml

The TOML is validated against `TrainConfig` before any heavy import
or device init. The actual training loop is dispatched to
`new_ime.train.run`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from new_ime.config import load_toml
from new_ime.config.train import TrainConfig


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: ime-train <config.toml>", file=sys.stderr)
        return 2

    config_path = Path(args[0]).resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    cfg = load_toml(config_path, TrainConfig)
    print(f"[train] loaded {config_path}")
    print(f"[train] run.name = {cfg.run.name}")
    print(f"[train] preset  = {cfg.model.preset}")
    print(f"[train] out_dir = {cfg.run.out_dir}")

    from new_ime.training.run import run as train_run

    return train_run(cfg, config_path)


if __name__ == "__main__":
    raise SystemExit(main())
