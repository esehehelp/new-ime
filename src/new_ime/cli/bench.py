"""Benchmark entry point. Single positional argument: a TOML config path.

Usage:
    python -m new_ime.cli.bench configs/bench/<bench>.toml
    ime-bench configs/bench/<bench>.toml

Output JSONs land at `<run.out_dir>/<bench_name>__<decode.mode>.json` and
a combined `<run.out_dir>/summary.json`. Format defined in
`docs/benchmark.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from new_ime.config import load_toml
from new_ime.config.bench import BenchConfig


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: ime-bench <config.toml>", file=sys.stderr)
        return 2

    config_path = Path(args[0]).resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    cfg = load_toml(config_path, BenchConfig)
    print(f"[bench] loaded {config_path}")
    print(f"[bench] run.name = {cfg.run.name}")
    print(f"[bench] decode   = {cfg.decode.mode}")
    print(f"[bench] benches  = {list(cfg.benches.keys())}")

    from new_ime.eval import run as bench_run

    return bench_run(cfg, config_path)


if __name__ == "__main__":
    raise SystemExit(main())
