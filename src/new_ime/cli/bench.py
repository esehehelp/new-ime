"""Benchmark entry point.

Usage:
    ime-bench [-v] <config.toml>
    python -m new_ime.cli.bench [-v] <config.toml>

Options:
    -v / --verbose
        Write a per-item NDJSON log next to the per-bench JSON
        (`<out_dir>/<bench>__<mode>.full.jsonl`) containing the FULL
        candidate list for every input. Used for post-hoc audit and
        recomputing metrics without re-running the model.

Output JSONs land at `<run.out_dir>/<bench>__<decode.mode>.json` and a
combined `<run.out_dir>/summary.json`. Format defined in
`docs/benchmark.md`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from new_ime.config import load_toml
from new_ime.config.bench import BenchConfig


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ime-bench")
    p.add_argument("config", type=Path, help="path to a benchmark TOML")
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="write per-item NDJSON log of all backend outputs",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"config not found: {config_path}", file=sys.stderr)
        return 2

    cfg = load_toml(config_path, BenchConfig)
    print(f"[bench] loaded {config_path}", file=sys.stderr)
    print(f"[bench] run.name = {cfg.run.name}", file=sys.stderr)
    print(f"[bench] decode   = {cfg.decode.mode}", file=sys.stderr)
    print(f"[bench] benches  = {list(cfg.benches.keys())}", file=sys.stderr)
    if args.verbose:
        print("[bench] verbose: full per-item logs enabled", file=sys.stderr)

    from new_ime.eval import run as bench_run

    return bench_run(cfg, config_path, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
