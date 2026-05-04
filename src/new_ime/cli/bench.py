"""Benchmark entry point.

Usage:
    ime-bench [-v] [-m MODEL ...] [-t TEST ...] [-c CONFIG_DIR]
    python -m new_ime.cli.bench [-v] [-m MODEL ...] [-t TEST ...]

Discovers every `*.toml` in `configs/bench/` (override with `-c`), runs
each as one experiment. Each TOML declares one model (= `[run] name`)
and any number of bench datasets (= `[benches]` keys).

Filters:
    -m / --models  MODEL [MODEL ...]
        Whitelist of `[run] name`. If given, configs whose run.name is
        not in this list are skipped.
    -t / --tests   TEST [TEST ...]
        Whitelist of bench keys (e.g. `probe_v3`, `ajimee_jwtd`). If
        given, benches not in this list are skipped within each
        surviving config.
    -v / --verbose
        Write per-item NDJSON log of all backend outputs to
        `<out_dir>/<bench>__<mode>.full.jsonl`.

Output: each TOML writes its own `<run.out_dir>/<bench>__<mode>.json`
plus a per-TOML `summary.json`. Format: see `docs/benchmark.md`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from new_ime.config import load_toml
from new_ime.config.bench import BenchConfig


_DEFAULT_CONFIG_DIR = Path("configs/bench")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ime-bench")
    p.add_argument(
        "-c",
        "--config-dir",
        type=Path,
        default=_DEFAULT_CONFIG_DIR,
        help=f"directory of bench TOMLs (default: {_DEFAULT_CONFIG_DIR})",
    )
    p.add_argument(
        "-m",
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="run only configs whose [run] name is in this list",
    )
    p.add_argument(
        "-t",
        "--tests",
        nargs="+",
        metavar="TEST",
        default=None,
        help="run only [benches] keys in this list (e.g. probe_v3 ajimee_jwtd)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="write per-item NDJSON log of all backend outputs",
    )
    return p.parse_args(argv)


def _discover_configs(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        return []
    return sorted(p for p in config_dir.glob("*.toml") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    config_dir = args.config_dir.resolve()
    paths = _discover_configs(config_dir)
    if not paths:
        print(f"[bench] no TOML configs found under {config_dir}", file=sys.stderr)
        return 2

    # Load + validate everything up front so a typo in any config kills
    # the run before any model is loaded.
    loaded: list[tuple[Path, BenchConfig]] = []
    for path in paths:
        try:
            cfg = load_toml(path, BenchConfig)
        except Exception as e:  # noqa: BLE001 — surface validation error verbatim
            print(f"[bench] FAILED to load {path}: {e}", file=sys.stderr)
            return 2
        loaded.append((path, cfg))

    model_filter: set[str] | None = set(args.models) if args.models else None
    test_filter: set[str] | None = set(args.tests) if args.tests else None

    # Apply -m: whitelist by run.name.
    if model_filter is not None:
        kept = [(p, c) for p, c in loaded if c.run.name in model_filter]
        seen = {c.run.name for _, c in loaded}
        unknown = model_filter - seen
        if unknown:
            print(
                f"[bench] -m requested unknown models: {sorted(unknown)}; "
                f"available: {sorted(seen)}",
                file=sys.stderr,
            )
            return 2
        loaded = kept

    if not loaded:
        print("[bench] no configs match the model filter", file=sys.stderr)
        return 2

    # Apply -t: whitelist by [benches] keys (per-config). Validate that
    # at least one bench in each surviving config matches; if a TOML has
    # zero matches, drop it.
    if test_filter is not None:
        all_test_names = {k for _, c in loaded for k in c.benches.keys()}
        unknown = test_filter - all_test_names
        if unknown:
            print(
                f"[bench] -t requested unknown tests: {sorted(unknown)}; "
                f"available across configs: {sorted(all_test_names)}",
                file=sys.stderr,
            )
            return 2

    print(
        f"[bench] config_dir={config_dir}",
        f"models={sorted(c.run.name for _, c in loaded)}",
        sep="\n  ",
        file=sys.stderr,
    )
    if test_filter is not None:
        print(f"  tests={sorted(test_filter)}", file=sys.stderr)
    if args.verbose:
        print("  verbose=on", file=sys.stderr)

    from new_ime.eval import run as bench_run

    rc = 0
    for path, cfg in loaded:
        print(f"\n[bench] === {cfg.run.name} ({path.name}) ===", file=sys.stderr)
        try:
            this_rc = bench_run(
                cfg,
                path,
                verbose=args.verbose,
                test_filter=test_filter,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[bench] {cfg.run.name} FAILED: {e}", file=sys.stderr)
            this_rc = 1
        rc = rc or this_rc

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
