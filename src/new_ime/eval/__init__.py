"""Evaluation / benchmarking entry point.

Top-level dispatch: `run(cfg, config_path)` builds the backend from the
TOML, runs the configured bench suite, writes JSON output. Submodules:

    new_ime.eval.loaders        — probe / AJIMEE / general loaders
    new_ime.eval.metrics        — EM1 / EM5 / CharAcc, latency
    new_ime.eval.runner         — orchestration (evaluate, run_bench_suite)
    new_ime.eval.backend        — CTCNATBackend (loads .pt checkpoints)
"""
from __future__ import annotations

from pathlib import Path

from new_ime.config.bench import BenchConfig
from new_ime.eval.runner import run_bench_suite


def run(
    cfg: BenchConfig,
    config_path: Path,
    verbose: bool = False,
    test_filter: set[str] | None = None,
) -> int:
    # Backend factory is imported lazily so loaders/metrics testing does
    # not pull torch into process memory.
    from new_ime.eval.backend import build_backend

    benches = {k: Path(v) for k, v in cfg.benches.items()}
    if test_filter is not None:
        benches = {k: v for k, v in benches.items() if k in test_filter}
        if not benches:
            print(
                f"[bench] {cfg.run.name}: no benches match -t filter, skipping",
                file=__import__("sys").stderr,
            )
            return 0

    backend = build_backend(cfg)
    out_dir = cfg.run.out_dir
    if not out_dir.is_absolute():
        # Configs use repo-root-relative paths. Resolve them against the
        # cwd where ime-bench was invoked (the operator stands at repo
        # root by convention).
        out_dir = Path.cwd() / out_dir

    rows = run_bench_suite(
        backend=backend,
        benches=benches,
        out_dir=out_dir,
        decode_mode=cfg.decode.mode,
        top_k=cfg.decode.top_k,
        verbose=verbose,
    )

    print(
        f"[bench] {cfg.run.name}: wrote {out_dir / 'summary.json'} ({len(rows)} rows)",
        file=__import__("sys").stderr,
    )
    return 0
