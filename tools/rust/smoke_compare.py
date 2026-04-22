#!/usr/bin/env python
"""Compare Rust and Python trainer loss curves at equal config.

Optional companion to `smoke_train_cuda.sh`. Re-runs the Python
`train_ctc_nat` on the same shard/tokenizer for the same number of
optimizer steps, parses both loss logs, and asserts they stay within
a generous tolerance at the end of the burst.

This script is not run by default (requires Python + torch installed,
plus a copy of train_ctc_nat.py with its full stack). Invoke manually:

    python tools/rust/smoke_compare.py \
        --rust-log models/checkpoints/rust-smoke/fit.log \
        --python-log models/checkpoints/py-smoke/train.log

Prints a markdown summary table and exits non-zero if the absolute
difference at the final step exceeds `--tolerance` (default 0.5 —
deliberately loose since schedulers, seeds, and dropout can drift).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LOSS_RE_RUST = re.compile(r"fit_last_loss:\s*([0-9eE.+-]+)")
# Python trainer emits lines like `[step 1000] train_loss=6.81`
LOSS_RE_PY = re.compile(r"train_loss=([0-9eE.+-]+)")


def last_float(path: Path, pattern: re.Pattern[str]) -> float | None:
    value: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            try:
                value = float(m.group(1))
            except ValueError:
                continue
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rust-log", type=Path, required=True)
    parser.add_argument("--python-log", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.5)
    args = parser.parse_args()

    rust = last_float(args.rust_log, LOSS_RE_RUST)
    py = last_float(args.python_log, LOSS_RE_PY)

    print("| side | last_loss |")
    print("|---|---:|")
    print(f"| rust | {rust if rust is not None else 'n/a'} |")
    print(f"| python | {py if py is not None else 'n/a'} |")

    if rust is None or py is None:
        print("smoke_compare: one or both logs had no loss lines", file=sys.stderr)
        return 2
    diff = abs(rust - py)
    print(f"|diff| = {diff:.4f}  tolerance = {args.tolerance:.4f}")
    if diff > args.tolerance:
        print("SMOKE COMPARE FAIL", file=sys.stderr)
        return 1
    print("SMOKE COMPARE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
