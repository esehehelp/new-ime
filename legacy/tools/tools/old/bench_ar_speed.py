"""Compatibility wrapper for scripts.bench.bench_ar_speed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.bench_ar_speed import *  # noqa: F401,F403
from scripts.bench.bench_ar_speed import main


if __name__ == "__main__":
    main()
