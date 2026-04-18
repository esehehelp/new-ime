"""Compatibility wrapper for scripts.gold.gen_gold_final."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gold.gen_gold_final import *  # noqa: F401,F403
