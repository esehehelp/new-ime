"""Compatibility wrapper for scripts.manual.manual_test_beam."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manual.manual_test_beam import *  # noqa: F401,F403
from scripts.manual.manual_test_beam import main


if __name__ == "__main__":
    main()
