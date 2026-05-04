"""TOML config schemas. The single source of truth for what a config file
must contain. Loading any config goes through `load_toml(...)` so that
typos and missing fields fail fast before training/eval starts.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeVar

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_toml(path: str | Path, schema: type[T]) -> T:
    """Read TOML at `path` and validate against `schema`. Returns the
    parsed pydantic model. Raises ValidationError on schema violations.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return schema.model_validate(raw)
