"""Bench dataset loaders.

Each loader returns `list[BenchItem]` with fields:
    reading: str       (hiragana — katakana inputs are normalized via jaconv)
    context: str       (preceding-text context, "" if absent)
    references: list[str]  (one or more acceptable surface strings)
    source: str        (bench name)
    category: str | None   (probe only)
    index: str | None      (original index, for error tracing)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import jaconv


@dataclass
class BenchItem:
    reading: str
    context: str
    references: List[str]
    source: str
    category: Optional[str] = None
    index: Optional[str] = None
    extras: dict = field(default_factory=dict)


def load_general(path: str | Path) -> List[BenchItem]:
    """`datasets/eval/general/dev.jsonl`: one JSON per line with
    {reading, surface, context?, source?}.
    """
    items: List[BenchItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                BenchItem(
                    reading=d["reading"],
                    context=d.get("context", "") or "",
                    references=[d["surface"]],
                    source=d.get("source", "general"),
                )
            )
    return items


def load_probe(path: str | Path) -> List[BenchItem]:
    """`datasets/eval/probe/probe.json` (probe_v3, AJIMEE-compatible +
    `category` field). Inputs are katakana; we normalize to hiragana for
    the backend.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items: List[BenchItem] = []
    for r in raw:
        items.append(
            BenchItem(
                reading=jaconv.kata2hira(r["input"]),
                context=r.get("context_text", "") or "",
                references=list(r["expected_output"]),
                source="probe",
                category=r.get("category"),
                index=str(r.get("index", "")) or None,
            )
        )
    return items


def load_ajimee_jwtd(path: str | Path) -> List[BenchItem]:
    """AJIMEE-Bench JWTD_v2 `evaluation_items.json`. Same shape as probe
    minus the category field.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items: List[BenchItem] = []
    for r in raw:
        items.append(
            BenchItem(
                reading=jaconv.kata2hira(r["input"]),
                context=r.get("context_text", "") or "",
                references=list(r["expected_output"]),
                source="ajimee_jwtd",
                index=str(r.get("index", "")) or None,
            )
        )
    return items


# Registry: bench-name -> loader function.
LOADERS = {
    "probe_v3": load_probe,
    "ajimee_jwtd": load_ajimee_jwtd,
    "general": load_general,
}


def load_bench(name: str, path: str | Path) -> List[BenchItem]:
    """Dispatch to a registered loader by bench name."""
    if name not in LOADERS:
        raise KeyError(
            f"unknown bench {name!r}; known: {sorted(LOADERS)}. "
            "Add a new loader to new_ime.eval.loaders if needed."
        )
    return LOADERS[name](path)
