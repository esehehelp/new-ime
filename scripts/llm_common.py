"""Shared DeepInfra OpenAI-compat client for one-off LLM tasks.

These scripts exist to use up the API credit pool with project-relevant
work (probe failure analysis, synth data generation). They're one-offs;
real data pipeline tooling lives in Rust crates per project convention.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    """Load .env from project root if present (httpx-compatible env access)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()


def get_client() -> tuple[httpx.Client, str, str]:
    """Return (client, endpoint, model). Endpoint and model are read from
    DATA_ROW_LLM_* env vars so the active provider matches the TUI."""
    endpoint = os.environ.get(
        "DATA_ROW_LLM_ENDPOINT",
        "https://api.deepinfra.com/v1/openai/chat/completions",
    )
    model = os.environ.get("DATA_ROW_LLM_MODEL", "google/gemma-4-31B-it")
    token = os.environ.get("DATA_ROW_LLM_TOKEN") or os.environ.get(
        "DEEPINFRA_API_KEY"
    )
    if not token:
        raise RuntimeError("no DATA_ROW_LLM_TOKEN / DEEPINFRA_API_KEY set in .env")
    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {token}"},
    )
    return client, endpoint, model


def chat(
    client: httpx.Client,
    endpoint: str,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> tuple[str, dict[str, Any]]:
    """One-shot chat completion. Returns (content, usage)."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = client.post(endpoint, json=body)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) or {}
    return content, usage


def cost_estimate(usage: dict[str, Any]) -> float:
    """Pull `estimated_cost` if the provider reports it; else 0."""
    return float(usage.get("estimated_cost") or 0.0)


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = True) -> None:
    """Write rows as JSONL (Schema B). Append by default for crash safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
