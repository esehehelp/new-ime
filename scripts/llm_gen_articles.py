"""LLM article generation only — no reading attribution.

Pairs with `scripts/articles_to_bunsetsu.py` for the surface/reading split.
This script writes raw articles ({article_idx, topic_tag, topic_label,
text, usage}) and stops there. Run articles_to_bunsetsu.py afterward to
get Schema B rows.

Generic in two ways:
  1. Endpoint / model are env-driven so the same script can target Gemini,
     DeepInfra, DeepSeek, or any OpenAI-compat provider.
  2. Topic list is loaded from a JSON file via --topics so we can swap
     "tech-heavy", "casual", "fiction-heavy" mixes without editing code.

Usage:
    LLM_GEN_ENDPOINT=...  LLM_GEN_MODEL=...  LLM_GEN_TOKEN=... \\
        python scripts/llm_gen_articles.py \\
            --topics scripts/topics_default.json \\
            --output datasets/corpus/synth/_articles_xxx_raw.jsonl \\
            --rounds 8 [--min-interval-sec 5.0] [--temperature 0.95] \\
            [--max-tokens 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
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


SYSTEM_DEFAULT = (
    "日本語の自然な文章を生成するアシスタント。"
    "タイトル・見出し・箇条書き・コードブロック禁止、本文のみ。"
)


def gen_article(
    client: httpx.Client,
    endpoint: str,
    model: str,
    system_prompt: str,
    topic_label: str,
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"ジャンル: {topic_label}\n"
                    "上記ジャンルの 400-900 字程度の自然な日本語文を 1 本書いてください。"
                    "本文のみ、タイトル・見出し・箇条書き禁止。"
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = client.post(endpoint, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(60.0, 4.0 * (2 ** attempt))
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                try:
                    payload = r.json()
                    msg = payload.get("error", {}).get("message", "")
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                    if m:
                        wait = max(wait, float(m.group(1)) + 1.0)
                except Exception:
                    pass
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return content.strip(), usage
        except httpx.HTTPStatusError as e:
            last_err = e
            time.sleep(4.0 * (2 ** attempt))
    raise last_err if last_err else RuntimeError("gen_article retries exhausted")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topics", required=True, help="JSON file with [{tag,label}, ...]")
    ap.add_argument("--output", required=True, help="raw articles JSONL output")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--min-interval-sec", type=float, default=5.0)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--system-prompt", default=SYSTEM_DEFAULT)
    args = ap.parse_args()

    endpoint = os.environ.get("LLM_GEN_ENDPOINT") or os.environ.get(
        "DATA_ROW_LLM_ENDPOINT",
        "https://api.deepinfra.com/v1/openai/chat/completions",
    )
    model = os.environ.get("LLM_GEN_MODEL") or os.environ.get(
        "DATA_ROW_LLM_MODEL", "google/gemma-4-31B-it"
    )
    token = (
        os.environ.get("LLM_GEN_TOKEN")
        or os.environ.get("DATA_ROW_LLM_TOKEN")
        or os.environ.get("DEEPINFRA_API_KEY")
    )
    if not token:
        raise SystemExit("no LLM_GEN_TOKEN / DATA_ROW_LLM_TOKEN / DEEPINFRA_API_KEY in env")

    topics_path = Path(args.topics)
    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    if not isinstance(topics, list) or not topics:
        raise SystemExit(f"topics file must be a non-empty JSON array: {topics_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    print(f"endpoint={endpoint}", file=sys.stderr)
    print(f"model={model}", file=sys.stderr)
    print(f"topics={len(topics)} rounds={args.rounds} → {out_path}", file=sys.stderr)

    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {token}"},
    )

    last_call = 0.0
    article_idx = 0
    target = args.rounds * len(topics)
    with out_path.open("a", encoding="utf-8") as fout:
        for round_idx in range(args.rounds):
            for entry in topics:
                tag = entry.get("tag") or f"topic{article_idx}"
                label = entry.get("label") or tag
                gap = time.monotonic() - last_call
                if gap < args.min_interval_sec:
                    time.sleep(args.min_interval_sec - gap)
                last_call = time.monotonic()
                try:
                    text, usage = gen_article(
                        client, endpoint, model,
                        args.system_prompt, label,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"[gen] {tag} r{round_idx} error: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps({
                    "article_idx": article_idx,
                    "topic_tag": tag,
                    "topic_label": label,
                    "text": text,
                    "usage": usage,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                article_idx += 1
                print(
                    f"[gen] {article_idx}/{target} {tag} r{round_idx} chars={len(text)}",
                    file=sys.stderr,
                )
    client.close()
    print(f"done: {article_idx} articles → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
