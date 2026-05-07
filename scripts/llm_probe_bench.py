"""Run probe_v3 directly against an LLM for kana→kanji conversion EM1.

Sanity baseline: how does a frontier LLM do at the IME task without any
IME-specific training? Tells us a ceiling and where our specialized
30M-param CTC-NAT student stands relative to a 200B-param thinker.

Usage:
    LLM_BENCH_ENDPOINT=...  LLM_BENCH_MODEL=...  LLM_BENCH_TOKEN=... \\
        python scripts/llm_probe_bench.py --limit 50

Output: results/llm_bench/<model-tag>/probe_v3_em1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
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


SYSTEM_PROMPT = (
    "あなたは日本語のかな漢字変換アシスタント。"
    "ユーザが与えるかな (またはカタカナ) 読みを、文脈に合う最も自然な漢字混じり日本語に "
    "1 通りだけ変換し、余計な説明・引用符・ピリオド・改行なしで surface のみ 1 行で返してください。"
)


PROMPT_TEMPLATE = """前文脈: {context}
読み: {reading}

変換結果 (漢字混じり surface のみ、1 行):"""


def kata_to_hira(s: str) -> str:
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def call_llm(
    client: httpx.Client,
    endpoint: str,
    model: str,
    reading: str,
    context: str,
    extra: dict,
    max_tokens: int,
) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(context=context or "(無し)", reading=reading)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    body.update(extra)
    last_err: Exception | None = None
    for attempt in range(4):
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
                    msg = r.json().get("error", {}).get("message", "")
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                    if m:
                        wait = max(wait, float(m.group(1)) + 1.0)
                except Exception:
                    pass
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            # Reasoning-only models (GLM-5.1, sometimes Qwen3 with thinking
            # forced on) put the final answer at the end of reasoning_content
            # when content is empty. Take the last non-empty line as the
            # answer in that case.
            if not content:
                rc = (msg.get("reasoning_content") or "").strip()
                if rc:
                    lines = [l.strip() for l in rc.splitlines() if l.strip()]
                    content = lines[-1] if lines else rc
            usage = data.get("usage", {}) or {}
            return content, usage
        except httpx.HTTPStatusError as e:
            last_err = e
            time.sleep(4.0 * (2 ** attempt))
    raise last_err if last_err else RuntimeError("retries exhausted")


def clean_output(s: str) -> str:
    """Trim quotes / leading bullets / trailing punctuation noise."""
    s = s.strip()
    # Drop leading/trailing quotes
    for q in ['"', "'", "「", "」", "『", "』", "`"]:
        if s.startswith(q):
            s = s[len(q):]
        if s.endswith(q):
            s = s[: -len(q)]
    # Drop "変換結果:" / "答え:" prefixes if model added them
    s = re.sub(r"^(変換結果|答え|出力|回答)\s*[:：]\s*", "", s)
    # Take only the first line in case the model emitted extra trailing prose
    s = s.splitlines()[0].strip() if s else s
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=str(ROOT / "datasets/eval/probe/probe.json"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tag", default=None, help="output dir tag; defaults to model id slug")
    ap.add_argument(
        "--enable-thinking",
        choices=["yes", "no", "auto"],
        default="auto",
        help="set chat_template_kwargs.enable_thinking; auto omits the field",
    )
    ap.add_argument("--min-interval-sec", type=float, default=0.0)
    ap.add_argument(
        "--stratified",
        action="store_true",
        help="distribute --limit across probe categories (probe.json is grouped, "
        "so naive first-N hits one category only)",
    )
    args = ap.parse_args()

    endpoint = os.environ.get("LLM_BENCH_ENDPOINT") or os.environ.get(
        "DATA_ROW_LLM_ENDPOINT", "https://api.deepinfra.com/v1/openai/chat/completions"
    )
    model = os.environ.get("LLM_BENCH_MODEL") or os.environ.get(
        "DATA_ROW_LLM_MODEL", "google/gemma-4-31B-it"
    )
    token = (
        os.environ.get("LLM_BENCH_TOKEN")
        or os.environ.get("DATA_ROW_LLM_TOKEN")
        or os.environ.get("DEEPINFRA_API_KEY")
    )
    if not token:
        raise SystemExit("no token in env")

    extra: dict = {}
    if args.enable_thinking == "no":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.enable_thinking == "yes":
        extra["chat_template_kwargs"] = {"enable_thinking": True}

    tag = args.tag or model.replace("/", "_").replace("@", "_at_")
    out_dir = ROOT / "results/llm_bench" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    items = probe if isinstance(probe, list) else probe.get("items", [])
    if args.stratified:
        # Take an even distribution across `category` so a small --limit
        # still hits every probe bucket. probe.json is grouped by category
        # so the naive first-N slice misses most categories.
        from collections import defaultdict
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for it in items:
            by_cat[it.get("category", "?")].append(it)
        cats = sorted(by_cat.keys())
        sampled: list[dict] = []
        if args.limit > 0:
            per_cat = max(1, args.limit // len(cats))
            for cat in cats:
                sampled.extend(by_cat[cat][:per_cat])
            items = sampled[: args.limit]
        else:
            for cat in cats:
                sampled.extend(by_cat[cat])
            items = sampled
    elif args.limit > 0:
        items = items[: args.limit]
    print(f"endpoint={endpoint}", file=sys.stderr)
    print(f"model={model}", file=sys.stderr)
    print(f"items={len(items)}  thinking={args.enable_thinking}", file=sys.stderr)

    client = httpx.Client(
        timeout=httpx.Timeout(180.0),
        headers={"Authorization": f"Bearer {token}"},
    )

    em1 = 0
    em1_nfkc = 0
    failed = 0
    cost = 0.0
    out_path = out_dir / "probe_v3_em1.jsonl"
    summary_path = out_dir / "summary.json"
    last_call = 0.0
    by_cat: dict[str, list[int]] = {}  # cat -> [em1, total]

    with out_path.open("w", encoding="utf-8") as fout:
        for i, item in enumerate(items):
            if args.min_interval_sec > 0:
                gap = time.monotonic() - last_call
                if gap < args.min_interval_sec:
                    time.sleep(args.min_interval_sec - gap)
                last_call = time.monotonic()
            # probe.json schema (jinen / AJIMEE-aligned):
            #   input (KATAKANA), context_text, expected_output[]
            # but we also accept reading/context/references for portability.
            raw_reading = item.get("input") or item.get("reading") or ""
            reading = kata_to_hira(raw_reading)
            context = item.get("context_text") or item.get("context") or ""
            references = (
                item.get("expected_output")
                or item.get("references")
                or item.get("surface")
                or []
            )
            if isinstance(references, str):
                references = [references]
            cat = item.get("category", "?")
            try:
                raw, usage = call_llm(
                    client, endpoint, model, reading, context, extra, args.max_tokens
                )
            except Exception as e:
                print(f"[{i}] error: {e}", file=sys.stderr)
                failed += 1
                continue
            cost += float(usage.get("estimated_cost") or 0.0)
            top1 = clean_output(raw)
            hit_em = int(top1 in references)
            hit_em_nfkc = int(any(nfkc(top1) == nfkc(r) for r in references))
            em1 += hit_em
            em1_nfkc += hit_em_nfkc
            by_cat.setdefault(cat, [0, 0])
            by_cat[cat][0] += hit_em_nfkc
            by_cat[cat][1] += 1
            fout.write(json.dumps({
                "i": i,
                "category": cat,
                "reading": reading,
                "context": context,
                "references": references,
                "top1": top1,
                "raw": raw,
                "em1": hit_em,
                "em1_nfkc": hit_em_nfkc,
                "usage": usage,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(
                    f"[{i + 1}/{len(items)}] em1={em1}  em1_nfkc={em1_nfkc}  "
                    f"failed={failed}  cost=${cost:.4f}",
                    file=sys.stderr,
                )
    client.close()

    n = len(items) - failed
    summary = {
        "model": model,
        "endpoint": endpoint,
        "thinking": args.enable_thinking,
        "n": len(items),
        "answered": n,
        "failed": failed,
        "em1": em1 / n if n else 0,
        "em1_nfkc": em1_nfkc / n if n else 0,
        "estimated_cost_usd": cost,
        "per_category": {
            c: {"n": v[1], "em1_nfkc": v[0] / v[1] if v[1] else 0}
            for c, v in sorted(by_cat.items())
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===SUMMARY===", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
