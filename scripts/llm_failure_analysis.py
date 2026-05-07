"""Categorize Suiko-v1 + Hatsuyume probe_v3 failures via LLM.

Reads the verbose probe outputs, filters em1_nfkc=0 rows, sends batches to
the LLM with a categorization prompt, then writes a markdown summary.

Output: docs/probe_failure_analysis.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from llm_common import ROOT, chat, cost_estimate, get_client


SOURCES = {
    "suiko-v1-greedy": ROOT / "results/bench/suiko-v1-small-greedy/probe_v3__greedy.full.jsonl",
    "suiko-v1-kenlm-moe": ROOT / "results/bench/suiko-v1-small-kenlm-6gram-q8-moe/probe_v3__beam.full.jsonl",
    "hatsuyume-greedy": ROOT / "results/bench/hatsuyume-greedy/probe_v3__greedy.full.jsonl",
    "hatsuyume-kenlm-moe": ROOT / "results/bench/hatsuyume-kenlm-moe/probe_v3__beam.full.jsonl",
}

CATEGORIES = [
    "homophone_choice",   # right reading, wrong kanji choice
    "kana_kanji_split",   # split position wrong (e.g. の/を boundaries)
    "okurigana",          # 送りがな (verb stem / inflection) wrong
    "named_entity",       # 人名 / 地名 / 組織名 wrong
    "numeric_format",     # 1 vs 一 / 数字単位 / SI 単位
    "katakana_loanword",  # カタカナ choice (ピンク vs ぴんく etc.)
    "punctuation",        # 句読点 / 引用符差
    "rare_kanji",         # 旧字体 / 異体字 / 外字
    "context_misuse",     # context があるのに別意味の漢字選択
    "model_garbage",      # surface 自体が破綻 (途中切れ / 別物)
    "reference_ambiguous",  # references 側が複数あって正解の選び損ね
    "other",
]


def load_failures(path: Path) -> list[dict]:
    if not path.exists():
        return []
    failures: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if int(row.get("em1_nfkc", 0)) == 0:
                failures.append(row)
    return failures


PROMPT_SYSTEM = "日本語かな漢字 IME の bench 失敗ケース分析アシスタント。各失敗を 1 行 1 カテゴリに分類する。"

PROMPT_USER_TEMPLATE = """以下は IME bench (probe_v3) で誤答した行です。各行 1 ケースとして失敗カテゴリを 1 つ選んで返してください。

カテゴリ (許可リスト):
- homophone_choice: 読み正解、漢字選択誤り
- kana_kanji_split: 区切り位置誤り
- okurigana: 送りがな誤り
- named_entity: 人名/地名/組織名の誤り
- numeric_format: 1↔一、数字単位/SI 単位
- katakana_loanword: カタカナ選択誤り (ピンク↔ぴんく等)
- punctuation: 句読点/引用符差
- rare_kanji: 旧字体/異体字
- context_misuse: context 提示でも別意味の漢字
- model_garbage: surface 自体破綻
- reference_ambiguous: references 複数で選び損ね
- other: 上記以外

出力形式 (説明なし、各ケース 1 行):
{{"i": <index>, "category": "<上記の1つ>"}}

ケース:
{cases}
"""


def format_case(row: dict, i: int) -> str:
    return json.dumps(
        {
            "i": i,
            "probe_cat": row.get("category"),
            "reading": row.get("reading"),
            "context": row.get("context") or "",
            "references": row.get("references") or [],
            "top1": (row.get("candidates") or [None])[0],
        },
        ensure_ascii=False,
    )


def parse_categories(content: str, n: int) -> list[str]:
    out: list[str | None] = [None] * n
    for line in content.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        i = obj.get("i")
        cat = obj.get("category")
        if isinstance(i, int) and 0 <= i < n and isinstance(cat, str):
            out[i] = cat
    return [c if c in CATEGORIES else "other" for c in out]


def run_for_source(name: str, path: Path, batch_size: int = 25) -> tuple[list[tuple[dict, str]], float]:
    failures = load_failures(path)
    print(f"[{name}] failures = {len(failures)} from {path.name}", file=sys.stderr)
    if not failures:
        return [], 0.0
    client, endpoint, model = get_client()
    results: list[tuple[dict, str]] = []
    cost = 0.0
    for offset in range(0, len(failures), batch_size):
        batch = failures[offset : offset + batch_size]
        cases = "\n".join(format_case(r, i) for i, r in enumerate(batch))
        try:
            content, usage = chat(
                client, endpoint, model,
                PROMPT_SYSTEM,
                PROMPT_USER_TEMPLATE.format(cases=cases),
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[{name}] batch @ {offset} error: {e}", file=sys.stderr)
            continue
        cost += cost_estimate(usage)
        cats = parse_categories(content, len(batch))
        for row, cat in zip(batch, cats):
            results.append((row, cat))
        print(
            f"[{name}] batch @ {offset}/{len(failures)} cost=${cost:.4f}",
            file=sys.stderr,
        )
    client.close()
    return results, cost


def write_summary(out_path: Path, all_results: dict[str, list[tuple[dict, str]]], total_cost: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# probe_v3 失敗カテゴリ分析\n")
    lines.append(f"LLM (DeepInfra Gemma 4 31B-it) による失敗分類。総コスト ${total_cost:.4f}。\n")
    lines.append("カテゴリ定義は `scripts/llm_failure_analysis.py` の CATEGORIES 参照。\n")
    for name, results in all_results.items():
        lines.append(f"\n## {name} (n={len(results)} failures)\n")
        cats = Counter(cat for _, cat in results)
        by_probe_cat: dict[str, Counter] = defaultdict(Counter)
        for row, cat in results:
            by_probe_cat[row.get("category", "?")][cat] += 1
        lines.append("| failure_category | count | % |")
        lines.append("|---|---:|---:|")
        total = sum(cats.values()) or 1
        for cat, n in cats.most_common():
            lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
        lines.append("\n### probe_v3 カテゴリ別の失敗内訳\n")
        for probe_cat in sorted(by_probe_cat.keys()):
            lines.append(f"- **{probe_cat}**: " + ", ".join(
                f"{c}({n})" for c, n in by_probe_cat[probe_cat].most_common()
            ))
        # Sample failure cases per category (top 3)
        lines.append("\n### サンプル (各失敗カテゴリ 上位 3 例)\n")
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for row, cat in results:
            if len(by_cat[cat]) < 3:
                by_cat[cat].append(row)
        for cat in sorted(by_cat.keys()):
            lines.append(f"\n#### {cat}\n")
            for row in by_cat[cat]:
                refs = " / ".join(row.get("references") or [])
                top1 = (row.get("candidates") or [None])[0]
                lines.append(
                    f"- reading=`{row.get('reading')}` context=`{row.get('context') or ''}` "
                    f"refs=`{refs}` top1=`{top1}`"
                )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}", file=sys.stderr)


def main() -> int:
    all_results: dict[str, list[tuple[dict, str]]] = {}
    total_cost = 0.0
    for name, path in SOURCES.items():
        results, cost = run_for_source(name, path)
        all_results[name] = results
        total_cost += cost
    out_path = ROOT / "docs/probe_failure_analysis.md"
    write_summary(out_path, all_results, total_cost)
    print(f"total cost ${total_cost:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
