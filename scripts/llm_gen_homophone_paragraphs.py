"""Homophone-disambiguation paragraph generator.

For each kana reading with multiple competing kanji writings, ask the LLM
to write a 4-6 sentence paragraph that NATURALLY uses ALL variants in
unambiguous contexts. The output is then chunked by fugashi into Schema B
rows where the bunsetsu containing the homophone has full context.

Surface-only generation here — readings come from fugashi later via
`scripts/articles_to_bunsetsu.py`. This avoids the LLM-direct-reading
quality pitfalls observed earlier (kanji-in-reading, semantic mismatch).

Output: `_llm_homophone_para_raw.jsonl` with one row per generated paragraph.
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


# Curated list — kana reading → list of kanji writings the model should
# disambiguate via context. Ordered roughly by IME relevance / observed
# probe failures. Keep the list focused so each paragraph really exercises
# disambiguation rather than name-dropping.
HOMOPHONES: list[tuple[str, list[str]]] = [
    ("きせい", ["規制", "帰省", "既製", "寄生", "気勢"]),
    ("こうせい", ["構成", "校正", "公正", "厚生", "更生", "後世"]),
    ("かんしん", ["関心", "感心", "歓心", "寒心"]),
    ("いがい", ["以外", "意外"]),
    ("いぎ", ["意義", "異議", "威儀"]),
    ("じてん", ["辞典", "事典", "時点", "字典"]),
    ("しょうがい", ["生涯", "障害", "傷害", "渉外"]),
    ("しょうかい", ["紹介", "照会", "商会", "哨戒"]),
    ("せいか", ["成果", "正解", "聖火", "青果", "盛夏"]),
    ("たいしょう", ["対象", "対称", "対照", "大将"]),
    ("たいせい", ["体制", "態勢", "大勢", "耐性", "体勢"]),
    ("ついきゅう", ["追求", "追究", "追及"]),
    ("ふしん", ["不振", "不審", "不信", "腐心"]),
    ("ようしき", ["様式", "洋式"]),
    ("へいこう", ["平行", "並行", "閉口", "平衡"]),
    ("かいしゅう", ["回収", "改修", "改宗", "会衆"]),
    ("けいき", ["景気", "契機", "計器", "刑期"]),
    ("しんちょう", ["慎重", "身長", "新調", "深長"]),
    ("せいさく", ["政策", "製作", "制作"]),
    ("ほうこう", ["方向", "芳香", "咆哮", "奉公"]),
    ("ほしょう", ["保証", "保障", "補償"]),
    ("もと", ["元", "本", "下", "基", "素"]),
    ("あつい", ["暑い", "熱い", "厚い"]),
    ("はやい", ["早い", "速い"]),
    ("あう", ["会う", "合う", "遭う"]),
    ("とる", ["取る", "撮る", "採る", "捕る", "執る"]),
    ("みる", ["見る", "観る", "診る", "看る"]),
    ("きく", ["聞く", "聴く", "効く", "利く"]),
    ("つく", ["着く", "付く", "就く", "突く", "点く"]),
    ("おさめる", ["収める", "納める", "治める", "修める"]),
    ("こうえん", ["公演", "講演", "公園", "後援", "好演"]),
    ("せいき", ["世紀", "正規", "性器"]),
    ("じき", ["時期", "次期", "磁気", "時機", "直", "直き"]),
    ("きかい", ["機会", "機械", "器械", "奇怪"]),
    ("そうい", ["相違", "創意", "総意", "僧位"]),
]


SYSTEM_PROMPT = (
    "日本語の同音異義語を文脈で使い分けた自然な段落を書く文章作家。"
    "段落本文のみ、前置き・引用符・コードフェンス禁止。"
)

USER_TEMPLATE = """同音異義語 `{reading}` の以下の漢字表記を、すべて 1 つの段落の中で自然に使い分けた段落 (4-6 文、400-700 字) を 1 本書いてください。

漢字候補:
{variants_lines}

要件:
- 各漢字をそれぞれ少なくとも 1 回ずつ含める
- 文脈から「どの意味の {reading} か」が読み手にとって明らかになる
- 同じ話題でつなぎ、不自然な羅列にしない
- 一段落のまとまった文章として完結
- タイトル・前書き・引用符 / コードフェンス禁止
- 一気に段落本文だけ出力

段落本文:"""


def gen_paragraph(
    client: httpx.Client,
    endpoint: str,
    model: str,
    reading: str,
    variants: list[str],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    variants_lines = "\n".join(f"- {v}" for v in variants)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(reading=reading, variants_lines=variants_lines),
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
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return content.strip(), usage
        except httpx.HTTPStatusError as e:
            last_err = e
            time.sleep(4.0 * (2 ** attempt))
    raise last_err if last_err else RuntimeError("retries exhausted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--rounds", type=int, default=4, help="paragraphs per homophone")
    ap.add_argument("--min-interval-sec", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    endpoint = os.environ.get("LLM_GEN_ENDPOINT") or os.environ.get(
        "DATA_ROW_LLM_ENDPOINT", "https://api.deepinfra.com/v1/openai/chat/completions"
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
        raise SystemExit("no token in env")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {token}"},
    )
    last_call = 0.0
    article_idx = 0
    target = args.rounds * len(HOMOPHONES)
    print(f"endpoint={endpoint}", file=sys.stderr)
    print(f"model={model}", file=sys.stderr)
    print(f"target paragraphs: {target}", file=sys.stderr)

    with out_path.open("a", encoding="utf-8") as fout:
        for round_idx in range(args.rounds):
            for reading, variants in HOMOPHONES:
                gap = time.monotonic() - last_call
                if gap < args.min_interval_sec:
                    time.sleep(args.min_interval_sec - gap)
                last_call = time.monotonic()
                try:
                    text, usage = gen_paragraph(
                        client, endpoint, model, reading, variants,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    print(f"[homo] {reading} r{round_idx} error: {e}", file=sys.stderr)
                    continue
                fout.write(json.dumps({
                    "article_idx": article_idx,
                    "topic_tag": f"homo_{reading}",
                    "topic_label": f"homophone {reading} ({'/'.join(variants)})",
                    "text": text,
                    "usage": usage,
                }, ensure_ascii=False) + "\n")
                fout.flush()
                article_idx += 1
                print(
                    f"[homo] {article_idx}/{target} {reading} r{round_idx} chars={len(text)}",
                    file=sys.stderr,
                )
    client.close()
    print(f"done: {article_idx} paragraphs → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
