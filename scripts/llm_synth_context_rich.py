"""Generate fresh JP IME training rows with rich left context.

The bunsetsu pool's main weakness is fragmented left_context_surface
(often 0-5 chars). This script generates fresh sample rows where
left_context is guaranteed substantive (15-30 chars), targeting the
distribution gap that bunsetsu alone can't fill.

Output: datasets/corpus/synth/llm_context_rich.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_common import ROOT, chat, cost_estimate, get_client


# Genre buckets for rotation. Each bucket round produces N rows.
GENRE_BUCKETS: list[tuple[str, str]] = [
    ("news_economy", "経済ニュース風 (株価・景気・企業動向・金融政策)"),
    ("news_tech", "技術系ニュース風 (AI・半導体・新製品・研究発表)"),
    ("news_politics", "政治・行政ニュース風 (国会・自治体・国際関係)"),
    ("news_sports", "スポーツニュース風 (野球・サッカー・五輪・選手談)"),
    ("news_culture", "文化・芸能ニュース風 (映画・音楽・出版・受賞)"),
    ("blog_lifestyle", "個人ブログ風・生活 (料理・育児・旅行・趣味)"),
    ("blog_review", "レビューブログ風 (書評・商品レビュー・映画感想)"),
    ("essay_literary", "エッセイ・随筆風 (季節・日常観察・思索)"),
    ("technical_doc", "技術文書風 (ソフトウェア仕様・マニュアル・解説)"),
    ("academic_humanities", "人文系論文・解説風 (歴史・哲学・文学批評)"),
    ("academic_science", "理工系論文・解説風 (物理・生物・医学)"),
    ("dialog_business", "ビジネス会話風 (会議・メール・電話応対)"),
    ("dialog_casual", "日常会話風 (友人・家族・SNS 投稿)"),
    ("fiction_dialogue", "小説の地の文 + 会話混在 (場面描写)"),
    ("instructional", "実用ハウツー風 (料理レシピ・DIY・旅行案内)"),
]

PROMPT_SYSTEM = (
    "日本語かな漢字 IME 学習データ生成アシスタント。"
    "指定ジャンルの 2-3 文構成のミニ文書を生成し、最後の 1 文を IME 学習対象 (surface)、"
    "それより前の文を left context として切り出す。"
)

PROMPT_USER_TEMPLATE = """ジャンル「{genre_label}」の 2-3 文程度の自然な日本語ミニ文書を {n} 個生成してください。

各ミニ文書から学習サンプル 1 つを切り出し、次の形式の JSON 1 行で出力 (説明・コードフェンス禁止):
{{"reading":"...対象文(末文)の全文かな...","surface":"...対象文の漢字混じり全文...","left_context_surface":"...対象文の前にある文を 15-30 文字で繋いだもの...","left_context_reading":"...前文の全文かな..."}}

要件:
- 対象 (surface) は 15-50 文字程度で完結した 1 文
- left_context_surface は対象文の直前にある自然な前文 (15-30 文字)、絶対空にしない
- reading / left_context_reading は対応する全文かな
- ジャンルらしい語彙・文体・話題を反映
- 同一テーマの繰り返しは避け、{n} 個間で話題を多様化

出力 ({n} 行):"""


def parse_rows(content: str, source_tag: str, *, min_ctx_len: int = 8) -> list[dict]:
    out: list[dict] = []
    for line in content.strip().splitlines():
        line = line.strip().lstrip("- ").lstrip("•").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        reading = (obj.get("reading") or "").strip()
        surface = (obj.get("surface") or "").strip()
        ctx = (obj.get("left_context_surface") or "").strip()
        if not reading or not surface:
            continue
        if len(ctx) < min_ctx_len:
            # Drop rows where the model didn't produce real context.
            continue
        out.append({
            "reading": reading,
            "surface": surface,
            "left_context_surface": ctx,
            "left_context_reading": (obj.get("left_context_reading") or "").strip(),
            "source": source_tag,
            "span_bunsetsu": 1,
        })
    return out


def main() -> int:
    out_path = ROOT / "datasets/corpus/synth/llm_context_rich.jsonl"
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client, endpoint, model = get_client()
    total_cost = 0.0
    total_rows = 0
    rounds_per_bucket = 5  # 15 buckets × 5 rounds × 10 = ~750 rows
    n_per_round = 10
    with out_path.open("a", encoding="utf-8") as f:
        for tag, label in GENRE_BUCKETS:
            for round_idx in range(rounds_per_bucket):
                try:
                    content, usage = chat(
                        client, endpoint, model,
                        PROMPT_SYSTEM,
                        PROMPT_USER_TEMPLATE.format(genre_label=label, n=n_per_round),
                        max_tokens=4096,
                        temperature=0.9,
                    )
                except Exception as e:
                    print(f"[ctx] {tag} round {round_idx} error: {e}", file=sys.stderr)
                    continue
                total_cost += cost_estimate(usage)
                rows = parse_rows(content, source_tag="synth_llm_context_rich")
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += len(rows)
                print(
                    f"[ctx] {tag} r{round_idx}: {len(rows)} rows  cost=${total_cost:.4f}  total_rows={total_rows}",
                    file=sys.stderr,
                )
                f.flush()
    client.close()
    print(f"[ctx] done: {total_rows} rows, ${total_cost:.4f} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
