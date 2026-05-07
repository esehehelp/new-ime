"""Generate JP sentences featuring named entities (人名/地名/組織名).

The probe `names` category sits at 0.473 (Suiko-v1) / 0.473 (Hatsuyume),
worst after homophone. Existing synth_name.jsonl is template-based; this
script asks the LLM for naturally-flowing sentences with NE coverage.

Output: datasets/corpus/synth/llm_names.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_common import ROOT, chat, cost_estimate, get_client


# Categories the model should rotate through. Each round asks for diverse
# entity-bearing sentences in that category.
NE_BUCKETS: list[tuple[str, str]] = [
    ("japanese_person_common", "日常的な日本人姓名 (鈴木 / 佐藤 / 田中 / 高橋 / 山田 などフルネーム)"),
    ("japanese_person_rare", "やや珍しい日本人姓名 (佐々木 / 一ノ瀬 / 御手洗 / 五十嵐 / 海老原 など)"),
    ("foreign_person_kana", "外国人姓名のカタカナ表記 (スティーブ・ジョブズ / ナタリー・ポートマン など)"),
    ("japanese_place_pref", "都道府県・市区町村 (神奈川県横浜市 / 福岡県北九州市 / 北海道函館市 など)"),
    ("japanese_place_landmark", "観光地・建造物 (浅草寺 / 東京タワー / 厳島神社 / 黒部ダム など)"),
    ("foreign_place", "海外の地名 (パリ / ニューヨーク / イスタンブール / シンガポール など)"),
    ("organization_corp", "日本企業名 (トヨタ自動車 / ソニーグループ / 三菱重工業 など)"),
    ("organization_school", "学校・大学 (東京大学 / 早稲田大学 / 開成高等学校 など)"),
    ("organization_govt", "政府機関・自治体 (経済産業省 / 国税庁 / 横浜市役所 など)"),
    ("product_brand", "ブランド・商品名 (ユニクロ / ガリガリ君 / シャネル など)"),
    ("event_period", "歴史/時代/イベント名 (明治維新 / 万国博覧会 / 高度経済成長期 など)"),
    ("media_title", "作品タイトル (源氏物語 / 鬼滅の刃 / 千と千尋の神隠し など)"),
]

PROMPT_SYSTEM = (
    "日本語かな漢字 IME 学習データ生成アシスタント。"
    "指定されたカテゴリの固有名詞を含む自然な短文を多様に生成する。"
)

PROMPT_USER_TEMPLATE = """カテゴリ「{cat_label}」の固有名詞を含む自然な日本語の文を {n} 個生成してください。

各文 1 行 JSON (説明・コードフェンス禁止):
{{"reading":"...全文かな読み...","surface":"...対応する漢字混じり全文...","left_context_surface":"...10-25 文字の自然な前文...","left_context_reading":"...前文の全文かな..."}}

要件:
- surface に上記カテゴリの固有名詞 (実在/架空問わず、自然なもの) を必ず含む
- 同一エンティティの重複生成は避け、固有名詞自体を毎回変える
- reading は surface の正確な全文かな
- left_context は同じ話題の自然な前文 (空にしない)
- 文体は新聞・小説・ブログ風など実コーパス的に自然
- 固有名詞の難易度は混在 (易しい / 中 / 難)

出力 ({n} 行):"""


def parse_rows(content: str, source_tag: str) -> list[dict]:
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
        if not reading or not surface:
            continue
        out.append({
            "reading": reading,
            "surface": surface,
            "left_context_surface": (obj.get("left_context_surface") or "").strip(),
            "left_context_reading": (obj.get("left_context_reading") or "").strip(),
            "source": source_tag,
            "span_bunsetsu": 1,
        })
    return out


def main() -> int:
    out_path = ROOT / "datasets/corpus/synth/llm_names.jsonl"
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client, endpoint, model = get_client()
    total_cost = 0.0
    total_rows = 0
    rounds_per_bucket = 4  # 12 buckets × 4 rounds × 12 sentences = ~576 rows
    n_per_round = 12
    with out_path.open("a", encoding="utf-8") as f:
        for tag, label in NE_BUCKETS:
            for round_idx in range(rounds_per_bucket):
                try:
                    content, usage = chat(
                        client, endpoint, model,
                        PROMPT_SYSTEM,
                        PROMPT_USER_TEMPLATE.format(cat_label=label, n=n_per_round),
                        max_tokens=3500,
                        temperature=0.95,
                    )
                except Exception as e:
                    print(f"[names] {tag} round {round_idx} error: {e}", file=sys.stderr)
                    continue
                total_cost += cost_estimate(usage)
                rows = parse_rows(content, source_tag="synth_llm_names")
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += len(rows)
                print(
                    f"[names] {tag} r{round_idx}: {len(rows)} rows  cost=${total_cost:.4f}  total_rows={total_rows}",
                    file=sys.stderr,
                )
                f.flush()
    client.close()
    print(f"[names] done: {total_rows} rows, ${total_cost:.4f} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
