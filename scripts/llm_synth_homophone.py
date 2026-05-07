"""Generate diverse homophone-bearing JP sentences via LLM.

Existing mozc_homophone.jsonl is a flat dictionary template (393k pairs but
no sentence context). This script asks the LLM to generate natural sentences
that exercise specific homophones with surrounding context, output as
Schema-B rows ready to mix in.

Output: datasets/corpus/synth/llm_homophone.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_common import ROOT, chat, cost_estimate, get_client


# Frequent homophones the model should disambiguate. Each entry: reading
# → list of plausible kanji surface choices. Curated for IME relevance.
HOMOPHONES: dict[str, list[str]] = {
    "きせい": ["規制", "帰省", "既製", "寄生", "気勢"],
    "こうせい": ["構成", "校正", "公正", "厚生", "更生", "後世", "攻勢"],
    "かんしん": ["関心", "感心", "歓心", "寒心"],
    "いがい": ["以外", "意外"],
    "いぎ": ["意義", "異議", "威儀"],
    "じてん": ["辞典", "事典", "時点", "字典"],
    "しょうがい": ["生涯", "障害", "傷害", "渉外"],
    "しょうかい": ["紹介", "照会", "商会", "哨戒"],
    "せいか": ["成果", "正解", "聖火", "青果", "盛夏"],
    "たいしょう": ["対象", "対称", "対照", "大将", "大正"],
    "たいせい": ["体制", "態勢", "大勢", "耐性", "体勢"],
    "ついきゅう": ["追求", "追究", "追及"],
    "ふしん": ["不振", "不審", "不信", "腐心"],
    "ようしき": ["様式", "洋式"],
    "へいこう": ["平行", "並行", "閉口", "平衡"],
    "かいしゅう": ["回収", "改修", "改宗", "会衆"],
    "けいき": ["景気", "契機", "計器", "刑期"],
    "しんちょう": ["慎重", "身長", "新調", "深長"],
    "せいさく": ["政策", "製作", "制作"],
    "ほうこう": ["方向", "芳香", "咆哮", "奉公"],
    "ほしょう": ["保証", "保障", "補償"],
    "もと": ["元", "本", "下", "基", "素"],
    "あつい": ["暑い", "熱い", "厚い"],
    "はやい": ["早い", "速い"],
    "あう": ["会う", "合う", "遭う"],
    "とる": ["取る", "撮る", "採る", "捕る", "執る"],
    "みる": ["見る", "観る", "診る", "看る"],
    "きく": ["聞く", "聴く", "効く", "利く"],
    "つく": ["着く", "付く", "就く", "突く", "点く"],
    "おさめる": ["収める", "納める", "治める", "修める"],
}

PROMPT_SYSTEM = (
    "日本語かな漢字 IME 学習データ生成アシスタント。"
    "指定された同音異義語ごとに、文脈で意味が確定する自然な短文を生成する。"
)

PROMPT_USER_TEMPLATE = """同音異義語 `{reading}` の下記の異なる漢字表記それぞれについて、その意味が文脈で曖昧でなく確定する自然な日本語の文を {n_per} 個生成してください。

漢字候補: {variants}

各文は次の形式の JSON 1 行で出力 (説明・コードフェンス禁止):
{{"reading":"...全文かな読み...","surface":"...対応する漢字混じり全文...","left_context_surface":"...10-25 文字程度の自然な前文...","left_context_reading":"...前文の全文かな..."}}

要件:
- surface は対象同音異義語を必ず含み、文として完結
- reading は surface の正確な全文かな (脱字・余分なし)
- left_context_surface / left_context_reading は同一意味の前文 (空にしない)
- 文体は実際の日本語コーパス的に自然 (新聞・小説・ブログ風混在可)
- 同じ漢字選択でも文の話題は重複しないよう多様化

出力 ({n_per} 文 × {n_variants} 漢字 = 計 {total} 行):"""


def build_user_prompt(reading: str, variants: list[str], n_per: int = 3) -> str:
    return PROMPT_USER_TEMPLATE.format(
        reading=reading,
        variants=" / ".join(variants),
        n_per=n_per,
        n_variants=len(variants),
        total=n_per * len(variants),
    )


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
    out_path = ROOT / "datasets/corpus/synth/llm_homophone.jsonl"
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client, endpoint, model = get_client()
    total_cost = 0.0
    total_rows = 0
    with out_path.open("a", encoding="utf-8") as f:
        for reading, variants in HOMOPHONES.items():
            try:
                content, usage = chat(
                    client, endpoint, model,
                    PROMPT_SYSTEM,
                    build_user_prompt(reading, variants, n_per=3),
                    max_tokens=3072,
                    temperature=0.8,
                )
            except Exception as e:
                print(f"[homophone] {reading} error: {e}", file=sys.stderr)
                continue
            cost = cost_estimate(usage)
            total_cost += cost
            rows = parse_rows(content, source_tag="synth_llm_homophone")
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += len(rows)
            print(
                f"[homophone] {reading}: {len(rows)} rows  cost=${total_cost:.4f}  total_rows={total_rows}",
                file=sys.stderr,
            )
            f.flush()
    client.close()
    print(f"[homophone] done: {total_rows} rows, ${total_cost:.4f} → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
