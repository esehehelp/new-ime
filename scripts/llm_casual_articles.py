"""Casual / SNS / chat-register article generation, then bunsetsu chunk.

The existing corpus (Hibiki) is wiki/news-heavy. Real IME usage skews
toward casual messages, social media posts, dialogue, blog comments.
Hatsuyume's per-category probe results show particular weakness on
homophone (which often needs casual context) and general (broad register).

Reuses the article-bunsetsu pipeline with casual-only topics + a system
prompt that nudges toward natural informal register.

Output: datasets/corpus/synth/llm_casual_bunsetsu.jsonl
"""

from __future__ import annotations

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

# Casual runs on DeepInfra Gemma 4 31B by default so it can run in
# parallel with the Gemini-flash-based llm_articles_bunsetsu without
# stepping on Gemini's free-tier RPM cap.
ENDPOINT = os.environ.get(
    "CASUAL_LLM_ENDPOINT",
    "https://api.deepinfra.com/v1/openai/chat/completions",
)
MODEL = os.environ.get("CASUAL_LLM_MODEL", "google/gemma-4-31B-it")
TOKEN = os.environ.get("CASUAL_LLM_TOKEN") or os.environ.get(
    "DEEPINFRA_API_KEY"
)
if not TOKEN:
    raise SystemExit(
        "no CASUAL_LLM_TOKEN / DEEPINFRA_API_KEY set in .env"
    )


TOPICS: list[tuple[str, str]] = [
    ("sns_post_short", "SNS の短文投稿 4-6 個 (Twitter/X 風、150 字以内ずつ、絵文字なし、改行で区切る)"),
    ("chat_friend", "友人との LINE/メッセージ会話のログ (10-20 ターン、Aさん:〜 Bさん:〜 形式)"),
    ("chat_family", "家族間の LINE 会話ログ (10-15 ターン、買い物・予定調整など日常)"),
    ("blog_diary_long", "日記風ブログ (1 日の出来事を 600-800 字、口語混じり)"),
    ("blog_complaint", "愚痴・お悩み相談ブログ (口語多め、感情表現あり)"),
    ("blog_review_food", "外食レビュー (評価 + 感想、雑な話し言葉)"),
    ("dialogue_couple", "カップル/夫婦の会話シーン (10-15 ターン、生活の話題)"),
    ("dialogue_workmate", "同僚との雑談 (休憩中の会話、10-15 ターン)"),
    ("dialogue_classmate", "学生同士の会話 (授業や部活、10-15 ターン)"),
    ("monologue_self", "ひとりごと・思考の流れ (口語的な独白 500 字程度)"),
    ("review_consumer", "口語的なネット商品レビュー (ガジェット/化粧品/書籍/ゲーム)"),
    ("question_qa", "知恵袋風の質問本文 + 回答 (口語、生活相談、5 ペア程度)"),
    ("step_recipe_casual", "ノリで書いた料理レシピ (材料リストなし、本文のみ口語)"),
    ("travel_diary_casual", "旅行記 (口語混じりで具体地名混在)"),
    ("opinion_rant", "ネット民の意見・主張 (やや過激な口語、論理は雑)"),
    ("kakikomi_2ch", "雑談スレ風カキコミ (匿名掲示板、5-10 レス連続、>>1 形式 OK)"),
    ("nichijou_observation", "日常の小さな観察 (200-400 字、つぶやき風)"),
    ("haiku_diary", "短歌・俳句風の感想を散りばめた日記 (口語と詩情の混在)"),
    ("kids_voice", "子供の発言 (台詞引用形式、3-5 個まとめて)"),
    ("elder_voice", "年配の方の語り (戦後の思い出・近況など、口語)"),
]

SYSTEM_PROMPT = (
    "日本語の口語・カジュアル文章を多く生成するアシスタント。"
    "敬体・常体・タメ口を混在させ、「〜だよね」「〜じゃん」「〜って」「〜的な」"
    "「〜系」「〜みたいな」など実際のチャット・SNS・口語特有の表現を取り入れる。"
    "顔文字・絵文字は使わず、文字情報のみで生き生きとした話し言葉を表現する。"
    "タイトル・見出し・箇条書き・コードブロック禁止、本文のみ。"
)


def gen_article(client: httpx.Client, topic_label: str) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ジャンル: {topic_label}\n"
                    "上記ジャンルの自然な日本語を生成。500-900 字程度。"
                    "口語・カジュアル多用 OK、本文のみ。"
                ),
            },
        ],
        "max_tokens": 4096,
        "temperature": 1.05,
    }
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = client.post(ENDPOINT, json=body)
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
    raise last_err if last_err else RuntimeError("gen_article exhausted retries")


CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭辞"}


def kata_to_hira(s: str) -> str:
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def bunsetsu_chunks(tagger, text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    cur_surf: list[str] = []
    cur_read: list[str] = []
    for token in tagger(text):
        feat = token.feature
        pos = getattr(feat, "pos1", None) or getattr(feat, "pos", None) or ""
        kana_kata = (
            getattr(feat, "kana", None)
            or getattr(feat, "pron", None)
            or token.surface
        )
        kana = kata_to_hira(kana_kata)
        if pos in CONTENT_POS and cur_surf:
            chunks.append(("".join(cur_surf), "".join(cur_read)))
            cur_surf = []
            cur_read = []
        cur_surf.append(token.surface)
        cur_read.append(kana)
    if cur_surf:
        chunks.append(("".join(cur_surf), "".join(cur_read)))
    return chunks


def emit_rows(
    chunks: list[tuple[str, str]],
    *,
    article_idx: int,
    source_tag: str = "synth_llm_casual",
    max_window: int = 4,
) -> list[dict]:
    rows: list[dict] = []
    for k in range(len(chunks)):
        surface, reading = chunks[k]
        if not surface.strip() or not reading.strip():
            continue
        left_start = max(0, k - max_window)
        left_chunks = chunks[left_start:k]
        lcs = "".join(s for s, _ in left_chunks)
        lcr = "".join(r for _, r in left_chunks)
        rows.append({
            "reading": reading,
            "surface": surface,
            "left_context_surface": lcs,
            "left_context_reading": lcr,
            "source": source_tag,
            "sentence_id": f"{source_tag}:{article_idx}#{k}",
            "span_bunsetsu": 1,
        })
    return rows


def main() -> int:
    out_path = ROOT / "datasets/corpus/synth/llm_casual_bunsetsu.jsonl"
    raw_path = ROOT / "datasets/corpus/synth/_llm_casual_raw.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    if raw_path.exists():
        raw_path.unlink()

    rounds_per_topic = int(os.environ.get("CASUAL_ROUNDS_PER_TOPIC", "10"))
    target_articles = rounds_per_topic * len(TOPICS)
    min_interval = float(os.environ.get("CASUAL_MIN_INTERVAL_SEC", "5.0"))
    print(f"target articles: {target_articles}", file=sys.stderr)

    import fugashi
    tagger = fugashi.Tagger()

    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )

    last_call = 0.0
    article_idx = 0
    total_rows = 0
    raw_f = raw_path.open("a", encoding="utf-8")
    out_f = out_path.open("a", encoding="utf-8")
    try:
        for round_idx in range(rounds_per_topic):
            for tag, label in TOPICS:
                gap = time.monotonic() - last_call
                if gap < min_interval:
                    time.sleep(min_interval - gap)
                last_call = time.monotonic()
                try:
                    article, usage = gen_article(client, label)
                except Exception as e:
                    print(f"[casual] {tag} r{round_idx} error: {e}", file=sys.stderr)
                    continue
                raw_f.write(json.dumps({
                    "article_idx": article_idx,
                    "topic_tag": tag,
                    "topic_label": label,
                    "text": article,
                    "usage": usage,
                }, ensure_ascii=False) + "\n")
                raw_f.flush()
                chunks = bunsetsu_chunks(tagger, article)
                rows = emit_rows(chunks, article_idx=article_idx)
                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                total_rows += len(rows)
                article_idx += 1
                print(
                    f"[casual] {article_idx}/{target_articles} {tag} r{round_idx}: "
                    f"chars={len(article)} rows={len(rows)} total={total_rows}",
                    file=sys.stderr,
                )
    finally:
        raw_f.close()
        out_f.close()
        client.close()
    print(f"done: {article_idx} articles → {total_rows} rows.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
