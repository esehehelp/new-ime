"""Gemini-generate JP articles → fugashi tokenize → bunsetsu chunks → Schema B.

User strategy: low-medium quality LLM articles (volume + diversity matter
more than per-row polish), then deterministic MeCab/fugashi bunsetsu
segmentation handles the structure. One ~500-token article expands to ~50
bunsetsu rows with natural multi-bunsetsu left context, far more efficient
than asking the LLM to emit one polished row at a time.

Output: datasets/corpus/synth/llm_articles_bunsetsu.jsonl
"""

from __future__ import annotations

import json
import os
import sys
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


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = os.environ.get("GEMINI_ARTICLE_MODEL", "gemini-2.5-flash")
GEMINI_KEY = os.environ.get("GOOGLE_AISTUDIO_API_KEY")
if not GEMINI_KEY:
    raise SystemExit("GOOGLE_AISTUDIO_API_KEY not set in .env")


# Article topics. Each round picks one and asks for a 400-700 char article.
# Mix is intentionally broad: tech-heavy buckets target the weak `tech`
# probe category, daily/news buckets fill general distribution.
TOPICS: list[tuple[str, str]] = [
    ("tech_software_casual", "プログラミング言語・ソフトウェア開発の話題 (やや雑な個人ブログ風)"),
    ("tech_software_formal", "ソフトウェア技術解説 (硬めの技術記事風)"),
    ("tech_hardware", "PC・スマホ・電子機器のレビュー / 解説"),
    ("tech_ai_ml", "AI / 機械学習 / 深層学習の解説 (中級者向け)"),
    ("tech_security", "セキュリティ・暗号・認証の解説"),
    ("tech_web", "Web 技術 (HTTP / CSS / JavaScript / フロントエンド)"),
    ("tech_database", "データベース・SQL・データ基盤の解説"),
    ("tech_network", "ネットワーク・インフラ・クラウドの解説"),
    ("science_physics", "物理 / 工学トピックの一般向け解説"),
    ("science_biology", "生物・医学の一般向け解説"),
    ("science_math", "数学・統計トピックの解説"),
    ("news_economy", "経済・市場・企業動向のニュース風記事"),
    ("news_politics", "政治・行政ニュース風記事"),
    ("news_culture", "文化・芸能・出版ニュース風記事"),
    ("news_sports", "スポーツニュース風記事 (野球 / サッカー / 五輪)"),
    ("blog_food", "料理 / グルメ / レシピのブログ"),
    ("blog_travel", "旅行記 / 観光地紹介のブログ"),
    ("blog_diary", "日常生活エッセイ・雑記"),
    ("essay_literary", "随筆 / 文学的エッセイ"),
    ("instructional_howto", "ハウツー記事 (DIY / 生活術 / 手続き解説)"),
    ("review_product", "商品レビュー (家電・ガジェット・本・映画)"),
    ("dialogue_business", "ビジネス会話シーン (議事録風 / メール本文)"),
    ("fiction_short", "ショートフィクション・物語の地の文+会話"),
    ("history_explanatory", "歴史トピックの一般向け解説"),
]


SYSTEM_PROMPT = (
    "日本語の文章生成アシスタント。指定ジャンルの 400-700 字程度の自然な記事本文を 1 本だけ書く。"
    "タイトル・見出し・箇条書き・コードブロック禁止、地の文と会話文のみ。"
    "句点で文を区切り、書き言葉として自然な日本語にする。"
)


def gen_article(client: httpx.Client, topic_label: str) -> tuple[str, dict]:
    body = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ジャンル: {topic_label}\n"
                    "上記ジャンルの 400-700 字程度の自然な日本語記事を 1 本書いてください。"
                    "本文のみ、タイトル・見出し・箇条書き禁止。"
                ),
            },
        ],
        "max_tokens": 4096,
        "temperature": 0.95,
    }
    # Retry-with-backoff on 429 / 5xx — Gemini free tier RPM is tight
    # (smoke run hit 429 on ~3/24 requests under unthrottled bursts).
    import time
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = client.post(GEMINI_ENDPOINT, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(60.0, 4.0 * (2 ** attempt))
                # Honour Retry-After if present.
                ra = r.headers.get("retry-after")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                # Pull the suggested retryDelay from the response body if present.
                try:
                    payload = r.json()
                    msg = payload.get("error", {}).get("message", "")
                    if "retry in" in msg.lower():
                        import re
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


# Bunsetsu segmentation ---------------------------------------------------

CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭辞"}


def kata_to_hira(s: str) -> str:
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # standard katakana → hiragana
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def bunsetsu_chunks(tagger, text: str) -> list[tuple[str, str]]:
    """Return [(surface, reading)] one entry per bunsetsu."""
    chunks: list[tuple[str, str]] = []
    cur_surf: list[str] = []
    cur_read: list[str] = []
    for token in tagger(text):
        feat = token.feature
        # unidic feature schema: feat.pos1, feat.kana / feat.pron
        pos = getattr(feat, "pos1", None) or getattr(feat, "pos", None) or ""
        kana_kata = (
            getattr(feat, "kana", None)
            or getattr(feat, "pron", None)
            or token.surface
        )
        kana = kata_to_hira(kana_kata)
        # New bunsetsu starts when we see a content POS *and* we already
        # have something in flight (the leading particle/punct goes with
        # the previous bunsetsu).
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
    source_tag: str = "synth_llm_articles",
    max_window: int = 4,
) -> list[dict]:
    """For each bunsetsu position k, emit a row with that bunsetsu as the
    (reading, surface) target and the preceding `max_window` bunsetsu as
    left_context."""
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
    out_path = ROOT / "datasets/corpus/synth/llm_articles_bunsetsu.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    raw_articles_path = ROOT / "datasets/corpus/synth/_llm_articles_raw.jsonl"
    if raw_articles_path.exists():
        raw_articles_path.unlink()

    rounds_per_topic = int(os.environ.get("ARTICLE_ROUNDS_PER_TOPIC", "8"))
    target_articles = rounds_per_topic * len(TOPICS)
    print(f"target articles: {target_articles}", file=sys.stderr)

    # Lazy import — keeps script importable without fugashi installed for
    # quick edits, but main run needs it.
    import fugashi
    tagger = fugashi.Tagger()

    client = httpx.Client(
        timeout=httpx.Timeout(120.0),
        headers={"Authorization": f"Bearer {GEMINI_KEY}"},
    )

    import time
    min_interval = float(os.environ.get("ARTICLE_MIN_INTERVAL_SEC", "5.0"))
    last_call = 0.0
    article_idx = 0
    total_rows = 0
    total_completion_tokens = 0
    raw_f = raw_articles_path.open("a", encoding="utf-8")
    out_f = out_path.open("a", encoding="utf-8")
    try:
        for round_idx in range(rounds_per_topic):
            for tag, label in TOPICS:
                # Throttle to stay under free-tier RPM cap.
                gap = time.monotonic() - last_call
                if gap < min_interval:
                    time.sleep(min_interval - gap)
                last_call = time.monotonic()
                try:
                    article, usage = gen_article(client, label)
                except httpx.HTTPStatusError as e:
                    print(f"[gen] {tag} r{round_idx} HTTP {e.response.status_code}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[gen] {tag} r{round_idx} error: {e}", file=sys.stderr)
                    continue
                total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                raw_f.write(json.dumps({
                    "article_idx": article_idx,
                    "topic_tag": tag,
                    "topic_label": label,
                    "text": article,
                    "usage": usage,
                }, ensure_ascii=False) + "\n")
                raw_f.flush()
                # Tokenize + bunsetsu + emit.
                chunks = bunsetsu_chunks(tagger, article)
                rows = emit_rows(chunks, article_idx=article_idx)
                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                total_rows += len(rows)
                article_idx += 1
                print(
                    f"[gen] {article_idx}/{target_articles} {tag} r{round_idx}: "
                    f"chars={len(article)} chunks={len(chunks)} rows={len(rows)} "
                    f"total_rows={total_rows} comp_tok={total_completion_tokens}",
                    file=sys.stderr,
                )
    finally:
        raw_f.close()
        out_f.close()
        client.close()
    print(f"done: {article_idx} articles → {total_rows} rows. raw={raw_articles_path.name}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
