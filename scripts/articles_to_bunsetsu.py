"""Tokenize raw articles → Schema B bunsetsu rows. Pure rule-based reading.

Reads JSONL with `{text, [topic_tag], [article_idx]}` and emits Schema B
training rows. The reading attribution is fully deterministic (fugashi +
unidic-lite morphological analysis), independent of whatever LLM produced
the source text. This mirrors how the corpus was meant to flow:

    LLM → text (surface only)         ← creative, license-relevant step
    morphological analyser → reading  ← deterministic, reproducible

Splitting these means you can:
  - re-chunk the same articles with different window sizes
  - rerun the chunker after a unidic version bump without paying the LLM
  - mix article sources (Gemini + DeepInfra + manual) into one chunk pass
  - validate the chunker independently of the gen step

Usage:
    python scripts/articles_to_bunsetsu.py \\
        --input  datasets/corpus/synth/_llm_articles_raw.jsonl \\
        --output datasets/corpus/synth/llm_articles_bunsetsu.jsonl \\
        --source-tag synth_llm_articles \\
        [--max-window 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    """[(surface, reading)] one entry per bunsetsu."""
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
    source_tag: str,
    max_window: int,
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="raw articles JSONL")
    ap.add_argument("--output", required=True, help="Schema B output JSONL")
    ap.add_argument("--source-tag", required=True, help="row.source value")
    ap.add_argument("--max-window", type=int, default=4)
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--idx-field", default="article_idx")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import fugashi
    tagger = fugashi.Tagger()

    n_articles = 0
    n_rows = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(args.text_field) or ""
            if not text.strip():
                continue
            article_idx = obj.get(args.idx_field, n_articles)
            chunks = bunsetsu_chunks(tagger, text)
            rows = emit_rows(
                chunks,
                article_idx=article_idx,
                source_tag=args.source_tag,
                max_window=args.max_window,
            )
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += len(rows)
            n_articles += 1
            if n_articles % 50 == 0:
                print(f"[chunk] {n_articles} articles → {n_rows} rows", file=sys.stderr)
    print(f"done: {n_articles} articles → {n_rows} rows → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
