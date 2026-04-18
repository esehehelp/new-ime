"""v2 corpus の文レベル JSONL を bunsetsu 単位 + 2-bunsetsu 塊に展開。

入力:
    datasets/v2/*.clean.jsonl, aozora_dialogue.jsonl, tatoeba_v2.jsonl

出力:
    datasets/v2_bunsetsu/{source}.jsonl
    各行:
        {
          "reading":              目的句の yomi (ひらがな),
          "surface":              正解表記,
          "left_context_surface": 直前 1-2 bunsetsu の表記 (無ければ ""),
          "left_context_reading": 同 yomi,
          "span_bunsetsu":        1 or 2,
          "source":               "aozora_dialogue" 等,
          "sentence_id":          "{source}:{row}#{offset}"
        }

スパン戦略:
    各文について
      - 単独 bunsetsu: offset 0..N-1 で「i 番目を target, i-1 を context」
      - 2-bunsetsu 塊: offset 0..N-2 で「i..i+1 を target, i-1 を context」
    両方を同じ JSONL に emit (span_bunsetsu で区別)。

品質フィルタ:
    - target の reading/surface は必ず存在
    - target reading はひらがなのみ (句読点は末尾で剥がす)
    - target surface は漢字かカタカナを少なくとも 1 含む (ひらがなだけの
      句は edge 以外では学習価値が薄いため)
    - 長さは target 1-30 字、context 0-30 字
    - reading 桁数 == surface 桁数範囲 (明らかな崩れを捨てる、具体的な
      下限は hiragana >= 1)

Ginza の split_mode='C' (粗) を採用。default だと compound_splitter が
None で落ちるため spacy.load(config=...) で上書き。

Usage:
    uv run python -m tools.corpus_v2.bunsetsu_split \
        --src datasets/v2/tatoeba_v2.jsonl \
        --out datasets/v2_bunsetsu/tatoeba_v2.jsonl \
        --source tatoeba_v2 \
        --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import spacy
from ginza import bunsetu_spans

KATA_RANGE = (0x30A1, 0x30F6)
HIRA_RE = re.compile(r"^[\u3041-\u309F\u30FC]+$")
KANJI_RE = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
KATA_RE = re.compile(r"[\u30A1-\u30FA\u30FC]")
PUNCT_TAIL = re.compile(r"[。、．，．・！？\!\?\s]+$")


def kata_to_hira(s: str) -> str:
    out = []
    for c in s:
        code = ord(c)
        if KATA_RANGE[0] <= code <= KATA_RANGE[1]:
            out.append(chr(code - 0x60))
        else:
            out.append(c)
    return "".join(out)


def load_ginza():
    return spacy.load(
        "ja_ginza",
        config={"components": {"compound_splitter": {"split_mode": "C"}}},
    )


def span_reading(span) -> str:
    """Ginza span の reading を kana→hira 変換で抽出。"""
    parts = []
    for t in span:
        r = t.morph.get("Reading")
        parts.append(r[0] if r else t.text)
    return kata_to_hira("".join(parts))


def clean_target(reading: str, surface: str) -> tuple[str, str] | None:
    """Target の reading/surface を正規化。None なら reject。"""
    # strip trailing punct from both
    r = PUNCT_TAIL.sub("", reading).strip()
    s = PUNCT_TAIL.sub("", surface).strip()
    if not r or not s:
        return None
    # reading should be pure hiragana
    if not HIRA_RE.match(r):
        return None
    # surface must contain kanji or katakana (skip pure hiragana targets)
    if not (KANJI_RE.search(s) or KATA_RE.search(s)):
        return None
    if not (1 <= len(s) <= 30) or not (1 <= len(r) <= 30):
        return None
    return r, s


def clean_context(reading: str, surface: str) -> tuple[str, str]:
    """Context は ASCII/記号は残すが長さは 30 までにクリップ。"""
    r = reading[-30:] if len(reading) > 30 else reading
    s = surface[-30:] if len(surface) > 30 else surface
    return r, s


def items_from_doc(doc, source: str, sentence_id: str) -> list[dict]:
    """1 文 (Ginza doc) から bunsetsu spans を抽出し JSONL 行を生成。"""
    spans = list(bunsetu_spans(doc))
    n = len(spans)
    if n == 0:
        return []

    # 前もって reading/surface をキャッシュ。
    span_info = []
    for sp in spans:
        surf = sp.text
        read = span_reading(sp)
        span_info.append((surf, read))

    out: list[dict] = []

    # 単独 bunsetsu
    for i in range(n):
        ctx_s = span_info[i - 1][0] if i > 0 else ""
        ctx_r = span_info[i - 1][1] if i > 0 else ""
        ctx_s, ctx_r = clean_context(ctx_r, ctx_s) if False else (ctx_s, ctx_r)
        tgt = clean_target(span_info[i][1], span_info[i][0])
        if tgt is None:
            continue
        r, s = tgt
        out.append({
            "reading": r,
            "surface": s,
            "left_context_surface": ctx_s,
            "left_context_reading": ctx_r,
            "span_bunsetsu": 1,
            "source": source,
            "sentence_id": f"{sentence_id}#{i}",
        })

    # 2-bunsetsu
    for i in range(n - 1):
        ctx_s = span_info[i - 1][0] if i > 0 else ""
        ctx_r = span_info[i - 1][1] if i > 0 else ""
        combined_surf = span_info[i][0] + span_info[i + 1][0]
        combined_read = span_info[i][1] + span_info[i + 1][1]
        tgt = clean_target(combined_read, combined_surf)
        if tgt is None:
            continue
        r, s = tgt
        out.append({
            "reading": r,
            "surface": s,
            "left_context_surface": ctx_s,
            "left_context_reading": ctx_r,
            "span_bunsetsu": 2,
            "source": source,
            "sentence_id": f"{sentence_id}#{i}-{i+1}",
        })

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-sentences", type=int, default=0, help="0 = no cap")
    ap.add_argument("--report-every", type=int, default=5000)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Ginza...", flush=True)
    t0 = time.perf_counter()
    nlp = load_ginza()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Stream sentences from JSONL, tag each with its row index.
    def sent_stream():
        with src.open(encoding="utf-8") as f:
            for row, line in enumerate(f):
                if args.max_sentences and row >= args.max_sentences:
                    break
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                surf = rec.get("surface", "")
                if not surf or len(surf) > 200:
                    continue
                yield surf, row

    processed = 0
    emitted = 0
    t_start = time.perf_counter()
    with out.open("w", encoding="utf-8") as g:
        # Use nlp.pipe for batching. Pass (text, row_idx) as-tuple via
        # as_tuples so we can recover the sentence_id per doc.
        stream = sent_stream()
        for doc, row in nlp.pipe(stream, batch_size=args.batch_size,
                                  as_tuples=True):
            sentence_id = f"{args.source}:{row}"
            try:
                items = items_from_doc(doc, args.source, sentence_id)
            except Exception:
                items = []
            for item in items:
                g.write(json.dumps(item, ensure_ascii=False) + "\n")
                emitted += 1
            processed += 1
            if processed % args.report_every == 0:
                elapsed = time.perf_counter() - t_start
                rate = processed / elapsed if elapsed else 0
                print(
                    f"  sent={processed:,} emitted={emitted:,} "
                    f"rate={rate:.1f} sent/s ({emitted/max(1,processed):.2f} items/sent)",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    rate = processed / elapsed if elapsed else 0
    print(
        f"\ndone: sent={processed:,} emitted={emitted:,} "
        f"({emitted/max(1,processed):.2f} items/sent)  "
        f"elapsed={elapsed:.0f}s rate={rate:.1f} sent/s -> {out}"
    )


if __name__ == "__main__":
    main()
