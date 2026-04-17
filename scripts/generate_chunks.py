"""Generate short-text (bunsetsu-level) training data from clean JSONL.

Splits sentences into bunsetsu (文節) chunks using MeCab POS info,
then generates 1-4 bunsetsu combinations as training pairs.

This addresses the "short text problem" where the model fails on
inputs like やまにのぼる, ほんをかう, etc. — because training data
is predominantly long sentences from Wikipedia/Aozora.

Usage:
    uv run python scripts/generate_chunks.py \
        --input datasets/wiki_clean_v3.jsonl datasets/aozora_clean.jsonl \
        --output datasets/chunks_v3.jsonl \
        --max-samples 5000000 \
        --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from multiprocessing import Pool
from pathlib import Path

import jaconv
import MeCab


def _worker_init():
    global _tagger
    _tagger = MeCab.Tagger()
    _tagger.parse("")


def _split_bunsetsu(text: str) -> list[tuple[str, str]]:
    """Split text into bunsetsu (文節) units using MeCab POS.

    Returns list of (surface, reading) tuples per bunsetsu.

    Rule: A bunsetsu boundary occurs AFTER a particle (助詞),
    auxiliary verb (助動詞), or punctuation, when followed by
    a content word (名詞, 動詞, 形容詞, 副詞, 連体詞, 接続詞, 感動詞).
    """
    global _tagger

    node = _tagger.parseToNode(text)
    morphemes = []
    while node:
        if node.surface:
            features = node.feature.split(",")
            pos = features[0]
            # Get reading from features[17] (仮名形出現形)
            reading = None
            if len(features) >= 18 and features[17] != "*":
                reading = features[17]
            elif len(features) >= 7 and features[6] != "*":
                reading = features[6]

            if reading:
                reading_hira = jaconv.kata2hira(reading)
            elif all("\u3040" <= c <= "\u309f" or not c.strip() for c in node.surface):
                reading_hira = node.surface
            elif all(c in "、。！？「」『』（）・…―─　\n\r\t " for c in node.surface):
                reading_hira = node.surface
            else:
                reading_hira = None

            morphemes.append((node.surface, reading_hira, pos))
        node = node.next

    if not morphemes:
        return []

    # Group morphemes into bunsetsu
    bunsetsu_list: list[tuple[str, str]] = []
    current_surface = ""
    current_reading = ""
    has_content = False

    CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞", "接頭辞"}
    BOUNDARY_POS = {"助詞", "助動詞"}

    prev_pos = ""
    for surface, reading, pos in morphemes:
        if reading is None:
            # Unknown reading: flush current and skip
            if current_surface and has_content:
                bunsetsu_list.append((current_surface, current_reading))
            current_surface = ""
            current_reading = ""
            has_content = False
            prev_pos = pos
            continue

        # Detect bunsetsu boundary: after function word, before content word
        if (prev_pos in BOUNDARY_POS and pos in CONTENT_POS
                and current_surface and has_content):
            bunsetsu_list.append((current_surface, current_reading))
            current_surface = ""
            current_reading = ""
            has_content = False

        current_surface += surface
        current_reading += reading
        if pos in CONTENT_POS:
            has_content = True
        prev_pos = pos

    # Flush last bunsetsu
    if current_surface and has_content:
        bunsetsu_list.append((current_surface, current_reading))

    return bunsetsu_list


def _process_sentence(text: str) -> list[dict]:
    """Generate chunk training pairs from a sentence.

    For each sentence, produces:
    - Individual bunsetsu (1-chunk)
    - 2-bunsetsu combinations
    - 3-bunsetsu combinations
    - 4-bunsetsu combinations (if available)
    """
    bunsetsu = _split_bunsetsu(text)
    if len(bunsetsu) < 1:
        return []

    pairs = []

    # Filter: reading must be hiragana-only
    valid_re = re.compile(r"^[\u3040-\u309f\u30fc、。！？「」『』（）・…―─　\s]+$")

    for window in range(1, min(5, len(bunsetsu) + 1)):
        for i in range(len(bunsetsu) - window + 1):
            chunk = bunsetsu[i:i + window]
            surface = "".join(s for s, _ in chunk)
            reading = "".join(r for _, r in chunk)

            # Skip if too short or too long
            if len(surface) < 2 or len(surface) > 60:
                continue
            if not reading or not valid_re.match(reading):
                continue
            # Skip if reading == surface (no kanji to convert)
            if reading == surface:
                continue

            pairs.append({
                "reading": reading,
                "surface": surface,
                "context": "",  # No context for chunk training
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_workers = args.workers or max(1, os.cpu_count() - 2)
    rng = random.Random(args.seed)

    # Collect all surface texts
    print("Loading sentences...")
    sentences = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                sentences.append(d["surface"])
    print(f"Total sentences: {len(sentences):,}")

    # Shuffle and limit
    rng.shuffle(sentences)

    # Process with multiprocessing
    print(f"Generating chunks with {num_workers} workers...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    seen = set()

    with open(output_path, "w", encoding="utf-8") as out, \
         Pool(processes=num_workers, initializer=_worker_init) as pool:

        for pairs in pool.imap_unordered(_process_sentence, sentences, chunksize=500):
            for pair in pairs:
                # Dedup
                key = pair["reading"] + "|" + pair["surface"]
                if key in seen:
                    continue
                seen.add(key)

                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_chunks += 1

                if total_chunks >= args.max_samples:
                    break

            if total_chunks >= args.max_samples:
                break

            if total_chunks % 500000 == 0 and total_chunks > 0:
                print(f"  {total_chunks:,} chunks...", flush=True)

    print(f"\nDone: {len(sentences):,} sentences → {total_chunks:,} chunks")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
