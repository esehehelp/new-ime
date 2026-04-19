#!/usr/bin/env bash
# v2 の全 pool を bunsetsu 化する orchestrator。
# 5 sources を順次実行。所要時間は CPU シングルコアで ~100 分想定。
set -e
cd "$(dirname "$0")/../.."

OUT_DIR="datasets/v2_bunsetsu"
mkdir -p "$OUT_DIR"

run_one() {
    local src="$1"
    local source_tag="$2"
    local out="$OUT_DIR/${source_tag}.jsonl"
    echo "====================================="
    echo "=== $source_tag : $src -> $out"
    echo "====================================="
    uv run python -m datasets.tools.corpus.bunsetsu_split \
        --src "$src" --out "$out" --source "$source_tag" \
        --batch-size 64 --report-every 10000
}

# 小さい順に。失敗したら分かりやすい。
run_one datasets/v2/wikinews_v2.clean.jsonl   wikinews_v2
run_one datasets/v2/aozora_dialogue.jsonl     aozora_dialogue
run_one datasets/v2/tatoeba_v2.jsonl          tatoeba_v2
run_one datasets/v2/wikibooks_v2.clean.jsonl  wikibooks_v2
run_one datasets/v2/wiktionary_v2.clean.jsonl wiktionary_v2

echo
echo "=== summary ==="
wc -l "$OUT_DIR"/*.jsonl
du -sh "$OUT_DIR"
