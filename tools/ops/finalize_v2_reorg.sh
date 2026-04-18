#!/usr/bin/env bash
# bunsetsu_split.py の全 pool 完了後に datasets/v2/ と datasets/v2_bunsetsu/ を
# datasets/corpus/v2/ 配下に移動 + rename (_v2 suffix drop)。
#
# Usage:
#   bash scripts/finalize_v2_reorg.sh
#
# 前提: bunsetsu_split プロセスが全て終了していること。
# 実行中は datasets/v2/*.clean.jsonl が読まれているため mv が失敗する。
set -e
cd "$(dirname "$0")/.."

# Running process check
if ps -ef 2>/dev/null | grep -q "bunsetsu_split" | grep -v grep; then
    echo "[error] bunsetsu_split is still running. Wait for it to finish."
    exit 1
fi

mkdir -p datasets/corpus/v2/sentences datasets/corpus/v2/bunsetsu datasets/corpus/v2/synth

echo "=== move v2 sentence pools (drop _v2 suffix) ==="
mv datasets/v2/wikinews_v2.jsonl        datasets/corpus/v2/sentences/wikinews.jsonl
mv datasets/v2/wikinews_v2.clean.jsonl  datasets/corpus/v2/sentences/wikinews.clean.jsonl
mv datasets/v2/wikibooks_v2.jsonl       datasets/corpus/v2/sentences/wikibooks.jsonl
mv datasets/v2/wikibooks_v2.clean.jsonl datasets/corpus/v2/sentences/wikibooks.clean.jsonl
mv datasets/v2/wiktionary_v2.jsonl      datasets/corpus/v2/sentences/wiktionary.jsonl
mv datasets/v2/wiktionary_v2.clean.jsonl datasets/corpus/v2/sentences/wiktionary.clean.jsonl
mv datasets/v2/tatoeba_v2.jsonl         datasets/corpus/v2/sentences/tatoeba.jsonl
mv datasets/v2/aozora_dialogue.jsonl    datasets/corpus/v2/sentences/aozora_dialogue.jsonl
rmdir datasets/v2 2>/dev/null || echo "  datasets/v2 not empty, leaving"

echo "=== move v2 bunsetsu + synth (drop _v2 suffix) ==="
mv datasets/v2_bunsetsu/wikinews_v2.jsonl     datasets/corpus/v2/bunsetsu/wikinews.jsonl
mv datasets/v2_bunsetsu/wikibooks_v2.jsonl    datasets/corpus/v2/bunsetsu/wikibooks.jsonl
mv datasets/v2_bunsetsu/wiktionary_v2.jsonl   datasets/corpus/v2/bunsetsu/wiktionary.jsonl
mv datasets/v2_bunsetsu/tatoeba_v2.jsonl      datasets/corpus/v2/bunsetsu/tatoeba.jsonl
mv datasets/v2_bunsetsu/aozora_dialogue.jsonl datasets/corpus/v2/bunsetsu/aozora_dialogue.jsonl
mv datasets/v2_bunsetsu/synth_numeric.jsonl      datasets/corpus/v2/synth/numeric.jsonl
mv datasets/v2_bunsetsu/synth_numeric_ext.jsonl  datasets/corpus/v2/synth/numeric_ext.jsonl
rmdir datasets/v2_bunsetsu 2>/dev/null || echo "  datasets/v2_bunsetsu not empty, leaving"

echo
echo "=== done ==="
ls datasets/corpus/v2/sentences/
echo "---"
ls datasets/corpus/v2/bunsetsu/
echo "---"
ls datasets/corpus/v2/synth/
