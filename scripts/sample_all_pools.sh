#!/usr/bin/env bash
# Run data-pool-sampler across every local raw pool.
#
# Outputs under datasets/audits/pool-qa/<pool>/round_N/samples.jsonl.
# Single-threaded per pool; small pools are near-instant, large pools
# (chunks_100m, fineweb2_ja, short/combined) stream the full file so they
# take a few minutes each.
set -eu

REPO="D:/Dev/new-ime"
BIN="$REPO/build/release/data-pool-sampler.exe"
OUT="$REPO/datasets/audits/pool-qa"
ROUNDS=${ROUNDS:-10}
SAMPLES=${SAMPLES:-100}
SEED=${SEED:-42}

run_pool() {
    local input="$1"
    local name="$2"
    if [ ! -f "$input" ]; then
        echo "[skip] $name: $input not found" >&2
        return 0
    fi
    echo "[pool] $name ($input)"
    "$BIN" \
        --input "$input" \
        --pool-name "$name" \
        --out-dir "$OUT" \
        --rounds "$ROUNDS" \
        --samples "$SAMPLES" \
        --seed "$SEED" \
        --force
}

# bunsetsu
run_pool "$REPO/datasets/corpus/bunsetsu/wikibooks.jsonl"          bunsetsu-wikibooks
run_pool "$REPO/datasets/corpus/bunsetsu/wiktionary.jsonl"         bunsetsu-wiktionary
run_pool "$REPO/datasets/corpus/bunsetsu/wikinews.jsonl"           bunsetsu-wikinews
run_pool "$REPO/datasets/corpus/bunsetsu/aozora_dialogue.jsonl"    bunsetsu-aozora-dialogue
run_pool "$REPO/datasets/corpus/bunsetsu/tatoeba.jsonl"            bunsetsu-tatoeba

# sentence
run_pool "$REPO/datasets/corpus/sentence/wikibooks.jsonl"          sentence-wikibooks
run_pool "$REPO/datasets/corpus/sentence/wikibooks.clean.jsonl"    sentence-wikibooks-clean
run_pool "$REPO/datasets/corpus/sentence/wiktionary.jsonl"         sentence-wiktionary
run_pool "$REPO/datasets/corpus/sentence/wiktionary.clean.jsonl"   sentence-wiktionary-clean
run_pool "$REPO/datasets/corpus/sentence/wikinews.jsonl"           sentence-wikinews
run_pool "$REPO/datasets/corpus/sentence/wikinews.clean.jsonl"     sentence-wikinews-clean
run_pool "$REPO/datasets/corpus/sentence/aozora_dialogue.jsonl"    sentence-aozora-dialogue
run_pool "$REPO/datasets/corpus/sentence/tatoeba.jsonl"            sentence-tatoeba
run_pool "$REPO/datasets/corpus/sentence/whitepaper.jsonl"         sentence-whitepaper

# synth
run_pool "$REPO/datasets/corpus/synth/homophone.jsonl"             synth-homophone
run_pool "$REPO/datasets/corpus/synth/name.jsonl"                  synth-name
run_pool "$REPO/datasets/corpus/synth/numeric.jsonl"               synth-numeric
run_pool "$REPO/datasets/corpus/synth/numeric_ext.jsonl"           synth-numeric-ext
run_pool "$REPO/datasets/corpus/synth/numeric_units.jsonl"         synth-numeric-units

# short
run_pool "$REPO/datasets/corpus/short/combined.jsonl"              short-combined

# legacy
run_pool "$REPO/datasets/corpus/legacy/aozora.jsonl"               legacy-aozora
run_pool "$REPO/datasets/corpus/legacy/chunks_100m.jsonl"          legacy-chunks-100m
run_pool "$REPO/datasets/corpus/legacy/fineweb2_ja.jsonl"          legacy-fineweb2-ja
run_pool "$REPO/datasets/corpus/legacy/hplt3_ja.jsonl"             legacy-hplt3-ja
run_pool "$REPO/datasets/corpus/legacy/livedoor.jsonl"             legacy-livedoor
run_pool "$REPO/datasets/corpus/legacy/livedoor_sentences.jsonl"   legacy-livedoor-sentences
run_pool "$REPO/datasets/corpus/legacy/tatoeba.jsonl"              legacy-tatoeba
run_pool "$REPO/datasets/corpus/legacy/tatoeba_sentences.jsonl"    legacy-tatoeba-sentences
run_pool "$REPO/datasets/corpus/legacy/wiki.jsonl"                 legacy-wiki
run_pool "$REPO/datasets/corpus/legacy/zenz_llmjp.jsonl"           legacy-zenz-llmjp

echo "[done] samples under $OUT"
