#!/usr/bin/env bash
# Apply data-pool-filter (rules_v3) to every raw pool, writing cleaned
# JSONL + per-pool reject log + report TSV under datasets/corpus/cleaned/
# and datasets/audits/cleaned/.
#
# Fast path (Rust, rayon-parallel). Largest pools (short-combined 234M,
# fineweb2 117M, zenz 110M, chunks 100M) take a few minutes each.
set -eu

REPO="D:/Dev/new-ime"
BIN="$REPO/build/release/data-pool-filter.exe"
CLEAN_DIR="$REPO/datasets/corpus/cleaned"
AUDIT_DIR="$REPO/datasets/audits/cleaned"

run_one() {
    local src="$1"; shift
    local rel="$1"; shift   # relative path inside cleaned/
    local name="$1"; shift
    if [ ! -f "$src" ]; then
        echo "[skip] $name: $src missing" >&2
        return 0
    fi
    local out="$CLEAN_DIR/$rel"
    local rej="$AUDIT_DIR/${name}.rejects.jsonl"
    local rep="$AUDIT_DIR/${name}.report.tsv"
    mkdir -p "$(dirname "$out")" "$AUDIT_DIR"
    echo "[pool] $name  -> $rel"
    "$BIN" --input "$src" --output "$out" --rejects "$rej" --report "$rep"
}

# bunsetsu
run_one "$REPO/datasets/corpus/bunsetsu/wikibooks.jsonl"        bunsetsu/wikibooks.jsonl         bunsetsu-wikibooks
run_one "$REPO/datasets/corpus/bunsetsu/wiktionary.jsonl"       bunsetsu/wiktionary.jsonl        bunsetsu-wiktionary
run_one "$REPO/datasets/corpus/bunsetsu/wikinews.jsonl"         bunsetsu/wikinews.jsonl          bunsetsu-wikinews
run_one "$REPO/datasets/corpus/bunsetsu/aozora_dialogue.jsonl"  bunsetsu/aozora_dialogue.jsonl   bunsetsu-aozora-dialogue
run_one "$REPO/datasets/corpus/bunsetsu/tatoeba.jsonl"          bunsetsu/tatoeba.jsonl           bunsetsu-tatoeba

# sentence
run_one "$REPO/datasets/corpus/sentence/wikibooks.jsonl"        sentence/wikibooks.jsonl         sentence-wikibooks
run_one "$REPO/datasets/corpus/sentence/wikibooks.clean.jsonl"  sentence/wikibooks-clean.jsonl   sentence-wikibooks-clean
run_one "$REPO/datasets/corpus/sentence/wiktionary.jsonl"       sentence/wiktionary.jsonl        sentence-wiktionary
run_one "$REPO/datasets/corpus/sentence/wiktionary.clean.jsonl" sentence/wiktionary-clean.jsonl  sentence-wiktionary-clean
run_one "$REPO/datasets/corpus/sentence/wikinews.jsonl"         sentence/wikinews.jsonl          sentence-wikinews
run_one "$REPO/datasets/corpus/sentence/wikinews.clean.jsonl"   sentence/wikinews-clean.jsonl    sentence-wikinews-clean
run_one "$REPO/datasets/corpus/sentence/aozora_dialogue.jsonl"  sentence/aozora_dialogue.jsonl   sentence-aozora-dialogue
run_one "$REPO/datasets/corpus/sentence/tatoeba.jsonl"          sentence/tatoeba.jsonl           sentence-tatoeba
run_one "$REPO/datasets/corpus/sentence/whitepaper.jsonl"       sentence/whitepaper.jsonl        sentence-whitepaper

# synth
run_one "$REPO/datasets/corpus/synth/homophone.jsonl"           synth/homophone.jsonl            synth-homophone
run_one "$REPO/datasets/corpus/synth/name.jsonl"                synth/name.jsonl                 synth-name
run_one "$REPO/datasets/corpus/synth/numeric.jsonl"             synth/numeric.jsonl              synth-numeric
run_one "$REPO/datasets/corpus/synth/numeric_ext.jsonl"         synth/numeric_ext.jsonl          synth-numeric-ext
run_one "$REPO/datasets/corpus/synth/numeric_units.jsonl"       synth/numeric_units.jsonl        synth-numeric-units

# short
run_one "$REPO/datasets/corpus/short/combined.jsonl"            short/combined.jsonl             short-combined

# legacy
run_one "$REPO/datasets/corpus/legacy/aozora.jsonl"             legacy/aozora.jsonl              legacy-aozora
run_one "$REPO/datasets/corpus/legacy/chunks_100m.jsonl"        legacy/chunks_100m.jsonl         legacy-chunks-100m
run_one "$REPO/datasets/corpus/legacy/fineweb2_ja.jsonl"        legacy/fineweb2_ja.jsonl         legacy-fineweb2-ja
run_one "$REPO/datasets/corpus/legacy/hplt3_ja.jsonl"           legacy/hplt3_ja.jsonl            legacy-hplt3-ja
run_one "$REPO/datasets/corpus/legacy/livedoor.jsonl"           legacy/livedoor.jsonl            legacy-livedoor
run_one "$REPO/datasets/corpus/legacy/livedoor_sentences.jsonl" legacy/livedoor_sentences.jsonl  legacy-livedoor-sentences
run_one "$REPO/datasets/corpus/legacy/tatoeba.jsonl"            legacy/tatoeba.jsonl             legacy-tatoeba
run_one "$REPO/datasets/corpus/legacy/tatoeba_sentences.jsonl"  legacy/tatoeba_sentences.jsonl   legacy-tatoeba-sentences
run_one "$REPO/datasets/corpus/legacy/wiki.jsonl"               legacy/wiki.jsonl                legacy-wiki
run_one "$REPO/datasets/corpus/legacy/zenz_llmjp.jsonl"         legacy/zenz_llmjp.jsonl          legacy-zenz-llmjp

# J-E parallel (already filtered, but run through for consistency)
run_one "$REPO/datasets/corpus/parallel/tatoeba-ja-en.jsonl"    parallel/tatoeba-ja-en.jsonl     parallel-tatoeba-ja-en

echo "[done] cleaned pools under $CLEAN_DIR"
echo "[done] audit under $AUDIT_DIR"
