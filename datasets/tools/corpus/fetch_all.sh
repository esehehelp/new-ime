#!/bin/bash
# One-shot fetcher for the v2 corpus expansion sources. All six sources
# are permissively licensed (Creative Commons, CC0, PD, or GPL for the
# Linux kernel translations) — see docs/corpus_candidates_v2.md for the
# license and rationale on each pick.
#
# Outputs land in datasets/raw_current/. Each source is downloaded once;
# re-running is a no-op.

set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
DST="$REPO/datasets/raw_current"
mkdir -p "$DST"

fetch() {
    local name=$1 url=$2 out=$3
    if [ ! -e "$out" ]; then
        echo "[fetch] $name -> $out"
        curl -sSL -o "$out" "$url"
    else
        echo "[skip]  $name (exists)"
    fi
}

# Wikimedia dumps — Wikibooks / Wikinews / Wiktionary ja.
fetch "Wikibooks ja"   "https://dumps.wikimedia.org/jawikibooks/latest/jawikibooks-latest-pages-articles.xml.bz2"     "$DST/jawikibooks-latest-pages-articles.xml.bz2"
fetch "Wikinews ja"    "https://dumps.wikimedia.org/jawikinews/latest/jawikinews-latest-pages-articles.xml.bz2"       "$DST/jawikinews-latest-pages-articles.xml.bz2"
fetch "Wiktionary ja"  "https://dumps.wikimedia.org/jawiktionary/latest/jawiktionary-latest-pages-articles.xml.bz2"   "$DST/jawiktionary-latest-pages-articles.xml.bz2"

# Tatoeba — full sentences CSV (id<TAB>lang<TAB>text).
mkdir -p "$DST/tatoeba"
fetch "Tatoeba full"   "https://downloads.tatoeba.org/exports/sentences.tar.bz2"                                       "$DST/tatoeba/sentences.tar.bz2"

# Intentionally skipped:
#   * OpenSubtitles — OPUS aggregation is CC-BY-SA but individual subtitles
#     remain copyrighted by translators / studios. Too risky for an IME
#     we might distribute.
#   * Linux kernel ja (GPL-2.0) — training-data use is the industry norm,
#     but if the model ever emits docs verbatim the output is GPL-bound.
#     Volume is tiny (~1 MB) so the ROI doesn't justify the policy risk.
# The aozora_dialogue extraction below covers the colloquial-register
# gap without the licensing ambiguity.

# Extract archives the downstream pipeline needs as plain files.
cd "$DST"
for f in jawikibooks jawikinews jawiktionary; do
    if [ ! -e "${f}-latest-pages-articles.xml" ]; then
        echo "[extract] ${f} XML"
        bunzip2 -k "${f}-latest-pages-articles.xml.bz2"
    fi
done
if [ ! -e "tatoeba/sentences.csv" ]; then
    echo "[extract] tatoeba CSV"
    tar -xjf tatoeba/sentences.tar.bz2 -C tatoeba/
fi
if [ ! -e "opensubtitles/ja.txt" ]; then
    echo "[extract] opensubtitles ja.txt"
    gunzip -k opensubtitles/ja.txt.gz
fi

# Aozora dialogue extraction (PD, no mecab pass needed). Re-runs are cheap
# because the source aozora_clean.jsonl is already on disk and the script
# is a single pass filter.
if [ -e "$REPO/datasets/aozora_clean.jsonl" ]; then
    echo "[extract] aozora dialogue"
    uv run python -m datasets.tools.corpus.extract_aozora_dialogue \
        --src "$REPO/datasets/aozora_clean.jsonl" \
        --out "$REPO/datasets/v2/aozora_dialogue.jsonl"
else
    echo "[warn] aozora_clean.jsonl not present — skipping dialogue extraction"
fi

echo
echo "== Sizes =="
ls -lh "$DST"/*.xml "$DST/tatoeba/sentences.csv" 2>/dev/null | awk '{print $5, $9}'
ls -lh "$REPO/datasets/v2"/*.jsonl 2>/dev/null | awk '{print $5, $9}'
