#!/bin/bash
# One-shot fetcher for the v2 corpus expansion sources. All six sources
# are permissively licensed (Creative Commons, CC0, PD, or GPL for the
# Linux kernel translations) — see docs/corpus_candidates_v2.md for the
# license and rationale on each pick.
#
# Outputs land in datasets/raw_v2/. Each source is downloaded once;
# re-running is a no-op.

set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
DST="$REPO/datasets/raw_v2"
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

# OpenSubtitles monolingual Japanese from the OPUS project.
mkdir -p "$DST/opensubtitles"
fetch "OpenSubtitles ja" "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ja.txt.gz"                         "$DST/opensubtitles/ja.txt.gz"

# Linux kernel Japanese translations via sparse checkout.
if [ ! -d "$DST/linux_kernel_ja/.git" ]; then
    echo "[fetch] Linux kernel ja (sparse)"
    mkdir -p "$DST/linux_kernel_ja"
    pushd "$DST/linux_kernel_ja" >/dev/null
    git init -q
    git remote add origin https://github.com/torvalds/linux.git
    git config core.sparseCheckout true
    echo "Documentation/translations/ja_JP/*" > .git/info/sparse-checkout
    git fetch --depth 1 origin master 2>&1 | tail -1
    git checkout master 2>&1 | tail -1
    popd >/dev/null
else
    echo "[skip]  Linux kernel ja (exists)"
fi

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

echo
echo "== Sizes =="
ls -lh "$DST"/*.xml "$DST/tatoeba/sentences.csv" "$DST/opensubtitles/ja.txt" 2>/dev/null | awk '{print $5, $9}'
