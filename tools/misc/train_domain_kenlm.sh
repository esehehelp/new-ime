#!/usr/bin/env bash
# Train domain-specific KenLM 4-gram models (tech, entity) to pair with the
# existing general KenLM for MoE shallow fusion.
#
# Inputs:
#   datasets/kenlm_corpora/tech.txt    — char-separated surface lines
#   datasets/kenlm_corpora/entity.txt  — char-separated surface lines
# Outputs:
#   models/kenlm/kenlm_tech_4gram.bin
#   models/kenlm/kenlm_entity_4gram.bin
#
# Requires WSL + /home/esehe/kenlm/build/bin/{lmplz,build_binary}.
set -e
cd /mnt/d/Dev/new-ime

LMPLZ=/home/esehe/kenlm/build/bin/lmplz
BUILD_BIN=/home/esehe/kenlm/build/bin/build_binary
mkdir -p models/kenlm

for DOMAIN in tech entity; do
    IN=datasets/kenlm_corpora/${DOMAIN}.txt
    ARPA=models/kenlm/kenlm_${DOMAIN}_4gram.arpa
    BIN=models/kenlm/kenlm_${DOMAIN}_4gram.bin

    if [ ! -f "$IN" ]; then
        echo "[warn] $IN not found; skipping"
        continue
    fi

    echo "=== training ${DOMAIN}-LM ==="
    wc -l "$IN"

    $LMPLZ -o 4 --prune 0 0 1 1 --text "$IN" --arpa "$ARPA"
    $BUILD_BIN probing "$ARPA" "$BIN"
    rm -f "$ARPA"
    ls -lh "$BIN"
done
echo DONE
