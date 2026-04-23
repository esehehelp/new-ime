#!/usr/bin/env bash
set -e
cd /mnt/d/Dev/new-ime
MODEL="ctc-nat-30m-student"
LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT="results/probe_v3_30mv2_kenlm_sweep"
mkdir -p "$OUT"

echo "=== baseline greedy ==="
python3 -m datasets.tools.probe.run --models "$MODEL" --beam 1 \
    --out-dir "$OUT/greedy" 2>&1 | tail -15 | head -10

echo "=== beam=5 no LM ==="
python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
    --out-dir "$OUT/beam5_nolm" 2>&1 | tail -15 | head -10

for A in 0.2 0.4 0.6; do
    for B in 0.3 0.6; do
        echo "=== alpha=$A beta=$B ==="
        python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
            --lm-path "$LM" --alpha "$A" --beta "$B" \
            --out-dir "$OUT/a${A}_b${B}" 2>&1 | tail -15 | head -10
    done
done
echo DONE
