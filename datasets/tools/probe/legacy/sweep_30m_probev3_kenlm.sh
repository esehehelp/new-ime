#!/usr/bin/env bash
set -e
cd /mnt/d/Dev/new-ime
LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT="results/probe_v3_30m_kenlm_sweep"
mkdir -p "$OUT"

for MODEL in ctc_nat_30m-best ctc_nat_30m-step16000; do
    for A in 0.2 0.4 0.6; do
        for B in 0.3 0.6; do
            echo "=== $MODEL alpha=$A beta=$B ==="
            python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
                --lm-path "$LM" --alpha "$A" --beta "$B" \
                --out-dir "$OUT/${MODEL}_a${A}_b${B}" 2>&1 | grep -E "EM1=" | head -1
        done
    done
done
echo DONE
