#!/usr/bin/env bash
set -e
cd /mnt/d/Dev/new-ime
MODEL="ctc_nat_30m_v2-step49000"
LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT="results/probe_v2_30mv2_kenlm_sweep"
mkdir -p "$OUT"

echo "=== beam=5 no LM ==="
python3 -m tools.probe.run_probe_v2 --models "$MODEL" --beam 5 \
    --out-dir "$OUT/beam5_nolm" 2>&1 | grep -E "(EM1=|p50)" | tail -3

for A in 0.2 0.4 0.6; do
    for B in 0.0 0.3 0.6; do
        echo "=== alpha=$A beta=$B ==="
        python3 -m tools.probe.run_probe_v2 --models "$MODEL" --beam 5 \
            --lm-path "$LM" --alpha "$A" --beta "$B" \
            --out-dir "$OUT/a${A}_b${B}" 2>&1 | grep -E "(EM1=|p50)" | tail -3
    done
done
echo DONE
