#!/usr/bin/env bash
set -e
cd /mnt/d/Dev/new-ime
LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT="results/probe_v3_step160k_kenlm"
mkdir -p "$OUT"

# greedy baseline
echo "=== step160k greedy ==="
python3 -m tools.probe.run_probe_v3 --models ctc_nat_30m_v2-step160000 --beam 1 \
    --out-dir "$OUT/greedy" 2>&1 | grep -E "EM1=" | head -1

for A in 0.2 0.4 0.6; do
    for B in 0.3 0.6; do
        echo "=== step160k alpha=$A beta=$B ==="
        python3 -m tools.probe.run_probe_v3 --models ctc_nat_30m_v2-step160000 --beam 5 \
            --lm-path "$LM" --alpha "$A" --beta "$B" \
            --out-dir "$OUT/a${A}_b${B}" 2>&1 | grep -E "EM1=" | head -1
    done
done
echo DONE
