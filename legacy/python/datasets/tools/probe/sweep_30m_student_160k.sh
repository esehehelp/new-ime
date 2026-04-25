#!/usr/bin/env bash
set -e
cd /mnt/d/Dev/new-ime

LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT="results/probe_v3_30m_student_120_140k/step160000"
MODEL="ctc-nat-30m-student-step160000"
mkdir -p "$OUT"

echo "=== $MODEL : greedy ==="
python3 -m datasets.tools.probe.run --models "$MODEL" --beam 1 \
    --out-dir "$OUT/greedy" 2>&1 | tail -3

echo "=== $MODEL : beam5_nolm ==="
python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
    --out-dir "$OUT/beam5_nolm" 2>&1 | tail -3

for A in 0.2 0.4 0.6; do
    for B in 0.3 0.6; do
        echo "=== $MODEL : a=$A b=$B ==="
        python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
            --lm-path "$LM" --alpha "$A" --beta "$B" \
            --out-dir "$OUT/a${A}_b${B}" 2>&1 | tail -3
    done
done
echo DONE
