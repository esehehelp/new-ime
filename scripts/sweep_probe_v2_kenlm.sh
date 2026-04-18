#!/usr/bin/env bash
# α/β sweep on probe_v2 via WSL (needs kenlm).
# Usage: wsl -- bash -c "bash /mnt/d/Dev/new-ime/scripts/sweep_probe_v2_kenlm.sh [model]"
set -e
cd /mnt/d/Dev/new-ime

MODEL="${1:-ctc_nat_90m-step27500}"
LM="models/kenlm_eval_v3_train_4gram_probing.bin"
OUT="results/probe_v2_kenlm_sweep_${MODEL}"
mkdir -p "$OUT"

echo "=== MODEL: $MODEL ==="
echo "=== baseline: greedy, no LM ==="
python3 -m scripts.run_probe_v2 \
    --models "$MODEL" --beam 1 \
    --out-dir "$OUT/baseline" 2>&1 | tail -3

echo "=== beam=5, no LM ==="
python3 -m scripts.run_probe_v2 \
    --models "$MODEL" --beam 5 \
    --out-dir "$OUT/beam5" 2>&1 | tail -3

for A in 0.2 0.4 0.6; do
    for B in 0.0 0.3 0.6; do
        tag="a${A}_b${B}"
        echo "=== beam=5, α=$A, β=$B ==="
        python3 -m scripts.run_probe_v2 \
            --models "$MODEL" \
            --beam 5 --lm-path "$LM" \
            --alpha "$A" --beta "$B" \
            --out-dir "$OUT/$tag" 2>&1 | tail -3
    done
done
echo "done — results in $OUT/"
