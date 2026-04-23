#!/usr/bin/env bash
# probe_v3 sweep for ctc-nat-30m-bunsetsu-v3 checkpoints.
# CPU only (WSL torch cpu), 8 configs per checkpoint:
#   greedy | beam5_nolm | (α=0.2,0.4,0.6) × (β=0.3,0.6) with KenLM.
set -e
cd /mnt/d/Dev/new-ime

LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT_ROOT="results/probe_v3_bunsetsu_kenlm_sweep"
mkdir -p "$OUT_ROOT"

CKPTS=(
    "ctc-nat-30m-bunsetsu-v3-best"
    "ctc-nat-30m-bunsetsu-v3-step60000"
    "ctc-nat-30m-bunsetsu-v3-step73000"
)
ALPHAS=(0.2 0.4 0.6)
BETAS=(0.3 0.6)

for M in "${CKPTS[@]}"; do
    STEP_DIR="$OUT_ROOT/$M"
    mkdir -p "$STEP_DIR"

    echo "=== $M : greedy ==="
    python3 -m datasets.tools.probe.run --models "$M" --beam 1 \
        --out-dir "$STEP_DIR/greedy" 2>&1 | tail -3

    echo "=== $M : beam5_nolm ==="
    python3 -m datasets.tools.probe.run --models "$M" --beam 5 \
        --out-dir "$STEP_DIR/beam5_nolm" 2>&1 | tail -3

    for A in "${ALPHAS[@]}"; do
        for B in "${BETAS[@]}"; do
            echo "=== $M : α=$A β=$B ==="
            python3 -m datasets.tools.probe.run --models "$M" --beam 5 \
                --lm-path "$LM" --alpha "$A" --beta "$B" \
                --out-dir "$STEP_DIR/a${A}_b${B}" 2>&1 | tail -3
        done
    done
done
echo DONE
