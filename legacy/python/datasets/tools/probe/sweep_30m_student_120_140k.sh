#!/usr/bin/env bash
# 30m-v2 (ctc-nat-30m-student) step 120k-140k @5k × (No-KenLM, KenLM α/β sweep)
# CPU 実行 (WSL, torch CPU-only)。各 run が probe 全 348 items を流すため数分/run。
set -e
cd /mnt/d/Dev/new-ime

LM="models/kenlm/kenlm_general_train_4gram_probing.bin"
OUT_ROOT="results/probe_v3_30m_student_120_140k"
STEPS=(120000 125000 130000 135000 140000)
ALPHAS=(0.2 0.4 0.6)
BETAS=(0.3 0.6)

mkdir -p "$OUT_ROOT"

for S in "${STEPS[@]}"; do
    MODEL="ctc-nat-30m-student-step${S}"
    STEP_DIR="$OUT_ROOT/step${S}"
    mkdir -p "$STEP_DIR"

    echo "=== ${MODEL} : greedy (no LM, beam=1) ==="
    python3 -m datasets.tools.probe.run --models "$MODEL" --beam 1 \
        --out-dir "$STEP_DIR/greedy" 2>&1 | tail -20

    echo "=== ${MODEL} : beam5_nolm ==="
    python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
        --out-dir "$STEP_DIR/beam5_nolm" 2>&1 | tail -20

    for A in "${ALPHAS[@]}"; do
        for B in "${BETAS[@]}"; do
            echo "=== ${MODEL} : KenLM alpha=${A} beta=${B} beam=5 ==="
            python3 -m datasets.tools.probe.run --models "$MODEL" --beam 5 \
                --lm-path "$LM" --alpha "$A" --beta "$B" \
                --out-dir "$STEP_DIR/a${A}_b${B}" 2>&1 | tail -20
        done
    done
done

echo "DONE: $OUT_ROOT"
