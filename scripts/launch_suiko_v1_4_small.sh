#!/usr/bin/env bash
# Suiko-v1.4-small. Single-variable ablation vs v1.2 baseline:
#   bunsetsu paths: 14 -> 5 (drop sentence/_v2 paths, like v1-small).
# All other settings identical to v1.2 (synth 0.10, KD start 12000, etc).
set -eu

if [ -d "D:/Dev/new-ime" ]; then
    REPO="D:/Dev/new-ime"
else
    REPO="$(cd "$(dirname "$0")/.." && pwd)"
fi
RUN_DIR="$REPO/models/checkpoints/Suiko-v1.4-small"
TEACHER="$REPO/models/checkpoints/ctc-nat-41m-maskctc-student-wp/checkpoint_step_100000.pt"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"
: > "$LOG"
echo "[launch] $(date)  — Suiko-v1.4-small (bunsetsu 5 paths)" > "$LOG"

cd "$REPO/legacy/python"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1

if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
    RUNNER=(uv run python -u)
else
    RUNNER=(python -u)
fi

"${RUNNER[@]}" -m models.src.training.train_ctc_nat \
  --train  "$REPO/datasets/mixes/student-v1_4-500m.kkc" \
  --dev    "$REPO/datasets/eval/general/dev.jsonl" \
  --output "$RUN_DIR" \
  --preset phase3_30m \
  --tokenizer-path "$REPO/datasets/tokenizers/char-5k.json" \
  --batch-size 64 --eval-batch-size 64 --grad-accum 2 \
  --max-steps 300000 --max-seq-len 128 \
  --lr 3e-4 --warmup-steps 4000 --weight-decay 0.01 --grad-clip 1.0 \
  --max-train-samples 0 --max-dev-samples 2000 \
  --max-context 32 \
  --warmup-short-sample-steps 2000 --warmup-short-sample-max-chars 24 \
  --fp16 --compile \
  --num-workers 0 \
  --log-every 500 --eval-every 1000 --checkpoint-every 10000 --keep-last-k 3 \
  --print-samples 5 \
  --use-cvae \
  --refine-loss-weight 1.0 --refine-warmup-steps 5000 --refine-mask-ratio 0.3 \
  --kd-teacher-type ctc \
  --kd-teacher-path "$TEACHER" \
  --kd-alpha 0.1 --kd-alpha-final 0.02 \
  --kd-start-step 12000 --kd-warmup-steps 6000 \
  --kd-alpha-decay-start 18000 --kd-alpha-decay-steps 20000 \
  --kd-every 16 --kd-gate-mode low_conf \
  --seed 52 >>"$LOG" 2>&1
