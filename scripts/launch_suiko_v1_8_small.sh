#!/usr/bin/env bash
# Suiko-v1.8-small. Single-variable ablation vs v1.5 baseline:
#   KD: on -> off  (drop --kd-* flags except --kd-alpha=0)
# Reuses v1.7 shard (chunks 0.10 was reverted to default; v1.7 has chunks
# 0.20 → not single-variable. To keep ablation clean we use v1_4 mix again
# but it was deleted. Pragmatic compromise: reuse v1.7 shard. Net change
# from v1.7 to v1.8 is then KD off only.
set -eu

if [ -d "D:/Dev/new-ime" ]; then
    REPO="D:/Dev/new-ime"
else
    REPO="$(cd "$(dirname "$0")/.." && pwd)"
fi
RUN_DIR="$REPO/models/checkpoints/Suiko-v1.8-small"

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"
: > "$LOG"
echo "[launch] $(date)  — Suiko-v1.8-small (KD off, baseline=v1.7 mix)" > "$LOG"

cd "$REPO/legacy/python"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1

if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
    RUNNER=(uv run python -u)
else
    RUNNER=(python -u)
fi

"${RUNNER[@]}" -m models.src.training.train_ctc_nat \
  --train  "$REPO/datasets/mixes/student-v1_7-500m.kkc" \
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
  --refine-loss-weight 1.0 --refine-warmup-steps 5000 --refine-mask-ratio 0.3 \
  --kd-alpha 0.0 \
  --seed 52 >>"$LOG" 2>&1
