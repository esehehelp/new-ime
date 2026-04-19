#!/bin/bash
# Train on sentence-level data (20M)
# Usage: bash docker/train_sentence.sh [--batch-size 128] [--max-samples 0]
set -e
cd /workspace/new-ime

# Pull latest code
git pull origin main 2>/dev/null || true

BATCH_SIZE=${1:-128}
MAX_SAMPLES=${2:-0}

echo "=== Sentence training: batch=$BATCH_SIZE max_samples=$MAX_SAMPLES ==="

uv run python -m src.training.train_ar \
  --train datasets/eval/general/train.jsonl \
  --dev datasets/eval/general/dev.jsonl \
  --output checkpoints/ar_sentence \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --grad-accum 1 \
  --lr 3e-4 \
  --max-seq-len 256 \
  --hidden-size 512 \
  --num-layers 8 \
  --num-heads 8 \
  --checkpoint-every 10000 \
  --eval-every 5000 \
  --log-every 500 \
  --warmup-steps 2000 \
  --num-workers 8 \
  --max-samples "$MAX_SAMPLES"

echo "=== Uploading checkpoints ==="
uv run hf upload esehe/new-ime-dataset checkpoints/ar_sentence/ checkpoints/ar_sentence/ --repo-type dataset

echo "=== Done ==="
