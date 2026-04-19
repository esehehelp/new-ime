#!/bin/bash
# Train on mixed sentence + chunk data
# Usage: bash docker/train_mixed.sh [--batch-size 128] [--chunk-ratio 0.5]
set -e
cd /workspace/new-ime

# Pull latest code
git pull origin main 2>/dev/null || true

BATCH_SIZE=${1:-128}
CHUNK_SAMPLES=${2:-5000000}

echo "=== Preparing mixed dataset ==="
# Take N samples from chunks, combine with full sentence data
head -"$CHUNK_SAMPLES" datasets/eval/general/chunks_v3_100m.jsonl > /tmp/chunks_sample.jsonl
cat datasets/eval/general/train.jsonl /tmp/chunks_sample.jsonl | shuf > datasets/mixes/train-mixed.jsonl
wc -l datasets/mixes/train-mixed.jsonl
rm /tmp/chunks_sample.jsonl

echo "=== Mixed training: batch=$BATCH_SIZE ==="

uv run python -m src.training.train_ar \
  --train datasets/mixes/train-mixed.jsonl \
  --dev datasets/eval/general/dev.jsonl \
  --output checkpoints/ar_mixed \
  --epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --grad-accum 1 \
  --lr 3e-4 \
  --max-samples 5000000 \
  --max-seq-len 256 \
  --hidden-size 512 \
  --num-layers 8 \
  --num-heads 8 \
  --checkpoint-every 5000 \
  --eval-every 2000 \
  --log-every 500 \
  --warmup-steps 1000 \
  --num-workers 8

echo "=== Uploading checkpoints ==="
uv run hf upload esehe/new-ime-dataset checkpoints/ar_mixed/ checkpoints/ar_mixed/ --repo-type dataset

echo "=== Done ==="
