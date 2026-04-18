#!/bin/bash
# Training script for vast.ai instance
# Usage: Run on vast.ai after instance is up
#   bash vast_train.sh
set -e

echo "=== Setup ==="
pip install -q uv
cd /workspace

# Clone repo
git clone https://github.com/esehehelp/new-ime.git
cd new-ime

# Install deps
uv sync --group dev

# Download dataset from HuggingFace
echo "=== Downloading dataset ==="
mkdir -p datasets/eval_v2
uv run hf download esehe/new-ime-dataset train.jsonl --repo-type dataset --local-dir datasets/eval_v2
uv run hf download esehe/new-ime-dataset dev.jsonl --repo-type dataset --local-dir datasets/eval_v2
uv run hf download esehe/new-ime-dataset test.jsonl --repo-type dataset --local-dir datasets/eval_v2

echo "=== Dataset ready ==="
wc -l datasets/eval_v2/*.jsonl

# Train AR baseline (Phase 2) with full data
echo "=== Training AR baseline ==="
uv run python -m models.src.training.train_ar \
  --train datasets/eval_v2/train.jsonl \
  --dev datasets/eval_v2/dev.jsonl \
  --output checkpoints/ar_baseline_v2 \
  --epochs 1 \
  --batch-size 64 \
  --grad-accum 2 \
  --lr 3e-4 \
  --max-seq-len 256 \
  --hidden-size 512 \
  --num-layers 8 \
  --num-heads 8 \
  --checkpoint-every 10000 \
  --eval-every 5000 \
  --log-every 500 \
  --warmup-steps 2000 \
  --num-workers 8

echo "=== Training complete ==="

# Upload checkpoints to HuggingFace
echo "=== Uploading checkpoints ==="
uv run hf upload esehe/new-ime-dataset checkpoints/ar_baseline_v2/ checkpoints/ar_baseline_v2/ --repo-type dataset

echo "=== Done! Destroy this instance now ==="
