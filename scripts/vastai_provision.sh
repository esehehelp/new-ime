#!/usr/bin/env bash
# Runs on the vast.ai instance (vastai/pytorch image) after scp of
# transfer/ and this script. Idempotent.
#
# Assumes:
#   - image has working `python`, `pip`, `nvidia-smi`, `tmux`, `zstd`, `tar`, `git`
#   - GPU driver supports sm_120 (5090). For older GPUs, pin torch+cu wheel
#     via /opt/vastai_pytorch/ or reinstall explicitly (not handled here).
#   - this script is invoked from the repo root: `bash scripts/vastai_provision.sh`

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[provision] host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[provision] torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())' 2>&1 | head -1)"

# --- dependencies -------------------------------------------------------
# Image has torch pre-installed. Install the remaining deps directly.
# Avoid `uv sync` because it would pin torch to a specific cu wheel that
# may conflict with the image's build.
pip install --no-cache-dir --upgrade pip >/dev/null
pip install --no-cache-dir \
    "transformers>=4.40" \
    tokenizers \
    sentencepiece \
    mecab-python3 \
    unidic-lite \
    jaconv \
    accelerate \
    zstandard \
    numpy \
    protobuf \
    2>&1 | tail -5

# Optional: triton (for torch.compile). Skip silently if install fails —
# training will fall back to eager.
pip install --no-cache-dir triton 2>&1 | tail -2 || echo "[provision] triton install skipped"

# --- unpack data --------------------------------------------------------
TRANSFER="$REPO_ROOT/transfer"
DATA_DIR="$REPO_ROOT/datasets"
CKPT_DIR="$REPO_ROOT/models/checkpoints/ctc-nat-41m-maskctc-student-wp"
MIX_DIR="$DATA_DIR/mixes"
TOK_DIR="$DATA_DIR/tokenizers"
EVAL_DIR="$DATA_DIR/eval/general"

mkdir -p "$CKPT_DIR" "$MIX_DIR" "$TOK_DIR" "$EVAL_DIR"

# 1. assemble shard
if [ ! -f "$MIX_DIR/student-cleaned-500m.kkc" ]; then
    echo "[provision] assembling shard from $TRANSFER/_split/"
    (cd "$TRANSFER/_split" && sha256sum -c "$TRANSFER/MANIFEST.sha256") \
        || { echo "[provision] manifest check FAILED"; exit 1; }
    cat "$TRANSFER/_split/"chunk_* > "$TRANSFER/student-cleaned-500m.kkc.zst"
    zstd -d --long=27 --rm "$TRANSFER/student-cleaned-500m.kkc.zst" \
        -o "$MIX_DIR/student-cleaned-500m.kkc"
    echo "[provision] shard: $(du -h "$MIX_DIR/student-cleaned-500m.kkc" | cut -f1)"
fi

# 2. unpack aux bundle (teacher ckpt + tokenizer + dev + meta)
if [ ! -f "$CKPT_DIR/checkpoint_step_100000.pt" ]; then
    echo "[provision] unpacking aux.tar.zst"
    tmp=$(mktemp -d)
    tar -xf "$TRANSFER/aux.tar.zst" --use-compress-program="zstd -d" -C "$tmp"
    mv "$tmp/checkpoint_step_100000.pt"                  "$CKPT_DIR/"
    mv "$tmp/checkpoint_step_100000_tokenizer.json"      "$CKPT_DIR/" 2>/dev/null || true
    mv "$tmp/char-5k.json"                               "$TOK_DIR/"
    mv "$tmp/dev.jsonl"                                  "$EVAL_DIR/"
    mv "$tmp/student-cleaned-500m.kkc.meta.json"         "$MIX_DIR/"
    rm -rf "$tmp"
fi

# 3. clean up compressed chunks to free disk (optional; comment out to keep)
echo "[provision] keeping transfer/ for safety; manual 'rm -rf transfer/' to reclaim 17GB"

# --- smoke -------------------------------------------------------------
echo "[provision] import smoke test..."
python -c "
import torch
import MeCab  # noqa: F401
print('OK torch=', torch.__version__, 'cuda=', torch.cuda.is_available(), 'arch=', torch.cuda.get_arch_list())
"

echo "[provision] ready. launch with: tmux new -d -s train 'bash scripts/launch_suiko_v1_2_medium.sh 2>&1 | tee -a models/checkpoints/Suiko-v1.2-medium/train.log'"
