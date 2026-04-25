#!/bin/bash
# One-shot deploy of a Phase 3 training run to a vast.ai instance.
#
# Usage:
#   scripts/deploy_vastai.sh <host> <port> [--config configs/train_*.env] [--no-train]
#
# Example:
#   scripts/deploy_vastai.sh 219.122.229.5 46851 \
#       --config configs/train_phase3_90m_baseline.env
#
# Idempotent phases:
#   1. pre-flight     — SSH reachable, GPU arch, disk, Python, uv
#   2. compress       — ensure datasets/phase3/train.jsonl.zst exists (zstd -9)
#   3. remote setup   — clone repo, uv sync, torch (cu128 for Blackwell)
#   4. upload         — small files + 4 parallel scp of .zst chunks
#   5. remote finalize— cat chunks, zstd -d, cleanup
#   6. run script     — write scripts/run_<NAME>.sh on remote from env
#   7. launch         — tmux -d training session
#   8. next steps     — reminder to start scripts/mirror_checkpoints.sh
#
# This script is bash-heavy so it runs on Windows git-bash and Linux alike.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <host> <port> [--config FILE] [--no-train]" >&2
    exit 2
fi

HOST=$1
PORT=$2
shift 2

CONFIG=configs/train_phase3_90m_baseline.env
LAUNCH_TRAIN=1

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG=$2; shift 2 ;;
        --no-train) LAUNCH_TRAIN=0; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 2
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "deploy_vastai.sh starting"
log "  host=$HOST port=$PORT config=$CONFIG launch=$LAUNCH_TRAIN"

# Load config into env.
# shellcheck disable=SC1090
source "$CONFIG"
log "  loaded config: NAME=$NAME PRESET=$PRESET OUTPUT_SUBDIR=$OUTPUT_SUBDIR"

SSH_BASE="ssh -p $PORT -c chacha20-poly1305@openssh.com"
SCP_BASE="scp -P $PORT -c chacha20-poly1305@openssh.com"
REMOTE=root@$HOST

# =========================================================================
# Phase 1: pre-flight
# =========================================================================
log "Phase 1: pre-flight (SSH reach, GPU, disk)"
$SSH_BASE -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$REMOTE" bash <<'REMOTE_PRE'
set -e
echo "[remote] GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "[remote] cores=$(nproc) mem=$(free -h | awk '/Mem:/ {print $2}')"
echo "[remote] disk:"
df -h / 2>&1 | head -2
mkdir -p /workspace
echo "[remote] Python / uv:"
uv --version 2>&1 || { echo "ERROR: uv not found"; exit 1; }
echo "[remote] pre-flight OK"
REMOTE_PRE
log "Phase 1 done"

# Snapshot compute_cap so we can pick torch index.
log "Phase 1b: probing compute_cap"
COMPUTE_CAP=$($SSH_BASE "$REMOTE" 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader' | head -1 | tr -d ' ')
log "  compute_cap=$COMPUTE_CAP"

# =========================================================================
# Phase 2: local compress of train.jsonl (cache .zst)
# =========================================================================
ZST_PATH="${TRAIN}.zst"
if [ ! -f "$ZST_PATH" ] || [ "$TRAIN" -nt "$ZST_PATH" ]; then
    log "Phase 2: compress $TRAIN -> $ZST_PATH (zstd -9, multi-thread, ~5 min)"
    zstd -T0 -9 -f "$TRAIN" -o "$ZST_PATH"
    log "  done: $(stat -c %s "$ZST_PATH") bytes"
else
    log "Phase 2: reusing existing $ZST_PATH ($(stat -c %s "$ZST_PATH") bytes)"
fi
ZST_SIZE=$(stat -c %s "$ZST_PATH")
log "Phase 2 done"

# =========================================================================
# Phase 3: remote setup
# =========================================================================
log "Phase 3: remote setup (clone + uv sync + torch)"
$SSH_BASE "$REMOTE" bash <<REMOTE_SETUP
set -e
mkdir -p $REMOTE_HOME $(dirname $REMOTE_HOME)
if [ ! -d $REMOTE_HOME/.git ]; then
    git clone $REMOTE_REPO $REMOTE_HOME
fi
cd $REMOTE_HOME
git fetch --tags
git pull --ff-only || echo "(pull skipped; local changes?)"

mkdir -p datasets/phase3 datasets/eval/general datasets/tokenizers checkpoints/ar-31m-scratch logs

uv sync 2>&1 | tail -3

# Blackwell (compute_cap 12.x) needs cu128 torch.
case "$COMPUTE_CAP" in
    12.*)
        echo "=== installing cu128 torch for Blackwell ==="
        uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade \
            "torch>=2.7" "torchvision>=0.22" "torchaudio>=2.7" 2>&1 | tail -3
        ;;
    *)
        echo "=== keeping project-default torch (compute_cap=$COMPUTE_CAP) ==="
        ;;
esac

# Sanity
.venv/bin/python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'torch={torch.__version__} cuda={torch.version.cuda} device={p.name} vram={p.total_memory/1024**3:.1f}GB cc={p.major}.{p.minor}')
x = torch.randn(64,64,device='cuda',dtype=torch.float16)
print('fp16 matmul ok')
"
REMOTE_SETUP

# =========================================================================
# Phase 4: upload
# =========================================================================
log "Phase 3 done"

log "Phase 4a: small assets (eval sets, tokenizer, AR teacher) — ~30s"
$SCP_BASE \
    datasets/eval/general/dev.jsonl datasets/eval/general/test.jsonl \
    "$REMOTE:$REMOTE_HOME/datasets/eval/general/"
$SCP_BASE \
    datasets/tokenizers/char-5k.json \
    "$REMOTE:$REMOTE_HOME/datasets/tokenizers/"
$SCP_BASE \
    checkpoints/ar-31m-scratch/best.pt checkpoints/ar-31m-scratch/best_vocab.json \
    "$REMOTE:$REMOTE_HOME/checkpoints/ar-31m-scratch/"

log "Phase 4a done"

log "Phase 4b: train.jsonl.zst ($((ZST_SIZE / 1024 / 1024)) MB, 4-way parallel scp — ~1-3 min)"
SPLIT_DIR="${ZST_PATH}.split"
mkdir -p "$SPLIT_DIR"
rm -f "$SPLIT_DIR"/tp_* 2>/dev/null || true
split -n 4 -d "$ZST_PATH" "$SPLIT_DIR/tp_"

# Parallel upload.
for i in 00 01 02 03; do
    $SCP_BASE "$SPLIT_DIR/tp_$i" \
        "$REMOTE:$REMOTE_HOME/datasets/phase3/tp_$i" &
done
wait
log "Phase 4b: all chunks uploaded"

# =========================================================================
# Phase 5: remote finalize (cat + decompress + verify)
# =========================================================================
log "Phase 5: remote cat + zstd -d"
EXPECTED_BYTES=$(stat -c %s "$TRAIN")
$SSH_BASE "$REMOTE" bash <<REMOTE_FINALIZE
set -e
cd $REMOTE_HOME/datasets/phase3
cat tp_00 tp_01 tp_02 tp_03 > train.jsonl.zst
rm tp_0?
zstd -d -f -T0 train.jsonl.zst -o train.jsonl
ACTUAL=\$(stat -c %s train.jsonl)
if [ "\$ACTUAL" != "$EXPECTED_BYTES" ]; then
    echo "SIZE MISMATCH: expected $EXPECTED_BYTES got \$ACTUAL" >&2
    exit 3
fi
echo "train.jsonl: \$ACTUAL bytes OK"
wc -l train.jsonl
rm train.jsonl.zst
df -h / | tail -1
REMOTE_FINALIZE

rm -rf "$SPLIT_DIR"

# =========================================================================
# Phase 6: generate run script on remote
# =========================================================================
log "Phase 6: emitting remote run script"
FP16_FLAG=""
[ "${FP16:-}" = "true" ] && FP16_FLAG="--fp16"

cat <<EOF_RUN | $SSH_BASE "$REMOTE" "cat > $REMOTE_HOME/scripts/run_${NAME}.sh && chmod +x $REMOTE_HOME/scripts/run_${NAME}.sh"
#!/bin/bash
set -eo pipefail
cd $REMOTE_HOME
LOG="logs/train_\$(date +%Y%m%d_%H%M%S)_${NAME}.log"
echo "log: \$LOG"

.venv/bin/python3 -u -m models.src.training.train_ctc_nat \\
    --train "$TRAIN" \\
    --dev "$DEV" \\
    --preset "$PRESET" \\
    --tokenizer-path "$TOKENIZER" \\
    --batch-size $BATCH_SIZE --eval-batch-size $EVAL_BATCH_SIZE \\
    --grad-accum $GRAD_ACCUM \\
    $FP16_FLAG \\
    --max-context $MAX_CONTEXT \\
    --warmup-short-sample-steps $WARMUP_SHORT_SAMPLE_STEPS \\
    --warmup-short-sample-max-chars $WARMUP_SHORT_SAMPLE_MAX_CHARS \\
    --num-workers $NUM_WORKERS \\
    --lr $LR --warmup-steps $WARMUP_STEPS \\
    --weight-decay $WEIGHT_DECAY --grad-clip $GRAD_CLIP \\
    --checkpoint-every $CHECKPOINT_EVERY --eval-every $EVAL_EVERY --log-every $LOG_EVERY \\
    --max-train-samples $MAX_TRAIN_SAMPLES \\
    --max-dev-samples $MAX_DEV_SAMPLES \\
    --max-steps $MAX_STEPS --epochs $EPOCHS \\
    --kd-teacher-path "$KD_TEACHER_PATH" \\
    --kd-alpha $KD_ALPHA --kd-hard-threshold $KD_HARD_THRESHOLD \\
    --kd-gate-mode $KD_GATE_MODE \\
    --kd-start-step $KD_START_STEP --kd-warmup-steps $KD_WARMUP_STEPS \\
    --kd-every $KD_EVERY --kd-max-new-tokens $KD_MAX_NEW_TOKENS \\
    --output "checkpoints/$OUTPUT_SUBDIR" 2>&1 | tee "\$LOG"
EOF_RUN

# =========================================================================
# Phase 7: launch
# =========================================================================
if [ "$LAUNCH_TRAIN" = "1" ]; then
    log "Phase 7: tmux new -d -s train"
    $SSH_BASE "$REMOTE" bash <<REMOTE_LAUNCH
cd $REMOTE_HOME
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train 'bash scripts/run_${NAME}.sh'
sleep 2
tmux ls
REMOTE_LAUNCH
else
    log "Phase 7: --no-train, skipping launch"
fi

# =========================================================================
# Phase 8: next steps
# =========================================================================
cat <<NEXT

[$(date +%H:%M:%S)] deploy complete.

To monitor in real time (in a separate shell):
    ssh -p $PORT $REMOTE 'tmux attach -t train'

To mirror checkpoints to local every 5 minutes:
    bash scripts/mirror_checkpoints.sh $HOST $PORT $OUTPUT_SUBDIR

To check progress from this shell:
    ssh -p $PORT $REMOTE 'tail -n 30 \$(ls -t $REMOTE_HOME/logs/train_*.log | head -1)'

NEXT
