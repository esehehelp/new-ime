#!/usr/bin/env bash
# Upload transfer/ + clone instructions to a vast.ai instance.
# Run locally (Windows git-bash OK). Set HOST / PORT / KEY / REMOTE below.
#
# Usage:
#   HOST=ssh.vast.ai PORT=12345 ./scripts/vastai_upload.sh
#
# Requires `scp` and an SSH key. Uses 8-way parallel scp for chunks (§§ 2.2
# of vastai-ops playbook) — single scp is throttled per-connection.

set -eu

HOST="${HOST:?Set HOST=<vast.ai ssh host>}"
PORT="${PORT:?Set PORT=<ssh port>}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
USER="${USER:-root}"
REMOTE="${REMOTE:-/workspace/new-ime}"
BRANCH="${BRANCH:-dev}"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i "$KEY" -p "$PORT")
SCP_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i "$KEY" -P "$PORT" -q)

echo "[upload] host=$USER@$HOST:$PORT  remote=$REMOTE"

# --- step 1: git clone on remote (small) -------------------------------
ssh "${SSH_OPTS[@]}" "$USER@$HOST" "
    mkdir -p $REMOTE
    cd $REMOTE
    if [ ! -d .git ]; then
        git clone -b $BRANCH https://github.com/esehehelp/new-ime.git .
    else
        git fetch origin $BRANCH && git checkout $BRANCH && git pull --ff-only
    fi
    mkdir -p transfer/_split datasets/mixes datasets/tokenizers datasets/eval/general models/checkpoints/ctc-nat-41m-maskctc-student-wp
" 2>&1 | tail -20

# --- step 2: parallel scp of chunks (15 x 1.2GB = 17GB) ----------------
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
TRANSFER_DIR="$REPO_LOCAL/transfer"

echo "[upload] pushing 15 chunks in 8-way parallel batches"
N_CHUNKS=$(ls "$TRANSFER_DIR/_split/" | wc -l)
BATCH=8
idx=0
for chunk in "$TRANSFER_DIR/_split/"chunk_*; do
    scp "${SCP_OPTS[@]}" "$chunk" "$USER@$HOST:$REMOTE/transfer/_split/" &
    idx=$((idx + 1))
    if (( idx % BATCH == 0 )); then
        wait
        echo "[upload] $idx / $N_CHUNKS chunks uploaded"
    fi
done
wait
echo "[upload] all chunks uploaded"

# --- step 3: aux bundle + manifest + scripts ---------------------------
scp "${SCP_OPTS[@]}" \
    "$TRANSFER_DIR/aux.tar.zst" \
    "$TRANSFER_DIR/MANIFEST.sha256" \
    "$TRANSFER_DIR/assemble.sh" \
    "$USER@$HOST:$REMOTE/transfer/"

# --- step 4: provision on remote ---------------------------------------
echo "[upload] running provision on remote"
ssh "${SSH_OPTS[@]}" "$USER@$HOST" "
    cd $REMOTE
    bash scripts/vastai_provision.sh 2>&1 | tail -40
"

echo "[upload] done. to launch:"
echo "  ssh ${SSH_OPTS[*]} $USER@$HOST \"cd $REMOTE && tmux new -d -s train 'bash scripts/launch_suiko_v1_2_medium.sh'\""
