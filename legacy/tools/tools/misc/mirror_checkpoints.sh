#!/bin/bash
# Continuously pull checkpoint_step_*.pt + tokenizer JSON + logs from a
# running vast.ai training session into a local mirror directory, so a
# sudden instance death costs at most --interval seconds of progress.
#
# Usage:
#   scripts/mirror_checkpoints.sh <host> <port> <subdir> [--interval 300]
#
# Writes to ./checkpoints/vast_mirror/<subdir>/ and ./logs/vast_mirror/<subdir>/.
# --ignore-existing: only pulls newly-appearing files; does NOT re-fetch files
# that change in-place (best.pt is overwritten by the trainer, so we force
# that one via a separate rule).

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "usage: $0 <host> <port> <subdir> [--interval SEC]" >&2
    exit 2
fi

HOST=$1
PORT=$2
SUBDIR=$3
shift 3

INTERVAL=300
while [ $# -gt 0 ]; do
    case "$1" in
        --interval) INTERVAL=$2; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 2 ;;
    esac
done

REMOTE=root@$HOST
REMOTE_HOME=/workspace/new-ime
LOCAL_CK="./checkpoints/vast_mirror/$SUBDIR"
LOCAL_LOG="./logs/vast_mirror/$SUBDIR"

mkdir -p "$LOCAL_CK" "$LOCAL_LOG"

log() { echo "[$(date +%H:%M:%S)] $*"; }

SSH_OPTS="-p $PORT -c chacha20-poly1305@openssh.com -o LogLevel=QUIET -o ConnectTimeout=30"
# scp takes -P (capital) for port; other flags are shared.
SCP_OPTS="-P $PORT -c chacha20-poly1305@openssh.com -o LogLevel=QUIET -o ConnectTimeout=30"

# SCP-based fetch. rsync isn't always available on the Windows side; scp is
# everywhere ssh is.
fetch() {
    local remote_glob=$1
    local local_dir=$2
    # List matching files and download those that don't already exist locally.
    local files
    files=$(ssh $SSH_OPTS "$REMOTE" "ls -1 $remote_glob 2>/dev/null" || true)
    if [ -z "$files" ]; then
        return 0
    fi
    while IFS= read -r remote_path; do
        [ -z "$remote_path" ] && continue
        local base
        base=$(basename "$remote_path")
        # best.pt is overwritten in place — always refresh it.
        if [ "$base" = "best.pt" ] || [ "$base" = "best_tokenizer.json" ] || [ ! -e "$local_dir/$base" ]; then
            log "pull $base"
            scp $SCP_OPTS "$REMOTE:$remote_path" "$local_dir/$base" >/dev/null 2>&1 || \
                log "  (pull failed, will retry next cycle)"
        fi
    done <<< "$files"
}

log "mirror: host=$HOST port=$PORT subdir=$SUBDIR interval=${INTERVAL}s"
log "  local_checkpoints=$LOCAL_CK"
log "  local_logs=$LOCAL_LOG"

while true; do
    fetch "$REMOTE_HOME/checkpoints/$SUBDIR/checkpoint_step_*.pt" "$LOCAL_CK" || true
    fetch "$REMOTE_HOME/checkpoints/$SUBDIR/checkpoint_step_*_tokenizer.json" "$LOCAL_CK" || true
    fetch "$REMOTE_HOME/checkpoints/$SUBDIR/best.pt" "$LOCAL_CK" || true
    fetch "$REMOTE_HOME/checkpoints/$SUBDIR/best_tokenizer.json" "$LOCAL_CK" || true
    fetch "$REMOTE_HOME/logs/train_*.log" "$LOCAL_LOG" || true
    sleep "$INTERVAL"
done
