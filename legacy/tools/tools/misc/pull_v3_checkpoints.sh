#!/usr/bin/env bash
# Pull v3 remote checkpoints at 10k-step cadence.
#
# Policy:
#   - On every poll (INTERVAL sec): scp best.pt + final.pt (lightweight, keep fresh)
#   - When a new checkpoint_step_N.pt where N % 10000 == 0 exists remotely
#     and not locally, scp it (+ its _tokenizer.json sidecar)
#   - Local retention after each pull:
#       * Keep all checkpoint_step_N.pt where N % 50000 == 0  (milestones)
#       * Among non-milestone numbered ckpts, keep only the 3 most recent
#       * Never delete best.pt / final.pt / *_tokenizer.json
#
# No auto-restart of training. No alerting.

set -u

HOST="${PULL_HOST:-root@89.221.67.149}"
PORT="${PULL_PORT:-36694}"
REMOTE_DIR="${PULL_REMOTE_DIR:-/workspace/new-ime/models/checkpoints/ctc-nat-30m-bunsetsu-v3}"
LOCAL_DIR="${PULL_LOCAL_DIR:-D:/Dev/new-ime/models/checkpoints/ctc-nat-30m-bunsetsu-v3}"
INTERVAL="${PULL_INTERVAL:-300}"   # sec (5 min)
PULL_MULTIPLE="${PULL_MULTIPLE:-10000}"
KEEP_MULTIPLE="${PULL_KEEP_MULTIPLE:-50000}"
KEEP_LAST_N="${PULL_KEEP_LAST_N:-10}"

mkdir -p "$LOCAL_DIR"

ssh_cmd() { ssh -p "$PORT" -o ConnectTimeout=15 -o ServerAliveInterval=60 "$HOST" "$@"; }
scp_one() { scp -q -P "$PORT" "$HOST:$1" "$2" 2>/dev/null; }

remote_steps() {
    # Echo space-separated step numbers of checkpoint_step_N.pt on remote.
    ssh_cmd "ls -1 ${REMOTE_DIR} 2>/dev/null | sed -n 's/^checkpoint_step_\\([0-9]\\+\\)\\.pt$/\\1/p'" 2>/dev/null \
        | tr '\n' ' '
}

local_steps() {
    ls -1 "$LOCAL_DIR" 2>/dev/null | sed -n 's/^checkpoint_step_\([0-9]\+\)\.pt$/\1/p' | tr '\n' ' '
}

iter() {
    local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Always refresh best.pt / final.pt (cheap, may change every eval).
    for base in best final; do
        scp_one "${REMOTE_DIR}/${base}.pt" "$LOCAL_DIR/"
        scp_one "${REMOTE_DIR}/${base}_tokenizer.json" "$LOCAL_DIR/"
    done

    # Numbered checkpoints: pull any N%PULL_MULTIPLE==0 new on remote.
    local r_steps l_steps pulled=()
    r_steps=$(remote_steps)
    l_steps=" $(local_steps)"
    for n in $r_steps; do
        if [ $((n % PULL_MULTIPLE)) -ne 0 ]; then continue; fi
        if echo "$l_steps" | grep -q " $n "; then continue; fi
        scp_one "${REMOTE_DIR}/checkpoint_step_${n}.pt" "$LOCAL_DIR/"
        scp_one "${REMOTE_DIR}/checkpoint_step_${n}_tokenizer.json" "$LOCAL_DIR/"
        pulled+=("$n")
    done

    # Retention: always keep N%KEEP_MULTIPLE==0; otherwise keep last KEEP_LAST_N by step.
    local all new_local non_milestone deletable
    all=$(ls -1 "$LOCAL_DIR" 2>/dev/null | sed -n 's/^checkpoint_step_\([0-9]\+\)\.pt$/\1/p' | sort -n)
    non_milestone=$(for n in $all; do
        if [ $((n % KEEP_MULTIPLE)) -ne 0 ]; then echo "$n"; fi
    done)
    # Among non-milestone, delete all but last KEEP_LAST_N
    local count; count=$(echo -n "$non_milestone" | grep -c '^' || true)
    if [ "${count:-0}" -gt "$KEEP_LAST_N" ]; then
        deletable=$(echo "$non_milestone" | head -n -"$KEEP_LAST_N")
        for n in $deletable; do
            rm -f "$LOCAL_DIR/checkpoint_step_${n}.pt" "$LOCAL_DIR/checkpoint_step_${n}_tokenizer.json"
        done
        echo "[$ts] retention: pruned $(echo "$deletable" | wc -l) non-milestone ckpt(s): $(echo "$deletable" | tr '\n' ' ')"
    fi

    if [ "${#pulled[@]}" -gt 0 ]; then
        echo "[$ts] pulled step(s): ${pulled[*]}"
    else
        echo "[$ts] no new 10k-boundary ckpt (best/final refreshed if present)"
    fi
}

echo "pull loop: host=$HOST remote=$REMOTE_DIR local=$LOCAL_DIR interval=${INTERVAL}s"
echo "  pull_multiple=${PULL_MULTIPLE}  keep_multiple=${KEEP_MULTIPLE}  keep_last_n=${KEEP_LAST_N}"
while true; do
    iter || echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iter failed"
    sleep "$INTERVAL"
done
