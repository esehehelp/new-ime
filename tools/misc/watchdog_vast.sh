#!/usr/bin/env bash
# v3 vast.ai training watchdog.
#
# Responsibilities:
#   1. Every INTERVAL seconds, ssh to the instance and check:
#        - tmux session alive
#        - latest training log line age (stall detection)
#        - latest step number (progress)
#        - grep for CUDA OOM / error markers
#   2. scp-pull the latest best.pt / checkpoint_step_*.pt + sidecars.
#   3. Append JSON status lines to $LOG so any dashboard can read them.
#
# No auto-restart: on anomaly it prints a warning and continues polling. The
# user decides whether to intervene (e.g. tmux attach + inspect, restart).

set -u

HOST="${WATCH_HOST:-root@89.221.67.149}"
PORT="${WATCH_PORT:-36694}"
REMOTE_DIR="${WATCH_REMOTE_DIR:-/workspace/new-ime/models/checkpoints/ctc-nat-30m-bunsetsu-v3}"
REMOTE_LOG="${WATCH_REMOTE_LOG:-/workspace/new-ime/logs/train_v3.log}"
REMOTE_TMUX="${WATCH_REMOTE_TMUX:-train}"
LOCAL_DIR="${WATCH_LOCAL_DIR:-D:/Dev/new-ime/models/checkpoints/ctc-nat-30m-bunsetsu-v3}"
LOG="${WATCH_LOG:-D:/Dev/new-ime/logs/watchdog_vast.jsonl}"
INTERVAL="${WATCH_INTERVAL:-300}"   # seconds (5 min)
STALL_LIMIT="${WATCH_STALL_LIMIT:-900}"  # seconds (15 min no log → warn)

mkdir -p "$(dirname "$LOG")" "$LOCAL_DIR"

ssh_cmd() {
    ssh -p "$PORT" -o ConnectTimeout=15 -o ServerAliveInterval=30 "$HOST" "$@"
}

scp_pull_dir() {
    # Pull newest best.pt + latest numbered checkpoint (+ sidecars) via scp.
    # Avoids rsync dependency on the Windows side.
    local remote_dir="$1"
    local local_dir="$2"
    # Find the latest step .pt via ssh; scp it + its tokenizer if present.
    local latest
    latest=$(ssh_cmd "ls -1 ${remote_dir} 2>/dev/null | grep -E '^checkpoint_step_[0-9]+\.pt$' | sed 's/[^0-9]*//' | sort -n | tail -1") || return 1
    if [ -n "$latest" ]; then
        local stem="checkpoint_step_${latest}"
        scp -q -P "$PORT" "$HOST:${remote_dir}/${stem}.pt" "$local_dir/" 2>/dev/null || true
        scp -q -P "$PORT" "$HOST:${remote_dir}/${stem}_tokenizer.json" "$local_dir/" 2>/dev/null || true
    fi
    # best.pt / final.pt (+ tokenizer) if present.
    for f in best.pt best_tokenizer.json final.pt final_tokenizer.json; do
        scp -q -P "$PORT" "$HOST:${remote_dir}/${f}" "$local_dir/" 2>/dev/null || true
    done
    echo "$latest"
}

iter() {
    local ts alive stall_sec last_line last_step err_lines latest_pulled
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    # Tmux liveness + log stat
    local probe
    probe=$(ssh_cmd "tmux has-session -t ${REMOTE_TMUX} 2>&1; echo '---SEP---'; stat -c '%Y' ${REMOTE_LOG} 2>/dev/null; echo '---SEP---'; tail -n 300 ${REMOTE_LOG} 2>/dev/null" 2>/dev/null) || probe=""
    if echo "$probe" | head -1 | grep -q "can't find session"; then
        alive="false"
    else
        alive="true"
    fi

    local log_mtime
    log_mtime=$(echo "$probe" | awk '/---SEP---/{f++;next} f==1{print;exit}')
    if [ -z "$log_mtime" ]; then
        stall_sec=-1
    else
        local now_epoch
        now_epoch=$(date +%s)
        stall_sec=$((now_epoch - log_mtime))
    fi

    local log_tail
    log_tail=$(echo "$probe" | awk 'BEGIN{f=0} /---SEP---/{f++;next} f==2{print}')
    last_line=$(echo "$log_tail" | tail -1)
    last_step=$(echo "$log_tail" | grep -Eo 'step=[0-9]+' | tail -1 | sed 's/step=//')
    [ -z "$last_step" ] && last_step=$(echo "$log_tail" | grep -Eo '\[eval [0-9]+\]' | tail -1 | grep -Eo '[0-9]+')
    err_lines=$(echo "$log_tail" | grep -iE 'traceback|cuda out of memory|runtimeerror|cuda error|nan|inf detected' | head -3)

    latest_pulled=$(scp_pull_dir "$REMOTE_DIR" "$LOCAL_DIR")

    # Emit JSON status line.
    printf '{"ts":"%s","tmux_alive":%s,"log_stall_sec":%s,"last_step":"%s","latest_pulled":"%s","errors":%s,"last_log":"%s"}\n' \
        "$ts" "$alive" "$stall_sec" "${last_step:-?}" "${latest_pulled:-}" \
        "$(printf '%s' "$err_lines" | jq -Rs . 2>/dev/null || echo '""')" \
        "$(printf '%s' "$last_line" | head -c 200 | sed 's/"/\\"/g')" \
        >> "$LOG"

    # Human-readable status on stdout.
    local warn=""
    if [ "$alive" != "true" ]; then warn="$warn [TMUX DEAD]"; fi
    if [ "$stall_sec" -gt "$STALL_LIMIT" ] 2>/dev/null; then warn="$warn [STALL ${stall_sec}s]"; fi
    if [ -n "$err_lines" ]; then warn="$warn [ERRORS]"; fi
    printf '[%s] tmux=%s stall=%ss step=%s pulled=%s%s\n' \
        "$ts" "$alive" "${stall_sec:-?}" "${last_step:-?}" "${latest_pulled:-?}" \
        "$warn"
    if [ -n "$err_lines" ]; then
        echo "$err_lines" | sed 's/^/    /'
    fi
}

echo "watchdog starting: host=$HOST port=$PORT interval=${INTERVAL}s log=$LOG"
while true; do
    iter || echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] iter failed"
    sleep "$INTERVAL"
done
