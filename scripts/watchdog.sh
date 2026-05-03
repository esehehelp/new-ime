#!/usr/bin/env bash
# Watchdog for Suiko-v1.1-small training.
#
# Detects CRASH / STALL / SLOW on the train.log, auto-relaunches on
# CRASH and STALL (up to MAX_RESTARTS), warns on SLOW. Emits one
# stdout line per trigger — plugged into Claude's Monitor tool, each
# line becomes a notification.
#
# Not committed (lives under target/ which is gitignored).

set -u

RUN_DIR="D:/Dev/new-ime/models/checkpoints/Suiko-v1.1-small"
LOG="$RUN_DIR/train.log"
RESTART_STATE="$RUN_DIR/.watchdog_restarts"

STALL_SECONDS=300         # 5 min with no new log line -> STALL
SLOW_RATE=0.3             # < 0.3 steps/s -> SLOW (3-sample moving window)
SLOW_WINDOW=3
HEARTBEAT_SECONDS=600     # heartbeat summary every 10 min
POLL_SECONDS=30
MAX_RESTARTS=3
MAX_STEPS=200000

# Training command — must match the original launch so restarts use the
# same hyperparameters. Kept in one place so watchdog + user agree.
relaunch_training() {
    local reason="$1"
    local restarts
    restarts=$(cat "$RESTART_STATE" 2>/dev/null || echo 0)
    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        echo "[watchdog] FATAL ${reason}: already restarted ${restarts}/${MAX_RESTARTS}, giving up"
        return 1
    fi
    restarts=$((restarts + 1))
    echo "$restarts" > "$RESTART_STATE"
    echo "[watchdog] RESTART #${restarts}/${MAX_RESTARTS} reason=${reason}"

    # Kill any existing python processes first (orphans from prior run).
    # Use powershell so we don't depend on pid file bookkeeping.
    powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | ForEach-Object { try { \$_.Kill(); Write-Host \"killed PID \$(\$_.Id)\" } catch {} }" >/dev/null 2>&1 || true

    # Append a restart marker; use shell-level append so it flushes immediately.
    {
        echo ""
        echo "[watchdog-relaunch #${restarts}] $(date) reason=${reason}"
    } >> "$LOG"

    # Launch in background. Detached via disown so this script survives.
    (
        cd D:/Dev/new-ime/legacy/python || exit 1
        export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
        uv run python -u -m models.src.training.train_ctc_nat \
            --train D:/Dev/new-ime/datasets/mixes/student-300m-short.kkc \
            --dev   D:/Dev/new-ime/datasets/eval/general/dev.jsonl \
            --output "$RUN_DIR" \
            --preset phase3_30m \
            --tokenizer-path D:/Dev/new-ime/datasets/tokenizers/char-5k.json \
            --batch-size 64 --eval-batch-size 64 --grad-accum 2 \
            --max-steps $MAX_STEPS --max-seq-len 128 \
            --lr 3e-4 --warmup-steps 1000 --weight-decay 0.01 --grad-clip 1.0 \
            --max-train-samples 0 --max-dev-samples 2000 \
            --max-context 32 \
            --warmup-short-sample-steps 1000 --warmup-short-sample-max-chars 24 \
            --fp16 \
            --num-workers 0 \
            --log-every 100 --eval-every 5000 --checkpoint-every 10000 --keep-last-k 3 \
            --seed 52 \
            --resume "$RUN_DIR" \
            >>"$LOG" 2>&1
    ) &
    disown
    echo "[watchdog] relaunched python PID=$! reason=${reason}"
    return 0
}

# Returns 0 if a python process seems to be the trainer.
is_trainer_alive() {
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_ctc_nat*' }).Count" 2>/dev/null | tr -d '\r' | head -c 8
}

get_last_rate() {
    # Pull the last 3 rate= fields from [step ...] lines.
    grep -oE 'rate=[0-9]+\.[0-9]+' "$LOG" 2>/dev/null | tail -n $SLOW_WINDOW | awk -F= '{print $2}'
}

get_last_step() {
    grep -oE '\[step [0-9]+\]' "$LOG" 2>/dev/null | tail -n 1 | tr -dc '0-9'
}

log_age_seconds() {
    # mtime delta in seconds from now.
    if [ ! -f "$LOG" ]; then echo 99999; return; fi
    local mtime now
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    echo $((now - mtime))
}

slow_count=0
last_heartbeat=0
echo "[watchdog] started poll=${POLL_SECONDS}s stall=${STALL_SECONDS}s slow<${SLOW_RATE} window=${SLOW_WINDOW}"

while true; do
    sleep $POLL_SECONDS
    now=$(date +%s)
    age=$(log_age_seconds)
    alive_count=$(is_trainer_alive || echo 0)
    alive_count=${alive_count:-0}
    step=$(get_last_step)
    step=${step:-0}

    # CRASH: no trainer running but we are nowhere near MAX_STEPS.
    if [ "$alive_count" = "0" ] && [ "$step" -lt "$MAX_STEPS" ]; then
        echo "[watchdog] CRASH detected step=${step}/${MAX_STEPS} no python proc"
        tail -5 "$LOG" 2>/dev/null | sed 's/^/[watchdog] tail: /'
        if ! relaunch_training "CRASH"; then exit 1; fi
        slow_count=0
        last_heartbeat=$now
        continue
    fi

    # STALL: python up but log stale.
    if [ "$age" -ge "$STALL_SECONDS" ]; then
        echo "[watchdog] STALL detected age=${age}s step=${step} alive=${alive_count}"
        tail -5 "$LOG" 2>/dev/null | sed 's/^/[watchdog] tail: /'
        if ! relaunch_training "STALL"; then exit 1; fi
        slow_count=0
        last_heartbeat=$now
        continue
    fi

    # SLOW: moving window of rate readings, no auto-action, emit alert only.
    rates=$(get_last_rate)
    if [ -n "$rates" ]; then
        window=$(echo "$rates" | wc -l | tr -d ' ')
        if [ "$window" -ge "$SLOW_WINDOW" ]; then
            avg=$(echo "$rates" | awk '{s+=$1; n++} END{ if(n>0) printf "%.3f", s/n; else print "0" }')
            below=$(awk -v avg="$avg" -v th="$SLOW_RATE" 'BEGIN{ print (avg < th) ? 1 : 0 }')
            if [ "$below" = "1" ]; then
                slow_count=$((slow_count + 1))
                if [ $((slow_count % 4)) = "1" ]; then
                    echo "[watchdog] SLOW step=${step} avg_rate=${avg} steps/s (threshold ${SLOW_RATE})"
                fi
            else
                slow_count=0
            fi
        fi
    fi

    # Heartbeat.
    if [ $((now - last_heartbeat)) -ge $HEARTBEAT_SECONDS ]; then
        last_rate=$(echo "$rates" | tail -n 1)
        echo "[watchdog] OK step=${step} rate=${last_rate:-n/a} age=${age}s alive=${alive_count}"
        last_heartbeat=$now
    fi
done
