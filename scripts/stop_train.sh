#!/usr/bin/env bash
# Kill switch for train_ctc_nat.py. Writes a `STOP` sentinel file in the
# run dir; the Python loop checks each microbatch and raises
# KeyboardInterrupt cleanly (saves `interrupted_step_<N>.pt`).
#
# More reliable than `kill -9 <winpid>` because cygwin `kill` does not
# always reach native-Windows `python.exe` spawned via uv.
#
# Usage:
#   bash scripts/stop_train.sh [run_dir]
#   bash scripts/stop_train.sh                  # stops every active run
set -eu

stop_one() {
    local rundir="$1"
    [ -d "$rundir" ] || return 0
    local stopf="$rundir/STOP"
    local pidf="$rundir/train.pid"
    # Skip dirs without a live training process.
    [ -f "$pidf" ] || return 0
    touch "$stopf"
    echo "[stop] requested STOP at $stopf"
    # Wait up to 60s for the python loop to react.
    for i in $(seq 1 12); do
        sleep 5
        if [ ! -f "$pidf" ]; then
            echo "[stop] $rundir: train.pid removed (clean exit)"
            return 0
        fi
    done
    echo "[stop] $rundir: WARN train.pid still present after 60s"
    echo "[stop] $rundir: STOP signal pending; if log not advancing, native"
    echo "[stop] $rundir: python.exe may be hung. Run from a Windows shell:"
    echo "[stop] $rundir:   taskkill /F /PID $(cat "$pidf")"
}

if [ $# -ge 1 ]; then
    stop_one "$1"
else
    found=0
    for d in /d/Dev/new-ime/models/checkpoints/*/; do
        [ -f "$d/train.pid" ] || continue
        stop_one "$d"
        found=1
    done
    [ "$found" = "0" ] && echo "[stop] no active train.pid files found"
fi
