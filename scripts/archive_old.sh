#!/usr/bin/env bash
# Archive + compress old checkpoints, rust-train safetensors, reject logs,
# and _split chunks. Writes .zst in place; removes originals only after
# the corresponding .zst is present and non-empty.
#
# Keeps active files:
# - ctc-nat-41m-maskctc-student-wp/checkpoint_step_100000.pt (Suiko-v1.2 KD teacher)
# - datasets/corpus/cleaned/** (needed for future mix rebuilds)
# - datasets/mixes/student-cleaned-500m.jsonl (currently being shard-compiled)
set -eu

REPO="D:/Dev/new-ime"

compress_file() {
    local f="$1"
    [ -f "$f" ] || return 0
    [ -f "$f.zst" ] && { echo "[skip] $(basename $f) already .zst"; return 0; }
    echo "[zst] $(basename $(dirname $f))/$(basename $f) ($(du -h "$f" | cut -f1))"
    zstd -T4 -3 --rm "$f" -o "$f.zst" 2>/dev/null
}

# ---- old checkpoints (.pt) ------------------------------------------------
for dir in \
    "$REPO/models/checkpoints/ar-31m-scratch" \
    "$REPO/models/checkpoints/ctc-nat-30m-scratch" \
    "$REPO/models/checkpoints/ctc-nat-30m-student" \
    "$REPO/models/checkpoints/ctc-nat-90m-scratch" \
    "$REPO/models/checkpoints/teacher-150m-teacher" \
; do
    [ -d "$dir" ] || continue
    echo "=== $dir ==="
    for pt in "$dir"/*.pt; do
        [ -f "$pt" ] || continue
        compress_file "$pt"
    done
done

# Suiko-v1-small (ctc-nat-41m-maskctc-student-wp): keep step_100000 as KD teacher
CKPT="$REPO/models/checkpoints/ctc-nat-41m-maskctc-student-wp"
if [ -d "$CKPT" ]; then
    echo "=== $CKPT (keep step_100000) ==="
    for pt in "$CKPT"/*.pt; do
        [ -f "$pt" ] || continue
        case "$(basename $pt)" in
            checkpoint_step_100000.pt) echo "[keep] $(basename $pt) (v1.2 KD teacher)";;
            *) compress_file "$pt";;
        esac
    done
fi

# ---- rust-train safetensors (deprecated) ---------------------------------
SAFE="$REPO/models/checkpoints/suiko-v2-small__suiko-corpus-v2-300m"
if [ -d "$SAFE" ]; then
    echo "=== $SAFE → tar.zst ==="
    if [ ! -f "$SAFE.tar.zst" ]; then
        tar -I "zstd -T4 -3" -cf "$SAFE.tar.zst" -C "$(dirname "$SAFE")" "$(basename "$SAFE")" \
            && rm -rf "$SAFE" \
            && echo "[archived] $SAFE.tar.zst"
    else
        echo "[skip] $SAFE.tar.zst exists"
    fi
fi

# ---- mixes/_split chunks -------------------------------------------------
SPLIT="$REPO/datasets/mixes/_split"
if [ -d "$SPLIT" ] && [ -n "$(ls -A "$SPLIT" 2>/dev/null)" ]; then
    echo "=== $SPLIT → tar.zst ==="
    if [ ! -f "$SPLIT.tar.zst" ]; then
        tar -I "zstd -T4 -3" -cf "$SPLIT.tar.zst" -C "$(dirname "$SPLIT")" "$(basename "$SPLIT")" \
            && rm -rf "$SPLIT" \
            && echo "[archived] $SPLIT.tar.zst"
    fi
fi

# ---- audits/cleaned rejects ----------------------------------------------
AUDIT="$REPO/datasets/audits/cleaned"
if [ -d "$AUDIT" ]; then
    echo "=== $AUDIT rejects ==="
    for f in "$AUDIT"/*.rejects.jsonl; do
        [ -f "$f" ] || continue
        compress_file "$f"
    done
fi

# ---- old .pre-shardfix training log --------------------------------------
for log in "$REPO/models/checkpoints"/*/train.log.pre-shardfix; do
    [ -f "$log" ] || continue
    compress_file "$log"
done

echo "[done] see: df -h D:/"
df -h D:/ | tail -2
