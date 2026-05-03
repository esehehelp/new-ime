#!/usr/bin/env bash
# Sequential 10k-step Suiko-small improvement loop.
#
# Goal:
#   dev EM > 0.15 at step 10000
#
# This script runs short, comparable 10k experiments and stops as soon as a
# run reaches the target. It assumes the currently available shard is
# datasets/mixes/student-v1_7-500m.kkc.
set -eu

TARGET_EM="${TARGET_EM:-0.15}"

if [ -d "D:/Dev/new-ime" ]; then
    REPO="D:/Dev/new-ime"
else
    REPO="$(cd "$(dirname "$0")/.." && pwd)"
fi

TRAIN="$REPO/datasets/mixes/student-v1_7-500m.kkc"
DEV="$REPO/datasets/eval/general/dev.jsonl"
TOKENIZER="$REPO/datasets/tokenizers/char-5k.json"
TEACHER="$REPO/models/checkpoints/ctc-nat-41m-maskctc-student-wp/checkpoint_step_100000.pt"
ROOT="$REPO/models/checkpoints"

if [ ! -f "$TRAIN" ]; then
    echo "[fatal] missing train shard: $TRAIN" >&2
    exit 2
fi
if [ ! -f "$DEV" ]; then
    echo "[fatal] missing dev set: $DEV" >&2
    exit 2
fi

cd "$REPO/legacy/python"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1

if [ -x ".venv/Scripts/python.exe" ]; then
    RUNNER=(.venv/Scripts/python.exe -u)
elif [ -x ".venv/bin/python" ]; then
    RUNNER=(.venv/bin/python -u)
elif command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
    RUNNER=(uv run python -u)
elif command -v python >/dev/null 2>&1; then
    RUNNER=(python -u)
elif command -v py >/dev/null 2>&1; then
    RUNNER=(py -3 -u)
else
    echo "[fatal] neither uv, python, nor py is available on PATH" >&2
    exit 2
fi

last_em() {
    local log="$1"
    python - "$log" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
best = ""
pat = re.compile(r"^\[eval 10000\].*\bEM=([0-9.]+)")
if log.exists():
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            best = m.group(1)
print(best)
PY
}

reached() {
    local em="$1"
    python - "$em" "$TARGET_EM" <<'PY'
import sys
em = float(sys.argv[1] or 0.0)
target = float(sys.argv[2])
raise SystemExit(0 if em > target else 1)
PY
}

run_variant() {
    local name="$1"
    shift
    local out="$ROOT/$name"
    local log="$out/train.log"
    mkdir -p "$out"
    rm -f "$out/STOP"
    : > "$log"
    echo "[launch] $(date) -- $name target_em>$TARGET_EM@10k" | tee -a "$log"

    "${RUNNER[@]}" -m models.src.training.train_ctc_nat \
      --train "$TRAIN" \
      --dev "$DEV" \
      --output "$out" \
      --preset phase3_30m \
      --tokenizer-path "$TOKENIZER" \
      --batch-size 64 --eval-batch-size 64 --grad-accum 2 \
      --max-steps 10000 --max-seq-len 128 \
      --max-train-samples 0 --max-dev-samples 2000 \
      --max-context 32 \
      --fp16 --compile \
      --num-workers 0 \
      --log-every 500 --eval-every 1000 --checkpoint-every 10000 --keep-last-k 2 \
      --print-samples 5 \
      --seed 52 \
      "$@" >>"$log" 2>&1

    local em
    em="$(last_em "$log")"
    echo "[result] $name eval10000_em=${em:-missing}" | tee -a "$log"
    if reached "${em:-0}"; then
        echo "[success] $name reached target: EM=$em > $TARGET_EM"
        exit 0
    fi
}

# Baseline-derived short-horizon variants. All keep CVAE off because v1.5 was
# the strongest 10k run among existing logs.

run_variant "Suiko-v1.9-small-10k-fastlr" \
  --lr 5e-4 --warmup-steps 1000 --weight-decay 0.01 --grad-clip 1.0 \
  --warmup-short-sample-steps 1000 --warmup-short-sample-max-chars 24 \
  --refine-loss-weight 1.0 --refine-warmup-steps 2000 \
  --refine-mask-ratio-min 0.15 --refine-mask-ratio-max 0.35 \
  --kd-alpha 0.0

run_variant "Suiko-v1.10-small-10k-earlykd" \
  --lr 5e-4 --warmup-steps 1000 --weight-decay 0.01 --grad-clip 1.0 \
  --warmup-short-sample-steps 1000 --warmup-short-sample-max-chars 24 \
  --refine-loss-weight 1.0 --refine-warmup-steps 2000 \
  --refine-mask-ratio-min 0.15 --refine-mask-ratio-max 0.35 \
  --kd-teacher-type ctc \
  --kd-teacher-path "$TEACHER" \
  --kd-alpha 0.05 --kd-alpha-final 0.02 \
  --kd-start-step 1000 --kd-warmup-steps 2000 \
  --kd-alpha-decay-start 6000 --kd-alpha-decay-steps 4000 \
  --kd-every 4 --kd-gate-mode all

run_variant "Suiko-v1.11-small-10k-noshort" \
  --lr 5e-4 --warmup-steps 1000 --weight-decay 0.01 --grad-clip 1.0 \
  --refine-loss-weight 1.0 --refine-warmup-steps 2000 \
  --refine-mask-ratio-min 0.15 --refine-mask-ratio-max 0.35 \
  --kd-teacher-type ctc \
  --kd-teacher-path "$TEACHER" \
  --kd-alpha 0.05 --kd-alpha-final 0.02 \
  --kd-start-step 1000 --kd-warmup-steps 2000 \
  --kd-alpha-decay-start 6000 --kd-alpha-decay-steps 4000 \
  --kd-every 4 --kd-gate-mode all

echo "[fail] no variant reached target EM>$TARGET_EM @ 10k"
exit 1
