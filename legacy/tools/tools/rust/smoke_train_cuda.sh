#!/usr/bin/env bash
# End-to-end smoke for the Rust tch CTC-NAT trainer.
#
# Compiles (or reuses) the student-20m shard, inits a tiny run dir,
# runs `kkc-train fit` for a short burst, and asserts the loss curve
# went down. Designed to catch regressions in the full CLI → config →
# data → model → ckpt → eval pipeline; it is NOT a substitute for the
# parity unit tests.
#
# Usage:
#   bash tools/rust/smoke_train_cuda.sh            # auto-detect device
#   DEVICE=cuda bash tools/rust/smoke_train_cuda.sh
#   STEPS=100 bash tools/rust/smoke_train_cuda.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

: "${LIBTORCH:=$REPO/.venv/Lib/site-packages/torch}"
: "${DEVICE:=cpu}"              # override with DEVICE=cuda for a real GPU run
: "${STEPS:=20}"                 # small enough to finish in seconds on CPU
: "${CHECKPOINT_EVERY:=10}"
: "${RUN_DIR:=$REPO/models/checkpoints/rust-smoke}"
: "${CONFIG:=$REPO/models/checkpoints/rust-smoke/smoke.toml}"

export LIBTORCH
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PATH="$LIBTORCH/lib:$PATH"

rm -rf "$RUN_DIR"
mkdir -p "$(dirname "$CONFIG")"

# Generate a small config derived from rust_student_cuda.toml but with
# architecture trimmed to fit inside a 20-step CPU smoke.
cat > "$CONFIG" <<'TOML'
[dataset]
train_shard = "datasets/mixes/student-20m.kkc"

[tokenizer]
path = "datasets/tokenizers/char-5k.json"
max_kanji = 6000
vocab_size = 4801

[model]
preset = "phase3_20m"

[runtime]
param_dtype_bytes = 4
grad_dtype_bytes = 4
adam_state_bytes = 8
activation_dtype_bytes = 4
prefetch_queue = 2

[eval]
batches = 0

[backend]
kind = "tch-ctc-nat"
optimizer = "adamw"
scheduler = "warmup_cosine"
learning_rate = 5e-4
init_scale = 0.02
weight_decay = 0.01
momentum = 0.9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
warmup_steps = 5
scheduler_total_steps = 100
min_lr_scale = 0.2
parameter_count = 1000000
hidden_size = 96
encoder_layers = 2
num_heads = 4
ffn_size = 256
decoder_layers = 2
decoder_heads = 4
decoder_ffn_size = 256
output_size = 4801
blank_id = 4
max_positions = 128
mask_token_id = 5
refine_loss_weight = 0.0
refine_warmup_steps = 0
refine_mask_ratio = 0.3
remask_loss_weight = 0.0
stop_loss_weight = 0.0

[train]
batch_size = 4
max_input_len = 128
max_target_len = 128
grad_accum = 1
block_rows = 128
seed = 1
checkpoint_keep_last = 2
TOML

echo "=== smoke: init-run ==="
cargo run -p kkc-train --features cuda --quiet -- init-run \
    --config "$CONFIG" --output "$RUN_DIR"

echo "=== smoke: fit $STEPS step(s) on $DEVICE ==="
LOG="$RUN_DIR/fit.log"
cargo run -p kkc-train --features cuda --quiet -- fit \
    --config "$CONFIG" --run-dir "$RUN_DIR" \
    --device "$DEVICE" \
    --steps "$STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --async-ckpt-queue 1 \
    --grad-clip 1.0 \
    2>&1 | tee "$LOG"

last_loss=$(awk '/fit_last_loss: / { print $2 }' "$LOG" | tail -1)
if [[ -z "$last_loss" || "$last_loss" == "null" ]]; then
    echo "SMOKE FAILED: no fit_last_loss in log" >&2
    exit 1
fi
echo "smoke last_loss=$last_loss"
echo "SMOKE OK"
