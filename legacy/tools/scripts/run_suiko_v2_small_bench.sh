#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin_path="$repo_root/build/release/kkc-bench.exe"

export PATH="$repo_root/.venv/Scripts:$repo_root/.venv/Lib/site-packages/torch/lib:$PATH"
export LIBTORCH="$repo_root/.venv/Lib/site-packages/torch"
export LIBTORCH_USE_PYTORCH=1
export PYTHON="$repo_root/.venv/Scripts/python.exe"

if [[ ! -x "$bin_path" ]]; then
  echo "[run_suiko_v2_small_bench] building release binary"
  (cd "$repo_root" && cargo build -p kkc-bench --release)
fi

if [[ $# -eq 0 ]]; then
  set -- \
    --config "$repo_root/configs/suiko_v2_small__suiko_corpus_v2_300m.toml" \
    --checkpoint "$repo_root/models/checkpoints/ctc-nat-30m-student/checkpoint_step_160000.pt" \
    --out-dir "$repo_root/results/eval_runs_rust_ctc30m_step160000" \
    --markdown "$repo_root/results/eval_runs_rust_ctc30m_step160000/benchmark_tables.md" \
    --model-name "ctc-nat-30m-student greedy rust-native" \
    --num-beams 1 \
    --num-return 1
fi

echo "[run_suiko_v2_small_bench] exec: $bin_path $*"
if command -v stdbuf >/dev/null 2>&1; then
  exec stdbuf -oL -eL "$bin_path" "$@" 2>&1
else
  exec "$bin_path" "$@" 2>&1
fi
