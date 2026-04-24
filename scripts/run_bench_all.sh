#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench_bin="$repo_root/build/release/kkc-bench.exe"
runner_py="$repo_root/models/tools/eval/run_bench_all_from_config.py"
config_path="$repo_root/configs/benchmark_models.json"

export PATH="$repo_root/.venv/Scripts:$repo_root/.venv/Lib/site-packages/torch/lib:$PATH"
export LIBTORCH="$repo_root/.venv/Lib/site-packages/torch"
export LIBTORCH_USE_PYTORCH=1
export PYTHON="$repo_root/.venv/Scripts/python.exe"

if [[ ! -x "$bench_bin" ]]; then
  echo "[run_bench_all] building release kkc-bench"
  (cd "$repo_root" && cargo build -p kkc-bench --release)
fi

if [[ $# -eq 0 ]]; then
  set -- --config "$config_path"
fi

cmd=("$PYTHON" "$runner_py" "$@")
echo "[run_bench_all] exec: ${cmd[*]}"
exec "${cmd[@]}"
