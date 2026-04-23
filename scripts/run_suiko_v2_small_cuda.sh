#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
default_config="$repo_root/configs/suiko_v2_small__suiko_corpus_v2_300m.toml"
default_run_dir="D:/Dev/new-ime/models/checkpoints/suiko-v2-small__suiko-corpus-v2-300m"
default_steps="200000"
default_checkpoint_every="2000"
default_device="cuda"
bin_path="$repo_root/build/release/kkc-train.exe"
feature_stamp="$repo_root/build/release/.kkc-train-features"

export PATH="$repo_root/.venv/Scripts:$repo_root/.venv/Lib/site-packages/torch/lib:$PATH"
export LIBTORCH="$repo_root/.venv/Lib/site-packages/torch"
export LIBTORCH_USE_PYTORCH=1
export PYTHON="$repo_root/.venv/Scripts/python.exe"

cd "$repo_root"

needs_cuda_build=0
if [[ ! -x "$bin_path" ]]; then
  needs_cuda_build=1
elif [[ ! -f "$feature_stamp" ]] || ! grep -qx 'cuda' "$feature_stamp"; then
  needs_cuda_build=1
fi

if [[ $needs_cuda_build -eq 1 ]]; then
  echo "[run_suiko_v2_small_cuda] building kkc-train release with --features cuda" >&2
  cargo build -p kkc-train --release --features cuda
  mkdir -p "$(dirname "$feature_stamp")"
  printf 'cuda\n' > "$feature_stamp"
fi

inject_default_flag() {
  local flag="$1"
  local value="$2"
  shift 2
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "$flag" ]]; then
      printf '%s\0' "${args[@]}"
      return
    fi
  done
  args+=("$flag" "$value")
  printf '%s\0' "${args[@]}"
}

if [[ $# -eq 0 ]]; then
  set -- train
fi

case "$1" in
  plan|init-run|train|eval|peek-batches|scan-epoch|dry-train)
    subcmd="$1"
    shift
    args=("$@")
    mapfile -d '' -t args < <(inject_default_flag --config "$default_config" "${args[@]}")
    if [[ "$subcmd" == "train" ]]; then
      mapfile -d '' -t args < <(inject_default_flag --run-dir "$default_run_dir" "${args[@]}")
      mapfile -d '' -t args < <(inject_default_flag --device "$default_device" "${args[@]}")
      mapfile -d '' -t args < <(inject_default_flag --steps "$default_steps" "${args[@]}")
      mapfile -d '' -t args < <(inject_default_flag --checkpoint-every "$default_checkpoint_every" "${args[@]}")
    fi
    set -- "$subcmd" "${args[@]}"
    ;;
  *)
    args=("train" "$@")
    mapfile -d '' -t args < <(inject_default_flag --config "$default_config" "${args[@]}")
    mapfile -d '' -t args < <(inject_default_flag --run-dir "$default_run_dir" "${args[@]}")
    mapfile -d '' -t args < <(inject_default_flag --device "$default_device" "${args[@]}")
    mapfile -d '' -t args < <(inject_default_flag --steps "$default_steps" "${args[@]}")
    mapfile -d '' -t args < <(inject_default_flag --checkpoint-every "$default_checkpoint_every" "${args[@]}")
    set -- "${args[@]}"
    ;;
esac

printf '[run_suiko_v2_small_cuda] exec: %q' "$bin_path" >&2
for arg in "$@"; do
  printf ' %q' "$arg" >&2
done
printf '\n' >&2

if command -v stdbuf >/dev/null 2>&1; then
  exec stdbuf -oL -eL "$bin_path" "$@" 2>&1
else
  exec "$bin_path" "$@" 2>&1
fi
