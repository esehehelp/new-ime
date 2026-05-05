#!/usr/bin/env bash
# Pre-warm the DAT CUDA kernel JIT build (Linux / vast.ai).
#
# `torch.utils.cpp_extension.load(...)` compiles the kernel on first
# import, which adds 60-180 sec to the first training step. Running this
# script ahead of `ime-train` shifts that one-time cost out of the hot
# path so step rate logging starts immediately.
#
# Cache location: `$TORCH_EXTENSIONS_DIR` if set, otherwise
# `~/.cache/torch_extensions/<TORCH_VERSION>/new_ime_dat_kernel/`.
# Override `TORCH_EXTENSIONS_DIR` if `~/.cache/` is on a read-only
# filesystem (some HPC envs).
#
# Requirements:
#   * CUDA toolkit (nvcc) on $PATH
#   * `uv pip install -e ".[cuda]"` to pull in `ninja` (already pulled
#     into the project venv if you ran `uv sync --extra cuda`)
#
# Failure modes:
#   * nvcc missing  → kernel build fails; loader falls back to
#     pure-PyTorch DP at runtime (slow, but training still works).
#   * GPU arch not in `TORCH_CUDA_ARCH_LIST` → pass it explicitly:
#     export TORCH_CUDA_ARCH_LIST="12.0"   # for sm_120 (5090)
#
# Usage:
#   bash scripts/setup_dat_cuda_linux.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[setup_dat_cuda_linux] WARNING: nvcc not on PATH; build will fail."
    echo "[setup_dat_cuda_linux] Install CUDA toolkit, or accept the pure-PyTorch fallback."
fi

NEW_IME_DAT_VERBOSE=1 uv run python -c "
from new_ime.training.loss.dat_cuda import load_dat_kernel
import sys
kernel = load_dat_kernel()
if kernel is None:
    print('[setup_dat_cuda_linux] FAILED: kernel did not build (see log above).', file=sys.stderr)
    print('[setup_dat_cuda_linux] Training will fall back to pure-PyTorch DP.', file=sys.stderr)
    sys.exit(1)
print('[setup_dat_cuda_linux] kernel ready')
"
