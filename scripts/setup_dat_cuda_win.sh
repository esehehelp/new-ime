#!/usr/bin/env bash
# Pre-warm the DAT CUDA kernel JIT build (Windows / MSVC).
#
# Same purpose as setup_dat_cuda_linux.sh: shift the 60-180 sec first-time
# build out of the hot training path.
#
# This script auto-loads MSVC env via vcvars64.bat (located via vswhere)
# by generating a temp .bat shim, so you don't need a "Developer Command
# Prompt".
#
# Requirements:
#   * Visual Studio 2022 Build Tools with C++ workload
#     (already installed if you can build the Rust crates)
#   * CUDA Toolkit (matching the torch wheel CUDA version, currently cu130).
#     Download: https://developer.nvidia.com/cuda-13-0-0-download-archive
#   * `uv sync --extra cuda` to install ninja
#
# Usage (from any shell, MSVC env loads automatically):
#   bash scripts/setup_dat_cuda_win.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
if [ ! -x "$VSWHERE" ]; then
    echo "[setup_dat_cuda_win] vswhere.exe not found at $VSWHERE" >&2
    echo "[setup_dat_cuda_win] Install Visual Studio 2022 Build Tools with C++ workload." >&2
    exit 1
fi

VS_INSTALL=$("$VSWHERE" -latest -products '*' \
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
    -property installationPath)
if [ -z "$VS_INSTALL" ]; then
    echo "[setup_dat_cuda_win] No VS install with C++ tools found." >&2
    exit 1
fi
VCVARS="$VS_INSTALL\\VC\\Auxiliary\\Build\\vcvars64.bat"
echo "[setup_dat_cuda_win] using MSVC env from: $VCVARS"

# Auto-detect CUDA Toolkit if env not already set.
CUDA_HINT=""
if [ -z "${CUDA_PATH:-}" ] && [ -z "${CUDA_HOME:-}" ]; then
    LATEST_CUDA=$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"/v* 2>/dev/null | sort -V | tail -1 || true)
    if [ -n "$LATEST_CUDA" ]; then
        CUDA_HINT=$(cygpath -w "$LATEST_CUDA")
        echo "[setup_dat_cuda_win] auto-detected CUDA at: $CUDA_HINT"
    else
        echo "[setup_dat_cuda_win] WARNING: CUDA Toolkit not found." >&2
        echo "[setup_dat_cuda_win] Install from https://developer.nvidia.com/cuda-13-0-0-download-archive" >&2
        echo "[setup_dat_cuda_win] Build will fail and the loader will fall back to pure-PyTorch DP." >&2
    fi
fi

# Generate a temp .bat that loads MSVC env + sets CUDA env + invokes the
# kernel build. cmd /c invocation avoids the bash↔cmd quoting cliff.
TMP_BAT=$(mktemp --suffix=.bat)
cleanup() { rm -f "$TMP_BAT"; }
trap cleanup EXIT

{
    echo "@echo off"
    echo "call \"$VCVARS\" >nul"
    echo "if errorlevel 1 exit /b 1"
    if [ -n "$CUDA_HINT" ]; then
        echo "set \"CUDA_PATH=$CUDA_HINT\""
        echo "set \"CUDA_HOME=$CUDA_HINT\""
    fi
    echo "set NEW_IME_DAT_VERBOSE=1"
    echo "uv run python -c \"from new_ime.training.loss.dat_cuda import load_dat_kernel; import sys; k = load_dat_kernel(); sys.exit(0 if k is not None else 1); \""
    echo "exit /b %errorlevel%"
} > "$TMP_BAT"

TMP_BAT_WIN=$(cygpath -w "$TMP_BAT")
if cmd //c "$TMP_BAT_WIN"; then
    echo "[setup_dat_cuda_win] kernel ready"
else
    echo "[setup_dat_cuda_win] FAILED: kernel did not build (see log above)." >&2
    echo "[setup_dat_cuda_win] Training will fall back to pure-PyTorch DP." >&2
    exit 1
fi
