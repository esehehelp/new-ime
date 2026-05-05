#!/usr/bin/env bash
# CMake build of references/kenlm/ for Linux/WSL.
#
# Outputs:
#   build/kenlm_linux/lib/libkenlm.a
#   build/kenlm_linux/lib/libkenlm_util.a
#
# `crates/new-ime-engine-core/build.rs` picks these up automatically (Linux
# branch). The KENLM_LIBS_ONLY patch in references/kenlm (applied by
# setup_kenlm_win.sh — patches are idempotent so this script doesn't
# re-apply them; if you cloned a fresh kenlm into references/, run
# setup_kenlm_win.sh first to install the patches).
#
# Boost / lmplz / test deps are skipped via -DKENLM_LIBS_ONLY=ON, matching
# the Windows build. KENLM_MAX_ORDER=6 matches the Windows shim and the
# 6-gram models in assets/kenlm/.
#
# Usage:
#   bash scripts/setup_kenlm_linux.sh
#   (or via WSL: wsl.exe -e bash -lc "cd /mnt/d/Dev/new-ime && bash scripts/setup_kenlm_linux.sh")

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
KENLM_DIR="$REPO_ROOT/references/kenlm"
BUILD_DIR="$REPO_ROOT/build/kenlm_linux"

if [ ! -d "$KENLM_DIR" ]; then
    echo "[ERR] $KENLM_DIR not found. Run setup_kenlm_win.sh first to clone+patch kenlm." >&2
    exit 1
fi

if ! grep -q "KENLM_LIBS_ONLY" "$KENLM_DIR/CMakeLists.txt"; then
    echo "[ERR] $KENLM_DIR/CMakeLists.txt is unpatched. Run setup_kenlm_win.sh first." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

cmake -S "$KENLM_DIR" -B "$BUILD_DIR" \
    -DKENLM_LIBS_ONLY=ON \
    -DKENLM_MAX_ORDER=6 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF

cmake --build "$BUILD_DIR" --config Release --parallel \
    --target kenlm kenlm_util

# Mirror the Windows layout (build/kenlm_win/lib/Release/) but flat so
# build.rs only needs one Linux-side path.
mkdir -p "$BUILD_DIR/lib"
find "$BUILD_DIR" -maxdepth 4 -name "libkenlm.a" -exec cp -f {} "$BUILD_DIR/lib/" \;
find "$BUILD_DIR" -maxdepth 4 -name "libkenlm_util.a" -exec cp -f {} "$BUILD_DIR/lib/" \;

echo
echo "[setup_kenlm_linux] artifacts:"
ls -l "$BUILD_DIR/lib/libkenlm.a" "$BUILD_DIR/lib/libkenlm_util.a"
