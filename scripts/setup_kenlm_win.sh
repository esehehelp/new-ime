#!/usr/bin/env bash
# Clone KenLM and apply patches that drop the Boost / lmplz / test deps,
# leaving just `kenlm` + `kenlm_util` runtime libs (which is all the TSF
# DLL needs for shallow fusion).
#
# Adapted from the historical engine/server/third_party/setup_kenlm.sh on
# the `main` branch. The dev branch doesn't carry that tree, so this
# version targets `references/kenlm/` (already gitignored). Patches are
# applied via Python (instead of patch.exe) since the embedded diff format
# in the historical script trips msys patch.exe over blank context lines.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
KENLM_DIR="$REPO_ROOT/references/kenlm"

mkdir -p "$REPO_ROOT/references"
if [ ! -d "$KENLM_DIR" ]; then
    git clone --depth=1 https://github.com/kpu/kenlm.git "$KENLM_DIR"
fi

# Idempotent in-place edits. Each block: (filename, find, replace).
# `find` is matched as an exact substring; if not found, we assume the patch
# was already applied and continue. If neither old nor new variant is in the
# file we error out.
python - "$KENLM_DIR" <<'PY'
import sys
from pathlib import Path

kenlm = Path(sys.argv[1])

patches = [
    (
        kenlm / "CMakeLists.txt",
        # marker that already-patched files contain
        "option(KENLM_LIBS_ONLY",
        # find
        """# We need boost
find_package(Boost 1.41.0 REQUIRED COMPONENTS
  program_options
  system
  thread
  unit_test_framework
)

# Define where include files live
include_directories(${Boost_INCLUDE_DIRS})""",
        # replace
        """# Boost is only needed for the command-line binaries (lmplz, query, etc)
# and the test suite. The kenlm/kenlm_util libs themselves link fine
# without it. -DKENLM_LIBS_ONLY=ON skips the find_package, drops the
# binary and test targets, and still produces kenlm.lib + kenlm_util.lib.
option(KENLM_LIBS_ONLY "Skip Boost / binaries / tests, build libs only" OFF)
if(NOT KENLM_LIBS_ONLY)
  find_package(Boost 1.41.0 REQUIRED COMPONENTS
    program_options
    system
    thread
    unit_test_framework
  )
  include_directories(${Boost_INCLUDE_DIRS})
endif()""",
    ),
    (
        kenlm / "util" / "CMakeLists.txt",
        "if(NOT KENLM_LIBS_ONLY)\n  add_subdirectory(stream)",
        """add_subdirectory(double-conversion)
add_subdirectory(stream)""",
        """add_subdirectory(double-conversion)
# util/stream pulls Boost.thread (multi_progress) which isn't required for
# Model loading, so skip it in the libs-only configuration.
if(NOT KENLM_LIBS_ONLY)
  add_subdirectory(stream)
endif()

if(KENLM_LIBS_ONLY)
  set(KENLM_UTIL_STREAM_SOURCE "")
endif()""",
    ),
    (
        kenlm / "lm" / "CMakeLists.txt",
        "if(NOT KENLM_LIBS_ONLY)\n  add_subdirectory(builder)",
        """# This directory has children that need to be processed
add_subdirectory(builder)
add_subdirectory(filter)
add_subdirectory(interpolate)""",
        """# This directory has children that need to be processed
# Skip Boost/Eigen-dependent subdirs and binaries when only the libs are requested.
if(NOT KENLM_LIBS_ONLY)
  add_subdirectory(builder)
  add_subdirectory(filter)
  add_subdirectory(interpolate)
endif()""",
    ),
    (
        kenlm / "lm" / "CMakeLists.txt",
        "if(NOT KENLM_LIBS_ONLY)\n  AddExes(EXES",
        """AddExes(EXES ${EXE_LIST}
        LIBRARIES ${LM_LIBS})""",
        """if(NOT KENLM_LIBS_ONLY)
  AddExes(EXES ${EXE_LIST}
          LIBRARIES ${LM_LIBS})
endif()""",
    ),
    (
        kenlm / "lm" / "common" / "CMakeLists.txt",
        "if(KENLM_LIBS_ONLY)\n  set(KENLM_LM_COMMON_SOURCE",
        """set(KENLM_LM_COMMON_SOURCE
\t\t${CMAKE_CURRENT_SOURCE_DIR}/model_buffer.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/print.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/renumber.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/size_option.cc
  PARENT_SCOPE)""",
        """# The common sources (model_buffer, print, renumber, size_option) are only
# used by lmplz / interpolation / the CLI binaries and pull Boost
# (program_options, util/stream -> boost/thread). The run-time Model API
# does not need them - skip the list when the caller asked for a
# Boost-free libs-only build.
if(KENLM_LIBS_ONLY)
  set(KENLM_LM_COMMON_SOURCE "" PARENT_SCOPE)
else()
  set(KENLM_LM_COMMON_SOURCE
\t\t${CMAKE_CURRENT_SOURCE_DIR}/model_buffer.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/print.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/renumber.cc
\t\t${CMAKE_CURRENT_SOURCE_DIR}/size_option.cc
    PARENT_SCOPE)
endif()""",
    ),
]

for path, marker, find, replace in patches:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"[skip] {path.relative_to(kenlm)} (already patched)")
        continue
    if find not in text:
        print(f"[ERR ] {path.relative_to(kenlm)}: neither marker nor find-text present")
        sys.exit(1)
    path.write_text(text.replace(find, replace, 1), encoding="utf-8")
    print(f"[ok  ] {path.relative_to(kenlm)}")
PY

echo "[setup_kenlm_win] kenlm ready at $KENLM_DIR - configure with -DKENLM_LIBS_ONLY=ON"
