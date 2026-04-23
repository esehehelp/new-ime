#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

active_text_roots=(.cargo Cargo.toml README.md configs crates)

check_absent_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    echo "forbidden active path present: $path" >&2
    exit 1
  fi
}

check_absent_path CMakeLists.txt
check_absent_path pyproject.toml
check_absent_path docs/old
check_absent_path datasets/tools
check_absent_path models/src
check_absent_path tools/old
check_absent_path engine/src
check_absent_path engine/server
check_absent_path engine/win32
check_absent_path engine/tools
check_absent_path engine/win32-rs
check_absent_path models/rust

if find .cargo Cargo.toml README.md configs crates docs scripts datasets -type f \
  \( -name '*.py' -o -name 'pyproject.toml' -o -name 'CMakeLists.txt' \) \
  -print | grep -q .; then
  echo "forbidden file type found in active tree" >&2
  find .cargo Cargo.toml README.md configs crates docs scripts datasets -type f \
    \( -name '*.py' -o -name 'pyproject.toml' -o -name 'CMakeLists.txt' \) \
    -print >&2
  exit 1
fi

if rg -n 'python -m|pytest|kkc-|interactive-rs|engine/win32-rs|engine/tools/|models/rust/' "${active_text_roots[@]}"; then
  echo "forbidden legacy text found in active tree" >&2
  exit 1
fi
