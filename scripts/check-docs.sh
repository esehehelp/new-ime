#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if rg -n \
  'python -m|pytest|models/src|datasets/tools|engine/src|engine/server|engine/win32|interactive-rs' \
  README.md docs; then
  echo "stale docs or script reference found" >&2
  exit 1
fi
