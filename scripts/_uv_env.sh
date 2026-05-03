#!/usr/bin/env bash
# Source from any bash/zsh script that needs uv to pick up the right
# project venv per OS. Sets UV_PROJECT_ENVIRONMENT so subsequent
# `uv run ...` lands on the appropriate Windows / Linux venv.
#
# Usage:
#   source "$(dirname "$0")/_uv_env.sh"
#   uv run python -m new_ime.cli.bench ...

if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    # Running inside WSL: park the venv on the Linux native filesystem
    # (~/.venvs/...) instead of /mnt/d, because cross-filesystem I/O via
    # 9P is ~10x slower for `uv sync` and per-process import time.
    # The venv contents are platform-specific anyway (Linux ELF).
    export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/new-ime"
elif [[ "${OSTYPE:-}" == "msys" ]] \
  || [[ "${OSTYPE:-}" == "cygwin" ]] \
  || [[ "${OS:-}" == "Windows_NT" ]]; then
    # Git Bash / Cygwin / MSYS on Windows: keep the venv next to the
    # project so editor tooling (Pylance / Pyright) finds it without
    # extra config.
    export UV_PROJECT_ENVIRONMENT=".venv-windows"
else
    # Plain Linux (CI etc.).
    export UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/new-ime"
fi
