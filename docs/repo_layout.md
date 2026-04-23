# Repository Layout

## Active Directories

- `crates/`: the only location for active Rust code.
- `configs/`: active Rust-facing configuration files.
- `datasets/`: data and generated artifacts only. No code.
- `docs/`: active project policy and benchmark contract.
- `scripts/`: shell-only maintenance checks.
- `legacy/`: archived Python, C++, and historical material.

## Naming Rules

- `rust-*`: model, training, evaluation, and model-side utilities.
- `data-*`: extraction, synthesis, auditing, and processing crates.
- `new-ime-*`: runtime and IME integration crates.

Directory name, Cargo package name, and primary binary name must match.

## Hard Rules

- Do not place Rust code outside `crates/`.
- Do not place Python files in the active tree.
- Do not place `CMakeLists.txt` or `pyproject.toml` in the active tree.
- Do not add `kkc-*`, `*-rs`, or other retired package names to active code or
  docs.
- Do not put code under `datasets/`.

## Legacy Policy

- `legacy/python/` stores retired Python workflows and configs.
- `legacy/tools/` stores historical helper scripts and non-active runtime
  support files.
- `legacy/docs/` stores superseded plans, reports, and benchmark notes.

Archived material is preserved for reference, not as a supported workflow.
