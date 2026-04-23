# Development

The active development surface is Cargo plus shell checks.

## Canonical Commands

```bash
cargo metadata --format-version 1 --no-deps
cargo check --workspace
cargo run -p rust-train -- --help
cargo run -p rust-bench -- --help
cargo run -p new-ime-interactive -- --help
scripts/check-hygiene.sh
scripts/check-docs.sh
```

For Windows-only TSF validation:

```bash
cargo check -p new-ime-tsf --target x86_64-pc-windows-msvc
```

## Expectations

- Prefer `cargo` subcommands over ad hoc tooling.
- Keep active automation in shell scripts under `scripts/`.
- Provide external `LIBTORCH` only when opting into `rust-train --features cuda`
  or `rust-bench --features native-tch`; do not pin it in repo config.
- Archive Python or C++ experiments under `legacy/` instead of reviving them
  in the active tree.

## Workspace Policy

- New crates belong under `crates/`.
- Shared dependency versions belong in the root workspace manifest.
- Active docs must stay aligned with the Rust-only repository contract.
