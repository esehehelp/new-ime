# Vision

The active repository is Rust-only.

- Python training, evaluation, and helper flows are retired from the active
  tree and archived under `legacy/python/`.
- C++ runtime and integration code are removed from the active tree.
- The workspace is rebuilt around `crates/*` with fixed naming prefixes:
  `rust-*`, `data-*`, and `new-ime-*`.
- IME work stays in the workspace, but this pass only guarantees structural
  consistency and build/check entrypoints.

## Current Direction

- `rust-*` crates own model, training, evaluation, and model-adjacent tools.
- `data-*` crates own extraction, synthesis, auditing, and processing crates.
- `new-ime-*` crates own runtime and IME integration boundaries.

## Non-Goals For This Reset

- Shipping a completed IME runtime.
- Preserving Python or C++ compatibility shims.
- Keeping stale benchmark tables in active docs.

## Success Criteria

- `cargo metadata` and `cargo check --workspace` are the canonical validation
  entrypoints.
- Active docs describe only the Rust-only layout and workflow.
- Archived material stays reachable under `legacy/` without leaking back into
  the active tree.
