# Benchmark Contract

This document defines the active benchmark contract only. Historical result
tables and sweep notes live under `legacy/docs/`.

## Canonical Inputs

- `datasets/eval/probe/probe.json`
- `references/AJIMEE-Bench/JWTD_v2/v1/evaluation_items.json`

## Canonical Decoding Settings

- device: CPU
- beams: `5`
- return count: `5`

## Canonical Metrics

- `EM1`
- `EM5`
- `CharAcc`
- `p50 latency`

## Regeneration Rules

- Use the active Rust workspace entrypoints.
- Enable model execution with `cargo run -p rust-bench --features native-tch -- ...`
  only when `LIBTORCH` is provided externally.
- Record the exact config, checkpoint or run directory, and decoding settings
  with each run.
- Treat any measurement outside this contract as exploratory, not canonical.

## Publishing Rules

- Keep the active doc contract-only.
- Move one-off measurements, stale comparisons, and superseded tables into
  `legacy/docs/`.
