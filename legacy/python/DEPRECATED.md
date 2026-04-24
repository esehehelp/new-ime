# Deprecated Python scripts (superseded by Rust crates)

This file inventories Python scripts in `legacy/python/` whose
functionality has been reimplemented as Rust crates in the active
workspace. The Python versions are **retained for reference only** —
new corpus / dataset / bench work should go through the Rust CLIs.

Kept separately in `dev`-branch Phase D. Deletion of these scripts is
a follow-up task gated on confirming each Rust CLI covers every option
and env that the Python version exposed.

## Superseded scripts

| Python script (legacy) | Rust replacement (active) | Notes |
|---|---|---|
| `datasets/tools/corpus/bunsetsu_split.py` | `crates/data-chunk-generator` | MeCab TSV → bunsetsu chunk JSONL. The Rust CLI covers `dedup` / `max_window` and is the canonical path. |
| `datasets/tools/corpus/synth_numeric.py` | `crates/data-synth-numeric-units` | Numeric + unit synthesis with deterministic seeding. |
| `datasets/tools/corpus/synth_numeric_ext.py` | `crates/data-synth-numeric-units` | Same; the Rust crate subsumes both the base and the "ext" generators. |
| `models/src/data/dataset.py::_build_offset_index` (fallback) | `crates/data-offset-index` | The Rust binary is already the primary path — the Python fallback in `dataset.py` stays only for environments without the binary. |
| in-Python tokenize + JSON parse on the dataloader hot path (`CTCCollator.__call__`) | `crates/rust-data compile` + `KanaKanjiShardDataset` / `CTCShardCollator` | Phase C (dev branch). Use `rust-data compile` once to produce a `.kkc` shard, then set the trainer's `--train` to that shard. |

## Rust-only from inception (no Python predecessor)

These crates never had a Python version — listed here so readers
don't look for one:

- `crates/data-audit` — pool audit / contamination report
- `crates/data-extract-domain` — domain splitter
- `crates/data-extract-short` — short-phrase filter
- `crates/data-mix` — weighted mix builder with n-gram contamination exclusion
- `crates/data-process-whitepaper` — ministry whitepaper pipeline
- `crates/data-process-zenz` — zenz-v2.5 preprocessing
- `crates/data-synth-homophone` — homophone pair mining
- `crates/data-synth-name` — proper-noun synthesis
- `crates/data-bench-onnx` — ONNX CTC-NAT benchmark
- `crates/rust-audit-tokenizer` — tokenizer byte-fallback audit
- `crates/rust-build-vocab` — vocab construction from JSONL character frequency
- `crates/rust-postprocess` — JSONL postprocess / dedup filter

## Still Python-only (no Rust replacement planned)

- `legacy/python/models/src/training/train_ctc_nat.py` — training entrypoint
  (retained per dev-branch policy; Rust-train was retired in commit
  `c9980f3`)
- `legacy/python/models/src/model/*.py` — `torch.nn` model definitions
- `legacy/python/models/src/training/kd.py` — KD teacher routing
- `legacy/python/datasets/tools/corpus/clean_v2.py`,
  `extract_aozora_dialogue.py`, `process_tatoeba.py`,
  `process_wikimedia.py`, `sample_and_classify.py` — one-off corpus
  shims, not worth porting until corpus is regenerated
- `legacy/python/datasets/tools/probe/*.py` — probe generation / sweeps
  (evaluation-side Python scripts, not on the dataloader hot path)
