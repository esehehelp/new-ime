# Data and Artifact Licensing

This repository separates code licensing from model/data artifact licensing.

- Code: [MIT](LICENSE)
- Model and data artifacts produced by training or mixing pipelines: [CC BY-SA 4.0](MODEL_LICENSE), subject to upstream notices below

This document is an operational summary for this repository. It is not legal advice.

## Current policy

The project is currently managed as a research / experimental prototype.
Quality and empirical validation are prioritized over keeping every training
artifact MIT-compatible.

As a result:

- training mixtures may include sources that are not MIT-compatible
- mixed training files and derived model artifacts must not be assumed to be MIT
- upstream attribution and source terms must be preserved

## Source categories

### Sources generally treated as permissive for training

- `HPLT/HPLT3.0`: CC0-1.0
- `HuggingFaceFW/fineweb-2`: ODC-By 1.0, also subject to Common Crawl terms
- `Miwa-Keita/zenz-v2.5-dataset` `train_llm-jp-corpus-v3.jsonl` subset: treated in this repo as ODC-By + Common Crawl terms, with caution
- Aozora Bunko public-domain works: Public Domain, where applicable

### Sources that trigger ShareAlike concerns

- Wikipedia-derived corpora
- datasets generated from Wikipedia-derived corpora
- chunk corpora generated from Wikipedia-derived sentence pairs
- zenz model weights themselves (`Miwa-Keita/zenz-v2.5-*`): CC-BY-SA 4.0

## Important consequence

If a training mix includes Wikipedia-derived content, or if a model is
materially derived from such a mix, that artifact must not be treated as MIT.
In this repository, such artifacts are handled under `MODEL_LICENSE` plus any
additional upstream requirements.

## Repository-local examples

- `datasets/wiki_clean_v3.jsonl`: Wikipedia-derived, ShareAlike-sensitive
- `datasets/chunks_v3_100m.jsonl`: may inherit upstream obligations if built
  from Wikipedia-derived inputs
- `datasets/phase3/train.jsonl`: mixed artifact; inspect the exact pool recipe
  used to build it before assigning downstream rights
- `checkpoints/*.pt`: derived model artifacts; do not assume MIT

## Practical rule

Before publishing a dataset or model artifact, record:

1. the exact input sources
2. the mixing script / command
3. the applicable upstream licenses
4. the intended outbound license for the artifact

If any input source is unknown, do not publish the artifact as MIT.
