# Attribution and Upstream Notices

This file records the main upstream sources and licensing notes relevant to
training, evaluation, and model comparison in this repository.

This is a repository-maintained notice file for operational use. It is not a
complete substitute for upstream model cards, dataset cards, or legal texts.

## Code

- Repository source code: MIT, see [LICENSE](LICENSE)

## Models and derived training artifacts

- Repository-trained model weights, checkpoints, exported artifacts, mixed
  training JSONL files, and distilled supervision artifacts:
  see [MODEL_LICENSE](MODEL_LICENSE)

## Upstream sources

### Wikimedia / Wikipedia text

- Wikipedia text is reused under Creative Commons Attribution Share-Alike.
- Wikimedia guidance states that most Wikimedia text is licensed under
  CC BY-SA 4.0 International, with additional reuse guidance and attribution
  expectations.

References:

- https://wikimediafoundation.org/our-work/wikimedia-projects/wikipedia/
- https://foundation.wikimedia.org/wiki/Legal:Wikimedia_Developer_App_Guidelines

### zenz-v2.5 models

- `Miwa-Keita/zenz-v2.5-xsmall`: CC-BY-SA 4.0
- `Miwa-Keita/zenz-v2.5-small`: CC-BY-SA 4.0
- `Miwa-Keita/zenz-v2.5-medium`: CC-BY-SA 4.0

References:

- https://huggingface.co/Miwa-Keita/zenz-v2.5-xsmall
- https://huggingface.co/Miwa-Keita/zenz-v2.5-medium

### zenz-v2.5 dataset

- `Miwa-Keita/zenz-v2.5-dataset`
- Repository usage has treated the `train_llm-jp-corpus-v3.jsonl` subset as
  ODC-By plus Common Crawl terms, while Wikipedia-derived subsets are not
  treated as permissive.

Reference:

- https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset

### FineWeb2

- `HuggingFaceFW/fineweb-2`: ODC-By 1.0
- Use is also subject to Common Crawl Terms of Use

Reference:

- https://huggingface.co/datasets/HuggingFaceFW/fineweb-2

### HPLT

- `HPLT/HPLT3.0`: CC0-1.0

Reference:

- https://huggingface.co/datasets/HPLT/HPLT3.0

### Aozora Bunko

- Public-domain works are used where applicable.
- Users must still verify work-specific status before redistribution.

Reference:

- https://www.aozora.gr.jp/

## Experimental status

This repository currently positions trained models and mixed datasets as
research / experimental artifacts.

Do not assume that all generated outputs are suitable for relicensing under
MIT. Check the exact input sources and artifact lineage first.
