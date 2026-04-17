"""Download the ODC-BY subset of zenz-v2.5-dataset.

Only ``train_llm-jp-corpus-v3.jsonl`` is downloaded — the companion
``train_wikipedia.jsonl`` inherits CC-BY-SA 4.0 from Wikipedia and is not
MIT-compatible (see ``docs/dataset_candidates.md`` row #18).

The file is ~30 GB, already in (input, output, left_context) form. No MeCab
or reading re-derivation is needed; downstream ``process_zenz_subset.py``
renames the fields to match our SharedCharTokenizer training schema.

Usage:
    uv run python scripts/download_zenz_subset.py \
        --out datasets/src/zenz_llmjp/ \
        [--revision main]

Requires a working network connection and enough disk for 30 GB. Resumes if
the partial file exists (Hugging Face client handles ranged transfer).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "Miwa-Keita/zenz-v2.5-dataset"
TARGET_FILE = "train_llm-jp-corpus-v3.jsonl"
# Companion file we refuse to touch (CC-BY-SA).
FORBIDDEN_FILES = ("train_wikipedia.jsonl",)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="datasets/src/zenz_llmjp/", help="Output directory")
    parser.add_argument("--revision", default=None, help="Optional dataset revision/commit")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TARGET_FILE} from {REPO_ID} → {out_dir}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=TARGET_FILE,
        repo_type="dataset",
        local_dir=str(out_dir),
        revision=args.revision,
    )
    print(f"Done: {path}")

    # Double-check the forbidden file was not materialised by any mirroring.
    for forbidden in FORBIDDEN_FILES:
        forbidden_path = out_dir / forbidden
        if forbidden_path.exists():
            raise SystemExit(
                f"Refusing: forbidden file {forbidden_path} is present. "
                "That subset is CC-BY-SA 4.0 (Wikipedia-derived) and must not "
                "be used for training in this MIT-licensed project."
            )

    # Attribution reminder — ODC-BY requires attribution on redistribution.
    attribution_hint = out_dir / "ATTRIBUTION.md"
    attribution_hint.write_text(
        "Source: https://huggingface.co/datasets/Miwa-Keita/zenz-v2.5-dataset\n"
        f"File:   {TARGET_FILE}\n"
        "License: ODC-BY 1.0 (llm-jp-corpus-v3 subset)\n"
        "Attribution required on redistribution of derivatives.\n",
        encoding="utf-8",
    )
    print(f"Attribution note written: {attribution_hint}")


if __name__ == "__main__":
    main()
