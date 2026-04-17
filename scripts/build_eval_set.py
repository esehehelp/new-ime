"""Build evaluation sets (dev/test) from clean JSONL data.

Splits data at article/work level to prevent contamination.
Produces stratified samples by domain and sentence length.

Usage:
    uv run python scripts/build_eval_set.py \
        --wiki datasets/wiki_clean.jsonl \
        --aozora datasets/aozora_clean.jsonl \
        --dev-size 2000 \
        --test-size 10000 \
        --output-dir datasets/eval
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def assign_length_bin(surface: str) -> str:
    """Categorize by surface length."""
    n = len(surface)
    if n <= 15:
        return "short"
    elif n <= 40:
        return "medium"
    else:
        return "long"


def stratified_sample(
    data: list[dict],
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Stratified sample by length bin, maintaining proportions."""
    rng = random.Random(seed)

    # Group by length bin
    bins: defaultdict[str, list[dict]] = defaultdict(list)
    for item in data:
        b = assign_length_bin(item["surface"])
        bins[b].append(item)

    # Sample proportionally
    total = len(data)
    sampled = []
    for bin_name, bin_data in bins.items():
        bin_n = max(1, round(n * len(bin_data) / total))
        if bin_n >= len(bin_data):
            sampled.extend(bin_data)
        else:
            sampled.extend(rng.sample(bin_data, bin_n))

    # Trim or pad to exact size
    rng.shuffle(sampled)
    return sampled[:n]


def split_by_context_group(data: list[dict]) -> list[list[dict]]:
    """Group consecutive sentences that share context chains.

    This approximates article-level grouping: sentences with context
    that references a previous sentence in the same article are grouped together.
    We split when context is empty (new article boundary).
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    for item in data:
        if not item.get("context") and current:
            groups.append(current)
            current = []
        current.append(item)

    if current:
        groups.append(current)

    return groups


def main():
    parser = argparse.ArgumentParser(description="Build evaluation sets")
    parser.add_argument("--wiki", required=True, help="Wiki clean JSONL")
    parser.add_argument("--aozora", required=True, help="Aozora clean JSONL")
    parser.add_argument("--dev-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading Wikipedia data...")
    wiki_data = load_jsonl(args.wiki)
    print(f"  {len(wiki_data):,} sentences")

    print("Loading Aozora data...")
    aozora_data = load_jsonl(args.aozora)
    print(f"  {len(aozora_data):,} sentences")

    # Group by article/work boundaries
    print("Grouping by article boundaries...")
    wiki_groups = split_by_context_group(wiki_data)
    aozora_groups = split_by_context_group(aozora_data)
    print(f"  Wiki: {len(wiki_groups):,} groups")
    print(f"  Aozora: {len(aozora_groups):,} groups")

    # Shuffle groups and split: first N groups for eval, rest for training
    rng.shuffle(wiki_groups)
    rng.shuffle(aozora_groups)

    # Reserve ~5% of groups for eval pool
    wiki_eval_groups = wiki_groups[:len(wiki_groups) // 20]
    wiki_train_groups = wiki_groups[len(wiki_groups) // 20:]
    aozora_eval_groups = aozora_groups[:len(aozora_groups) // 20]
    aozora_train_groups = aozora_groups[len(aozora_groups) // 20:]

    # Flatten eval pools
    wiki_eval_pool = [s for g in wiki_eval_groups for s in g]
    aozora_eval_pool = [s for g in aozora_eval_groups for s in g]
    print(f"\nEval pool: {len(wiki_eval_pool):,} wiki + {len(aozora_eval_pool):,} aozora")

    # Build dev set: 70% wiki, 30% aozora (proportional to dataset sizes)
    wiki_dev_n = int(args.dev_size * 0.7)
    aozora_dev_n = args.dev_size - wiki_dev_n
    dev_wiki = stratified_sample(wiki_eval_pool, wiki_dev_n, seed=args.seed)
    dev_aozora = stratified_sample(aozora_eval_pool, aozora_dev_n, seed=args.seed + 1)
    dev_set = dev_wiki + dev_aozora
    rng.shuffle(dev_set)

    # Tag with source
    for item in dev_wiki:
        item["source"] = "wiki"
    for item in dev_aozora:
        item["source"] = "aozora"

    # Build test set: same proportions
    wiki_test_n = int(args.test_size * 0.7)
    aozora_test_n = args.test_size - wiki_test_n

    # Use separate portion of eval pool for test (no overlap with dev)
    dev_surfaces = {item["surface"] for item in dev_set}
    wiki_eval_remaining = [s for s in wiki_eval_pool if s["surface"] not in dev_surfaces]
    aozora_eval_remaining = [s for s in aozora_eval_pool if s["surface"] not in dev_surfaces]

    test_wiki = stratified_sample(wiki_eval_remaining, wiki_test_n, seed=args.seed + 2)
    test_aozora = stratified_sample(aozora_eval_remaining, aozora_test_n, seed=args.seed + 3)
    for item in test_wiki:
        item["source"] = "wiki"
    for item in test_aozora:
        item["source"] = "aozora"
    test_set = test_wiki + test_aozora
    rng.shuffle(test_set)

    # Build training set (everything not in eval pool)
    train_data = [s for g in wiki_train_groups for s in g] + \
                 [s for g in aozora_train_groups for s in g]

    # Save
    def save_jsonl(data: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(dev_set, output_dir / "dev.jsonl")
    save_jsonl(test_set, output_dir / "test.jsonl")
    save_jsonl(train_data, output_dir / "train.jsonl")

    # Stats
    print("\n=== Output ===")
    print(f"Dev:   {len(dev_set):,} sentences → {output_dir / 'dev.jsonl'}")
    print(f"Test:  {len(test_set):,} sentences → {output_dir / 'test.jsonl'}")
    print(f"Train: {len(train_data):,} sentences → {output_dir / 'train.jsonl'}")

    # Length distribution
    for name, dataset in [("Dev", dev_set), ("Test", test_set)]:
        bins = defaultdict(int)
        sources = defaultdict(int)
        for item in dataset:
            bins[assign_length_bin(item["surface"])] += 1
            sources[item.get("source", "unknown")] += 1
        print(f"\n{name} distribution:")
        print(f"  Length: {dict(bins)}")
        print(f"  Source: {dict(sources)}")


if __name__ == "__main__":
    main()
