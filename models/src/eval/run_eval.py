"""Run evaluation on dev/test sets against one or more conversion backends.

Backends implement ConversionBackend protocol: convert(reading, context) -> list[str]

Usage:
    uv run python -m src.eval.run_eval \
        --eval-set datasets/eval/dev.jsonl \
        --backend mock \
        --output results/dev_mock.json
"""

from __future__ import annotations

import argparse
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

from models.src.eval.metrics import EvalResult


class ConversionBackend(ABC):
    """Interface for kana-kanji conversion backends."""

    @abstractmethod
    def convert(self, reading: str, context: str) -> list[str]:
        """Convert reading to kanji candidates.

        Args:
            reading: Hiragana reading.
            context: Left context (previous text).

        Returns:
            List of candidate strings, ordered by score (best first).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockBackend(ConversionBackend):
    """Mock backend that returns reading as-is (no conversion). For testing."""

    def convert(self, reading: str, context: str) -> list[str]:
        return [reading]

    @property
    def name(self) -> str:
        return "mock"


class IdentityBackend(ConversionBackend):
    """Backend that returns the reading unchanged. Measures baseline (all-hiragana)."""

    def convert(self, reading: str, context: str) -> list[str]:
        return [reading]

    @property
    def name(self) -> str:
        return "identity"


def load_eval_set(path: str) -> list[dict]:
    """Load evaluation JSONL."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_evaluation(
    backend: ConversionBackend,
    eval_data: list[dict],
    verbose: bool = False,
) -> dict:
    """Run evaluation and return results dict."""
    result = EvalResult()
    latencies: list[float] = []

    for i, item in enumerate(eval_data):
        reading = item["reading"]
        context = item.get("context", "")
        reference = item["surface"]

        t0 = time.perf_counter()
        candidates = backend.convert(reading, context)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms
        result.add(reference, candidates)

        if verbose and i < 5:
            print(f"  [{i}] reading: {reading[:30]}")
            print(f"       ref:     {reference[:30]}")
            print(f"       pred:    {candidates[0][:30] if candidates else '(none)'}")
            print()

    # Latency stats
    latencies.sort()
    n = len(latencies)
    latency_stats = {
        "p50_ms": round(latencies[n // 2], 2) if n else 0,
        "p95_ms": round(latencies[int(n * 0.95)] if n else 0, 2),
        "p99_ms": round(latencies[int(n * 0.99)] if n else 0, 2),
        "mean_ms": round(sum(latencies) / n if n else 0, 2),
    }

    summary = result.summary()
    summary["backend"] = backend.name
    summary["latency"] = latency_stats
    return summary


def get_backend(name: str) -> ConversionBackend:
    """Factory for backends."""
    if name == "mock":
        return MockBackend()
    if name == "identity":
        return IdentityBackend()
    raise ValueError(f"Unknown backend: {name}. Available: mock, identity")


def main():
    parser = argparse.ArgumentParser(description="Run IME evaluation")
    parser.add_argument("--eval-set", required=True, help="Eval JSONL path")
    parser.add_argument("--backend", default="identity", help="Backend name")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    eval_data = load_eval_set(args.eval_set)
    print(f"Loaded {len(eval_data)} eval samples from {args.eval_set}")

    backend = get_backend(args.backend)
    print(f"Backend: {backend.name}")
    print()

    results = run_evaluation(backend, eval_data, verbose=args.verbose)

    # Print report
    print(f"=== {backend.name} on {args.eval_set} ===")
    print(f"Samples: {results['total']}")
    for k in [1, 5, 10]:
        ca = results.get(f"char_acc_top{k}", 0)
        em = results.get(f"exact_match_top{k}", 0)
        print(f"  Top-{k}: CharAcc={ca:.4f}  ExactMatch={em:.4f}")
    lat = results["latency"]
    print(f"  Latency: p50={lat['p50_ms']}ms p95={lat['p95_ms']}ms p99={lat['p99_ms']}ms")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
