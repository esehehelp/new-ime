"""Evaluation metrics for kana-kanji conversion.

Core metrics:
    - Character Accuracy (Top-K): edit-distance based
    - Segmentation Accuracy: bunsetsu boundary P/R/F1
    - Homophone Accuracy: subset accuracy on ambiguous readings
"""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[len(b)]


def character_accuracy(reference: str, hypothesis: str) -> float:
    """Character-level accuracy based on edit distance.

    Returns 1.0 for perfect match, 0.0 for completely wrong.
    """
    if not reference:
        return 1.0 if not hypothesis else 0.0
    dist = edit_distance(reference, hypothesis)
    return max(0.0, 1.0 - dist / len(reference))


def top_k_character_accuracy(
    reference: str,
    candidates: list[str],
    k: int = 1,
) -> float:
    """Best character accuracy among top-K candidates."""
    if not candidates:
        return 0.0
    best = 0.0
    for cand in candidates[:k]:
        acc = character_accuracy(reference, cand)
        best = max(best, acc)
    return best


def exact_match(reference: str, hypothesis: str) -> bool:
    """Whether the hypothesis exactly matches the reference."""
    return reference == hypothesis


def top_k_exact_match(reference: str, candidates: list[str], k: int = 1) -> bool:
    """Whether any of the top-K candidates exactly matches the reference."""
    return any(c == reference for c in candidates[:k])


class EvalResult:
    """Aggregated evaluation results over a dataset."""

    def __init__(self) -> None:
        self.char_acc_sum: dict[int, float] = {}  # k -> sum of top-k char acc
        self.exact_match_sum: dict[int, int] = {}  # k -> count of exact matches
        self.total = 0

    def add(self, reference: str, candidates: list[str]) -> None:
        """Add one evaluation sample."""
        self.total += 1
        for k in [1, 5, 10]:
            acc = top_k_character_accuracy(reference, candidates, k)
            self.char_acc_sum[k] = self.char_acc_sum.get(k, 0.0) + acc
            em = top_k_exact_match(reference, candidates, k)
            self.exact_match_sum[k] = self.exact_match_sum.get(k, 0) + int(em)

    def summary(self) -> dict:
        """Return summary statistics."""
        if self.total == 0:
            return {}
        result = {"total": self.total}
        for k in [1, 5, 10]:
            ca = self.char_acc_sum.get(k, 0.0) / self.total
            em = self.exact_match_sum.get(k, 0) / self.total
            result[f"char_acc_top{k}"] = round(ca, 4)
            result[f"exact_match_top{k}"] = round(em, 4)
        return result

    def report(self) -> str:
        """Return human-readable report."""
        s = self.summary()
        if not s:
            return "No samples evaluated."
        lines = [f"Evaluated {s['total']} samples:"]
        for k in [1, 5, 10]:
            ca = s.get(f"char_acc_top{k}", 0)
            em = s.get(f"exact_match_top{k}", 0)
            lines.append(f"  Top-{k}:  CharAcc={ca:.4f}  ExactMatch={em:.4f}")
        return "\n".join(lines)
