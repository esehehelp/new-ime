"""Edit-distance based metrics for kana-kanji conversion.

Compatible with the pre-v2 implementation in
`legacy/python/models/src/eval/metrics.py` (see archive/pre-v2). Output
field names are preserved so downstream JSON consumers do not break.

NFKC variant: AJIMEE-Bench's "実運用上の値" (Acc@1 NFKC) compares NFKC-
normalised candidate against NFKC-normalised reference. Half-width vs
full-width digits / katakana are unified, which catches ~5pt of EM that
the raw match misses for typical IME outputs. `nfkc_normalize` is the
single normalisation used; runner.py computes both raw and NFKC numbers
in one pass.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List


def nfkc_normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[len(b)]


def character_accuracy(reference: str, hypothesis: str) -> float:
    """1 - normalized edit distance, clamped to [0, 1]."""
    if not reference:
        return 1.0 if not hypothesis else 0.0
    dist = edit_distance(reference, hypothesis)
    return max(0.0, 1.0 - dist / len(reference))


def top_k_character_accuracy(
    reference: str, candidates: List[str], k: int = 1
) -> float:
    """Best CharAcc among the top-K candidates."""
    if not candidates:
        return 0.0
    best = 0.0
    for cand in candidates[:k]:
        acc = character_accuracy(reference, cand)
        if acc > best:
            best = acc
    return best


class EvalResult:
    """Accumulator over a bench. Tracks CharAcc and ExactMatch at K=1/5/10.

    `add_multi(refs, cands)` accepts a list of acceptable references and
    a top-N candidate list. CharAcc is the max over (ref, cand) pairs in
    top-k. ExactMatch is True if any top-k candidate is in references.
    """

    K_VALUES = (1, 5, 10)

    def __init__(self) -> None:
        self.char_acc_sum: Dict[int, float] = {}
        self.exact_match_sum: Dict[int, int] = {}
        self.total = 0

    def add(self, reference: str, candidates: List[str]) -> None:
        self.add_multi([reference], candidates)

    def add_multi(self, references: List[str], candidates: List[str]) -> None:
        self.total += 1
        if not references:
            return
        for k in self.K_VALUES:
            best_acc = 0.0
            for ref in references:
                acc = top_k_character_accuracy(ref, candidates, k)
                if acc > best_acc:
                    best_acc = acc
            self.char_acc_sum[k] = self.char_acc_sum.get(k, 0.0) + best_acc
            em = any(c in references for c in candidates[:k])
            self.exact_match_sum[k] = self.exact_match_sum.get(k, 0) + int(em)

    def summary(self) -> dict:
        if self.total == 0:
            return {}
        result: dict = {"total": self.total}
        for k in self.K_VALUES:
            ca = self.char_acc_sum.get(k, 0.0) / self.total
            em = self.exact_match_sum.get(k, 0) / self.total
            result[f"char_acc_top{k}"] = round(ca, 4)
            result[f"exact_match_top{k}"] = round(em, 4)
        return result


def percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile, matching the pre-v2 behavior of
    `lat[int(n * pct)]` (no interpolation, simple index).
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if pct >= 1.0:
        return s[-1]
    idx = int(n * pct)
    if idx >= n:
        idx = n - 1
    return s[idx]


def latency_summary(latencies_ms: List[float]) -> dict:
    """Returns {p50, p95, mean} in ms, rounded to 1 decimal."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": round(percentile(latencies_ms, 0.50), 1),
        "p95": round(percentile(latencies_ms, 0.95), 1),
        "mean": round(sum(latencies_ms) / len(latencies_ms), 1),
    }
