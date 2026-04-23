"""KenLM-MoE (mixture) scorer for CTC-NAT shallow fusion.

Given N domain-specific KenLM models + a category estimator that looks at
the *reading* (hiragana input), return a weighted-sum log-prob in natural
log, caching per-prefix just like `KenLMCharScorer`.

Design:
    score(prefix) = Σ_d  w_d · logp_d(prefix)

    where the weights w_d are set per *input reading* at backend construction
    / convert() time by the `CategoryEstimator` (heuristic from the input
    reading: ratios of ASCII / katakana / kanji / digit / hiragana).

    The scorer holds N `KenLMCharScorer` instances internally; the active
    weight vector is set via `set_weights(dict[str, float])` before beam
    search. Beam search then calls `score(prefix)` as usual.

    CTCNATBackend calls `set_weights(estimator.weights(reading))` once per
    `_decode_one`, just before entering `prefix_beam_search`. The mixture
    then behaves as a single `PrefixLMScorer` for the beam search loop.

Weight semantics:
    Weights are linear coefficients over log-prob, interpretable as a
    log-linear interpolation (equivalent to geometric mean in prob space).
    They do not need to sum to 1; the KenLM shallow-fusion `α` hyper-
    parameter in `prefix_beam_search` multiplies the final mixture score
    anyway, so w_d absolute scale trades against α.
"""

from __future__ import annotations

import math
from typing import Tuple


class KenLMMixtureScorer:
    """Convex combination over multiple char-level KenLM scorers.

    Each inner scorer is loaded lazily; once loaded, `.score(prefix)` per
    domain is cached in that inner scorer's own LRU. The mixture keeps a
    second-level cache keyed by (prefix, weight-hash) so repeated queries
    with the same active weights are free.
    """

    def __init__(self, model_paths: dict[str, str], tokenizer):
        try:
            import kenlm  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "kenlm is not installed in this environment. "
                "Use WSL or a kenlm wheel compatible with the active python."
            ) from e
        from models.src.eval.kenlm_scorer import KenLMCharScorer

        self.scorers: dict[str, KenLMCharScorer] = {}
        for domain, path in model_paths.items():
            self.scorers[domain] = KenLMCharScorer(path, tokenizer)
        self.tokenizer = tokenizer
        # default: even weights
        self.weights: dict[str, float] = {d: 1.0 / len(self.scorers) for d in self.scorers}
        self._cache: dict[tuple[Tuple[int, ...], tuple[tuple[str, float], ...]], float] = {}

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update the active domain weights. Cache is keyed by the weight
        tuple so an update invalidates only stale entries, not everything."""
        # Clone + sort so the cache key is deterministic.
        self.weights = {d: float(weights.get(d, 0.0)) for d in self.scorers}

    def _weight_key(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self.weights.items()))

    def score(self, prefix: Tuple[int, ...]) -> float:
        key = (prefix, self._weight_key())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        total = 0.0
        for domain, scorer in self.scorers.items():
            w = self.weights.get(domain, 0.0)
            if w == 0.0:
                continue
            total += w * scorer.score(prefix)
        self._cache[key] = total
        return total


class CategoryEstimator:
    """Heuristic category weight estimator from a hiragana input reading.

    Returns a weight dict over the active domains. Simple, deterministic,
    no model. The heuristic uses character-class ratios computed on the
    *reading* (input), which correlates with the likely output domain:

        - katakana-heavy reading → probably tech / loanword output
        - digit-heavy reading    → probably numeric (no special boost, handled
                                    by general LM's digit n-grams)
        - proper-noun hint (trailing 「氏 / 市 / 区 / ...」 in surface context
          or specific prefixes in reading) → entity

    This is a *prior* over domains; the KenLM shallow-fusion coefficient α
    (set in backend) controls how strongly any LM contributes.
    """

    # Default per-profile weights. Tune by observed EM1 per category in
    # tools/misc/aggregate_v3_sweep.py outputs.
    PROFILES: dict[str, dict[str, float]] = {
        "generic":  {"general": 1.0, "tech": 0.0, "entity": 0.0},
        "tech":     {"general": 0.3, "tech": 0.7, "entity": 0.0},
        "entity":   {"general": 0.4, "tech": 0.0, "entity": 0.6},
        "tech_entity": {"general": 0.2, "tech": 0.5, "entity": 0.3},
    }

    KATA_TECH_HINTS = {
        # Common loan-word head that strongly implies tech domain.
        "プログラム", "システム", "データ", "ソフト", "ハード", "デジタル",
        "コンピュータ", "ネットワーク", "アルゴリズム", "クラウド", "サーバ",
        "スマート", "アップデート", "セキュリティ", "インターネット",
    }

    # Probe readings arrive as hiragana (kata2hira applied by load_probe).
    # These patterns catch loan-word / tech hiragana forms that would
    # otherwise look generic under char-class ratios.
    HIRA_LOAN_MARKERS = [
        "ー",                      # long-vowel kana (strong loan marker)
        "てぃ", "でぃ",             # ti / di
        "ふぁ", "ふぃ", "ふぇ", "ふぉ",
        "うぃ", "うぇ", "うぉ",
        "しぇ", "じぇ", "ちぇ",
        "つぁ", "つぇ", "つぉ",
    ]
    HIRA_TECH_HINTS = {
        # Stems — substring match so "ぷろぐら" catches ぷろぐらむ / ぷろぐらみんぐ
        "ぷろぐら", "しすて", "でーた", "そふと", "はーど", "でじた",
        "こんぴゅ", "ねっと", "あるごり", "くらうど", "さーば",
        "すまーと", "あっぷ", "せきゅ", "いんたー", "ぐらふ",
        "ろぼっ", "あぷり", "ふぁい", "ふぉる", "でーたべ",
        "あぷりけーしょん", "いんたふぇーす",
    }

    def __init__(self, available_domains: set[str] | None = None) -> None:
        self.available = available_domains or {"general", "tech", "entity"}

    @staticmethod
    def _char_class_ratios(s: str) -> dict[str, float]:
        if not s:
            return {"ascii": 0.0, "kata": 0.0, "hira": 0.0, "digit": 0.0, "other": 0.0}
        n = len(s)
        ascii_ct = sum(1 for c in s if c.isascii() and c.isalpha())
        digit_ct = sum(1 for c in s if c.isdigit())
        kata_ct = sum(1 for c in s if 0x30A1 <= ord(c) <= 0x30FA or ord(c) == 0x30FC)
        hira_ct = sum(1 for c in s if 0x3041 <= ord(c) <= 0x3096)
        return {
            "ascii": ascii_ct / n,
            "digit": digit_ct / n,
            "kata": kata_ct / n,
            "hira": hira_ct / n,
            "other": (n - ascii_ct - digit_ct - kata_ct - hira_ct) / n,
        }

    def estimate(self, reading: str, context: str = "") -> dict[str, float]:
        """Return per-domain weights for the input. Unavailable domains
        absent from the returned dict (caller treats missing as 0)."""
        r = self._char_class_ratios(reading)

        has_tech_hint = (
            any(h in reading for h in self.KATA_TECH_HINTS)
            or any(h in reading for h in self.HIRA_TECH_HINTS)
            or any(m in reading for m in self.HIRA_LOAN_MARKERS)
        )
        tech_signal = r["ascii"] >= 0.05 or r["kata"] >= 0.30 or has_tech_hint

        # Entity: ascii uppercase unlikely in reading (it's hiragana input), so
        # defer to context markers + common name morphology signals.
        entity_signal = False
        if context:
            for marker in ("氏", "市", "区", "町", "村", "県", "府", "都",
                          "駅", "線", "社", "部", "院", "党", "神", "寺",
                          "藩", "郡", "省", "庁", "山", "川", "島"):
                if marker in context:
                    entity_signal = True
                    break

        if tech_signal and entity_signal:
            profile = "tech_entity"
        elif tech_signal:
            profile = "tech"
        elif entity_signal:
            profile = "entity"
        else:
            profile = "generic"

        weights = self.PROFILES[profile].copy()
        # Mask out unavailable domains, renormalise residual mass to 'general'.
        residual = 0.0
        for d in list(weights.keys()):
            if d not in self.available:
                residual += weights[d]
                weights[d] = 0.0
        if residual > 0 and "general" in self.available:
            weights["general"] = weights.get("general", 0.0) + residual
        return weights
