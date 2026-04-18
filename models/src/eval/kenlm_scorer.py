"""KenLM-backed PrefixLMScorer for CTC prefix beam shallow fusion.

Assumes the KenLM model was trained on character-level corpus where each
line is a Japanese sentence with one ASCII space between every character
(e.g. ``"消 防 職 員"``) so that KenLM's whitespace-split tokenization
yields one token per glyph. `score(prefix)` decodes token ids to chars
via the tokenizer and returns the full-prefix natural-log probability.

Scores are cached per prefix tuple so each beam extension only does one
KenLM full-sentence re-score — cheap in practice because most prefixes
differ by a single trailing char.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple


class KenLMCharScorer:
    def __init__(self, model_path: str, tokenizer) -> None:
        try:
            import kenlm  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "kenlm is not installed in this environment. "
                "On Windows the native build is finicky; use WSL or the "
                "kenlm wheel built for the active interpreter."
            ) from e
        self.model = kenlm.Model(model_path)
        self.tokenizer = tokenizer
        # `score` is not a free call (C++ bindings + UTF-8 conv), so cache.
        # Size bounded by beam * seq_len * depth; 2**15 is plenty.
        self._cache: dict[Tuple[int, ...], float] = {}

    def score(self, prefix: Tuple[int, ...]) -> float:
        cached = self._cache.get(prefix)
        if cached is not None:
            return cached
        chars = self.tokenizer.decode(list(prefix))
        spaced = " ".join(chars)
        # KenLM returns log10-prob; convert to natural log so it composes
        # with CTC logp (which is in natural log from log_softmax).
        log10 = self.model.score(spaced, bos=True, eos=False)
        natlog = log10 * math.log(10.0)
        self._cache[prefix] = natlog
        return natlog
