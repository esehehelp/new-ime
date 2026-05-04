"""Short→long curriculum sampling for the warmup phase.

For the first `warmup_short_sample_steps` optimizer steps, the collator
filters out rows whose reading/surface length exceeds
`short_sample_max_chars`. After warmup the cap is lifted (set to 0).

Pre-v2 reported this stabilizes early CTC blank-collapse on long sequences.
The collator filter is implemented in `data/shards.py:CTCShardCollator`
(`short_sample_max_chars` field); this module supplies the step→cap policy
and is invoked from the loop's per-step hook in `training/run.py`.
"""

from __future__ import annotations

from new_ime.data.shards import CTCShardCollator


def apply_short_sample_warmup(
    collator: CTCShardCollator,
    *,
    step: int,
    warmup_steps: int,
    short_max_chars: int,
) -> None:
    """Set collator.short_sample_max_chars based on the current step."""
    if warmup_steps <= 0 or short_max_chars <= 0:
        collator.short_sample_max_chars = 0
        return
    collator.short_sample_max_chars = (
        short_max_chars if step < warmup_steps else 0
    )
