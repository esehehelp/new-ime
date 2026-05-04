"""AR cross-entropy (label smoothing). v1.1 — placeholder.

The AR model class itself has not been ported from pre-v2; once it lands,
this module should expose `build_ar_loss_fn(cfg) -> Callable` returning
the standard teacher-forced CE with optional label smoothing.
"""

from __future__ import annotations


def build_ar_loss_fn(*_args, **_kwargs):
    raise NotImplementedError(
        "AR loss is not implemented in v1.0; see the train/research/ar-arch "
        "sidebranch in plans/train-linear-blanket.md."
    )
