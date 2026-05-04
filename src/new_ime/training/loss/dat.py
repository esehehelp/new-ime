"""DAT alignment loss. v1.1 — placeholder.

Once a DAT model class lands, this module should expose
`build_dat_loss_fn(cfg) -> Callable` returning the alignment objective.
"""

from __future__ import annotations


def build_dat_loss_fn(*_args, **_kwargs):
    raise NotImplementedError(
        "DAT loss is not implemented in v1.0; see the train/research/dat-arch "
        "sidebranch in plans/train-linear-blanket.md."
    )
