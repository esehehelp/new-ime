"""Architecture-grouped loss helpers.

    ctc.py    — CTC loss + blank-fraction utility (CTC family in general)
    refine.py — mask-CTC refine 3-part loss (CTC-NAT)
    kd.py     — KL / CTC teacher distillation (logits-based, arch-agnostic)
    ar.py     — AR cross-entropy (skeleton; v1.1)
    dat.py    — DAT alignment loss (skeleton; v1.1)

The loop calls these via callbacks plumbed through `training/run.py`,
keeping `training/loop.py` unaware of any specific architecture.
"""
