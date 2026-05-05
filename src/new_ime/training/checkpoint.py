"""Checkpoint save / load / rolling-keep.

The checkpoint blob is arch-agnostic: model-specific fields (preset,
vocab_size, use_cvae, arch_tag) are pulled from `model.checkpoint_metadata()`
so adding new architectures (AR / DAT) requires no change here.

A tokenizer sidecar `<ckpt_stem>_tokenizer.json` is written next to the
.pt blob. The export pipeline (scripts/export_onnx.py) and the resume
path both rely on this exact naming.

`rolling_keep(out_dir, keep_last_k)` deletes old `checkpoint_step_<N>.pt`
(and their tokenizer sidecars) when more than `keep_last_k` exist. best.pt
and final.pt are never touched; keep_last_k=0 disables the policy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

_CKPT_STEP_RE = re.compile(r"^checkpoint_step_(\d+)\.pt$")


def _tokenizer_sidecar(ckpt_path: Path) -> Path:
    return ckpt_path.with_name(ckpt_path.stem + "_tokenizer.json")


def save(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    epoch: int,
    best_metric: float,
    tokenizer: Any,
    extra: dict | None = None,
) -> None:
    """Write a checkpoint .pt + tokenizer sidecar.

    `extra` is merged into the top-level blob (e.g. KD config snapshot).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arch_meta = (
        model.checkpoint_metadata()
        if hasattr(model, "checkpoint_metadata")
        else {}
    )
    blob: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "step": int(step),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    blob.update(arch_meta)
    if extra:
        blob.update(extra)

    torch.save(blob, path)
    tokenizer.save(_tokenizer_sidecar(path))


def load(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    reset_scheduler: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load weights into model and optionally optimizer/scheduler. Returns the blob."""
    path = Path(path)
    blob = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in blob:
        optimizer.load_state_dict(blob["optimizer_state_dict"])
    if (
        scheduler is not None
        and blob.get("scheduler_state_dict") is not None
        and not reset_scheduler
    ):
        scheduler.load_state_dict(blob["scheduler_state_dict"])
    return blob


def validate_resume_compatibility(
    blob: dict,
    model: torch.nn.Module,
) -> None:
    """Strict check: ckpt arch metadata must match the freshly built model.

    Compares the keys returned by `model.checkpoint_metadata()` against the
    same keys stored in the checkpoint blob. A mismatch raises ValueError so
    a stale resume cannot silently load the wrong vocab / arch / preset.
    """
    expected = (
        model.checkpoint_metadata()
        if hasattr(model, "checkpoint_metadata")
        else {}
    )
    mismatches: list[str] = []
    for key, current in expected.items():
        if key not in blob:
            continue
        ckpt_val = blob[key]
        if ckpt_val != current:
            mismatches.append(f"{key}: ckpt={ckpt_val!r} current={current!r}")
    if mismatches:
        raise ValueError(
            "Resume checkpoint incompatible with current model: "
            + "; ".join(mismatches)
        )


def rolling_keep(out_dir: str | Path, keep_last_k: int) -> None:
    """Delete old numbered checkpoints, keeping only the last K (by step).

    `best.pt`, `final.pt` and any non-matching files are left alone.
    Tokenizer sidecars are removed alongside their .pt.
    """
    if keep_last_k <= 0:
        return
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return
    entries: list[tuple[int, Path]] = []
    for child in out_dir.iterdir():
        m = _CKPT_STEP_RE.match(child.name)
        if m:
            entries.append((int(m.group(1)), child))
    if len(entries) <= keep_last_k:
        return
    entries.sort(key=lambda x: x[0])
    to_delete = entries[: len(entries) - keep_last_k]
    for _step, pt_path in to_delete:
        for p in (pt_path, _tokenizer_sidecar(pt_path)):
            try:
                os.remove(p)
            except OSError:
                pass
