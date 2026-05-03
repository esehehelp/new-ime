"""Training config schema. One TOML file = one experiment.

All fields are required unless marked Optional. Unknown fields raise
ValidationError to catch typos before training starts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSection(_Strict):
    name: str
    seed: int = 52
    out_dir: Path  # checkpoints + logs land here


class ModelSection(_Strict):
    # Preset names match the legacy phase3_* labels stored in checkpoints.
    preset: Literal["phase3_20m", "phase3_30m", "phase3_90m"]
    max_seq_len: int = 128
    max_context: int = 32


class DataSection(_Strict):
    train: Path
    dev: Path
    tokenizer: Path
    max_train_samples: int = 0  # 0 = unlimited
    max_dev_samples: int = 2000


class OptimSection(_Strict):
    lr: float
    warmup_steps: int
    schedule: Literal["cosine", "linear", "constant"] = "cosine"
    lr_min_ratio: float = 0.10
    weight_decay: float = 0.01
    grad_clip: float = 1.0


class LoopSection(_Strict):
    batch_size: int
    eval_batch_size: int = 64
    grad_accum: int = 1
    max_steps: int
    fp16: bool = True
    bf16: bool = False
    compile: bool = False
    num_workers: int = 0  # Windows: must be 0 (memory: feedback_windows_num_workers)

    @field_validator("num_workers")
    @classmethod
    def _windows_zero(cls, v: int) -> int:
        # Allow >0 explicitly; warn at runtime via the trainer.
        return v


class LoggingSection(_Strict):
    log_every: int = 500
    eval_every: int = 1000
    checkpoint_every: int = 10000
    keep_last_k: int = 5
    print_samples: int = 5


class ProbeSection(_Strict):
    path: Path
    every: int = 10000
    limit: int = 0  # 0 = full


class RefineSection(_Strict):
    loss_weight: float = 0.7
    warmup_steps: int = 5000
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.35


class KdSection(_Strict):
    teacher_type: Literal["ctc", "ar"]
    teacher_path: Path
    alpha: float
    alpha_final: float = 0.0
    start_step: int = 0
    warmup_steps: int = 0
    alpha_decay_start: int = 0
    alpha_decay_steps: int = 0
    every: int = 1
    gate_mode: Literal["all", "hard"] = "all"


class TrainConfig(_Strict):
    run: RunSection
    model: ModelSection
    data: DataSection
    optim: OptimSection
    loop: LoopSection
    logging: LoggingSection = LoggingSection()
    probe: Optional[ProbeSection] = None
    refine: Optional[RefineSection] = None
    kd: Optional[KdSection] = None
