"""Training config schema. One TOML file = one experiment.

All fields are required unless marked Optional. Unknown fields raise
ValidationError to catch typos before training starts.

v2.5-train scope: CTC-NAT only, with optional refine + KD. DAT,
deep_supervision, GLAT, cosine_warm_restarts, curriculum, and CTC-teacher
KD have been removed; reintroduce in v2.6+ if a measured benefit is shown.
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
    use_cvae: bool = False
    cvae_kl_weight: float = 0.1
    # arch is preserved as a forward-compat dispatch tag. v2.5 only ships
    # ctc-nat. AR / DAT will reintroduce additional Literal options when
    # those archs land.
    arch: Literal["ctc-nat"] = "ctc-nat"


class DataSection(_Strict):
    train: Path  # JSONL path (no shards in v2.5)
    dev: Path  # JSONL path
    tokenizer: Path
    max_train_samples: int = 0  # 0 = unlimited (full file in RAM)
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
    num_workers: int = 0  # Windows + max_train_samples=0 → must be 0


class LoggingSection(_Strict):
    log_every: int = 500
    eval_every: int = 1000
    checkpoint_every: int = 10000
    keep_last_k: int = 5
    print_samples: int = 5
    # Adaptive log cadence: for the first `early_log_steps` optimizer steps,
    # log every `early_log_every` steps so initial loss / rate / blank-fraction
    # is visible at high resolution. Set `early_log_every = 0` to disable.
    early_log_every: int = 10
    early_log_steps: int = 200


class ResumeSection(_Strict):
    checkpoint: Path
    reset_optimizer: bool = False
    reset_scheduler: bool = False
    reset_best_metric: bool = False


class ProbeSection(_Strict):
    path: Path
    every: int = 10000
    limit: int = 0  # 0 = full
    metric_priority: Literal[
        "probe_em1", "exact_match_top1", "char_acc_top1", "loss_neg"
    ] = "probe_em1"


class RefineSection(_Strict):
    """Mask-CTC refine head training (CTC-NAT only)."""

    loss_weight: float = 0.7
    warmup_steps: int = 5000
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.35
    # "target": mask + refill from gold target. "proposal": mask + refill from
    # the CTC argmax proposal (more realistic but requires a no-grad proposal
    # forward pass).
    refine_source: Literal["target", "proposal"] = "target"
    remask_loss_weight: float = 0.1
    stop_loss_weight: float = 0.1


class KdSection(_Strict):
    """Text-roundtrip knowledge distillation.

    The teacher generates `texts` from `(contexts, readings)`; the student
    is then trained with an additional CTC loss against the teacher text
    (gated by teacher confidence).

    Two teacher backends, dispatched by `teacher_type`:
        "zenz" → HuggingFace GPT-2 directory (e.g. references/zenz-v3.1-small).
                 teacher_path = directory; teacher_vocab/hidden/etc unused.
        "ar"   → SimpleGPT2 + char-level vocab (.pt + _vocab.json).
                 Requires teacher_vocab + teacher_hidden/layers/heads/max_seq_len.
    """

    teacher_type: Literal["zenz", "ar"] = "zenz"
    teacher_path: Path
    teacher_vocab: Optional[Path] = None
    teacher_hidden: int = 512
    teacher_layers: int = 8
    teacher_heads: int = 8
    teacher_max_seq_len: int = 192
    max_new_tokens: int = 48

    alpha: float
    alpha_final: float = 0.0
    start_step: int = 0
    warmup_steps: int = 0
    alpha_decay_start: int = 0
    alpha_decay_steps: int = 0
    # Run KD every N optimizer steps (1 = every step). Higher reduces
    # teacher inference cost.
    every: int = 4
    # "all": no gating. "low_conf": teacher mean prob < threshold (focus on
    # teacher's hard examples). "high_conf": ≥ threshold (stabilization).
    gate_mode: Literal["all", "low_conf", "high_conf"] = "low_conf"
    hard_threshold: float = 0.95


class TrainConfig(_Strict):
    run: RunSection
    model: ModelSection
    data: DataSection
    optim: OptimSection
    loop: LoopSection
    logging: LoggingSection = LoggingSection()
    resume: Optional[ResumeSection] = None
    probe: Optional[ProbeSection] = None
    refine: Optional[RefineSection] = None
    kd: Optional[KdSection] = None
