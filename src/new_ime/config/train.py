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
    use_cvae: bool = False
    # CVAE KL weight (only consumed when use_cvae=True).
    cvae_kl_weight: float = 0.1
    # arch is loop-agnostic dispatch tag. ar/dat models are not implemented
    # in v1.0 — the loop wires them via the same Protocol.
    arch: Literal["ctc-nat", "ar", "dat"] = "ctc-nat"


class DataSection(_Strict):
    train: Path
    dev: Path
    tokenizer: Path
    max_train_samples: int = 0  # 0 = unlimited
    max_dev_samples: int = 2000


class OptimSection(_Strict):
    lr: float
    warmup_steps: int
    schedule: Literal["cosine", "linear", "constant", "cosine_warm_restarts"] = "cosine"
    lr_min_ratio: float = 0.10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Used only when schedule == "cosine_warm_restarts".
    lr_restart_period: int = 80000
    lr_restart_decay: float = 0.9


class LoopSection(_Strict):
    batch_size: int
    eval_batch_size: int = 64
    grad_accum: int = 1
    max_steps: int
    fp16: bool = True
    bf16: bool = False
    compile: bool = False
    num_workers: int = 0  # Windows: must be 0 (memory: feedback_windows_num_workers)
    # Curriculum: for the first N optimizer steps, restrict batches to rows
    # whose reading/surface character length is <= short_sample_max_chars.
    warmup_short_sample_steps: int = 0
    short_sample_max_chars: int = 16
    # Debug: when > 0, sample only this many rows from the head of the
    # train loader and overfit on them. Useful for verifying loss can
    # actually drop on a small batch.
    tiny_overfit_samples: int = 0

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
    # Adaptive log cadence (port of Rust-train's early_log pattern). For the
    # first `early_log_steps` optimizer steps, log every `early_log_every`
    # steps so initial loss / rate / blank-fraction is visible at high
    # resolution. Set `early_log_every = 0` to disable (use only `log_every`).
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
    # best.pt selection key. "loss_neg" picks the lowest dev loss.
    metric_priority: Literal[
        "probe_em1", "exact_match_top1", "char_acc_top1", "loss_neg"
    ] = "probe_em1"


class RefineSection(_Strict):
    loss_weight: float = 0.7
    warmup_steps: int = 5000
    mask_ratio_min: float = 0.15
    mask_ratio_max: float = 0.35
    refine_iterations: int = 1
    # "target": mask + refill from gold target (current default).
    # "proposal": mask + refill from CTC argmax proposal (more realistic
    # but requires the proposal pass to have completed).
    refine_source: Literal["target", "proposal"] = "target"
    remask_loss_weight: float = 0.1
    stop_loss_weight: float = 0.1
    # GLAT-style hint leakage during training (Phase 1 γ). After building the
    # masked hypothesis, run a no-grad refine pass and identify positions
    # where the prediction misses the target; replace mask_token with the
    # oracle id at `glance_ratio` × miss_count of those positions before the
    # graded forward. Curriculum form `"0.3:0.1@40k"` (start:end@steps) — see
    # `training/loss/dat.parse_anneal`. `"0.0"` disables.
    glat_p: str = "0.0"
    glance_strategy: Literal["number-random", "none"] = "number-random"


class DatSection(_Strict):
    """DA-Transformer (DAT) hyperparameters. Optional — only consumed when
    `model.arch == "dat"`. Mutually exclusive with `[refine]` (which is
    CTC-NAT-specific)."""

    upsample_scale: int = 4
    glat_p: str = "0.5:0.1@100k"
    glance_strategy: Literal["number-random", "none"] = "number-random"
    max_transition_length: int = -1  # -1 = full square (only mode supported in v1.0)
    links_feature: Literal["feature", "feature:position"] = "feature:position"
    num_link_heads: int = 4
    loss_factor: float = 1.0
    label_smoothing: float = 0.0
    decode_strategy: Literal["greedy", "lookahead", "viterbi"] = "lookahead"
    decode_beta: float = 1.0
    decode_viterbibeta: float = 1.0
    decode_upsample_scale: float = 4.0


class DeepSupervisionSection(_Strict):
    """Auxiliary CTC heads on intermediate decoder layers (Phase 1 β).

    Each `layers[i]` is a 0-indexed decoder layer at which to apply the
    shared `ctc_head` (after `final_norm`) and add a CTC loss with weight
    `weights[i]`. Setting `layers=[]` (default) disables the path entirely
    — model ignores the capture argument and inference cost is unchanged.

    The heads themselves are NOT new parameters: we reuse the main
    `ctc_head` (which is tied to the encoder embedding) so deep
    supervision adds zero learnable params and only ~30% training cost.
    """

    layers: list[int] = []
    weights: list[float] = []
    warmup_steps: int = 0

    @field_validator("weights")
    @classmethod
    def _len_match(cls, v: list[float], info) -> list[float]:
        layers = info.data.get("layers", [])
        if v and len(v) != len(layers):
            raise ValueError(
                f"deep_supervision.weights ({len(v)}) must match layers ({len(layers)})"
            )
        return v


class KdSection(_Strict):
    teacher_type: Literal["ctc", "ar", "seq2seq"]
    teacher_path: Path
    # Required when teacher_type != "ctc" (different tokenizer than student).
    teacher_vocab: Optional[Path] = None
    alpha: float
    alpha_final: float = 0.0
    start_step: int = 0
    warmup_steps: int = 0
    alpha_decay_start: int = 0
    alpha_decay_steps: int = 0
    every: int = 1
    # "all": no gating; "low_conf": teacher mean prob < threshold (focus on
    # the teacher's hard examples); "high_conf": ≥ threshold (stabilization).
    gate_mode: Literal["all", "low_conf", "high_conf"] = "all"
    hard_threshold: float = 0.5
    temperature: float = 1.0


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
    dat: Optional[DatSection] = None
    deep_supervision: Optional[DeepSupervisionSection] = None
