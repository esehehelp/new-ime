"""Curriculum sampling utilities for Phase 3 data-pool mixing.

The sampler operates on abstract pool names and weights so it can be tested
before the real datasets are wired up. Training code can later map pool names
to actual Dataset / DataPipe objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class CurriculumStage:
    """One stage in the Phase 3 pool-mixing curriculum."""

    name: str
    start_step: int
    end_step: int
    pool_weights: dict[str, float]
    kd_fraction: float = 0.0

    def contains(self, step: int) -> bool:
        return self.start_step <= step < self.end_step


class CurriculumSampler:
    """Chooses pools according to the active curriculum stage.

    The sampler is intentionally small and deterministic-under-seed so the
    schedule can be unit-tested independently of the eventual dataloader.
    """

    def __init__(self, stages: list[CurriculumStage], seed: int = 42) -> None:
        if not stages:
            raise ValueError("At least one curriculum stage is required.")
        self.stages = sorted(stages, key=lambda s: s.start_step)
        self._validate()
        self.rng = random.Random(seed)

    def _validate(self) -> None:
        previous_end = None
        for stage in self.stages:
            if stage.start_step >= stage.end_step:
                raise ValueError(f"Stage {stage.name} must satisfy start_step < end_step.")
            if previous_end is not None and stage.start_step < previous_end:
                raise ValueError("Curriculum stages must not overlap.")
            total = sum(stage.pool_weights.values())
            if total <= 0:
                raise ValueError(f"Stage {stage.name} must have positive total pool weight.")
            if not 0.0 <= stage.kd_fraction <= 1.0:
                raise ValueError(f"Stage {stage.name} kd_fraction must be in [0, 1].")
            previous_end = stage.end_step

    def stage_for_step(self, step: int) -> CurriculumStage:
        for stage in self.stages:
            if stage.contains(step):
                return stage
        return self.stages[-1]

    def normalized_weights_for_step(self, step: int) -> dict[str, float]:
        stage = self.stage_for_step(step)
        total = sum(stage.pool_weights.values())
        return {name: weight / total for name, weight in stage.pool_weights.items()}

    def sample_pool(self, step: int) -> str:
        weights = self.normalized_weights_for_step(step)
        names = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(names, weights=probs, k=1)[0]

    def sample_batch_assignment(self, step: int, batch_size: int) -> list[str]:
        return [self.sample_pool(step) for _ in range(batch_size)]

    def kd_mask_for_batch(self, step: int, batch_size: int) -> list[bool]:
        stage = self.stage_for_step(step)
        return [self.rng.random() < stage.kd_fraction for _ in range(batch_size)]


def build_phase3_curriculum() -> CurriculumSampler:
    """Default schedule from the current Phase 3 research plan."""

    stages = [
        CurriculumStage(
            name="S0_warmup",
            start_step=0,
            end_step=10_000,
            pool_weights={"P1": 1.0},
            kd_fraction=0.0,
        ),
        CurriculumStage(
            name="S1_base",
            start_step=10_000,
            end_step=80_000,
            pool_weights={"P1": 0.60, "P2": 0.30, "P3": 0.05, "P4": 0.05},
            kd_fraction=0.20,
        ),
        CurriculumStage(
            name="S2_diverse",
            start_step=80_000,
            end_step=180_000,
            pool_weights={"P1": 0.40, "P2": 0.30, "P3": 0.20, "P4": 0.10},
            kd_fraction=0.30,
        ),
        CurriculumStage(
            name="S3_clean_finetune",
            start_step=180_000,
            end_step=230_000,
            pool_weights={"P1": 0.80, "P2": 0.20},
            kd_fraction=0.10,
        ),
        CurriculumStage(
            name="S4_cvae",
            start_step=230_000,
            end_step=300_000,
            pool_weights={"P1": 0.60, "P2": 0.20, "P3": 0.10, "P4": 0.10},
            kd_fraction=0.20,
        ),
        CurriculumStage(
            name="S5_qat",
            start_step=300_000,
            end_step=360_000,
            pool_weights={"P1": 0.70, "P2": 0.30},
            kd_fraction=0.10,
        ),
    ]
    return CurriculumSampler(stages)
