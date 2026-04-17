from src.data.curriculum_sampler import (
    CurriculumSampler,
    CurriculumStage,
    build_phase3_curriculum,
)


def test_stage_lookup():
    sampler = build_phase3_curriculum()
    assert sampler.stage_for_step(0).name == "S0_warmup"
    assert sampler.stage_for_step(10_000).name == "S1_base"
    assert sampler.stage_for_step(300_000).name == "S5_qat"


def test_weights_normalize():
    sampler = build_phase3_curriculum()
    weights = sampler.normalized_weights_for_step(50_000)
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    assert set(weights) == {"P1", "P2", "P3", "P4"}


def test_kd_mask_respects_fraction():
    sampler = CurriculumSampler(
        [
            CurriculumStage(
                name="only",
                start_step=0,
                end_step=100,
                pool_weights={"P1": 1.0},
                kd_fraction=1.0,
            )
        ],
        seed=123,
    )
    mask = sampler.kd_mask_for_batch(step=0, batch_size=8)
    assert all(mask)


def test_invalid_stage_overlap_raises():
    try:
        CurriculumSampler(
            [
                CurriculumStage("a", 0, 10, {"P1": 1.0}),
                CurriculumStage("b", 9, 20, {"P1": 1.0}),
            ]
        )
    except ValueError:
        return
    assert False, "Expected overlapping curriculum stages to raise"
