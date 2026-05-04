"""v2 training subpackage.

Module map (target structure across all stages):
    run        — top-level entry: device / tokenizer / model / loader / loop
    loop       — step / microbatch / grad_accum / AMP / eval & ckpt rhythm
    optim      — optimizer + LambdaLR scheduler factories
    checkpoint — save / load / validate_resume / rolling_keep
    evaluate   — dev loss & metrics, probe EM1 (Stage 2+)
    refine     — refinement helpers (Stage 3, kept under loss/)
    memory     — VRAM / RAM accounting (Stage 4)
    curriculum — short→long sample annealing (Stage 4)
    loss/*     — arch-grouped loss helpers (ctc, refine, kd, ar, dat)
    teacher/*  — KD teacher Protocol + concrete teachers (Stage 5)

The loop is model-agnostic: it pulls outputs["loss"] (and aux losses via
model.compute_aux_losses) and never imports a specific arch class.
"""
