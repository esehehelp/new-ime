import argparse
import json
from pathlib import Path

import pytest

from src.data.tokenizer import SharedCharTokenizer
from src.training.train_ctc_nat import (
    build_tokenizer,
    build_model,
    estimate_training_memory,
    validate_resume_compatibility,
)


def test_estimate_training_memory_for_30m():
    tokenizer = SharedCharTokenizer(max_kanji=6000)
    model = build_model("phase3_30m", vocab_size=tokenizer.vocab_size, use_cvae=False)
    estimate = estimate_training_memory(
        model,
        preset_name="phase3_30m",
        batch_size=4,
        seq_len=64,
        fp16=True,
        use_adamw=True,
    )
    assert estimate.params_m > 20
    assert estimate.total_gb > estimate.param_gb


def test_estimate_training_memory_for_20m():
    tokenizer = SharedCharTokenizer(max_kanji=6000)
    model = build_model("phase3_20m", vocab_size=tokenizer.vocab_size, use_cvae=False)
    estimate = estimate_training_memory(
        model,
        preset_name="phase3_20m",
        batch_size=4,
        seq_len=64,
        fp16=True,
        use_adamw=True,
    )
    assert 15 < estimate.params_m < 25


def test_estimate_training_memory_for_90m_with_cvae():
    tokenizer = SharedCharTokenizer(max_kanji=6000)
    model = build_model("phase3_90m", vocab_size=tokenizer.vocab_size, use_cvae=True)
    estimate = estimate_training_memory(
        model,
        preset_name="phase3_90m",
        batch_size=2,
        seq_len=128,
        fp16=True,
        use_adamw=True,
    )
    assert estimate.params_m > 80
    assert estimate.activation_gb > 0


def test_validate_resume_compatibility_ok():
    args = argparse.Namespace(
        preset="phase3_20m",
        use_cvae=False,
        max_seq_len=128,
        max_kanji=6000,
        tokenizer_path="",
    )
    checkpoint = {
        "preset": "phase3_20m",
        "use_cvae": False,
        "max_seq_len": 128,
        "max_kanji": 6000,
        "vocab_size": 256,
    }
    tokenizer = SharedCharTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "[SEP]": 2, "[CLS]": 3, "[BLANK]": 4, "[MASK]": 5} | {f"<0x{b:02X}>": 6 + b for b in range(250)})
    validate_resume_compatibility(checkpoint, args, tokenizer=tokenizer)


def test_validate_resume_compatibility_raises_on_mismatch():
    args = argparse.Namespace(
        preset="phase3_20m",
        use_cvae=False,
        max_seq_len=128,
        max_kanji=6000,
        tokenizer_path="",
    )
    checkpoint = {
        "preset": "phase3_30m",
        "use_cvae": False,
        "max_seq_len": 128,
        "max_kanji": 6000,
    }
    with pytest.raises(ValueError):
        validate_resume_compatibility(checkpoint, args)


def test_validate_resume_compatibility_raises_on_kd_drift():
    """Resume must fail if KD hyperparameters differ from the checkpoint."""
    args = argparse.Namespace(
        preset="phase3_20m",
        use_cvae=False,
        max_seq_len=128,
        max_kanji=6000,
        tokenizer_path="",
        kd_teacher_path="checkpoints/ar_v3_vast/best.pt",
        kd_teacher_vocab="",
        kd_alpha=0.5,
        kd_hard_threshold=0.6,
        kd_start_step=0,
        kd_warmup_steps=0,
        kd_every=4,
        kd_max_new_tokens=96,
    )
    checkpoint = {
        "preset": "phase3_20m",
        "use_cvae": False,
        "max_seq_len": 128,
        "max_kanji": 6000,
        "kd": {
            "teacher_path": "checkpoints/ar_baseline/best.pt",  # different teacher
            "teacher_vocab": "",
            "alpha": 0.5,
            "hard_threshold": 0.6,
            "start_step": 0,
            "warmup_steps": 0,
            "every": 4,
            "max_new_tokens": 96,
        },
    }
    with pytest.raises(ValueError, match="kd.teacher_path"):
        validate_resume_compatibility(checkpoint, args)


def test_validate_resume_compatibility_accepts_matching_kd():
    args = argparse.Namespace(
        preset="phase3_20m",
        use_cvae=False,
        max_seq_len=128,
        max_kanji=6000,
        tokenizer_path="",
        kd_teacher_path="checkpoints/ar_v3_vast/best.pt",
        kd_teacher_vocab="",
        kd_alpha=0.3,
        kd_hard_threshold=0.7,
        kd_start_step=1000,
        kd_warmup_steps=2000,
        kd_every=4,
        kd_max_new_tokens=96,
    )
    checkpoint = {
        "preset": "phase3_20m",
        "use_cvae": False,
        "max_seq_len": 128,
        "max_kanji": 6000,
        "kd": {
            "teacher_path": "checkpoints/ar_v3_vast/best.pt",
            "teacher_vocab": "",
            "alpha": 0.3,
            "hard_threshold": 0.7,
            "start_step": 1000,
            "warmup_steps": 2000,
            "every": 4,
            "max_new_tokens": 96,
        },
    }
    validate_resume_compatibility(checkpoint, args)


def test_build_tokenizer_from_path(tmp_path: Path):
    tokenizer = SharedCharTokenizer(max_kanji=32)
    tokenizer_path = tmp_path / "shared_tokenizer.json"
    tokenizer.save(tokenizer_path)
    args = argparse.Namespace(tokenizer_path=str(tokenizer_path), max_kanji=9999)
    loaded = build_tokenizer(args)
    assert loaded.vocab_size == tokenizer.vocab_size
