"""Tests for models/tools/export/verify_onnx_pair.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

from models.tools.export.verify_onnx_pair import (
    EXPECTED_ARTIFACT_VERSION,
    verify,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_meta(
    dir_path: Path,
    *,
    proposal_data: bytes,
    refiner_data: bytes | None,
    artifact_version: str = EXPECTED_ARTIFACT_VERSION,
    override_proposal_sha: str | None = None,
    override_refiner_sha: str | None = None,
) -> Path:
    proposal_path = dir_path / "model.onnx"
    proposal_path.write_bytes(proposal_data)
    refiner_name = ""
    refiner_sha = ""
    if refiner_data is not None:
        refiner_path = dir_path / "model.refiner.onnx"
        refiner_path.write_bytes(refiner_data)
        refiner_name = refiner_path.name
        refiner_sha = override_refiner_sha or _sha(refiner_data)
    meta = dir_path / "model.meta.txt"
    meta.write_text(
        "\n".join(
            [
                f"artifact_version={artifact_version}",
                "preset=test",
                "vocab_size=100",
                "blank_id=4",
                "mask_id=5",
                "proposal_inputs=2",
                f"proposal_path={proposal_path.name}",
                f"proposal_sha256={override_proposal_sha or _sha(proposal_data)}",
                f"refiner_enabled={1 if refiner_data is not None else 0}",
                "refiner_inputs=4",
                f"refiner_path={refiner_name}",
                f"refiner_sha256={refiner_sha}",
                "step=1",
                "max_seq_len=128",
            ]
        ),
        encoding="utf-8",
    )
    return meta


def test_verify_accepts_matching_pair(tmp_path):
    meta = _write_meta(
        tmp_path,
        proposal_data=b"proposal-bytes",
        refiner_data=b"refiner-bytes",
    )
    assert verify(meta) == []


def test_verify_flags_proposal_sha_mismatch(tmp_path):
    meta = _write_meta(
        tmp_path,
        proposal_data=b"proposal-bytes",
        refiner_data=b"refiner-bytes",
        override_proposal_sha="deadbeef",
    )
    errs = verify(meta)
    assert any("proposal sha256 mismatch" in e for e in errs)


def test_verify_flags_missing_refiner_when_enabled(tmp_path):
    meta = _write_meta(
        tmp_path,
        proposal_data=b"proposal-bytes",
        refiner_data=b"refiner-bytes",
    )
    # Delete the refiner after writing meta.
    (tmp_path / "model.refiner.onnx").unlink()
    errs = verify(meta)
    assert any("refiner ONNX missing" in e for e in errs)


def test_verify_flags_stale_version(tmp_path):
    meta = _write_meta(
        tmp_path,
        proposal_data=b"p",
        refiner_data=None,
        artifact_version="0",
    )
    errs = verify(meta)
    assert any("artifact_version" in e for e in errs)
