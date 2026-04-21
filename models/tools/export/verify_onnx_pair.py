"""Verify that a proposal/refiner ONNX pair matches its meta.txt sidecar.

The Rust runtime performs an equivalent check when loading a session.
Running this script in CI (or before uploading an artifact set) catches
mismatched exports — e.g. someone re-exported the proposal graph but
forgot to regenerate the refiner, leaving the meta.txt stale.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


EXPECTED_ARTIFACT_VERSION = "2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_meta(meta_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def verify(meta_path: Path) -> list[str]:
    errors: list[str] = []
    if not meta_path.exists():
        return [f"meta file not found: {meta_path}"]
    meta = parse_meta(meta_path)
    version = meta.get("artifact_version", "")
    if version != EXPECTED_ARTIFACT_VERSION:
        errors.append(
            f"artifact_version={version!r} but this verifier expects "
            f"{EXPECTED_ARTIFACT_VERSION!r}; regenerate the export or update the verifier"
        )

    proposal_name = meta.get("proposal_path", "")
    if not proposal_name:
        errors.append("proposal_path missing in meta")
    else:
        proposal_file = meta_path.parent / proposal_name
        if not proposal_file.exists():
            errors.append(f"proposal ONNX missing: {proposal_file}")
        else:
            observed = sha256_file(proposal_file)
            expected = meta.get("proposal_sha256", "")
            if observed != expected:
                errors.append(
                    f"proposal sha256 mismatch: file={observed} meta={expected}"
                )

    if meta.get("refiner_enabled", "0") == "1":
        refiner_name = meta.get("refiner_path", "")
        if not refiner_name:
            errors.append("refiner_enabled=1 but refiner_path is empty")
        else:
            refiner_file = meta_path.parent / refiner_name
            if not refiner_file.exists():
                errors.append(f"refiner ONNX missing: {refiner_file}")
            else:
                observed = sha256_file(refiner_file)
                expected = meta.get("refiner_sha256", "")
                if observed != expected:
                    errors.append(
                        f"refiner sha256 mismatch: file={observed} meta={expected}"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to .meta.txt")
    args = parser.parse_args()
    errors = verify(Path(args.meta))
    if errors:
        for e in errors:
            print(f"ERR: {e}")
        return 1
    print("OK: artifact pair integrity verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
