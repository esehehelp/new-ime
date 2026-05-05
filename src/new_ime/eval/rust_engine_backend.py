"""ConversionBackend that drives the Rust engine daemon over JSONL stdio.

Spawns `new-ime-engine-cli daemon` once per backend instance, holds the
subprocess for the lifetime of the bench run, and serialises one request
per `convert()` call. Suiko 推論は Rust runtime に閉じ、Python 側は
orchestration と metrics に専念する。

Protocol:
    Startup (daemon → us): {"ready": true, "version": "...",
                            "artifact_format": "onnx-fp32"|"onnx-int8",
                            "beam_width": int, "top_k": int}
    Request  (us → daemon): {"id": int, "context": str, "reading": str}
    Response (daemon → us): {"id": int, "candidates": [str], "engine_ms": float}
                             ↳ on error: includes "error": str
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import count
from pathlib import Path
from typing import List, Optional

from new_ime.config.bench import BenchConfig


_DEFAULT_BIN_REL = ("build", "release", "new-ime-engine-cli")
_DEFAULT_BIN_REL_DEBUG = ("build", "debug", "new-ime-engine-cli")


def _resolve_engine_binary(cfg: BenchConfig) -> Path:
    # Priority: explicit cfg.engine.binary > env > release > debug.
    if cfg.engine is not None and cfg.engine.binary is not None:
        return Path(cfg.engine.binary)
    env = os.environ.get("NEW_IME_ENGINE_BIN")
    if env:
        return Path(env)
    exe = ".exe" if os.name == "nt" else ""
    repo_root = Path(__file__).resolve().parents[3]
    release = repo_root.joinpath(*_DEFAULT_BIN_REL).with_suffix(exe)
    if release.exists():
        return release
    debug = repo_root.joinpath(*_DEFAULT_BIN_REL_DEBUG).with_suffix(exe)
    return debug


class RustEngineBackend:
    """ConversionBackend backed by the Rust daemon binary."""

    def __init__(self, cfg: BenchConfig) -> None:
        assert cfg.model.type == "ctc-nat"
        self.name: str = cfg.run.name
        self.runtime: str = "rust-engine-daemon"
        self.artifact_format: str = ""  # filled from ready handshake

        bin_path = _resolve_engine_binary(cfg)
        if not bin_path.exists():
            raise FileNotFoundError(
                f"Rust engine binary not found at {bin_path}. "
                "Build with `cargo build --release -p new-ime-engine-cli` "
                "or set NEW_IME_ENGINE_BIN."
            )

        beam_width = (
            1 if cfg.decode.mode == "greedy" else int(cfg.decode.num_beams)
        )
        top_k = max(int(cfg.decode.top_k), int(cfg.decode.num_return), 1)

        # Determine artifact format from filename suffix; user can override
        # via cfg.engine.artifact_format if a non-conventional name is used.
        onnx = Path(cfg.model.onnx)
        if cfg.engine is not None and cfg.engine.artifact_format is not None:
            af = cfg.engine.artifact_format
        elif ".int8" in onnx.name:
            af = "int8"
        else:
            af = "fp32"

        argv: List[str] = [
            str(bin_path),
            "daemon",
            "--onnx", str(onnx),
            "--artifact-format", af,
            "--beam-width", str(beam_width),
            "--top-k", str(top_k),
        ]
        if cfg.model.vocab is not None:
            argv += ["--vocab", str(cfg.model.vocab)]
        if cfg.lm is not None:
            argv += [
                "--kenlm-alpha", str(cfg.lm.alpha),
                "--kenlm-beta", str(cfg.lm.beta),
            ]
            if cfg.lm.mode == "single":
                argv += ["--kenlm", str(cfg.lm.path)]
            else:
                for dom, path in (cfg.lm.paths_by_domain or {}).items():
                    argv += ["--kenlm-domain", f"{dom}={path}"]

        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None

        # Read the ready handshake. Daemon emits exactly one ready line
        # before any request response, so a single readline is sufficient.
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            raise RuntimeError("Rust engine daemon exited before sending ready")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Rust engine daemon emitted non-JSON ready line: {ready_line!r}"
            ) from e
        if not ready.get("ready"):
            raise RuntimeError(f"Rust engine daemon failed to start: {ready!r}")
        self.artifact_format = ready.get("artifact_format", f"onnx-{af}")

        self._ids = count(start=1)
        self._last_engine_ms: float = 0.0

    @property
    def last_engine_ms(self) -> float:
        return self._last_engine_ms

    def convert(self, reading: str, context: str) -> List[str]:
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"Rust engine daemon has exited (code={self._proc.returncode})"
            )
        req_id = next(self._ids)
        payload = json.dumps(
            {"id": req_id, "context": context, "reading": reading},
            ensure_ascii=False,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Rust engine daemon closed stdout mid-request")
        resp = json.loads(line)
        if resp.get("id") != req_id:
            raise RuntimeError(
                f"Rust engine daemon response id mismatch: "
                f"sent {req_id}, got {resp.get('id')}"
            )
        if resp.get("error"):
            raise RuntimeError(f"Rust engine error: {resp['error']}")
        self._last_engine_ms = float(resp.get("engine_ms", 0.0))
        return list(resp.get("candidates", []))

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()

    def __enter__(self) -> "RustEngineBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort: avoid leaking the daemon if the caller forgot close().
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
