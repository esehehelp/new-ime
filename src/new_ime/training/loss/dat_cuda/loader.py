"""JIT loader for the DAT CUDA kernel extension.

`load_dat_kernel()` returns the compiled `torch.utils.cpp_extension` module
the first time it is called, caching the result for subsequent calls. On
build failure (CUDA toolkit missing, MSVC version mismatch on Windows,
unsupported GPU arch, etc.) it logs a single warning and returns `None`,
letting the dispatcher in `dat_dp.py` fall back to the pure-PyTorch path.

The compiled extension lives under PyTorch's standard cache location
(`~/.cache/torch_extensions/<TORCH_VERSION>/<NAME>/` on Linux,
`%LOCALAPPDATA%\\torch_extensions\\...` on Windows). Override with
`TORCH_EXTENSIONS_DIR` if that path is read-only (some HPC envs).

Build flags follow the reference implementation
(`references/DA-Transformer/fs_plugins/custom_ops/dag_loss.py:36-63`):
    -DOF_SOFTMAX_USE_FAST_MATH -O3
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"

_KERNEL = None
_BUILD_FAILED = False
_LOCK = threading.Lock()


def _verbose() -> bool:
    return bool(os.environ.get("NEW_IME_DAT_VERBOSE"))


def _force_pytorch() -> bool:
    return bool(os.environ.get("NEW_IME_DAT_FORCE_PYTORCH"))


def _ensure_msvc_env() -> bool:
    """On Windows, populate `os.environ` with the MSVC build env (vcvars64).

    `torch.utils.cpp_extension.load()` calls `cl` to verify the host
    compiler ABI even when the build is cached, so any process that
    imports the kernel must have MSVC on PATH. We locate vcvars64.bat via
    `vswhere.exe`, run it in a subshell, and copy the resulting env vars
    into the current process. Idempotent: if `cl` is already reachable,
    we skip the subprocess hop.

    Returns True if cl is now (or was already) reachable, False otherwise.
    """
    import shutil
    import subprocess

    if shutil.which("cl") is not None:
        return True

    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if not vswhere.exists():
        if _verbose():
            print(f"[dat_cuda] vswhere.exe not found at {vswhere}", file=sys.stderr)
        return False

    try:
        vs_install = subprocess.check_output(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return False
    if not vs_install:
        return False

    vcvars = Path(vs_install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        return False

    # Run vcvars64.bat in cmd, then `set` to dump env, parse and inject.
    try:
        out = subprocess.check_output(
            f'cmd /s /c "call "{vcvars}" >nul && set"',
            shell=True, text=True,
        )
    except subprocess.CalledProcessError:
        return False

    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v

    if _verbose():
        print(f"[dat_cuda] loaded MSVC env from {vcvars}", file=sys.stderr)
    return shutil.which("cl") is not None


def _ensure_cuda_env() -> None:
    """Set CUDA_PATH / CUDA_HOME from the standard install location.

    PyTorch's `_join_cuda_home()` raises if neither var is set, so we
    auto-detect the latest installed toolkit on Windows. No-op on Linux
    (where CUDA is usually on PATH already) and when the env is set.
    """
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return
    if not _IS_WINDOWS:
        return
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not base.exists():
        return
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda p: tuple(int(x) for x in p.name.lstrip("v").split(".") if x.isdigit()),
    )
    if not candidates:
        return
    latest = candidates[-1]
    os.environ["CUDA_PATH"] = str(latest)
    os.environ["CUDA_HOME"] = str(latest)
    if _verbose():
        print(f"[dat_cuda] auto-detected CUDA_PATH={latest}", file=sys.stderr)


def load_dat_kernel():
    """Return the compiled CUDA extension, or `None` if unavailable.

    Thread-safe: the first concurrent caller triggers the build; others
    wait on the lock and return the cached result. Subsequent failures
    short-circuit immediately (no retry until process restart) so we
    don't spam warnings every step.
    """
    global _KERNEL, _BUILD_FAILED

    if _KERNEL is not None:
        return _KERNEL
    if _BUILD_FAILED:
        return None
    if _force_pytorch():
        return None

    with _LOCK:
        if _KERNEL is not None:
            return _KERNEL
        if _BUILD_FAILED:
            return None

        try:
            import torch
        except ImportError:
            _BUILD_FAILED = True
            return None

        if not torch.cuda.is_available():
            if _verbose():
                print("[dat_cuda] CUDA not available, skipping kernel build", file=sys.stderr)
            _BUILD_FAILED = True
            return None

        _ensure_cuda_env()

        if _IS_WINDOWS and not _ensure_msvc_env():
            print(
                "[dat_cuda] cl.exe not on PATH and MSVC auto-detect failed; "
                "falling back to pure-PyTorch DP. Install Visual Studio "
                "Build Tools or run from 'x64 Native Tools Command Prompt'.",
                file=sys.stderr,
            )
            _BUILD_FAILED = True
            return None

        try:
            from torch.utils.cpp_extension import load
        except ImportError:
            _BUILD_FAILED = True
            return None

        kernels_dir = Path(__file__).parent / "_kernels"
        sources = [
            str(kernels_dir / "dag_loss.cpp"),
            str(kernels_dir / "dag_loss.cu"),
            str(kernels_dir / "dag_best_alignment.cu"),
            str(kernels_dir / "logsoftmax_gather.cu"),
        ]

        for src in sources:
            if not Path(src).exists():
                print(
                    f"[dat_cuda] WARNING: kernel source missing: {src} "
                    "(falling back to PyTorch DP)",
                    file=sys.stderr,
                )
                _BUILD_FAILED = True
                return None

        if _verbose():
            print(
                "[dat_cuda] compiling CUDA kernels (first-time build, "
                "~60-180 sec on a fresh cache)...",
                file=sys.stderr,
                flush=True,
            )

        # CUDA 13's CCCL headers require MSVC's standard-conforming preprocessor
        # (/Zc:preprocessor); without it CUB/thrust headers fail with "expected
        # a {" everywhere. With /Zc:preprocessor alone, torch's compiled_autograd.h
        # hits C2872 'std' ambiguous, so we mirror PyTorch's own build flags
        # (/permissive- /EHsc /bigobj) which resolve the ambiguity.
        # See torch/share/cmake/Caffe2/public/utils.cmake:355-371.
        extra_cflags = ["-DOF_SOFTMAX_USE_FAST_MATH", "-O3"]
        extra_cuda_cflags = ["-DOF_SOFTMAX_USE_FAST_MATH", "-O3"]
        if _IS_WINDOWS:
            msvc_host_flags = ["/Zc:preprocessor", "/permissive-", "/EHsc", "/bigobj"]
            extra_cflags.extend(msvc_host_flags)
            for flag in msvc_host_flags:
                extra_cuda_cflags.extend(["-Xcompiler", flag])

        try:
            _KERNEL = load(
                name="new_ime_dat_kernel",
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=_verbose(),
            )
        except Exception as e:
            print(
                f"[dat_cuda] CUDA kernel build failed; falling back to "
                f"pure-PyTorch DP. Set NEW_IME_DAT_VERBOSE=1 to see the "
                f"build log. Error: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            _BUILD_FAILED = True
            return None

        if _verbose():
            print("[dat_cuda] kernel ready", file=sys.stderr)

        return _KERNEL
