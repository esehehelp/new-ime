"""CUDA-accelerated DAG loss kernels for DAT training.

The pure-PyTorch implementation in `new_ime.training.loss.dat_dp` is a
correctness reference but runs a 128-iteration Python DP loop per loss
call, which is unusably slow at production batch sizes. This package
provides a CUDA kernel port of the reference implementation
(`thu-coai/DA-Transformer`, Apache-2.0) and an autograd-aware Python
wrapper.

The dispatcher in `dat_dp.py` automatically selects this CUDA path when
all of the following hold:
    * `torch.cuda.is_available()`
    * the input tensors are on a CUDA device
    * the JIT extension built successfully on first import
    * the env var `NEW_IME_DAT_FORCE_PYTORCH` is unset

If any of these fail, the dispatcher transparently falls back to the
pure-PyTorch implementation. Set `NEW_IME_DAT_VERBOSE=1` to see the
JIT build log (useful for debugging Windows MSVC / CUDA toolkit
mismatches).
"""

from new_ime.training.loss.dat_cuda.loader import load_dat_kernel

__all__ = ["load_dat_kernel"]
