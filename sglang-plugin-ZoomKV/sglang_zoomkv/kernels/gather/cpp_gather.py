import os
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from sglang_zoomkv.kernels.jit import load_jit


@lru_cache(maxsize=2)
def load_gather_ext(verbose: bool = False) -> Any:
    """
    Build & load the CPU gather extension once per process.
    Set env var POLAR_GATHER_EXT_VERBOSE=1 to see build logs.
    """
    # Rely on PyTorch threadpool (at::parallel_for), no OpenMP required.
    return load_jit(
        "polar_gather_ext",
        "gather_cpu.cpp",
        source_dir=Path(__file__).parent,
        cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=verbose or (os.environ.get("POLAR_GATHER_EXT_VERBOSE", "0") == "1"),
    )

