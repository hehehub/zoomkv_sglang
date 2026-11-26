"""JIT loader for the Quest chunk-score kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from sglang_zoomkv.kernels.jit import load_jit


@lru_cache(maxsize=1)
def load_kernel_module() -> Any:
    return load_jit(
        "quest_ext",
        "quest.cu",
        source_dir=Path(__file__).parent,
        cflags=["-O3", "-std=c++17"],
        cuda_flags=["-O3", "-std=c++17", "--use_fast_math"],
        with_cuda=True,
    )
