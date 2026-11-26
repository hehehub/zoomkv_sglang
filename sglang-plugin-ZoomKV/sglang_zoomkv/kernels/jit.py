"""Shared JIT loader for ZoomKV native extensions.

The kernels in this package are migrated from the ZoomKV
``cache_hub/{kivi,rerank,topk,gather,gather_trans,lse_merge}`` source tree.

The upstream files did not include per-file license headers. Their original
implementation and comments are kept in the neighboring source/wrapper files.
"""

from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

# One compiled extension per logical name for the life of the process.
# Hot paths used to call load_jit on every kernel launch; that re-stated the
# .cu file and re-entered the loader even when PyTorch's own cache would hit.
_loaded_modules: dict[str, Any] = {}
_loaded_lock = threading.Lock()


def _as_tuple(value: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(value or ())
    return result or default


def _source_signature(sources: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for source in sources:
        path = Path(source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ZoomKV JIT source does not exist: {path}")
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()[:16]


def _cache_root() -> Path:
    configured = os.environ.get("SGLANG_ZOOMKV_JIT_CACHE")
    if configured:
        return Path(configured).expanduser()
    torch_root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if torch_root:
        return Path(torch_root).expanduser() / "sglang_zoomkv"
    return Path.home() / ".cache" / "torch_extensions" / "sglang_zoomkv"


@lru_cache(maxsize=None)
def _load_cached(
    logical_name: str,
    sources: tuple[str, ...],
    source_signature: str,
    cflags: tuple[str, ...],
    cuda_flags: tuple[str, ...],
    ldflags: tuple[str, ...],
    with_cuda: bool,
    verbose: bool,
    arch_list: str,
) -> Any:
    # Keep TORCH_CUDA_ARCH_LIST out of nvcc flags: PyTorch parses it and emits
    # the correct -gencode arguments (including "+PTX" when requested).
    del source_signature, arch_list
    from torch.utils.cpp_extension import load

    extension_name = f"sglang_zoomkv_{logical_name}"
    build_directory = _cache_root() / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)
    try:
        return load(
            name=extension_name,
            sources=list(sources),
            extra_cflags=list(cflags),
            extra_cuda_cflags=list(cuda_flags) if with_cuda else None,
            extra_ldflags=list(ldflags) or None,
            with_cuda=with_cuda,
            build_directory=str(build_directory),
            verbose=verbose,
        )
    except Exception as exc:
        arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or "<PyTorch auto-detect>"
        source_text = ", ".join(sources)
        raise RuntimeError(
            f"Failed to build ZoomKV extension '{logical_name}'. "
            f"sources=[{source_text}], build_directory='{build_directory}', "
            f"TORCH_CUDA_ARCH_LIST='{arch}'. Verify the CUDA toolkit, compiler, "
            f"PyTorch CUDA build, and requested GPU architectures. "
            f"Original error: {exc}"
        ) from exc


def load_jit(
    name: str,
    sources: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    source_dir: str | os.PathLike[str] | None = None,
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
    with_cuda: bool = True,
    verbose: bool | None = None,
) -> Any:
    """Compile/load one extension with process and disk caching.

    The first call for ``name`` compiles or loads the extension. Later calls
    return that same module and do not stat sources or enter the compiler.
    Relative sources are resolved against ``source_dir``. PyTorch's standard
    ``TORCH_CUDA_ARCH_LIST`` behavior is preserved. Set
    ``SGLANG_ZOOMKV_JIT_CACHE`` to override the persistent build cache.
    """
    ready = _loaded_modules.get(name)
    if ready is not None:
        return ready
    with _loaded_lock:
        ready = _loaded_modules.get(name)
        if ready is not None:
            return ready
        module = _load_jit_uncached(
            name,
            sources,
            source_dir=source_dir,
            cflags=cflags,
            cuda_flags=cuda_flags,
            ldflags=ldflags,
            with_cuda=with_cuda,
            verbose=verbose,
        )
        _loaded_modules[name] = module
        return module


def _load_jit_uncached(
    name: str,
    sources: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    source_dir: str | os.PathLike[str] | None = None,
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
    with_cuda: bool = True,
    verbose: bool | None = None,
) -> Any:
    if isinstance(sources, (str, os.PathLike)):
        sources = (sources,)
    base = Path(source_dir or os.curdir).resolve()
    resolved = tuple(
        str((base / source).resolve()) if not Path(source).is_absolute() else str(Path(source).resolve())
        for source in sources
    )
    signature = _source_signature(resolved)
    # Current PyTorch headers use C++20 constructs (including initialized
    # bit-fields). Keep host and CUDA compilation on the same language level.
    effective_cflags = _as_tuple(cflags, ("-O3", "-std=c++20"))
    effective_cuda_flags = _as_tuple(
        cuda_flags, ("-O3", "-std=c++20", "--use_fast_math")
    )
    effective_ldflags = tuple(ldflags or ())
    effective_verbose = (
        os.environ.get("SGLANG_ZOOMKV_JIT_VERBOSE", "0") == "1"
        if verbose is None
        else verbose
    )
    return _load_cached(
        name,
        resolved,
        signature,
        effective_cflags,
        effective_cuda_flags,
        effective_ldflags,
        with_cuda,
        effective_verbose,
        os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    )


__all__ = ["load_jit"]
