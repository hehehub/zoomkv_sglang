"""Register the ZoomKV attention backend with SGLang."""

from __future__ import annotations

import os
from typing import Any

from sglang_zoomkv.compat import (
    BACKEND_NAME,
    get_attention_backend_factory,
    get_runner_head_dim,
    load_backend_class,
    register_attention_backend,
)
from sglang_zoomkv.config import get_config


def create_zoomkv_backend(runner: Any) -> Any:
    """Validate runtime compatibility and construct ZoomKV with its fallback."""
    config = get_config()
    runner_head_dim = get_runner_head_dim(runner)
    if runner_head_dim is not None and runner_head_dim not in (128, 256):
        raise ValueError(
            "ZoomKV requires model head_dim=128 or head_dim=256, "
            f"but the model reports head_dim={runner_head_dim}"
        )

    fallback_name = os.getenv(
        "SGLANG_ZOOMKV_FALLBACK_BACKEND",
        os.getenv("POLAR_FALLBACK_BACKEND", "torch_native"),
    ).strip()
    if not fallback_name:
        raise ValueError("SGLANG_ZOOMKV_FALLBACK_BACKEND cannot be empty")
    if fallback_name == BACKEND_NAME:
        raise ValueError("SGLANG_ZOOMKV_FALLBACK_BACKEND cannot be 'zoomkv'")

    fallback = get_attention_backend_factory(fallback_name)(runner)
    backend_class = load_backend_class()
    return backend_class(runner, fallback)


register_attention_backend(BACKEND_NAME, create_zoomkv_backend)
