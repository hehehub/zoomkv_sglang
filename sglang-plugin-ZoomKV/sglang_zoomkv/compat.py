"""Small compatibility boundary around SGLang's attention registry."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

BACKEND_NAME = "zoomkv"


def register_attention_backend(name: str, factory: Callable[[Any], Any]) -> None:
    """Register an attention factory against the current SGLang API.

    Recent SGLang releases provide a decorator. Direct dictionary assignment is
    retained as a fallback for releases that exposed the registry but not the
    decorator.
    """
    registry = import_module("sglang.srt.layers.attention.attention_registry")
    decorator = getattr(registry, "register_attention_backend", None)
    if decorator is not None:
        decorator(name)(factory)
    else:
        backends = getattr(registry, "ATTENTION_BACKENDS", None)
        if backends is None:
            raise RuntimeError(
                "This SGLang version does not expose an attention backend registry"
            )
        backends[name] = factory

    # Current SGLang validates CLI values against a separately maintained list.
    try:
        server_args = import_module("sglang.srt.server_args")
    except ImportError:
        return
    choices = getattr(server_args, "ATTENTION_BACKEND_CHOICES", ())
    add_choices = getattr(server_args, "add_attention_backend_choices", None)
    if add_choices is not None and name not in choices:
        add_choices([name])


def get_attention_backend_factory(name: str) -> Callable[[Any], Any]:
    """Resolve an already-registered SGLang attention backend factory."""
    registry = import_module("sglang.srt.layers.attention.attention_registry")
    backends = getattr(registry, "ATTENTION_BACKENDS", None)
    if backends is None:
        raise RuntimeError(
            "This SGLang version does not expose an attention backend registry"
        )
    try:
        return backends[name]
    except KeyError as exc:
        available = ", ".join(sorted(backends))
        raise ValueError(
            f"Unknown ZoomKV fallback backend {name!r}; "
            f"available backends: {available}"
        ) from exc


def load_backend_class() -> type[Any]:
    """Load the backend lazily, after plugin registration has completed."""
    backend_module = import_module("sglang_zoomkv.attention.backend")
    for class_name in ("ZoomKVAttentionBackend", "ZoomKVAttnBackend"):
        backend_class = getattr(backend_module, class_name, None)
        if backend_class is not None:
            return backend_class
    raise ImportError(
        "sglang_zoomkv.attention.backend must define ZoomKVAttentionBackend"
    )


def get_runner_head_dim(runner: Any) -> int | None:
    """Best-effort extraction across current and older ModelConfig layouts."""
    model_config = getattr(runner, "model_config", None)
    if model_config is None:
        return None

    for candidate in (
        model_config,
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if candidate is None:
            continue
        for name in ("head_dim", "head_size"):
            value = getattr(candidate, name, None)
            if value is not None:
                return int(value)
        hidden_size = getattr(candidate, "hidden_size", None)
        num_heads = getattr(candidate, "num_attention_heads", None)
        if hidden_size is not None and num_heads:
            return int(hidden_size) // int(num_heads)
    return None
