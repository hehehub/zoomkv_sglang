"""Attention integration for the SGLang ZoomKV plugin."""

from __future__ import annotations

from typing import Any

__all__ = ["ZoomKVAttentionBackend"]


def __getattr__(name: str) -> Any:
    """Keep backend imports lazy so registration is CPU/toolchain independent."""
    if name == "ZoomKVAttentionBackend":
        from .backend import ZoomKVAttentionBackend

        return ZoomKVAttentionBackend
    raise AttributeError(name)
