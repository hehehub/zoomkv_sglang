"""ZoomKV cache implementations.

Adapted from the upstream ``ZoomKV-main/cache_hub`` package.
"""

__all__ = ["KV_Cache", "KVCache", "ZoomKVCache", "polar_cache"]


def __getattr__(name):
    """Load the production cache lazily so base-cache imports stay lightweight."""
    if name in {"KV_Cache", "KVCache"}:
        from .cache import KV_Cache, KVCache

        return {"KV_Cache": KV_Cache, "KVCache": KVCache}[name]
    if name in {"ZoomKVCache", "polar_cache"}:
        from .polar_cache_kivi_quest_density_seperate_kernel import (
            ZoomKVCache,
            polar_cache,
        )

        return {"ZoomKVCache": ZoomKVCache, "polar_cache": polar_cache}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
