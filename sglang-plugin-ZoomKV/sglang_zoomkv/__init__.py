"""SGLang ZoomKV plugin entry point."""

from __future__ import annotations

import importlib
import os
import threading

from sglang_zoomkv.config import ZoomKVConfig, get_config

__all__ = ["ZoomKVConfig", "get_config", "is_plugin_loaded", "load_plugin"]

_load_lock = threading.Lock()
_plugin_loaded = False


def _prepare_server_args(original_fn, server_args, *args, **kwargs):
    """Apply the safe MVP execution envelope before SGLang resolves arguments."""
    selected = {
        getattr(server_args, "attention_backend", None),
        getattr(server_args, "prefill_attention_backend", None),
        getattr(server_args, "decode_attention_backend", None),
    }
    if "zoomkv" in selected:
        if getattr(server_args, "tp_size", 1) not in (1, 2):
            raise ValueError("ZoomKV requires --tp-size 1 or 2")
        # End-to-end timing uses one full decode graph in both modes.
        # SGLANG_ZOOMKV_BREAK_DECODE_GRAPH=1 leaves attention outside the
        # graph so score, topk, rerank and UVA each show up in nsys.
        server_args.disable_cuda_graph = False
        server_args.disable_decode_cuda_graph = False
        break_decode = os.environ.get(
            "SGLANG_ZOOMKV_BREAK_DECODE_GRAPH", "0"
        ) not in ("0", "false", "False", "")
        server_args.cuda_graph_backend_decode = (
            "breakable" if break_decode else "full"
        )
        server_args.disable_radix_cache = True
        server_args.chunked_prefill_size = -1
        max_running = getattr(server_args, "max_running_requests", None)
        if max_running is None:
            server_args.max_running_requests = 64
        elif not 1 <= max_running <= 64:
            raise ValueError(
                "ZoomKV supports --max-running-requests in [1, 64]"
            )
    return original_fn(server_args, *args, **kwargs)


def _register_server_args_hook() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        _prepare_server_args,
        HookType.AROUND,
    )


def is_plugin_loaded() -> bool:
    """Return whether backend registration completed in this process."""
    return _plugin_loaded


def load_plugin() -> None:
    """Validate configuration and register ZoomKV exactly once per process."""
    global _plugin_loaded
    if _plugin_loaded:
        return

    with _load_lock:
        if _plugin_loaded:
            return

        # Fail before mutating SGLang's registry when configuration is invalid.
        get_config()
        importlib.import_module("sglang_zoomkv.attention.register")
        _register_server_args_hook()
        from sglang_zoomkv.attention.cuda_graph import install_decode_graph_break
        from sglang_zoomkv.schedule import install_prefill_all_first

        install_decode_graph_break()
        install_prefill_all_first()
        _plugin_loaded = True
