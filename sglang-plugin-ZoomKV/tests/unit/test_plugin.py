from types import SimpleNamespace

import pytest

import sglang_zoomkv


def test_load_plugin_is_idempotent(monkeypatch):
    calls = []
    monkeypatch.setattr(sglang_zoomkv, "_plugin_loaded", False)
    monkeypatch.setattr(
        sglang_zoomkv, "get_config", lambda: calls.append("config") or object()
    )
    monkeypatch.setattr(
        sglang_zoomkv.importlib,
        "import_module",
        lambda name: calls.append(("import", name)) or object(),
    )
    monkeypatch.setattr(
        sglang_zoomkv,
        "_register_server_args_hook",
        lambda: calls.append("hook"),
    )

    sglang_zoomkv.load_plugin()
    sglang_zoomkv.load_plugin()

    assert calls == [
        "config",
        ("import", "sglang_zoomkv.attention.register"),
        "hook",
    ]
    assert sglang_zoomkv.is_plugin_loaded()


def test_invalid_config_does_not_mutate_plugin_state(monkeypatch):
    monkeypatch.setattr(sglang_zoomkv, "_plugin_loaded", False)
    monkeypatch.setattr(
        sglang_zoomkv,
        "get_config",
        lambda: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(
        sglang_zoomkv.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )

    with pytest.raises(ValueError, match="bad config"):
        sglang_zoomkv.load_plugin()

    assert not sglang_zoomkv.is_plugin_loaded()


def test_server_args_hook_enforces_mvp_constraints():
    args = SimpleNamespace(
        attention_backend=None,
        prefill_attention_backend="torch_native",
        decode_attention_backend="zoomkv",
        tp_size=1,
        disable_cuda_graph=False,
        disable_radix_cache=False,
        chunked_prefill_size=8192,
        max_running_requests=32,
    )
    seen = []

    result = sglang_zoomkv._prepare_server_args(
        lambda server_args, marker: seen.append((server_args, marker)) or "done",
        args,
        "original-called",
    )

    assert result == "done"
    assert seen == [(args, "original-called")]
    assert args.disable_cuda_graph is True
    assert args.disable_radix_cache is True
    assert args.chunked_prefill_size == -1
    assert args.max_running_requests == 32


def test_server_args_hook_accepts_tp2_and_defaults_batch_capacity():
    args = SimpleNamespace(
        attention_backend="zoomkv",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        tp_size=2,
        max_running_requests=None,
    )

    assert sglang_zoomkv._prepare_server_args(lambda value: value, args) is args
    assert args.max_running_requests == 64


def test_server_args_hook_rejects_unsupported_parallelism_and_batch():
    args = SimpleNamespace(
        attention_backend="zoomkv",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        tp_size=4,
        max_running_requests=64,
    )

    with pytest.raises(ValueError, match="tp-size 1 or 2"):
        sglang_zoomkv._prepare_server_args(lambda value: value, args)

    args.tp_size = 2
    args.max_running_requests = 65
    with pytest.raises(ValueError, match="max-running-requests"):
        sglang_zoomkv._prepare_server_args(lambda value: value, args)


def test_server_args_hook_leaves_other_backends_unchanged():
    args = SimpleNamespace(
        attention_backend="flashinfer",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        tp_size=4,
        disable_cuda_graph=False,
    )

    assert sglang_zoomkv._prepare_server_args(lambda value: value, args) is args
    assert args.disable_cuda_graph is False
