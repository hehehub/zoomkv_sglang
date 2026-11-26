import pytest

from sglang_zoomkv.config import ZoomKVConfig, get_config


def test_defaults_match_supported_kernel_point():
    config = ZoomKVConfig.from_env({})

    assert config == ZoomKVConfig()
    assert config.top_k == 1000
    assert config.local_window == 512
    assert config.sink_tokens == 64
    assert config.update_interval == 512


def test_canonical_environment_takes_precedence_over_legacy():
    config = ZoomKVConfig.from_env(
        {
            "SGLANG_ZOOMKV_FINAL_TOPK": "256",
            "POLAR_FINAL_TOPK": "128",
            "SGLANG_ZOOMKV_CANDIDATE_RATIO": "0.35",
            "SGLANG_ZOOMKV_LOCAL_SIZE": "768",
            "POLAR_LOCAL_WINDOW": "384",
        }
    )

    assert config.final_topk == 256
    assert config.candidate_ratio == 0.35
    assert config.local_size == 768


def test_legacy_names_and_boolean_spellings_are_supported():
    config = get_config(
        {
            "POLAR_TOP_K": "144",
            "POLAR_STATIC_PATTERN_END": "640",
            "POLAR_ENABLE_OFFLOAD": "YES",
            "POLAR_UPDATE_INTERVAL": "128",
        }
    )

    assert config.final_topk == 144
    assert config.local_size == 640
    assert config.enable_offload is True
    assert config.dynamic_update_interval == 128


def test_offload_false_selects_gpu_only_mode():
    config = ZoomKVConfig.from_env({"SGLANG_ZOOMKV_OFFLOAD": "false"})

    assert config.offload is False
    assert config.enable_offload is False


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"SGLANG_ZOOMKV_HEAD_DIM": "not-an-int"}, "must be an integer"),
        ({"SGLANG_ZOOMKV_HEAD_DIM": "64"}, "head_dim=128"),
        ({"SGLANG_ZOOMKV_BITS": "2"}, "bits=4"),
        ({"SGLANG_ZOOMKV_OFFLOAD": "sometimes"}, "must be one of"),
        ({"SGLANG_ZOOMKV_SINK_SIZE": "-1"}, "sink_size"),
        ({"SGLANG_ZOOMKV_LOCAL_SIZE": "0"}, "local_size"),
        ({"SGLANG_ZOOMKV_FINAL_TOPK": "0"}, "final_topk"),
        ({"SGLANG_ZOOMKV_CANDIDATE_RATIO": "abc"}, "must be a number"),
        ({"SGLANG_ZOOMKV_CANDIDATE_RATIO": "0"}, "candidate_ratio"),
        ({"SGLANG_ZOOMKV_CANDIDATE_RATIO": "1.1"}, "candidate_ratio"),
        ({"SGLANG_ZOOMKV_DYNAMIC_UPDATE_INTERVAL": "0"}, "update_interval"),
        ({"SGLANG_ZOOMKV_FULL_ATTENTION_THRESHOLD": "-1"}, "threshold"),
        ({"SGLANG_ZOOMKV_MAX_NEW_TOKENS": "0"}, "max_new_tokens"),
    ],
)
def test_invalid_configuration_fails_early(environment, message):
    with pytest.raises(ValueError, match=message):
        ZoomKVConfig.from_env(environment)

