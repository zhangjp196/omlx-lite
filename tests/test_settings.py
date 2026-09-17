# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.settings module."""

import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.config import OMLXConfig
from omlx.settings import (
    BURST_DECODE_MODES,
    DEFAULT_BURST_DECODE_MODE,
    AuthSettings,
    CacheSettings,
    GlobalSettings,
    HuggingFaceSettings,
    LoggingSettings,
    MemorySettings,
    ModelSettings,
    SamplingSettings,
    SchedulerSettings,
    ServerSettings,
    UsageSettings,
    burst_decode_env,
    get_settings,
    get_ssd_capacity,
    get_system_memory,
    init_settings,
    reset_settings,
    resolve_default_base_path,
)


class TestServerSettings:
    """Tests for ServerSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ServerSettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.log_level == "info"
        assert settings.cors_origins == ["*"]
        assert settings.sse_keepalive_mode == "chunk"
        assert settings.auto_start_on_launch is True
        assert settings.burst_decode_mode == "balanced"
        assert settings.preserve_mid_system_cache is True
        assert settings.distributed_inference_enabled is False
        assert settings.max_audio_upload_size == "100MB"
        assert settings.max_audio_upload_bytes() == 100 * 1024 * 1024

    def test_custom_values(self):
        """Test custom values."""
        settings = ServerSettings(
            host="0.0.0.0",
            port=9000,
            log_level="debug",
            cors_origins=["https://example.com"],
        )
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.log_level == "debug"
        assert settings.cors_origins == ["https://example.com"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = ServerSettings(host="127.0.0.1", port=8000, log_level="info")
        result = settings.to_dict()
        assert result == {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
            "cors_origins": ["*"],
            "server_aliases": [],
            "sse_keepalive_mode": "chunk",
            "auto_start_on_launch": True,
            "burst_decode_mode": "balanced",
            "preserve_mid_system_cache": True,
            "distributed_inference_enabled": False,
            "max_audio_upload_size": "100MB",
        }

    def test_from_dict_distributed_inference_is_opt_in(self):
        assert ServerSettings.from_dict({}).distributed_inference_enabled is False
        assert (
            ServerSettings.from_dict(
                {"distributed_inference_enabled": True}
            ).distributed_inference_enabled
            is True
        )

    def test_from_dict_sse_keepalive_mode(self):
        """sse_keepalive_mode round-trips through from_dict / to_dict."""
        for mode in ("chunk", "comment", "off"):
            settings = ServerSettings.from_dict({"sse_keepalive_mode": mode})
            assert settings.sse_keepalive_mode == mode
            assert settings.to_dict()["sse_keepalive_mode"] == mode

    def test_from_dict_burst_decode_mode(self):
        """burst_decode_mode round-trips through from_dict / to_dict."""
        for mode in BURST_DECODE_MODES:
            settings = ServerSettings.from_dict({"burst_decode_mode": mode})
            assert settings.burst_decode_mode == mode
            assert settings.to_dict()["burst_decode_mode"] == mode

    def test_from_dict_burst_decode_mode_default(self):
        """A settings.json without burst_decode_mode falls back to the default."""
        settings = ServerSettings.from_dict({})
        assert settings.burst_decode_mode == DEFAULT_BURST_DECODE_MODE

    def test_from_dict_preserve_mid_system_cache(self):
        """preserve_mid_system_cache round-trips through from_dict / to_dict."""
        settings = ServerSettings.from_dict({"preserve_mid_system_cache": False})
        assert settings.preserve_mid_system_cache is False
        assert settings.to_dict()["preserve_mid_system_cache"] is False

    def test_from_dict_preserve_mid_system_cache_default(self):
        """Missing preserve_mid_system_cache keeps the cache-friendly default."""
        settings = ServerSettings.from_dict({})
        assert settings.preserve_mid_system_cache is True

    def test_from_dict_auto_start_on_launch(self):
        """auto_start_on_launch round-trips through from_dict / to_dict."""
        settings = ServerSettings.from_dict({"auto_start_on_launch": False})
        assert settings.auto_start_on_launch is False
        assert settings.to_dict()["auto_start_on_launch"] is False

    def test_from_dict_max_audio_upload_size(self):
        """max_audio_upload_size round-trips through from_dict / to_dict."""
        settings = ServerSettings.from_dict({"max_audio_upload_size": "500MB"})
        assert settings.max_audio_upload_size == "500MB"
        assert settings.max_audio_upload_bytes() == 500 * 1024 * 1024
        assert settings.to_dict()["max_audio_upload_size"] == "500MB"

    def test_from_dict_max_audio_upload_size_default(self):
        """A settings.json without max_audio_upload_size keeps the 100MB default."""
        settings = ServerSettings.from_dict({})
        assert settings.max_audio_upload_size == "100MB"

    def test_max_audio_upload_bytes_rejects_non_positive(self):
        """0MB / negative sizes parse as integers but are not usable limits."""
        with pytest.raises(ValueError, match="must be positive"):
            ServerSettings(max_audio_upload_size="0MB").max_audio_upload_bytes()
        with pytest.raises(ValueError, match="must be positive"):
            ServerSettings(max_audio_upload_size="-1MB").max_audio_upload_bytes()

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"host": "0.0.0.0", "port": 9000, "log_level": "debug"}
        settings = ServerSettings.from_dict(data)
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.log_level == "debug"
        assert settings.cors_origins == ["*"]  # default

    def test_from_dict_reads_bind_address_fallback(self):
        """bind_address is accepted only as a compatibility fallback."""
        settings = ServerSettings.from_dict({"bind_address": "0.0.0.0"})
        assert settings.host == "0.0.0.0"

    def test_from_dict_host_wins_over_bind_address(self):
        """host remains the canonical persisted/admin API key."""
        settings = ServerSettings.from_dict(
            {"host": "127.0.0.1", "bind_address": "0.0.0.0"}
        )
        assert settings.host == "127.0.0.1"

    def test_from_dict_with_cors_origins(self):
        """Test creation from dictionary with cors_origins."""
        data = {
            "host": "0.0.0.0",
            "port": 9000,
            "cors_origins": ["https://chat.example.com", "http://localhost:3000"],
        }
        settings = ServerSettings.from_dict(data)
        assert settings.cors_origins == [
            "https://chat.example.com",
            "http://localhost:3000",
        ]

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        data = {"port": 9000}
        settings = ServerSettings.from_dict(data)
        assert settings.host == "127.0.0.1"  # default
        assert settings.port == 9000
        assert settings.log_level == "info"  # default
        assert settings.cors_origins == ["*"]  # default


class TestBurstDecodeEnv:
    """Tests for the Burst Decode mode -> OMLX_DECODE_BURST_* env mapping."""

    def test_off_disables_bursting(self):
        """'off' caps max_steps at 1, which disables bursting in _step_burst."""
        assert burst_decode_env("off")["OMLX_DECODE_BURST_MAX_STEPS"] == "1"

    def test_levels_set_single_request_budget(self):
        """light / balanced / aggressive map to the documented budgets."""
        assert burst_decode_env("light")["OMLX_DECODE_BURST_BUDGET_SINGLE_S"] == "0.05"
        assert (
            burst_decode_env("balanced")["OMLX_DECODE_BURST_BUDGET_SINGLE_S"] == "0.1"
        )
        assert (
            burst_decode_env("aggressive")["OMLX_DECODE_BURST_BUDGET_SINGLE_S"] == "0.2"
        )

    def test_on_levels_keep_burst_enabled(self):
        """The on-levels keep max_steps above the disable threshold."""
        for mode in ("light", "balanced", "aggressive"):
            assert int(burst_decode_env(mode)["OMLX_DECODE_BURST_MAX_STEPS"]) > 1

    def test_unknown_mode_falls_back_to_default(self):
        """An unknown mode never disables bursting; it uses the default."""
        assert burst_decode_env("bogus") == burst_decode_env(DEFAULT_BURST_DECODE_MODE)

    def test_keys_match_engine_config_env_vars(self):
        """The mapping only sets the env vars EngineConfig actually reads."""
        assert set(burst_decode_env("balanced")) == {
            "OMLX_DECODE_BURST_MAX_STEPS",
            "OMLX_DECODE_BURST_BUDGET_SINGLE_S",
        }


class TestModelSettings:
    """Tests for ModelSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ModelSettings()
        assert settings.model_dirs == []
        assert settings.model_dir is None

    def test_get_model_dirs_default(self):
        """Test default model directories."""
        settings = ModelSettings()
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/tmp/omlx/models")]
        assert settings.get_model_dir(base_path) == Path("/tmp/omlx/models")

    def test_get_model_dirs_custom(self):
        """Test custom model directories."""
        settings = ModelSettings(model_dirs=["/custom/models"])
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/custom/models")]

    def test_get_model_dirs_multiple(self):
        """Test multiple model directories."""
        settings = ModelSettings(model_dirs=["/path/a", "/path/b"])
        base_path = Path("/tmp/omlx")
        result = settings.get_model_dirs(base_path)
        assert len(result) == 2
        assert result[0] == Path("/path/a")
        assert result[1] == Path("/path/b")
        # get_model_dir returns the first (primary) directory
        assert settings.get_model_dir(base_path) == Path("/path/a")

    def test_get_model_dirs_with_tilde(self):
        """Test model directory with tilde expansion."""
        settings = ModelSettings(model_dirs=["~/models"])
        base_path = Path("/tmp/omlx")
        result = settings.get_model_dirs(base_path)
        assert "~" not in str(result[0])  # Should be expanded

    def test_get_model_dirs_backward_compat(self):
        """Test backward compatibility: model_dir fallback when model_dirs is empty."""
        settings = ModelSettings(model_dir="/legacy/models")
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/legacy/models")]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = ModelSettings(model_dirs=["/models"])
        result = settings.to_dict()
        assert result == {
            "model_dirs": ["/models"],
            "model_dir": "/models",
            "model_fallback": False,
            "hide_helper_models": False,
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"model_dirs": ["/models"]}
        settings = ModelSettings.from_dict(data)
        assert settings.model_dirs == ["/models"]
        assert settings.model_fallback is False

    def test_from_dict_ignores_legacy_max_model_memory(self):
        """Legacy max_model_memory key in settings.json is silently ignored."""
        data = {"model_dirs": ["/models"], "max_model_memory": "32GB"}
        settings = ModelSettings.from_dict(data)
        assert settings.model_dirs == ["/models"]
        assert not hasattr(settings, "max_model_memory")

    def test_model_fallback_default(self):
        """Test model_fallback defaults to False."""
        settings = ModelSettings()
        assert settings.model_fallback is False

    def test_model_fallback_to_dict(self):
        """Test model_fallback is included in to_dict."""
        settings = ModelSettings(model_dirs=["/models"], model_fallback=True)
        result = settings.to_dict()
        assert result["model_fallback"] is True

    def test_model_fallback_from_dict(self):
        """Test model_fallback is loaded from dict."""
        data = {"model_dirs": ["/models"], "model_fallback": True}
        settings = ModelSettings.from_dict(data)
        assert settings.model_fallback is True

    def test_model_fallback_from_dict_missing(self):
        """Test model_fallback defaults to False when missing from dict."""
        data = {"model_dirs": ["/models"]}
        settings = ModelSettings.from_dict(data)
        assert settings.model_fallback is False

    def test_from_dict_backward_compat(self):
        """Test from_dict migrates old model_dir to model_dirs."""
        data = {"model_dir": "/legacy/models"}
        settings = ModelSettings.from_dict(data)
        assert settings.model_dirs == ["/legacy/models"]
        assert settings.model_dir == "/legacy/models"


class TestSchedulerSettings:
    """Tests for SchedulerSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = SchedulerSettings()
        assert settings.max_concurrent_requests == 8
        assert settings.embedding_batch_size == 32

    def test_custom_values(self):
        """Test custom values."""
        settings = SchedulerSettings(
            max_concurrent_requests=128, embedding_batch_size=16
        )
        assert settings.max_concurrent_requests == 128
        assert settings.embedding_batch_size == 16

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = SchedulerSettings()
        result = settings.to_dict()
        assert result == {
            "max_concurrent_requests": 8,
            "embedding_batch_size": 32,
            "chunked_prefill": False,
            "prefill_priority": "context",
            "decode_fairness": True,
        }

    def test_decode_fairness_from_dict(self):
        """Defaults on; explicit false round-trips."""
        assert SchedulerSettings.from_dict({}).decode_fairness is True
        assert (
            SchedulerSettings.from_dict(
                {"decode_fairness": False}
            ).decode_fairness
            is False
        )

    def test_prefill_priority_from_dict(self):
        """Valid values pass through; anything else falls back to context."""
        assert SchedulerSettings.from_dict({}).prefill_priority == "context"
        assert (
            SchedulerSettings.from_dict({"prefill_priority": "speed"}).prefill_priority
            == "speed"
        )
        assert (
            SchedulerSettings.from_dict({"prefill_priority": "bogus"}).prefill_priority
            == "context"
        )

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"max_concurrent_requests": 512}
        settings = SchedulerSettings.from_dict(data)
        assert settings.max_concurrent_requests == 512
        assert settings.embedding_batch_size == 32

        data = {"max_concurrent_requests": 512, "embedding_batch_size": 24}
        settings = SchedulerSettings.from_dict(data)
        assert settings.max_concurrent_requests == 512
        assert settings.embedding_batch_size == 24

    def test_from_dict_backwards_compat(self):
        """Test creation from dictionary with old keys."""
        data = {"max_num_seqs": 64}
        settings = SchedulerSettings.from_dict(data)
        assert settings.max_concurrent_requests == 64

        data = {"completion_batch_size": 32}
        settings = SchedulerSettings.from_dict(data)
        assert settings.max_concurrent_requests == 32


class TestCacheSettings:
    """Tests for CacheSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = CacheSettings()
        assert settings.enabled is True
        assert settings.ssd_cache_dir is None
        assert settings.ssd_cache_max_size == "auto"
        assert settings.gdn_ssd_split_enabled is None
        assert settings.get_gdn_snapshot_storage() == "auto"
        assert settings.get_gdn_ssd_split_enabled() is True
        assert settings.gdn_ssd_pending_max_size == "512MB"
        assert settings.gdn_sidecar_state_dtype == "fp32"
        assert settings.ane_compile_cache is False
        assert settings.initial_cache_blocks == 256

    def test_get_ssd_cache_dir_default(self):
        """Test default SSD cache directory."""
        settings = CacheSettings(ssd_cache_dir=None)
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_dir(base_path) == Path("/tmp/omlx/cache")

    def test_get_ssd_cache_dir_custom(self):
        """Test custom SSD cache directory."""
        settings = CacheSettings(ssd_cache_dir="/custom/cache")
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_dir(base_path) == Path("/custom/cache")

    def test_get_ssd_cache_max_size_bytes_auto(self):
        """Test auto SSD cache size calculation."""
        settings = CacheSettings(ssd_cache_max_size="auto")
        base_path = Path("/tmp/omlx")
        cache_dir = settings.get_ssd_cache_dir(base_path)
        expected = int(get_ssd_capacity(cache_dir) * 0.1)
        assert settings.get_ssd_cache_max_size_bytes(base_path) == expected

    def test_get_ssd_cache_max_size_bytes_explicit(self):
        """Test explicit SSD cache size."""
        settings = CacheSettings(ssd_cache_max_size="100GB")
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_max_size_bytes(base_path) == 100 * 1024**3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = CacheSettings(
            enabled=False, ssd_cache_dir="/cache", ssd_cache_max_size="50GB"
        )
        result = settings.to_dict()
        assert result == {
            "enabled": False,
            "hot_cache_only": False,
            "gdn_ssd_split_enabled": False,
            "gdn_snapshot_storage": "auto",
            "gdn_ssd_pending_max_size": "512MB",
            "gdn_sidecar_precision": "fp32",
            "ssd_cache_dir": "/cache",
            "ssd_cache_max_size": "50GB",
            "hot_cache_max_size": "0",
            "hot_cache_write_through": False,
            "ane_compile_cache": False,
            "initial_cache_blocks": 256,
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "ssd_cache_dir": "/cache",
            "ssd_cache_max_size": "200GB",
        }
        settings = CacheSettings.from_dict(data)
        assert settings.enabled is False
        assert settings.ssd_cache_dir == "/cache"
        assert settings.ssd_cache_max_size == "200GB"
        assert settings.initial_cache_blocks == 256  # default

    def test_from_dict_loads_gdn_split_settings(self):
        """GDN split settings round-trip through the settings file shape."""
        settings = CacheSettings.from_dict(
            {
                "gdn_ssd_split_enabled": True,
                "gdn_ssd_pending_max_size": "1GB",
                "gdn_sidecar_precision": "int8",
            }
        )
        assert settings.gdn_ssd_split_enabled is True
        assert settings.gdn_ssd_pending_max_size == "1GB"
        assert settings.gdn_sidecar_state_dtype == "int8"
        assert settings.to_dict()["gdn_ssd_split_enabled"] is True
        assert settings.to_dict()["gdn_ssd_pending_max_size"] == "1GB"

    @pytest.mark.parametrize(
        ("legacy_split", "expected_mode"),
        [(False, "embedded"), (True, "ssd_sidecar")],
    )
    def test_from_dict_preserves_legacy_mode_and_fp32_default(
        self, legacy_split, expected_mode
    ):
        settings = CacheSettings.from_dict(
            {"gdn_ssd_split_enabled": legacy_split}
        )
        assert settings.gdn_ssd_split_enabled is legacy_split
        assert settings.get_gdn_snapshot_storage() == expected_mode
        assert settings.gdn_sidecar_state_dtype == "fp32"

    def test_auto_policy_survives_dict_roundtrip(self):
        original = CacheSettings()
        restored = CacheSettings.from_dict(original.to_dict())
        assert restored.gdn_ssd_split_enabled is None
        assert restored.get_gdn_snapshot_storage() == "auto"
        assert restored.get_gdn_ssd_split_enabled() is True
        assert restored.gdn_sidecar_state_dtype == "fp32"

    def test_from_dict_rejects_conflicting_mode_and_legacy_bool(self):
        settings = CacheSettings.from_dict(
            {
                "gdn_snapshot_storage": "embedded",
                "gdn_ssd_split_enabled": True,
            }
        )
        global_settings = GlobalSettings(cache=settings)
        assert any("gdn_snapshot_storage" in error for error in global_settings.validate())

    def test_auto_policy_embeds_when_cache_is_disabled_or_hot_only(self):
        settings = CacheSettings(enabled=False)
        assert settings.get_gdn_ssd_split_enabled() is False
        settings.enabled = True
        settings.hot_cache_only = True
        assert settings.get_gdn_ssd_split_enabled() is False

    def test_from_dict_loads_rht_int8_gdn_state_dtype(self):
        settings = CacheSettings.from_dict(
            {
                "gdn_ssd_split_enabled": True,
                "gdn_sidecar_precision": "rht_int8",
            }
        )
        assert settings.gdn_sidecar_state_dtype == "rht_int8"
        assert settings.to_dict()["gdn_sidecar_precision"] == "rht_int8"

    def test_from_dict_ignores_v060_gdn_sidecar_state_dtype(self):
        """The v0.6.0 persisted lossy default must migrate back to fp32."""
        settings = CacheSettings.from_dict(
            {
                "gdn_ssd_split_enabled": True,
                "gdn_sidecar_state_dtype": "rht_int16",
            }
        )

        assert settings.gdn_sidecar_state_dtype == "fp32"
        assert "gdn_sidecar_state_dtype" not in settings.to_dict()
        assert settings.to_dict()["gdn_sidecar_precision"] == "fp32"

    def test_new_gdn_sidecar_precision_wins_over_v060_key(self):
        """An explicit selection made with the new key survives reloads."""
        settings = CacheSettings.from_dict(
            {
                "gdn_sidecar_state_dtype": "rht_int16",
                "gdn_sidecar_precision": "bf16",
            }
        )

        assert settings.gdn_sidecar_state_dtype == "bf16"

    @pytest.mark.parametrize(
        "raw", ["RHT_INT8", "Rht_Int8", "RHT_INT16", "INT8", "BF16", "FP32"]
    )
    def test_from_dict_normalizes_gdn_state_dtype_case(self, raw):
        settings = CacheSettings.from_dict(
            {"gdn_ssd_split_enabled": True, "gdn_sidecar_precision": raw}
        )
        assert settings.gdn_sidecar_state_dtype == raw.lower()

    @pytest.mark.parametrize("raw", [None, 8, 3.5, ["int8"]])
    def test_non_string_gdn_state_dtype_is_stringified_not_raised(self, raw):
        """from_dict never raises; validate() is where the value is judged."""
        settings = CacheSettings.from_dict(
            {"gdn_ssd_split_enabled": True, "gdn_sidecar_precision": raw}
        )
        assert isinstance(settings.gdn_sidecar_state_dtype, str)
        assert settings.gdn_sidecar_state_dtype not in {
            "fp32",
            "bf16",
            "int8",
            "rht_int8",
            "rht_int16",
        }

    def test_gdn_state_dtype_survives_a_dict_roundtrip(self):
        original = CacheSettings.from_dict(
            {
                "gdn_ssd_split_enabled": True,
                "gdn_sidecar_precision": "rht_int8",
                "gdn_ssd_pending_max_size": "512MB",
            }
        )
        restored = CacheSettings.from_dict(original.to_dict())
        assert restored.gdn_sidecar_state_dtype == "rht_int8"
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_with_initial_cache_blocks(self):
        """Test creation from dictionary with initial_cache_blocks."""
        data = {
            "enabled": True,
            "initial_cache_blocks": 16384,
        }
        settings = CacheSettings.from_dict(data)
        assert settings.initial_cache_blocks == 16384

    def test_from_dict_migrates_hot_cache_auto_to_disabled(self):
        """Legacy hot_cache_max_size=auto should load as disabled."""
        settings = CacheSettings.from_dict({"hot_cache_max_size": "auto"})
        assert settings.hot_cache_max_size == "0"

    def test_initial_cache_blocks_custom(self):
        """Test custom initial_cache_blocks value."""
        settings = CacheSettings(initial_cache_blocks=8192)
        assert settings.initial_cache_blocks == 8192
        result = settings.to_dict()
        assert result["initial_cache_blocks"] == 8192


class TestAuthSettings:
    """Tests for AuthSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = AuthSettings()
        assert settings.api_key is None
        assert settings.secret_key is None
        assert settings.sub_keys == []

    def test_custom_values(self):
        """Test custom values."""
        settings = AuthSettings(api_key="test-secret-key", secret_key="my-secret")
        assert settings.api_key == "test-secret-key"
        assert settings.secret_key == "my-secret"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = AuthSettings(api_key="my-key")
        result = settings.to_dict()
        assert result == {
            "api_key": "my-key",
            "secret_key": None,
            "skip_api_key_verification": False,
            "sub_keys": [],
        }

    def test_to_dict_with_sub_keys(self):
        """Test conversion to dictionary with sub keys."""
        from omlx.settings import SubKeyEntry

        settings = AuthSettings(
            api_key="my-key",
            sub_keys=[SubKeyEntry(key="sk1", name="Test", created_at="2024-01-01")],
        )
        result = settings.to_dict()
        assert len(result["sub_keys"]) == 1
        assert result["sub_keys"][0]["key"] == "sk1"
        assert result["sub_keys"][0]["name"] == "Test"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"api_key": "loaded-key", "secret_key": "loaded-secret"}
        settings = AuthSettings.from_dict(data)
        assert settings.api_key == "loaded-key"
        assert settings.secret_key == "loaded-secret"
        assert settings.sub_keys == []

    def test_from_dict_with_sub_keys(self):
        """Test creation from dictionary with sub keys."""
        data = {
            "api_key": "loaded-key",
            "sub_keys": [
                {"key": "sk1", "name": "My Key", "created_at": "2024-01-01"},
            ],
        }
        settings = AuthSettings.from_dict(data)
        assert len(settings.sub_keys) == 1
        assert settings.sub_keys[0].key == "sk1"
        assert settings.sub_keys[0].name == "My Key"

    def test_from_dict_backward_compat(self):
        """Test creation from dictionary without secret_key (backward compat)."""
        data = {"api_key": "loaded-key"}
        settings = AuthSettings.from_dict(data)
        assert settings.api_key == "loaded-key"
        assert settings.secret_key is None
        assert settings.sub_keys == []


class TestGlobalSettingsPersistence:
    """Tests for GlobalSettings persistence."""

    def test_global_settings_save_is_atomic(self, tmp_path):
        """save() must never leave a temp file or a torn settings.json."""
        gs = GlobalSettings(base_path=tmp_path)
        gs.save()

        settings_file = tmp_path / "settings.json"
        assert settings_file.exists()
        json.loads(settings_file.read_text())  # parses cleanly
        assert not list(tmp_path.glob("settings.json*.tmp"))
        if os.name == "posix":
            assert (settings_file.stat().st_mode & 0o777) == 0o600

    def test_global_settings_corrupt_file_is_moved_aside(self, tmp_path):
        """A corrupt settings.json is preserved as evidence, not silently eaten."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"server": {"port": 9999}}garbage-tail')

        restored = GlobalSettings.load(base_path=tmp_path)

        assert restored.server.port == 8000  # defaults, not the torn file
        assert not settings_file.exists()
        backups = list(tmp_path.glob("settings.json.corrupt-*"))
        assert len(backups) == 1
        assert "garbage-tail" in backups[0].read_text()


class TestUsageSettings:
    """Tests for UsageSettings and the usage_history toggle."""

    def test_default_values(self):
        assert UsageSettings().usage_history is True
        assert GlobalSettings().usage.usage_history is True

    def test_to_dict_from_dict_round_trip(self):
        settings = UsageSettings.from_dict({"usage_history": False})
        assert settings.usage_history is False
        assert settings.to_dict() == {"usage_history": False}
        assert UsageSettings.from_dict({}).usage_history is True

    def test_global_settings_save_load_round_trip(self, tmp_path):
        gs = GlobalSettings(base_path=tmp_path)
        gs.usage.usage_history = False
        gs.save()
        data = json.loads((tmp_path / "settings.json").read_text())
        assert data["usage"] == {"usage_history": False}
        assert gs.to_dict()["usage"] == {"usage_history": False}
        restored = GlobalSettings.load(base_path=tmp_path)
        assert restored.usage.usage_history is False

    def test_legacy_settings_file_defaults_on(self, tmp_path):
        """Settings files written before the toggle existed keep recording."""
        gs = GlobalSettings(base_path=tmp_path)
        gs.save()
        settings_file = tmp_path / "settings.json"
        data = json.loads(settings_file.read_text())
        del data["usage"]
        settings_file.write_text(json.dumps(data))
        assert GlobalSettings.load(base_path=tmp_path).usage.usage_history is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("0", False),
            ("false", False),
            ("off", False),
            ("1", True),
            ("on", True),
            ("TRUE", True),
        ],
    )
    def test_env_override(self, tmp_path, monkeypatch, value, expected):
        gs = GlobalSettings(base_path=tmp_path)
        gs.usage.usage_history = not expected
        gs.save()
        monkeypatch.setenv("OMLX_USAGE_HISTORY", value)
        assert GlobalSettings.load(base_path=tmp_path).usage.usage_history is expected


class TestHuggingFaceSettings:
    """Tests for HuggingFaceSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = HuggingFaceSettings()
        assert settings.endpoint == ""
        assert settings.hf_cache_enabled is True

    def test_custom_values(self):
        """Test custom values."""
        settings = HuggingFaceSettings(
            endpoint="https://hf-mirror.com",
            hf_cache_enabled=False,
        )
        assert settings.endpoint == "https://hf-mirror.com"
        assert settings.hf_cache_enabled is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = HuggingFaceSettings(
            endpoint="https://hf-mirror.com",
            hf_cache_enabled=False,
        )
        result = settings.to_dict()
        assert result == {
            "endpoint": "https://hf-mirror.com",
            "hf_cache_enabled": False,
        }

    def test_to_dict_empty(self):
        """Test conversion to dictionary with empty endpoint."""
        settings = HuggingFaceSettings()
        result = settings.to_dict()
        assert result == {"endpoint": "", "hf_cache_enabled": True}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"endpoint": "https://hf-mirror.com", "hf_cache_enabled": False}
        settings = HuggingFaceSettings.from_dict(data)
        assert settings.endpoint == "https://hf-mirror.com"
        assert settings.hf_cache_enabled is False

    def test_from_dict_defaults(self):
        """Test creation from empty dictionary uses defaults."""
        settings = HuggingFaceSettings.from_dict({})
        assert settings.endpoint == ""
        assert settings.hf_cache_enabled is True


class TestLoggingSettings:
    """Tests for LoggingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = LoggingSettings()
        assert settings.log_dir is None
        assert settings.retention_days == 7

    def test_custom_values(self):
        """Test custom values."""
        settings = LoggingSettings(log_dir="/custom/logs", retention_days=14)
        assert settings.log_dir == "/custom/logs"
        assert settings.retention_days == 14

    def test_get_log_dir_default(self):
        """Test default log directory."""
        settings = LoggingSettings(log_dir=None)
        base_path = Path("/tmp/omlx")
        assert settings.get_log_dir(base_path) == Path("/tmp/omlx/logs")

    def test_get_log_dir_custom(self):
        """Test custom log directory."""
        settings = LoggingSettings(log_dir="/custom/logs")
        base_path = Path("/tmp/omlx")
        assert settings.get_log_dir(base_path) == Path("/custom/logs")

    def test_get_log_dir_with_tilde(self):
        """Test log directory with tilde expansion."""
        settings = LoggingSettings(log_dir="~/logs")
        base_path = Path("/tmp/omlx")
        result = settings.get_log_dir(base_path)
        assert "~" not in str(result)  # Should be expanded

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = LoggingSettings(log_dir="/logs", retention_days=30)
        result = settings.to_dict()
        assert result == {"log_dir": "/logs", "retention_days": 30}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"log_dir": "/my/logs", "retention_days": 14}
        settings = LoggingSettings.from_dict(data)
        assert settings.log_dir == "/my/logs"
        assert settings.retention_days == 14

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        data = {"log_dir": "/custom"}
        settings = LoggingSettings.from_dict(data)
        assert settings.log_dir == "/custom"
        assert settings.retention_days == 7  # default


class TestMemorySettings:
    """Tests for MemorySettings dataclass."""

    def test_defaults(self):
        """Memory guard defaults to balanced tier and guard on."""
        settings = MemorySettings()
        assert settings.prefill_memory_guard is True
        assert settings.memory_guard_tier == "balanced"

    def test_to_dict(self):
        """Test serialization."""
        settings = MemorySettings(memory_guard_tier="safe")
        d = settings.to_dict()
        assert d["memory_guard_tier"] == "safe"
        assert d["prefill_memory_guard"] is True
        assert d["soft_threshold"] == 0.85
        assert d["hard_threshold"] == 0.95
        # Removed fields must not be present.
        assert "max_process_memory" not in d
        assert "max_process_memory_is_explicit" not in d

    def test_to_dict_guard_disabled(self):
        """Test serialization with prefill guard disabled."""
        settings = MemorySettings(prefill_memory_guard=False)
        d = settings.to_dict()
        assert d["prefill_memory_guard"] is False

    def test_from_dict(self):
        """Test deserialization picks up tier value."""
        settings = MemorySettings.from_dict({"memory_guard_tier": "aggressive"})
        assert settings.memory_guard_tier == "aggressive"
        assert settings.prefill_memory_guard is True  # default

    def test_from_dict_defaults(self):
        """Test deserialization with empty dict uses defaults."""
        settings = MemorySettings.from_dict({})
        assert settings.memory_guard_tier == "balanced"
        assert settings.prefill_memory_guard is True

    def test_from_dict_invalid_tier_falls_back_to_balanced(self):
        """Unknown tier values silently degrade to balanced."""
        settings = MemorySettings.from_dict({"memory_guard_tier": "wild"})
        assert settings.memory_guard_tier == "balanced"

    def test_from_dict_ignores_legacy_keys(self):
        """Legacy max_process_memory / is_explicit keys in old settings.json are ignored."""
        settings = MemorySettings.from_dict(
            {
                "max_process_memory": "80%",
                "max_process_memory_is_explicit": True,
                "memory_guard_tier": "safe",
            }
        )
        assert settings.memory_guard_tier == "safe"
        assert not hasattr(settings, "max_process_memory")

    def test_from_dict_guard_disabled(self):
        """Test deserialization with prefill guard disabled."""
        settings = MemorySettings.from_dict({"prefill_memory_guard": False})
        assert settings.prefill_memory_guard is False


class TestGlobalSettings:
    """Tests for GlobalSettings class."""

    def test_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            assert settings.server.host == "127.0.0.1"
            assert settings.server.port == 8000
            assert settings.memory.memory_guard_tier == "balanced"
            assert settings.scheduler.max_concurrent_requests == 8
            assert settings.scheduler.embedding_batch_size == 32
            assert settings.cache.enabled is True
            assert settings.auth.api_key is None

    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "127.12.34.56", "localhost", "LOCALHOST.", "::1"],
    )
    def test_loopback_bind_does_not_require_api_key(self, host):
        settings = GlobalSettings()
        settings.server.host = host

        assert not any("API key is required" in error for error in settings.validate())

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.10", "my-mac.local", "127.0.0.1,0.0.0.0"],
    )
    def test_network_bind_requires_api_key(self, host):
        settings = GlobalSettings()
        settings.server.host = host

        assert any("API key is required" in error for error in settings.validate())

    def test_network_bind_with_api_key_is_valid(self):
        settings = GlobalSettings()
        settings.server.host = "0.0.0.0"
        settings.auth.api_key = "secret-key"

        assert settings.validate() == []

    def test_network_bind_rejects_api_key_bypass(self):
        settings = GlobalSettings()
        settings.server.host = "0.0.0.0"
        settings.auth.api_key = "secret-key"
        settings.auth.skip_api_key_verification = True

        errors = settings.validate()

        assert any("cannot be skipped" in error for error in errors)

    @pytest.mark.parametrize("host", ["", "   ", "999.999.999.999"])
    def test_invalid_bind_host_is_reported(self, host):
        settings = GlobalSettings()
        settings.server.host = host

        errors = settings.validate()

        assert any("host" in error.lower() for error in errors)
        assert not any("API key is required" in error for error in errors)

    def test_get_effective_model_dirs_includes_hf_cache_between_dirs(
        self, tmp_path, monkeypatch
    ):
        """HF cache is inserted between primary and additional model dirs."""
        primary = tmp_path / "primary"
        additional = tmp_path / "additional"
        hf_cache = tmp_path / "hf" / "hub"
        primary.mkdir()
        additional.mkdir()
        hf_cache.mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))

        settings = GlobalSettings(base_path=tmp_path / "omlx")
        settings.model.model_dirs = [str(primary), str(additional)]

        assert settings.get_effective_model_dirs() == [
            primary.resolve(),
            hf_cache.resolve(),
            additional.resolve(),
        ]

    def test_get_effective_model_dirs_skips_disabled_hf_cache(
        self, tmp_path, monkeypatch
    ):
        """Disabled HF cache is not included in discovery dirs."""
        primary = tmp_path / "primary"
        hf_cache = tmp_path / "hf" / "hub"
        primary.mkdir()
        hf_cache.mkdir(parents=True)
        monkeypatch.setenv("HF_HUB_CACHE", str(hf_cache))

        settings = GlobalSettings(base_path=tmp_path / "omlx")
        settings.model.model_dirs = [str(primary)]
        settings.huggingface.hf_cache_enabled = False

        assert settings.get_effective_model_dirs() == [primary.resolve()]

    def test_cli_override_memory_guard_tier(self, tmp_path):
        """CLI memory guard tier should override loaded settings."""
        args = Namespace(memory_guard="safe", memory_guard_gb=None)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.memory_guard_tier == "safe"
        assert settings.memory.memory_guard_custom_ceiling_gb == 0.0

    def test_cli_override_memory_guard_gb_sets_custom_tier(self, tmp_path):
        """CLI memory guard GB should select custom tier automatically."""
        args = Namespace(memory_guard=None, memory_guard_gb=48.0)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.memory_guard_tier == "custom"
        assert settings.memory.memory_guard_custom_ceiling_gb == 48.0

    def test_cli_memory_guard_off_disables_the_guard(self, tmp_path):
        args = Namespace(memory_guard="off", memory_guard_gb=None)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.prefill_memory_guard is False
        # The tier is left alone so turning the guard back on restores it.
        assert settings.memory.memory_guard_tier == "balanced"

    def test_cli_tier_turns_a_disabled_guard_back_on(self, tmp_path):
        """A saved prefill_memory_guard=false used to make --memory-guard a
        silent no-op: the enforcer reports a ceiling of 0 with the guard off,
        so the requested tier governed nothing."""
        (tmp_path / "settings.json").write_text(
            json.dumps({"memory": {"prefill_memory_guard": False}})
        )
        args = Namespace(memory_guard="aggressive", memory_guard_gb=None)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.prefill_memory_guard is True
        assert settings.memory.memory_guard_tier == "aggressive"

    def test_cli_memory_guard_gb_turns_a_disabled_guard_back_on(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"memory": {"prefill_memory_guard": False}})
        )
        args = Namespace(memory_guard=None, memory_guard_gb=20.0)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.prefill_memory_guard is True
        assert settings.memory.memory_guard_tier == "custom"
        assert settings.memory.memory_guard_custom_ceiling_gb == 20.0

    def test_cli_memory_guard_absent_leaves_saved_state(self, tmp_path):
        """No flag means "use settings.json", both ways."""
        (tmp_path / "settings.json").write_text(
            json.dumps({"memory": {"prefill_memory_guard": False}})
        )
        args = Namespace(memory_guard=None, memory_guard_gb=None)
        settings = GlobalSettings.load(base_path=tmp_path, cli_args=args)

        assert settings.memory.prefill_memory_guard is False

    def test_load_from_file(self):
        """Test loading settings from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {"port": 9000},
                        "auth": {"api_key": "test-key"},
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000
            assert settings.auth.api_key == "test-key"

    def test_load_from_file_all_sections(self):
        """Test loading all settings sections from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {
                            "host": "0.0.0.0",
                            "port": 9000,
                            "log_level": "debug",
                        },
                        "model": {"model_dir": "/models"},
                        "memory": {"memory_guard_tier": "safe"},
                        "scheduler": {
                            "max_concurrent_requests": 128,
                            "embedding_batch_size": 24,
                        },
                        "cache": {
                            "enabled": False,
                            "ssd_cache_dir": "/cache",
                            "ssd_cache_max_size": "50GB",
                        },
                        "auth": {"api_key": "secret"},
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.host == "0.0.0.0"
            assert settings.server.port == 9000
            assert settings.server.log_level == "debug"
            assert settings.model.model_dirs == ["/models"]  # Migrated from model_dir
            assert settings.model.model_dir == "/models"  # Backward compat field
            assert settings.memory.memory_guard_tier == "safe"
            assert settings.scheduler.max_concurrent_requests == 128
            assert settings.scheduler.embedding_batch_size == 24
            assert settings.cache.enabled is False
            assert settings.cache.ssd_cache_dir == "/cache"
            assert settings.auth.api_key == "secret"

    def test_load_and_save_migrates_v060_gdn_sidecar_precision(
        self, tmp_path, monkeypatch
    ):
        """A v0.6.0 settings file resets its persisted lossy default."""
        monkeypatch.delenv("OMLX_GDN_SIDECAR_STATE_DTYPE", raising=False)
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "cache": {
                        "gdn_ssd_split_enabled": True,
                        "gdn_sidecar_state_dtype": "rht_int16",
                    },
                }
            )
        )

        settings = GlobalSettings.load(base_path=tmp_path)
        assert settings.cache.gdn_sidecar_state_dtype == "fp32"

        settings.save()
        persisted_cache = json.loads(settings_file.read_text())["cache"]
        assert "gdn_sidecar_state_dtype" not in persisted_cache
        assert persisted_cache["gdn_sidecar_precision"] == "fp32"

    def test_load_nonexistent_file_uses_defaults(self):
        """Test loading with no settings file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.host == "127.0.0.1"
            assert settings.server.port == 8000

    def test_load_invalid_json_uses_defaults(self):
        """Test loading invalid JSON file logs warning and uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text("{ invalid json }")

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should use defaults due to parse error
            assert settings.server.port == 8000

    def test_save(self):
        """Test saving settings to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.port = 9001
            settings.auth.api_key = "saved-key"
            settings.save()

            # Verify file was created
            settings_file = Path(tmpdir) / "settings.json"
            assert settings_file.exists()

            # Verify content
            data = json.loads(settings_file.read_text())
            assert data["version"] == "1.0"
            assert data["server"]["port"] == 9001
            assert data["auth"]["api_key"] == "saved-key"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
    def test_save_restricts_settings_file_permissions(self):
        """Credential-bearing settings are readable only by their owner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.save()

            settings_file = Path(tmpdir) / "settings.json"
            assert settings_file.stat().st_mode & 0o777 == 0o600

            settings_file.chmod(0o644)
            settings.save()
            assert settings_file.stat().st_mode & 0o777 == 0o600

    def test_save_and_load_cors_origins(self):
        """Test saving and loading cors_origins through settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.cors_origins = ["https://chat.example.com"]
            settings.save()

            loaded = GlobalSettings.load(base_path=tmpdir)
            assert loaded.server.cors_origins == ["https://chat.example.com"]

    def test_load_cors_origins_from_file(self):
        """Test loading cors_origins from settings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {
                            "port": 8000,
                            "cors_origins": [
                                "https://chat.example.com",
                                "http://localhost:3000",
                            ],
                        },
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.cors_origins == [
                "https://chat.example.com",
                "http://localhost:3000",
            ]

    def test_load_without_cors_origins_uses_default(self):
        """Test that missing cors_origins in settings.json uses default ['*']."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.cors_origins == ["*"]

    def test_save_creates_directory(self):
        """Test save creates base directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "nested" / "omlx"
            settings = GlobalSettings(base_path=base)
            settings.save()

            assert base.exists()
            assert (base / "settings.json").exists()

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "omlx"
            settings = GlobalSettings(base_path=base)
            settings.ensure_directories()

            assert base.exists()
            assert (base / "models").exists()
            assert (base / "cache").exists()
            assert (base / "logs").exists()

    def test_ensure_directories_custom_paths(self):
        """Test directory creation with custom paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "omlx"
            custom_models = Path(tmpdir) / "custom_models"
            custom_cache = Path(tmpdir) / "custom_cache"
            custom_logs = Path(tmpdir) / "custom_logs"

            settings = GlobalSettings(base_path=base)
            settings.model.model_dir = str(custom_models)
            settings.cache.ssd_cache_dir = str(custom_cache)
            settings.logging.log_dir = str(custom_logs)
            settings.ensure_directories()

            assert base.exists()
            assert custom_models.exists()
            assert custom_cache.exists()
            assert custom_logs.exists()

    def test_ensure_directories_unavailable_model_dir(self):
        """Test that unavailable model dirs are skipped instead of crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "omlx"
            valid_models = Path(tmpdir) / "valid_models"
            unavailable = Path("/Volumes/NonExistentDrive/Models")

            settings = GlobalSettings(base_path=base)
            settings.model.model_dirs = [str(valid_models), str(unavailable)]
            settings.ensure_directories()

            assert base.exists()
            assert valid_models.exists()
            # Unavailable path should be removed from model_dirs
            resolved_dirs = settings.model.get_model_dirs(base)
            assert len(resolved_dirs) == 1
            assert resolved_dirs[0] == valid_models.resolve()

    def test_ensure_directories_unreadable_model_dir(self, tmp_path, monkeypatch):
        """Test that existing but unreadable model dirs are skipped."""
        base = tmp_path / "omlx"
        valid_models = tmp_path / "valid_models"
        unreadable = tmp_path / "unreadable_models"
        unreadable.mkdir()

        original_iterdir = Path.iterdir

        def fake_iterdir(path):
            if path == unreadable.resolve():
                raise PermissionError("Operation not permitted")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)

        settings = GlobalSettings(base_path=base)
        settings.model.model_dirs = [str(valid_models), str(unreadable)]
        settings.ensure_directories()

        resolved_dirs = settings.model.get_model_dirs(base)
        assert resolved_dirs == [valid_models.resolve()]

    def test_validate_valid_settings(self):
        """Test validation with valid settings."""
        settings = GlobalSettings()
        errors = settings.validate()
        assert errors == []

    def test_validate_context_window_policy(self):
        """Sampling context policy must be positive when set."""
        settings = GlobalSettings()
        settings.sampling.max_context_window_policy = 128000
        assert settings.validate() == []

        settings.sampling.max_context_window_policy = 0
        errors = settings.validate()
        assert any("max_context_window_policy" in e for e in errors)

    def test_validate_invalid_port_low(self):
        """Test validation catches port below 1."""
        settings = GlobalSettings()
        settings.server.port = 0
        errors = settings.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_port_high(self):
        """Test validation catches port above 65535."""
        settings = GlobalSettings()
        settings.server.port = 70000
        errors = settings.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_log_level(self):
        """Test validation catches invalid log level."""
        settings = GlobalSettings()
        settings.server.log_level = "invalid"
        errors = settings.validate()
        assert any("log_level" in e.lower() for e in errors)

    def test_validate_valid_log_levels(self):
        """Test validation accepts all valid log levels."""
        for level in ["trace", "debug", "info", "warning", "error", "critical"]:
            settings = GlobalSettings()
            settings.server.log_level = level
            errors = settings.validate()
            assert errors == []

    def test_validate_invalid_sse_keepalive_mode(self):
        """Validation rejects unknown sse_keepalive_mode values."""
        settings = GlobalSettings()
        settings.server.sse_keepalive_mode = "bogus"
        errors = settings.validate()
        assert any("sse_keepalive_mode" in e for e in errors)

    def test_validate_valid_sse_keepalive_modes(self):
        """Validation accepts chunk / comment / off."""
        for mode in ("chunk", "comment", "off"):
            settings = GlobalSettings()
            settings.server.sse_keepalive_mode = mode
            errors = settings.validate()
            assert errors == []

    def test_validate_invalid_max_audio_upload_size(self):
        """Validation rejects unparseable or non-positive audio upload limits."""
        settings = GlobalSettings()
        settings.server.max_audio_upload_size = "bogus"
        errors = settings.validate()
        assert any("max_audio_upload_size" in e for e in errors)

        settings = GlobalSettings()
        settings.server.max_audio_upload_size = "0MB"
        errors = settings.validate()
        assert any("max_audio_upload_size" in e for e in errors)

    def test_validate_valid_max_audio_upload_size(self):
        """Validation accepts human-readable audio upload sizes."""
        settings = GlobalSettings()
        settings.server.max_audio_upload_size = "250MB"
        errors = settings.validate()
        assert not any("max_audio_upload_size" in e for e in errors)

    def test_validate_memory_guard_tier_valid(self):
        """Test validation accepts each known tier."""
        for tier in ("safe", "balanced", "aggressive"):
            settings = GlobalSettings()
            settings.memory.memory_guard_tier = tier
            errors = settings.validate()
            assert not any("memory_guard_tier" in e for e in errors)

        settings = GlobalSettings()
        settings.memory.memory_guard_tier = "custom"
        settings.memory.memory_guard_custom_ceiling_gb = 48.0
        errors = settings.validate()
        assert not any("memory_guard_tier" in e for e in errors)
        assert not any("memory_guard_custom_ceiling_gb" in e for e in errors)

    def test_validate_memory_guard_tier_invalid(self):
        """Test validation flags unknown tier values."""
        settings = GlobalSettings()
        settings.memory.memory_guard_tier = "extreme"  # type: ignore[assignment]
        errors = settings.validate()
        assert any("memory_guard_tier" in e for e in errors)

    def test_validate_invalid_scheduler_values(self):
        """Test validation catches invalid scheduler values."""
        settings = GlobalSettings()
        settings.scheduler.max_concurrent_requests = 0
        errors = settings.validate()
        assert any("max_concurrent_requests" in e.lower() for e in errors)

        settings = GlobalSettings()
        settings.scheduler.embedding_batch_size = 0
        errors = settings.validate()
        assert any("embedding_batch_size" in e.lower() for e in errors)

    def test_validate_invalid_cache_size(self):
        """Test validation catches invalid cache size."""
        settings = GlobalSettings()
        settings.cache.ssd_cache_max_size = "not-a-size"
        errors = settings.validate()
        assert any("ssd_cache_max_size" in e.lower() for e in errors)

    def test_validate_hot_cache_size(self):
        """Hot cache accepts explicit sizes only; auto is SSD-cache-only."""
        settings = GlobalSettings()
        settings.cache.hot_cache_max_size = "0"
        assert not any("hot_cache_max_size" in e for e in settings.validate())

        settings.cache.hot_cache_max_size = "8GB"
        assert not any("hot_cache_max_size" in e for e in settings.validate())

        settings.cache.hot_cache_max_size = "auto"
        errors = settings.validate()
        assert any("hot_cache_max_size" in e for e in errors)
        assert any("auto" in e for e in errors)

        settings.cache.hot_cache_max_size = "not-a-size"
        errors = settings.validate()
        assert any("hot_cache_max_size" in e for e in errors)

    def test_validate_gdn_split_requires_ssd_cache(self):
        """GDN split cannot be enabled in hot-cache-only mode."""
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.hot_cache_only = True
        errors = settings.validate()
        assert any("gdn_ssd_split_enabled" in e for e in errors)
        assert any("hot_cache_only" in e for e in errors)

    def test_validate_gdn_pending_size(self):
        """The pending write limit must be a positive size."""
        settings = GlobalSettings()
        settings.cache.gdn_ssd_pending_max_size = "not-a-size"
        errors = settings.validate()
        assert any("gdn_ssd_pending_max_size" in e for e in errors)

    def test_reduced_gdn_dtype_is_dormant_when_embedded(self):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = False
        settings.cache.gdn_sidecar_state_dtype = "int8"
        errors = settings.validate()
        assert not any("gdn_sidecar_state_dtype" in e for e in errors)

    def test_rht_int8_is_dormant_when_embedded(self):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = False
        settings.cache.gdn_sidecar_state_dtype = "rht_int8"
        errors = settings.validate()
        assert not any("gdn_sidecar_state_dtype" in e for e in errors)

    @pytest.mark.parametrize(
        "dtype", ["fp32", "bf16", "int8", "rht_int8", "rht_int16"]
    )
    def test_validate_accepts_every_dtype_with_split_enabled(self, dtype):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.hot_cache_only = False
        settings.cache.gdn_sidecar_state_dtype = dtype
        errors = settings.validate()
        assert not any("gdn_sidecar_state_dtype" in e for e in errors)

    @pytest.mark.parametrize("dtype", ["fp8", "int4", "", "none", "8"])
    def test_validate_rejects_unknown_dtype(self, dtype):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.gdn_sidecar_state_dtype = dtype
        errors = settings.validate()
        assert any("must be one of" in e for e in errors)

    def test_validate_reports_unknown_dtype_before_the_split_invariant(self):
        """An unknown value is not also blamed on the split setting."""
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = False
        settings.cache.gdn_sidecar_state_dtype = "fp8"
        errors = settings.validate()
        gdn_errors = [e for e in errors if "gdn_sidecar_state_dtype" in e]
        assert len(gdn_errors) == 1
        assert "must be one of" in gdn_errors[0]

    def test_legacy_config_gdn_env_and_validation(self):
        """The legacy config layer exposes the same GDN cache plumbing."""
        with patch.dict(
            os.environ,
            {
                "OMLX_GDN_SSD_SPLIT_ENABLED": "1",
                "OMLX_GDN_SSD_PENDING_MAX_SIZE": "768MB",
                "OMLX_GDN_SIDECAR_STATE_DTYPE": "bf16",
            },
            clear=False,
        ):
            config = OMLXConfig.from_env()
        assert config.paged_ssd_cache.gdn_ssd_split_enabled is True
        assert config.paged_ssd_cache.gdn_ssd_pending_max_size == "768MB"
        assert config.paged_ssd_cache.gdn_sidecar_state_dtype == "bf16"

        config.paged_ssd_cache.hot_cache_only = True
        errors = config.validate()
        assert any("gdn_ssd_split_enabled" in e for e in errors)

    def test_legacy_config_gdn_storage_mode_env(self):
        with patch.dict(
            os.environ,
            {
                "OMLX_GDN_SNAPSHOT_STORAGE": "embedded",
                "OMLX_GDN_SSD_SPLIT_ENABLED": "1",
            },
            clear=False,
        ):
            config = OMLXConfig.from_env()
        assert config.paged_ssd_cache.gdn_ssd_split_enabled is False
        assert config.paged_ssd_cache.gdn_snapshot_storage == "embedded"

    def test_validate_invalid_initial_cache_blocks(self):
        """Test validation catches invalid initial_cache_blocks."""
        settings = GlobalSettings()
        settings.cache.initial_cache_blocks = 0
        errors = settings.validate()
        assert any("initial_cache_blocks" in e.lower() for e in errors)

        settings = GlobalSettings()
        settings.cache.initial_cache_blocks = -1
        errors = settings.validate()
        assert any("initial_cache_blocks" in e.lower() for e in errors)

    def test_validate_multiple_errors(self):
        """Test validation returns multiple errors."""
        settings = GlobalSettings()
        settings.server.port = 0
        settings.scheduler.max_concurrent_requests = -1
        settings.memory.memory_guard_tier = "extreme"  # type: ignore[assignment]
        errors = settings.validate()
        assert len(errors) >= 3

    def test_validate_hf_endpoint_empty_ok(self):
        """Test empty HuggingFace endpoint passes validation."""
        settings = GlobalSettings()
        settings.huggingface.endpoint = ""
        errors = settings.validate()
        assert not any("huggingface" in e.lower() for e in errors)

    def test_validate_hf_endpoint_valid(self):
        """Test valid HuggingFace endpoint passes validation."""
        settings = GlobalSettings()
        settings.huggingface.endpoint = "https://hf-mirror.com"
        errors = settings.validate()
        assert not any("huggingface" in e.lower() for e in errors)

    def test_validate_hf_endpoint_invalid(self):
        """Test invalid HuggingFace endpoint fails validation."""
        settings = GlobalSettings()
        settings.huggingface.endpoint = "not-a-url"
        errors = settings.validate()
        assert any("huggingface" in e.lower() for e in errors)

    def test_env_override_server(self):
        """Test environment variable override for server settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_HOST": "0.0.0.0",
                    "OMLX_PORT": "9999",
                    "OMLX_LOG_LEVEL": "debug",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.server.host == "0.0.0.0"
                assert settings.server.port == 9999
                assert settings.server.log_level == "debug"

    def test_env_override_max_audio_upload_size(self):
        """OMLX_MAX_AUDIO_UPLOAD_SIZE overrides the default 100MB cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_MAX_AUDIO_UPLOAD_SIZE": "250MB"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.server.max_audio_upload_size == "250MB"
                assert settings.server.max_audio_upload_bytes() == 250 * 1024 * 1024

    def test_cli_override_max_audio_upload_size(self):
        """--max-audio-upload-size is applied via CLI overrides."""
        from argparse import Namespace

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings.load(
                base_path=tmpdir,
                cli_args=Namespace(max_audio_upload_size="500MB"),
            )
            assert settings.server.max_audio_upload_size == "500MB"
            assert settings.server.max_audio_upload_bytes() == 500 * 1024 * 1024

    def test_env_override_model(self):
        """Test environment variable override for model settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_MODEL_DIR": "/env/models",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.model.model_dir == "/env/models"

    def test_env_override_scheduler(self):
        """Test environment variable override for scheduler settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_MAX_CONCURRENT_REQUESTS": "512"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.scheduler.max_concurrent_requests == 512

    def test_env_override_embedding_batch_size(self):
        """Test environment variable override for embedding batch size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_EMBEDDING_BATCH_SIZE": "24"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.scheduler.embedding_batch_size == 24

    def test_env_override_scheduler_legacy_fallback(self):
        """Test legacy OMLX_MAX_NUM_SEQS env var is accepted as fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_MAX_NUM_SEQS": "256"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.scheduler.max_concurrent_requests == 256

    def test_env_override_cache(self):
        """Test environment variable override for cache settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_CACHE_ENABLED": "false",
                    "OMLX_SSD_CACHE_DIR": "/env/cache",
                    "OMLX_SSD_CACHE_MAX_SIZE": "200GB",
                    "OMLX_GDN_SSD_SPLIT_ENABLED": "true",
                    "OMLX_GDN_SSD_PENDING_MAX_SIZE": "1GB",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.cache.enabled is False
                assert settings.cache.ssd_cache_dir == "/env/cache"
                assert settings.cache.ssd_cache_max_size == "200GB"
                assert settings.cache.gdn_ssd_split_enabled is True
                assert settings.cache.gdn_ssd_pending_max_size == "1GB"

    def test_env_override_initial_cache_blocks(self):
        """Test environment variable override for initial_cache_blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_INITIAL_CACHE_BLOCKS": "16384"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.cache.initial_cache_blocks == 16384

    def test_env_override_cache_enabled_values(self):
        """Test various values for OMLX_CACHE_ENABLED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for value in ["true", "1", "yes"]:
                with patch.dict(os.environ, {"OMLX_CACHE_ENABLED": value}, clear=False):
                    settings = GlobalSettings.load(base_path=tmpdir)
                    assert settings.cache.enabled is True

            for value in ["false", "0", "no"]:
                with patch.dict(os.environ, {"OMLX_CACHE_ENABLED": value}, clear=False):
                    settings = GlobalSettings.load(base_path=tmpdir)
                    assert settings.cache.enabled is False

    def test_env_override_auth(self):
        """Test environment variable override for auth settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OMLX_API_KEY": "env-key"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.auth.api_key == "env-key"

    def test_env_override_hf_endpoint(self):
        """Test environment variable override for HuggingFace settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"OMLX_HF_ENDPOINT": "https://hf-mirror.com"},
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.huggingface.endpoint == "https://hf-mirror.com"

    def test_env_override_invalid_port_logs_warning(self):
        """Test invalid OMLX_PORT logs warning and keeps default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OMLX_PORT": "not-a-number"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                # Should keep default due to parse error
                assert settings.server.port == 8000

    def test_env_override_after_file(self):
        """Test env vars override file settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            with patch.dict(os.environ, {"OMLX_PORT": "8888"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                # Env should override file
                assert settings.server.port == 8888

    def test_cli_override(self):
        """Test CLI argument override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                port=8888,
                host="0.0.0.0",
                log_level="warning",
                model_dir="/cli/models",
                api_key="cli-key",
            )
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 8888
            assert settings.server.host == "0.0.0.0"
            assert settings.server.log_level == "warning"
            assert settings.model.model_dir == "/cli/models"
            assert settings.auth.api_key == "cli-key"

    def test_cli_override_partial(self):
        """Test CLI override with some None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(port=7777, host=None, log_level=None)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 7777
            assert settings.server.host == "127.0.0.1"  # default
            assert settings.server.log_level == "info"  # default

    def test_cli_override_scheduler(self):
        """Test CLI override for scheduler settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(max_concurrent_requests=64, embedding_batch_size=12)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.scheduler.max_concurrent_requests == 64
            assert settings.scheduler.embedding_batch_size == 12

    def test_cli_override_cache(self):
        """Test CLI override for cache settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                cache_enabled=False,
                ssd_cache_dir="/cli/cache",
                ssd_cache_max_size="500GB",
            )
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.cache.enabled is False
            assert settings.cache.ssd_cache_dir == "/cli/cache"
            assert settings.cache.ssd_cache_max_size == "500GB"

    def test_cli_override_paged_ssd_cache(self):
        """Public serve cache flags persist to the matching settings fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                paged_ssd_cache_dir="/cli/cache",
                paged_ssd_cache_max_size="500GB",
                hot_cache_max_size="8GB",
                no_cache=False,
            )
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.cache.enabled = False
            settings._apply_cli_overrides(args)
            assert settings.cache.enabled is True
            assert settings.cache.ssd_cache_dir == "/cli/cache"
            assert settings.cache.ssd_cache_max_size == "500GB"
            assert settings.cache.hot_cache_max_size == "8GB"

    def test_cli_override_no_cache(self):
        """--no-cache persists an explicit cache disablement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings.load(
                base_path=tmpdir,
                cli_args=Namespace(
                    paged_ssd_cache_dir="/cli/cache",
                    no_cache=True,
                ),
            )
            assert settings.cache.enabled is False
            assert settings.cache.ssd_cache_dir == "/cli/cache"

    def test_save_cli_overrides_preserves_runtime_secrets_and_env(self):
        """Saving a CLI setting must not serialize transient overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            stored = GlobalSettings(base_path=base_path)
            stored.auth.api_key = "stored-key"
            stored.cache.enabled = True
            stored.save()

            args = Namespace(port=8765, api_key="cli-key")
            with patch.dict(
                os.environ,
                {"OMLX_API_KEY": "env-key", "OMLX_CACHE_ENABLED": "false"},
                clear=False,
            ):
                runtime = GlobalSettings.load(base_path=base_path, cli_args=args)
                assert runtime.auth.api_key == "cli-key"
                assert runtime.cache.enabled is False
                runtime.save_cli_overrides(args)

            persisted = GlobalSettings.load(base_path=base_path)
            assert persisted.server.port == 8765
            assert persisted.auth.api_key == "stored-key"
            assert persisted.cache.enabled is True

    def test_cli_override_initial_cache_blocks(self):
        """Test CLI override for initial_cache_blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(initial_cache_blocks=4096)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.cache.initial_cache_blocks == 4096

    def test_priority_cli_over_env_over_file(self):
        """Test that CLI > env > file > defaults priority is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file with port 9000
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            # Set env to port 8888
            with patch.dict(os.environ, {"OMLX_PORT": "8888"}, clear=False):
                # CLI sets port to 7777
                args = Namespace(port=7777)
                settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
                # CLI should win
                assert settings.server.port == 7777

            # Without CLI, env should win over file
            settings = GlobalSettings.load(base_path=tmpdir)
            # Env is no longer set, so file should be used
            assert settings.server.port == 9000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.port = 9000
            result = settings.to_dict()

            assert result["version"] == "1.0"
            assert result["base_path"] == tmpdir
            assert result["server"]["port"] == 9000
            assert "model" in result
            assert "scheduler" in result
            assert "cache" in result
            assert "auth" in result

    def test_to_scheduler_config(self):
        """Test conversion to SchedulerConfig."""
        settings = GlobalSettings()
        settings.scheduler.max_concurrent_requests = 128
        settings.scheduler.embedding_batch_size = 12

        scheduler_config = settings.to_scheduler_config()
        assert scheduler_config.max_num_seqs == 128
        assert scheduler_config.completion_batch_size == 128
        assert scheduler_config.embedding_batch_size == 12
        assert scheduler_config.initial_cache_blocks == 256  # default
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_ssd_pending_max_bytes == 512 * 1024**2
        assert scheduler_config.gdn_sidecar_state_dtype == "fp32"

    def test_to_scheduler_config_initial_cache_blocks(self):
        """Test that initial_cache_blocks passes through to SchedulerConfig."""
        settings = GlobalSettings()
        settings.cache.initial_cache_blocks = 8192

        scheduler_config = settings.to_scheduler_config()
        assert scheduler_config.initial_cache_blocks == 8192

    def test_to_scheduler_config_gdn_split_settings(self):
        """GDN settings are attached to the scheduler config for later runtime use."""
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.gdn_ssd_pending_max_size = "1GB"
        settings.cache.gdn_sidecar_state_dtype = "int8"

        scheduler_config = settings.to_scheduler_config()
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_ssd_pending_max_bytes == 1024**3
        assert scheduler_config.gdn_sidecar_state_dtype == "int8"

    def test_to_scheduler_config_rht_int8_gdn_state_dtype(self):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.gdn_sidecar_state_dtype = "rht_int8"

        scheduler_config = settings.to_scheduler_config()
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_sidecar_state_dtype == "rht_int8"


class TestInitSettings:
    """Tests for init_settings and get_settings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Reset settings after each test."""
        reset_settings()

    def test_init_settings(self):
        """Test initializing global settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = init_settings(base_path=tmpdir)
            assert settings is not None
            assert get_settings() is settings

    def test_init_settings_with_cli_args(self):
        """Test initializing with CLI arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(port=9999)
            settings = init_settings(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 9999
            assert get_settings().server.port == 9999

    def test_get_settings_uninitialized_raises(self):
        """Test get_settings raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError, match="Settings not initialized"):
            get_settings()

    def test_reset_settings(self):
        """Test resetting settings allows re-initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize first
            init_settings(base_path=tmpdir)
            assert get_settings() is not None

            # Reset
            reset_settings()

            # Should raise now
            with pytest.raises(RuntimeError):
                get_settings()

            # Can initialize again
            settings = init_settings(base_path=tmpdir)
            assert settings is not None

    def test_multiple_init_overwrites(self):
        """Test calling init_settings multiple times overwrites."""
        with (
            tempfile.TemporaryDirectory() as tmpdir1,
            tempfile.TemporaryDirectory() as tmpdir2,
        ):
            settings1 = init_settings(base_path=tmpdir1)
            settings2 = init_settings(base_path=tmpdir2)

            assert get_settings() is settings2
            # Use resolve() to handle macOS /var -> /private/var symlink
            assert get_settings().base_path.resolve() == Path(tmpdir2).resolve()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_system_memory(self):
        """Test system memory detection."""
        memory = get_system_memory()
        assert memory > 0
        # Should be at least 1GB on any modern system
        assert memory >= 1024**3

    def test_get_system_memory_returns_int(self):
        """Test that get_system_memory returns an integer."""
        memory = get_system_memory()
        assert isinstance(memory, int)

    def test_get_system_memory_uses_sysconf_before_compat(self):
        """macOS should not depend on psutil's HOST_VM_INFO64 adapter."""

        def fake_sysconf(name):
            if name == "SC_PHYS_PAGES":
                return 123
            if name == "SC_PAGE_SIZE":
                return 4096
            raise ValueError(name)

        with (
            patch("omlx.settings.os.sysconf", side_effect=fake_sysconf),
            patch(
                "omlx.utils.psutil_compat.get_total_memory",
                side_effect=AssertionError("compat should not be called"),
            ),
        ):
            assert get_system_memory() == 123 * 4096

    def test_get_system_memory_falls_back_to_compat_when_sysconf_fails(self):
        with (
            patch("omlx.settings.os.sysconf", side_effect=ValueError("unsupported")),
            patch(
                "omlx.utils.psutil_compat.get_total_memory",
                return_value=32 * 1024**3,
            ),
        ):
            assert get_system_memory() == 32 * 1024**3

    def test_get_ssd_capacity(self):
        """Test SSD capacity detection."""
        capacity = get_ssd_capacity(Path("/"))
        assert capacity > 0

    def test_get_ssd_capacity_returns_int(self):
        """Test that get_ssd_capacity returns an integer."""
        capacity = get_ssd_capacity(Path("/"))
        assert isinstance(capacity, int)

    def test_get_ssd_capacity_nonexistent_path(self):
        """Test SSD capacity for non-existent path uses parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does" / "not" / "exist"
            capacity = get_ssd_capacity(nonexistent)
            assert capacity > 0

    def test_get_ssd_capacity_with_string_path(self):
        """Test SSD capacity with string path."""
        capacity = get_ssd_capacity("/")
        assert capacity > 0

    def test_get_ssd_capacity_with_tilde(self):
        """Test SSD capacity with tilde path expansion."""
        capacity = get_ssd_capacity("~/")
        assert capacity > 0


class TestResolveDefaultBasePath:
    """Tests for resolve_default_base_path()."""

    def test_falls_back_to_default_when_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("OMLX_BASE_PATH", raising=False)
        monkeypatch.setattr(
            "omlx.settings.BASE_PATH_BOOTSTRAP_FILE",
            Path("/nonexistent/oMLX/base-path"),
        )
        assert resolve_default_base_path() == Path.home() / ".omlx"

    def test_uses_bootstrap_file_when_present(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OMLX_BASE_PATH", raising=False)
        custom_base = tmp_path / "external-ssd" / "omlx-data"
        bootstrap_file = tmp_path / "base-path"
        bootstrap_file.write_text(f"{custom_base}\n", encoding="utf-8")
        monkeypatch.setattr("omlx.settings.BASE_PATH_BOOTSTRAP_FILE", bootstrap_file)

        assert resolve_default_base_path() == custom_base.resolve()

    def test_env_var_wins_over_bootstrap_file(self, monkeypatch, tmp_path):
        env_base = tmp_path / "env-base"
        bootstrap_base = tmp_path / "bootstrap-base"
        bootstrap_file = tmp_path / "base-path"
        bootstrap_file.write_text(str(bootstrap_base), encoding="utf-8")
        monkeypatch.setattr("omlx.settings.BASE_PATH_BOOTSTRAP_FILE", bootstrap_file)
        monkeypatch.setenv("OMLX_BASE_PATH", str(env_base))

        assert resolve_default_base_path() == env_base.resolve()

    def test_empty_bootstrap_file_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OMLX_BASE_PATH", raising=False)
        bootstrap_file = tmp_path / "base-path"
        bootstrap_file.write_text("   \n", encoding="utf-8")
        monkeypatch.setattr("omlx.settings.BASE_PATH_BOOTSTRAP_FILE", bootstrap_file)

        assert resolve_default_base_path() == Path.home() / ".omlx"

    def test_global_settings_load_uses_resolver_when_no_base_path_given(
        self, monkeypatch, tmp_path
    ):
        resolved = tmp_path / "resolved-base"
        monkeypatch.setattr(
            "omlx.settings.resolve_default_base_path", lambda: resolved
        )

        settings = GlobalSettings.load()

        assert settings.base_path == resolved


class TestSettingsVersionMigration:
    """Tests for settings version handling."""

    def test_load_different_version_logs_migration(self):
        """Test loading settings with different version logs migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "0.9", "server": {"port": 9000}})
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should still load the settings
            assert settings.server.port == 9000

    def test_load_without_version(self):
        """Test loading settings file without version field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(json.dumps({"server": {"port": 9000}}))

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000


class TestSettingsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_settings_file(self):
        """Test loading empty settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text("{}")

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should use all defaults
            assert settings.server.port == 8000
            assert settings.server.host == "127.0.0.1"

    def test_partial_section_in_file(self):
        """Test loading file with partial section data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {"port": 9000},
                        # host and log_level should use defaults
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000
            assert settings.server.host == "127.0.0.1"  # default
            assert settings.server.log_level == "info"  # default

    def test_cli_args_without_expected_attrs(self):
        """Test CLI args object without all expected attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Minimal args object
            args = Namespace(port=9000)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 9000
            # Other settings should be defaults
            assert settings.server.host == "127.0.0.1"

    def test_save_with_unicode_api_key(self):
        """Test saving settings with unicode in values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.auth.api_key = "key-with-unicode-\u4e2d\u6587"
            settings.save()

            # Reload and verify
            loaded = GlobalSettings.load(base_path=tmpdir)
            assert loaded.auth.api_key == "key-with-unicode-\u4e2d\u6587"


class TestSamplingSettings:
    """Tests for SamplingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = SamplingSettings()
        # Fallback default kept at 32768 so existing settings.json
        # files carrying the historical default keep working unchanged
        # after upgrade. ``max_context_window_policy`` is the explicit
        # operator policy cap (None by default).
        assert settings.max_context_window == 32768
        assert settings.max_context_window_policy is None
        assert settings.max_tokens == 32768
        assert settings.temperature == 1.0
        assert settings.top_p == 0.95
        assert settings.top_k == 0
        assert settings.repetition_penalty == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = SamplingSettings(max_context_window=4096)
        d = settings.to_dict()
        assert d["max_context_window"] == 4096
        assert "max_tokens" in d
        assert "repetition_penalty" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "max_context_window": 8192,
            "max_tokens": 1024,
            "repetition_penalty": 1.2,
        }
        settings = SamplingSettings.from_dict(data)
        assert settings.max_context_window == 8192
        assert settings.max_tokens == 1024
        assert settings.repetition_penalty == 1.2

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        settings = SamplingSettings.from_dict({})
        assert settings.max_context_window == 32768
        assert settings.max_context_window_policy is None
        assert settings.repetition_penalty == 1.0

    def test_policy_field_round_trip(self):
        """``max_context_window_policy`` must serialize and
        deserialize without losing its ``None`` semantics."""
        unset = SamplingSettings.from_dict({})
        assert unset.max_context_window_policy is None
        # to_dict preserves None
        d = unset.to_dict()
        assert d["max_context_window_policy"] is None
        # Setting an explicit value round-trips
        with_policy = SamplingSettings.from_dict({"max_context_window_policy": 128_000})
        assert with_policy.max_context_window_policy == 128_000
        assert with_policy.to_dict()["max_context_window_policy"] == 128_000


class TestCORSMiddleware:
    """Test that CORS middleware is correctly applied to the server."""

    def test_cors_preflight(self):
        """Test that CORS preflight requests get proper response headers."""
        from fastapi.testclient import TestClient

        from omlx.server import app, init_server

        # Reset middleware stack so add_middleware works even if app was
        # already started by another test in the same process.
        app.middleware_stack = None

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            init_server(
                model_dirs=[tmpdir],
                global_settings=settings,
            )

            client = TestClient(app)
            resp = client.options(
                "/v1/models",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp.status_code == 200
            assert "access-control-allow-origin" in resp.headers
            assert resp.headers["access-control-allow-origin"] == "*"
