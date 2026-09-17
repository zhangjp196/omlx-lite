# SPDX-License-Identifier: Apache-2.0
"""Tests for the remote model registry, chat routing, and admin endpoints."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

import omlx.server as server
from omlx.admin import routes as admin_routes
from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.model_settings import ModelSettingsManager
from omlx.remote_models import (
    RemoteChatClient,
    RemoteChatError,
    RemoteModelConfig,
    RemoteModelManager,
    get_remote_model_manager,
    init_remote_models,
    remote_request_messages,
    sanitize_model_id,
    set_remote_model_manager,
    to_external_endpoint_config,
)

# ---------------------------------------------------------------------------
# sanitize / config
# ---------------------------------------------------------------------------


def test_sanitize_model_id():
    assert sanitize_model_id("My Model") == "my-model"
    assert sanitize_model_id("  A_B.c  ") == "a_b.c"
    assert sanitize_model_id("!!!") == ""
    assert sanitize_model_id("a b c") == "a-b-c"
    assert sanitize_model_id("a--b__c") == "a--b__c"


def test_config_validation_id_sanitized():
    cfg = RemoteModelConfig(id="My Remote", base_url="https://x/v1", model="m")
    assert cfg.id == "my-remote"


def test_config_requires_http_base_url():
    with pytest.raises(ValidationError):
        RemoteModelConfig(id="x", base_url="ftp://nope", model="m")


def test_config_requires_model():
    with pytest.raises(ValidationError):
        RemoteModelConfig(id="x", base_url="https://x/v1", model="  ")


def test_config_base_url_strips_trailing_slash():
    cfg = RemoteModelConfig(id="x", base_url="https://api.openai.com/v1/", model="m")
    assert cfg.base_url == "https://api.openai.com/v1"


def test_to_public_dict_hides_key_when_requested():
    cfg = RemoteModelConfig(id="x", base_url="https://x/v1", model="m", api_key="sk-abc")
    pub = cfg.to_public_dict(include_api_key=False)
    assert "api_key" not in pub
    assert cfg.to_public_dict()["api_key"] == "sk-abc"


def test_config_supports_vision_default_and_roundtrip():
    assert RemoteModelConfig(id="x", base_url="https://x/v1", model="m").supports_vision is False
    cfg = RemoteModelConfig(id="x", base_url="https://x/v1", model="m", supports_vision=True)
    assert cfg.supports_vision is True
    assert cfg.to_public_dict()["supports_vision"] is True
    assert RemoteModelConfig(**cfg.to_public_dict()).supports_vision is True


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_manager_crud_and_persistence(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="gpt", base_url="https://api.openai.com/v1",
                              api_key="sk-test", model="gpt-4o", display_name="GPT-4o"))
    # A fresh manager over the same dir reloads it.
    mgr2 = RemoteModelManager(tmp_path)
    assert len(mgr2.list_all()) == 1
    assert mgr2.get("gpt").model == "gpt-4o"
    assert mgr2.get("gpt").api_key_value == "sk-test"


def test_manager_add_duplicate_raises(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="a", base_url="https://x/v1", model="m"))
    with pytest.raises(ValueError):
        mgr.add(RemoteModelConfig(id="a", base_url="https://x/v1", model="m"))


def test_manager_update_and_delete(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="a", base_url="https://x/v1", model="m"))
    cfg = mgr.update("a", {"display_name": "A", "enabled": False})
    assert cfg.display_name == "A" and cfg.enabled is False
    assert mgr.delete("a") is True
    assert mgr.get("a") is None
    assert mgr.delete("a") is False


def test_manager_get_case_insensitive(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="my-model", base_url="https://x/v1", model="m"))
    assert mgr.get("MY-MODEL").id == "my-model"


def test_manager_resolve_by_alias(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="my-model", base_url="https://x/v1", model="m"))
    sm = ModelSettingsManager(tmp_path)
    s = sm.get_settings("my-model")
    s.model_alias = "myalias"
    sm.set_settings("my-model", s)
    assert mgr.resolve("my-model", sm).id == "my-model"
    assert mgr.resolve("myalias", sm).id == "my-model"
    assert mgr.resolve("nope", sm) is None


def test_manager_list_enabled_only(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="a", base_url="https://x/v1", model="m"))
    mgr.add(RemoteModelConfig(id="b", base_url="https://x/v1", model="m", enabled=False))
    assert len(mgr.list_all()) == 2
    assert [m.id for m in mgr.list_all(enabled_only=True)] == ["a"]


def test_singleton_lifecycle():
    set_remote_model_manager(None)
    assert get_remote_model_manager() is None
    init_remote_models("/tmp/omlx-test-remote-registry")
    assert get_remote_model_manager() is not None
    set_remote_model_manager(None)
    assert get_remote_model_manager() is None


# ---------------------------------------------------------------------------
# message normalization + client body
# ---------------------------------------------------------------------------


def test_remote_request_messages_normalizes_parts():
    out = remote_request_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "yo"},
                                          {"type": "image_url", "image_url": {"url": "x"}}]},
    ])
    assert out == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]


def test_remote_request_messages_accepts_objects():
    out = remote_request_messages([Message(role="user", content="hello")])
    assert out == [{"role": "user", "content": "hello"}]


def test_remote_request_messages_preserve_images():
    parts = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    dropped = remote_request_messages([{"role": "user", "content": parts}])
    assert dropped == [{"role": "user", "content": "look"}]
    kept = remote_request_messages(
        [{"role": "user", "content": parts}], preserve_images=True
    )
    assert kept == [{"role": "user", "content": parts}]


def test_remote_request_messages_preserve_images_pydantic_parts():
    from omlx.api.openai_models import ContentPart, ImageURL

    msg = Message(
        role="user",
        content=[
            ContentPart(type="text", text="see"),
            ContentPart(type="image_url", image_url=ImageURL(url="data:image/png;base64,BBB")),
        ],
    )
    out = remote_request_messages([msg], preserve_images=True)
    assert out[0]["content"][0] == {"type": "text", "text": "see"}
    assert out[0]["content"][1]["type"] == "image_url"
    assert out[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,BBB"


def test_client_body_extrabody_and_stream_options():
    cfg = RemoteModelConfig(id="x", base_url="https://x/v1", model="m",
                            extra_body={"top_logprobs": 5})
    client = RemoteChatClient(cfg)
    body = client._body([{"role": "user", "content": "hi"}],
                        max_tokens=10, temperature=0.0, top_p=1.0, stream=True)
    assert body["model"] == "m"
    assert body["max_tokens"] == 10
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["top_logprobs"] == 5
    # Extra body cannot override core fields.
    cfg2 = RemoteModelConfig(id="x", base_url="https://x/v1", model="m",
                             extra_body={"model": "evil", "foo": 1})
    body2 = RemoteChatClient(cfg2)._body([{"role": "user", "content": "hi"}],
                                         max_tokens=1, temperature=0.0, top_p=1.0, stream=False)
    assert body2["model"] == "m"
    assert body2["foo"] == 1


def test_to_external_endpoint_config():
    cfg = RemoteModelConfig(id="x", base_url="https://x/v1", api_key="sk-1",
                            model="m", extra_body={"a": 1})
    assert to_external_endpoint_config(cfg) == {
        "base_url": "https://x/v1", "api_key": "sk-1", "model": "m", "extra_body": {"a": 1},
    }


# ---------------------------------------------------------------------------
# chat routing
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_state():
    """Snapshot and restore the global server state around a test."""
    snap = {
        "remote_model_manager": server._server_state.remote_model_manager,
        "settings_manager": server._server_state.settings_manager,
        "engine_pool": server._server_state.engine_pool,
        "sampling": server._server_state.sampling,
    }
    yield server._server_state
    server._server_state.remote_model_manager = snap["remote_model_manager"]
    server._server_state.settings_manager = snap["settings_manager"]
    server._server_state.engine_pool = snap["engine_pool"]
    server._server_state.sampling = snap["sampling"]


def _remote_manager(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="gpt", base_url="https://api.openai.com/v1",
                              api_key="sk-test", model="gpt-4o"))
    return mgr


@pytest.mark.asyncio
async def test_create_chat_completion_routes_to_remote(saved_state, tmp_path):
    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=False,
    )
    fake = MagicMock()
    fake.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    })
    fake.aclose = AsyncMock()

    with patch("omlx.server.make_chat_client", return_value=fake):
        resp = await server.create_chat_completion(request, MagicMock())

    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "Hello"
    assert body["usage"]["total_tokens"] == 3
    fake.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_chat_completion_remote_preserves_images_when_vision(saved_state, tmp_path):
    state = saved_state
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="vlm", base_url="https://x/v1", model="vlm-1",
                              supports_vision=True))
    state.remote_model_manager = mgr
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="vlm",
        messages=[Message(role="user", content=[
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZZZ"}},
        ])],
        stream=False,
    )
    fake = MagicMock()
    fake.chat = AsyncMock(return_value={
        "choices": [{"message": {"content": "a cat"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
    fake.aclose = AsyncMock()

    with patch("omlx.server.make_chat_client", return_value=fake):
        await server.create_chat_completion(request, MagicMock())

    sent = fake.chat.await_args.args[0]
    assert sent[0]["content"][0] == {"type": "text", "text": "what is this"}
    assert sent[0]["content"][1]["type"] == "image_url"
    assert sent[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,ZZZ"


@pytest.mark.asyncio
async def test_create_chat_completion_remote_error_is_502(saved_state, tmp_path):
    from fastapi import HTTPException

    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=False,
    )
    fake = MagicMock()
    fake.chat = AsyncMock(side_effect=RemoteChatError(500, "boom"))
    fake.aclose = AsyncMock()

    with (
        patch("omlx.server.make_chat_client", return_value=fake),
        pytest.raises(HTTPException) as exc,
    ):
        await server.create_chat_completion(request, MagicMock())
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_create_chat_completion_disabled_remote_falls_through(saved_state, tmp_path):
    state = saved_state
    mgr = _remote_manager(tmp_path)
    mgr.set_enabled("gpt", False)
    state.remote_model_manager = mgr
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=False,
    )
    fake = MagicMock()
    fake.chat = AsyncMock()
    fake.aclose = AsyncMock()

    class _LocalPathHitError(Exception):
        pass

    get_engine = AsyncMock(side_effect=_LocalPathHitError())
    with (
        patch("omlx.server.make_chat_client", return_value=fake),
        patch("omlx.server.get_engine_for_model", new=get_engine),
        pytest.raises(_LocalPathHitError),
    ):
        await server.create_chat_completion(request, MagicMock())
    # The request reached the local engine path, not the remote proxy.
    get_engine.assert_awaited_once()
    fake.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_remote_stream_chat_yields_openai_chunks(saved_state, tmp_path):
    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=True,
    )

    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        yield {"choices": [{"delta": {"content": "lo"}}]}
        yield {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}

    fake = MagicMock()
    fake.stream = fake_stream
    fake.aclose = AsyncMock()

    with patch("omlx.server.make_chat_client", return_value=fake):
        resp = await server.create_chat_completion(request, MagicMock())

    chunks = []
    async for event in resp.body_iterator:
        for line in event.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                chunks.append(line[len("data:"):].strip())
    # First chunk is the role-only chunk; then content chunks; then [DONE].
    assert "role" in chunks[0]
    assert any("Hel" in c for c in chunks)
    assert any("lo" in c for c in chunks)
    assert chunks[-1] == "[DONE]"


@pytest.mark.asyncio
async def test_remote_stream_chat_usage_chunk_has_timing_metrics(saved_state, tmp_path):
    """Remote streams must emit a choices:[] usage chunk with TPS so the chat
    performance panel does not render 0.0 for external models."""
    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=True,
    )

    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {"choices": [{"delta": {"content": "Hel"}}]}
        yield {"choices": [{"delta": {"content": "lo"}}]}
        yield {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    fake = MagicMock()
    fake.stream = fake_stream
    fake.aclose = AsyncMock()

    ticks = iter([0.0, 1.0, 2.0])
    with (
        patch("omlx.server.make_chat_client", return_value=fake),
        patch("omlx.server.time.perf_counter", side_effect=lambda: next(ticks)),
    ):
        resp = await server.create_chat_completion(request, MagicMock())
        payloads = []
        async for event in resp.body_iterator:
            for line in event.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw and raw != "[DONE]":
                        payloads.append(json.loads(raw))

    usage_payloads = [p for p in payloads if p.get("usage")]
    assert len(usage_payloads) == 1
    usage_chunk = usage_payloads[0]
    assert usage_chunk["choices"] == []
    usage = usage_chunk["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 2
    assert usage["prompt_eval_duration"] > 0
    assert usage["generation_duration"] > 0
    assert usage["prompt_tokens_per_second"] > 0
    assert usage["generation_tokens_per_second"] > 0


@pytest.mark.asyncio
async def test_remote_stream_chat_forwards_upstream_timing_metrics(saved_state, tmp_path):
    """When the upstream endpoint reports its own timing, forward it verbatim
    instead of the proxy-measured values."""
    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    state.settings_manager = None
    state.engine_pool = None

    request = ChatCompletionRequest(
        model="gpt", messages=[Message(role="user", content="hi")], stream=True,
    )

    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "hi"}}]}
        yield {
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "prompt_tokens_per_second": 123.45,
                "generation_tokens_per_second": 67.89,
                "time_to_first_token": 0.42,
            },
        }

    fake = MagicMock()
    fake.stream = fake_stream
    fake.aclose = AsyncMock()

    with patch("omlx.server.make_chat_client", return_value=fake):
        resp = await server.create_chat_completion(request, MagicMock())

    payloads = []
    async for event in resp.body_iterator:
        for line in event.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw and raw != "[DONE]":
                    payloads.append(json.loads(raw))

    usage = next(p["usage"] for p in payloads if p.get("usage"))
    assert usage["prompt_tokens_per_second"] == 123.45
    assert usage["generation_tokens_per_second"] == 67.89
    assert usage["time_to_first_token"] == 0.42


@pytest.mark.asyncio
async def test_remote_effective_params_priority(saved_state, tmp_path):
    state = saved_state
    state.remote_model_manager = _remote_manager(tmp_path)
    sm = ModelSettingsManager(tmp_path)
    s = sm.get_settings("gpt")
    s.max_tokens = 111
    s.temperature = 0.5
    sm.set_settings("gpt", s)
    state.settings_manager = sm

    # Request value wins over per-model setting.
    req = ChatCompletionRequest(model="gpt", messages=[Message(role="user", content="hi")],
                                max_tokens=5, temperature=0.1)
    mt, temp, top_p = server._remote_effective_params(req, "gpt")
    assert mt == 5 and temp == 0.1

    # Per-model setting wins over global default.
    req2 = ChatCompletionRequest(model="gpt", messages=[Message(role="user", content="hi")])
    mt2, temp2, top_p2 = server._remote_effective_params(req2, "gpt")
    assert mt2 == 111 and temp2 == 0.5


# ---------------------------------------------------------------------------
# resolve_model_id remote fallback
# ---------------------------------------------------------------------------


def test_resolve_model_id_remote_fallback(saved_state, tmp_path):
    state = saved_state
    state.engine_pool = None
    state.remote_model_manager = _remote_manager(tmp_path)
    assert server.resolve_model_id("gpt") == "gpt"
    assert server.resolve_model_id(None) is None
    assert server.resolve_model_id("not-registered") == "not-registered"


def test_resolve_model_id_remote_via_alias(saved_state, tmp_path):
    state = saved_state
    state.engine_pool = None
    state.remote_model_manager = _remote_manager(tmp_path)
    sm = ModelSettingsManager(tmp_path)
    s = sm.get_settings("gpt")
    s.model_alias = "gpt-alias"
    sm.set_settings("gpt", s)
    state.settings_manager = sm
    assert server.resolve_model_id("gpt-alias") == "gpt"


# ---------------------------------------------------------------------------
# /v1/models/status remote exposure
# ---------------------------------------------------------------------------


def _fake_pool():
    pool = MagicMock()
    pool.get_status.return_value = {
        "final_ceiling": 0,
        "current_model_memory": 0,
        "model_count": 0,
        "loaded_count": 0,
        "load_seconds_per_gb_estimate": 0.0,
        "load_time_observations": 0,
        "models": [],
    }
    pool.resolve_model_id.side_effect = lambda model_id, sm=None: model_id
    pool.get_entry.return_value = None
    return pool


def _sampling():
    return SimpleNamespace(
        max_tokens=8192, max_context_window=32768, max_context_window_policy=None
    )


@pytest.mark.asyncio
async def test_models_status_includes_remote_vision_model(saved_state, tmp_path):
    state = saved_state
    mgr = RemoteModelManager(tmp_path)
    mgr.add(
        RemoteModelConfig(
            id="remote-vlm",
            base_url="https://x/v1",
            model="vlm-1",
            supports_vision=True,
        )
    )
    state.remote_model_manager = mgr
    state.settings_manager = None
    state.engine_pool = _fake_pool()
    state.sampling = _sampling()

    status = await server.list_models_status(True)

    entry = next(m for m in status["models"] if m["id"] == "remote-vlm")
    assert entry["model_type"] == "vlm"
    assert entry["source_type"] == "remote"
    assert entry["engine_type"] == "remote"
    assert entry["is_remote"] is True
    assert entry["supports_vision"] is True


@pytest.mark.asyncio
async def test_models_status_remote_text_and_hidden_excluded(saved_state, tmp_path):
    state = saved_state
    mgr = RemoteModelManager(tmp_path)
    mgr.add(RemoteModelConfig(id="remote-text", base_url="https://x/v1", model="t"))
    mgr.add(RemoteModelConfig(id="remote-hidden", base_url="https://x/v1", model="h"))
    sm = ModelSettingsManager(tmp_path)
    hidden = sm.get_settings("remote-hidden")
    hidden.is_hidden = True
    sm.set_settings("remote-hidden", hidden)
    state.remote_model_manager = mgr
    state.settings_manager = sm
    state.engine_pool = _fake_pool()
    state.sampling = _sampling()

    status = await server.list_models_status(True)

    ids = {m["id"] for m in status["models"]}
    assert "remote-text" in ids
    assert "remote-hidden" not in ids
    entry = next(m for m in status["models"] if m["id"] == "remote-text")
    assert entry["model_type"] == "llm"
    assert entry["supports_vision"] is False


# ---------------------------------------------------------------------------
# admin endpoints
# ---------------------------------------------------------------------------


def _patch_remote_mgr(mgr, settings_manager=None):
    return (
        patch("omlx.remote_models.get_remote_model_manager", return_value=mgr),
        patch("omlx.admin.routes._get_settings_manager", return_value=settings_manager),
    )


@pytest.mark.asyncio
async def test_admin_list_remote_models(tmp_path):
    mgr = _remote_manager(tmp_path)
    sm = ModelSettingsManager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, sm)
    with p1, p2:
        entries = await admin_routes.list_remote_models(True)
    assert len(entries) == 1
    assert entries[0]["source_type"] == "remote"
    assert entries[0]["settings"]  # a ModelSettings dict


@pytest.mark.asyncio
async def test_admin_create_and_update_remote_model(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, None)
    req = admin_routes.RemoteModelRequest(
        id="My Model", display_name="GPT", base_url="https://api.openai.com/v1",
        api_key="sk-1", model="gpt-4o",
    )
    with p1, p2:
        created = await admin_routes.create_remote_model(req, True)
    assert created["id"] == "my-model"

    req2 = admin_routes.RemoteModelRequest(
        id="my-model", display_name="GPT-4o", base_url="https://api.openai.com/v1",
        api_key="sk-1", model="gpt-4o", enabled=False, supports_vision=True,
    )
    with p1, p2:
        updated = await admin_routes.update_remote_model("my-model", req2, True)
    assert updated["enabled"] is False
    assert updated["supports_vision"] is True


@pytest.mark.asyncio
async def test_admin_create_remote_model_invalid_id(tmp_path):
    mgr = RemoteModelManager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, None)
    from fastapi import HTTPException

    req = admin_routes.RemoteModelRequest(
        id="!!!", base_url="https://x/v1", model="m",
    )
    with p1, p2, pytest.raises(HTTPException) as exc:
        await admin_routes.create_remote_model(req, True)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_admin_delete_remote_model(tmp_path):
    mgr = _remote_manager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, None)
    from fastapi import HTTPException

    with p1, p2:
        result = await admin_routes.delete_remote_model("gpt", True)
        assert result["success"] is True
        with pytest.raises(HTTPException) as exc:
            await admin_routes.delete_remote_model("gpt", True)
        assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_admin_test_remote_model(tmp_path):
    mgr = _remote_manager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, None)

    async def fake_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        yield {"choices": [{"delta": {"content": "O"}}]}
        yield {
            "choices": [{"delta": {"content": "K"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    fake = MagicMock()
    fake.stream = fake_stream
    fake.chat = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})
    fake.aclose = AsyncMock()
    with p1, p2, patch("omlx.remote_models.make_chat_client", return_value=fake):
        result = await admin_routes.test_remote_model("gpt", True)
    assert result["success"] is True
    assert result["latency_ms"] is not None
    assert result["total_ms"] is not None
    assert result["completion_tokens"] == 2
    assert result["tokens_per_sec"] is not None
    fake.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_test_remote_model_stream_fallback(tmp_path):
    mgr = _remote_manager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, None)

    async def failing_stream(*args, **kwargs):
        raise RuntimeError("streaming unsupported")
        yield {}  # pragma: no cover

    fake = MagicMock()
    fake.stream = failing_stream
    fake.chat = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    fake.aclose = AsyncMock()
    with p1, p2, patch("omlx.remote_models.make_chat_client", return_value=fake):
        result = await admin_routes.test_remote_model("gpt", True)
    assert result["success"] is True
    assert result["latency_ms"] is not None
    assert result["tokens_per_sec"] is None
    fake.chat.assert_awaited_once()


# ---------------------------------------------------------------------------
# update_model_settings remote branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_remote_model_settings(tmp_path):
    mgr = _remote_manager(tmp_path)
    sm = ModelSettingsManager(tmp_path)
    p1, p2 = _patch_remote_mgr(mgr, sm)
    p3 = patch("omlx.admin.routes._get_engine_pool", return_value=MagicMock())
    req = admin_routes.ModelSettingsRequest(model_alias="gpt-alias", temperature=0.7,
                                            max_context_window=8192)
    with p1, p2, p3:
        result = await admin_routes.update_model_settings("gpt", req, is_admin=True)
    assert result["success"] is True
    assert result["engine_type"] == "remote"
    assert result["settings"]["model_alias"] == "gpt-alias"
    assert result["settings"]["temperature"] == 0.7
    # Persisted on the settings manager.
    assert sm.get_settings("gpt").model_alias == "gpt-alias"
    assert sm.get_settings("gpt").max_context_window == 8192
