# SPDX-License-Identifier: Apache-2.0
"""
Remote model registry for oMLX.

Lets operators register OpenAI-compatible ``/chat/completions`` endpoints
(remote omlx instances, vLLM, OpenAI, ...). Registered remote models are
surfaced in the model list, the chat model selector, and the per-model
settings UI, and chat requests addressed to them are routed straight to the
remote endpoint.

Persisted to ``~/.omlx/remote_models.json`` (the same base path as
``settings.json``).

Usage:
    from omlx.remote_models import init_remote_models, get_remote_model_manager

    # At server startup
    init_remote_models(base_path)

    # Anywhere else
    manager = get_remote_model_manager()
    manager.add(config)
    entry = manager.resolve(model_id)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

logger = logging.getLogger(__name__)

REMOTE_MODELS_VERSION = "1.0"
REMOTE_MODELS_FILENAME = "remote_models.json"

_ID_INVALID_RE = re.compile(r"[^a-z0-9._-]+")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sanitize_model_id(value: str) -> str:
    """Normalize a user-provided gateway model id.

    Lowercases and strips characters that would make the id awkward as an
    OpenAI-style model identifier. Returns the cleaned id.
    """
    cleaned = _ID_INVALID_RE.sub("-", (value or "").strip().lower()).strip("-._")
    return cleaned


class RemoteModelConfig(BaseModel):
    """A registered remote (OpenAI-compatible) model endpoint."""

    model_config = ConfigDict(extra="ignore")

    # Gateway model id: the identifier clients use in /v1/models and
    # /v1/chat/completions. Unique within the registry.
    id: str
    display_name: str = ""
    # OpenAI-compatible chat endpoint, e.g. https://api.openai.com/v1
    base_url: str
    api_key: SecretStr = SecretStr("")
    # The remote model id sent to the endpoint (may differ from the gateway id).
    model: str
    # Extra request-body fields merged into every chat request.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Disabled models are kept in the registry but hidden from listings/chat.
    enabled: bool = True
    # Whether the endpoint accepts multimodal (image) content parts.
    supports_vision: bool = False
    created_at: str = Field(default_factory=_now_iso)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        cleaned = sanitize_model_id(value)
        if not cleaned:
            raise ValueError("id must not be empty")
        return cleaned

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        if not (value or "").strip():
            raise ValueError("model must not be empty")
        return value.strip()

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        url = (value or "").strip().rstrip("/")
        if not url:
            raise ValueError("base_url must not be empty")
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return url

    @property
    def api_key_value(self) -> str:
        return self.api_key.get_secret_value()

    def to_public_dict(self, *, include_api_key: bool = True) -> dict[str, Any]:
        """Serialize for persistence / the admin API.

        The admin API is authenticated, so the key is returned in clear (same
        convention as the benchmark external-endpoint flow) to pre-fill edits.
        """
        data = {
            "id": self.id,
            "display_name": self.display_name,
            "base_url": self.base_url,
            "model": self.model,
            "extra_body": dict(self.extra_body or {}),
            "enabled": self.enabled,
            "supports_vision": self.supports_vision,
            "created_at": self.created_at,
        }
        if include_api_key:
            data["api_key"] = self.api_key_value
        return data


class RemoteModelManager:
    """In-memory registry of remote models, persisted to JSON."""

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self._file = self.base_path / REMOTE_MODELS_FILENAME
        self._models: dict[str, RemoteModelConfig] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", self._file, exc)
            return
        for entry in data.get("models", []):
            try:
                config = RemoteModelConfig(**entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid remote model %r: %s", entry, exc)
                continue
            self._models[config.id] = config

    def _save(self) -> None:
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": REMOTE_MODELS_VERSION,
                "models": [m.to_public_dict() for m in self._models.values()],
            }
            tmp = self._file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._file)
        except OSError as exc:
            logger.error("Failed to save %s: %s", self._file, exc)

    # -- queries -----------------------------------------------------------
    def list_all(self, *, enabled_only: bool = False) -> list[RemoteModelConfig]:
        models = list(self._models.values())
        if enabled_only:
            models = [m for m in models if m.enabled]
        return sorted(models, key=lambda m: m.display_name or m.id)

    def get(self, model_id: str) -> RemoteModelConfig | None:
        if not model_id:
            return None
        if model_id in self._models:
            return self._models[model_id]
        lowered = model_id.lower()
        for config in self._models.values():
            if config.id.lower() == lowered:
                return config
        return None

    def resolve(
        self,
        model_id: str | None,
        settings_manager: Any | None = None,
    ) -> RemoteModelConfig | None:
        """Resolve a gateway id (or a per-model alias) to a remote model.

        Matches the exact id, a case-insensitive id, or a model alias stored in
        the settings manager (mirrors how local model aliases resolve).
        """
        if not model_id:
            return None
        direct = self.get(model_id)
        if direct is not None:
            return direct
        if settings_manager is not None and hasattr(settings_manager, "get_settings"):
            for config in self._models.values():
                try:
                    ms = settings_manager.get_settings(config.id)
                except Exception:  # noqa: BLE001
                    continue
                if ms is not None and getattr(ms, "model_alias", None) == model_id:
                    return config
        return None

    # -- mutation ----------------------------------------------------------
    def add(self, config: RemoteModelConfig) -> RemoteModelConfig:
        if config.id in self._models:
            raise ValueError(f"Remote model already exists: {config.id}")
        self._models[config.id] = config
        self._save()
        return config

    def update(self, model_id: str, updates: dict[str, Any]) -> RemoteModelConfig:
        existing = self.get(model_id)
        if existing is None:
            raise KeyError(f"Remote model not found: {model_id}")
        data = existing.to_public_dict()
        for key, value in updates.items():
            if key in data or key in {"api_key", "extra_body"}:
                data[key] = value
        config = RemoteModelConfig(**data)
        self._models[config.id] = config
        self._save()
        return config

    def delete(self, model_id: str) -> bool:
        existing = self.get(model_id)
        if existing is None:
            return False
        del self._models[existing.id]
        self._save()
        return True

    def set_enabled(self, model_id: str, enabled: bool) -> RemoteModelConfig | None:
        existing = self.get(model_id)
        if existing is None:
            return None
        existing.enabled = enabled
        self._save()
        return existing

    def reset(self) -> None:
        """Clear the in-memory registry (test helper)."""
        self._models.clear()


# Module-level singleton ----------------------------------------------------
_manager: RemoteModelManager | None = None


def init_remote_models(base_path: Path | str) -> RemoteModelManager:
    """Initialize the global remote model registry (server startup)."""
    global _manager
    _manager = RemoteModelManager(base_path)
    return _manager


def get_remote_model_manager() -> RemoteModelManager | None:
    """Return the global remote model registry, if initialized."""
    return _manager


def set_remote_model_manager(manager: RemoteModelManager | None) -> None:
    """Override the global registry (used by tests)."""
    global _manager
    _manager = manager


def reset_remote_models() -> None:
    """Clear the global registry (test helper)."""
    global _manager
    _manager = None


class RemoteChatError(RuntimeError):
    """Raised when a remote endpoint returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Remote endpoint error {status_code}: {detail[:500]}")
        self.status_code = status_code
        self.detail = detail


def _part_to_dict(part: Any) -> Any:
    """Serialize a content part (pydantic model or dict) to a plain dict."""
    if isinstance(part, dict):
        return part
    dump = getattr(part, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    legacy_dump = getattr(part, "dict", None)
    if callable(legacy_dump):
        return legacy_dump(exclude_none=True)
    return part


def _pick_content(message: Any, *, preserve_images: bool = False) -> Any:
    """Extract OpenAI message content from a pydantic/dict message.

    When ``preserve_images`` is set the multimodal content parts are forwarded
    verbatim (OpenAI shape); otherwise only text is kept, since plain text
    endpoints do not consume the multimodal part shape.
    """
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, list):
        if preserve_images:
            return [_part_to_dict(part) for part in content]
        # Keep only text parts; remote text endpoints do not consume the
        # full multimodal part shape. Image/audio parts are dropped.
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return content if content is not None else ""


def remote_request_messages(
    messages: list[Any], *, preserve_images: bool = False
) -> list[dict[str, Any]]:
    """Convert incoming request messages to a plain OpenAI message list."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
        else:
            role = getattr(message, "role", None)
        if role is None:
            continue
        out.append(
            {"role": role, "content": _pick_content(message, preserve_images=preserve_images)}
        )
    return out


class RemoteChatClient:
    """Thin async client for an OpenAI-compatible chat endpoint."""

    def __init__(self, config: RemoteModelConfig, timeout: float = 600.0) -> None:
        self._config = config
        base = config.base_url.rstrip("/")
        self._chat_url = f"{base}/chat/completions"
        self._model = config.model
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = config.api_key_value
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(timeout))

    async def aclose(self) -> None:
        with suppress(Exception):
            await self._client.aclose()

    def _body(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self._model, "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        # Merge operator-provided extra fields (cannot override core fields).
        for key, value in (self._config.extra_body or {}).items():
            if key not in body:
                body[key] = value
        return body

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> dict[str, Any]:
        body = self._body(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=False
        )
        response = await self._client.post(self._chat_url, json=body)
        if response.status_code != 200:
            raise RemoteChatError(response.status_code, response.text)
        return response.json()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw provider ``chat.completion.chunk`` dicts as they arrive."""
        body = self._body(
            messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stream=True
        )
        async with self._client.stream("POST", self._chat_url, json=body) as response:
            if response.status_code != 200:
                error_body = (await response.aread()).decode("utf-8", "replace")
                raise RemoteChatError(response.status_code, error_body)
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except (ValueError, TypeError):
                    continue


def make_chat_client(config: RemoteModelConfig) -> RemoteChatClient:
    """Build a chat client for a remote model config."""
    return RemoteChatClient(config)


def to_external_endpoint_config(config: RemoteModelConfig) -> dict[str, Any]:
    """Shape a remote config into the benchmark external-endpoint payload."""
    return {
        "base_url": config.base_url,
        "api_key": config.api_key_value,
        "model": config.model,
        "extra_body": dict(config.extra_body or {}),
    }
