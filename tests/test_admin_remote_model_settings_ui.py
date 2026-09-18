"""Regression tests for the dedicated remote-model settings panel."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_dashboard_includes_remote_settings_modal():
    dashboard = _read("omlx/admin/templates/dashboard.html")
    assert "dashboard/_modal_remote_model_settings.html" in dashboard
    assert (ROOT / "omlx/admin/templates/dashboard/_modal_remote_model_settings.html").exists()


def test_remote_modal_is_separate_from_local_modal():
    remote = _read("omlx/admin/templates/dashboard/_modal_remote_model_settings.html")
    local = _read("omlx/admin/templates/dashboard/_modal_model_settings.html")

    # Remote panel has its own modal state, so it never renders local-only
    # engine/acceleration controls.
    assert "showRemoteSettingsModal" in remote
    assert "showRemoteSettingsModal" not in local
    for local_only in ("mtp_enabled", "dflash_enabled", "turboquant_kv_enabled", "specprefill_enabled"):
        assert local_only not in remote


def test_remote_modal_edits_endpoint_and_portable_settings():
    remote = _read("omlx/admin/templates/dashboard/_modal_remote_model_settings.html")
    for field in (
        "remoteSettingsForm.base_url",
        "remoteSettingsForm.api_key",
        "remoteSettingsForm.model",
        "remoteSettingsForm.extra_body",
        "remoteSettingsForm.supports_vision",
        "remoteSettings.model_alias",
        "remoteSettings.max_context_window",
        "remoteSettings.temperature",
        "remoteSettings.is_hidden",
        "remoteSettings.is_favorite",
    ):
        assert field in remote


def test_open_model_settings_routes_remote_models_to_remote_panel():
    script = _read("omlx/admin/static/js/dashboard.js")
    open_settings = script.split("async openModelSettings(model) {", 1)[1].split(
        "async importMtplxSidecar()", 1
    )[0]
    assert "this.isRemoteModel(model)" in open_settings
    assert "this.openRemoteModelSettings(model)" in open_settings
    assert "saveRemoteModelSettings()" in script
    assert "/admin/api/remote-models/" in script


def test_remote_settings_i18n_keys_exist_in_every_locale():
    keys = {
        "modal.remote_model_settings.section_label",
        "modal.remote_model_settings.generation_label",
        "modal.remote_model_settings.connection_label",
        "modal.remote_model_settings.availability_label",
        "modal.remote_model_settings.hidden",
        "modal.remote_model_settings.favorite",
        "modal.remote_model_settings.thinking_default",
    }
    for path in sorted((ROOT / "omlx/admin/i18n").glob("*.json")):
        catalog = json.loads(path.read_text())
        missing = keys - set(catalog)
        assert not missing, f"{path.name} is missing {sorted(missing)}"
