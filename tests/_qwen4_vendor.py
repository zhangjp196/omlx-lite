# SPDX-License-Identifier: Apache-2.0
"""Module-level guard for tests that exercise oMLX's vendored Qwen4-Exp.

The overlay under ``omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm``
targets a custom mlx-vlm build: it relies on helpers the pinned PyPI mlx-vlm
does not provide (a three-argument ``_target_verify_linear``,
``Qwen3_5GatedDeltaNet(gdn_sink=...)``, ``configure_ple_runtime``, ...). When
the overlay cannot be activated, upstream's own ``qwen4_exp`` is used instead;
the affected tests are skipped with an explanatory reason rather than erroring
during collection or asserting against that divergent implementation.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

_VENDOR_QWEN4_EXP = (
    Path(compat.__file__).resolve().parent
    / "vendor"
    / "mlx_vlm"
    / "models"
    / "qwen4_exp"
).resolve()


def skip_if_unavailable() -> None:
    """Skip the calling module when the vendored Qwen4-Exp overlay is inactive."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    try:
        module = importlib.import_module("mlx_vlm.models.qwen4_exp")
        active = Path(module.__file__).resolve().parent == _VENDOR_QWEN4_EXP
        reason = f"mlx-vlm ships its own qwen4_exp ({module.__file__})"
    except Exception as exc:  # noqa: BLE001
        active = False
        reason = str(exc)

    if not active:
        pytest.skip(
            "oMLX's vendored Qwen4-Exp overlay requires a custom mlx-vlm build "
            f"that this fork does not pin: {reason}",
            allow_module_level=True,
        )
