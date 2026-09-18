# SPDX-License-Identifier: Apache-2.0
"""Guard the pinned upstream versions the post-load patches target (#9).

The patches monkey-patch mlx-lm / mlx-vlm internals by name; a version drift
makes them silently no-op. These tests fail loudly instead.
"""

from __future__ import annotations

from omlx.patches._compat import REQUIRED_VERSIONS, check_pins


def test_runtime_matches_pinned_versions():
    mismatches = check_pins()
    assert not mismatches, (
        "runtime drifted from the versions the patches target: "
        f"{mismatches}. Re-verify the patches or fix the environment."
    )


def test_pins_cover_the_core_upstreams():
    assert {"mlx", "mlx-lm", "mlx-vlm"} <= set(REQUIRED_VERSIONS)


def test_llama4_attention_patch_takes_effect():
    """A representative unconditional patch must install its marker.

    If mlx-lm renames or reshapes ``llama4.Attention``, the patch no-ops and
    this fails -- which is the whole point of the smoke test.
    """
    from mlx_lm.models import llama4

    from omlx.patches.llama4_attention import (
        _PATCH_MARKER,
        apply_llama4_attention_patch,
    )

    apply_llama4_attention_patch()
    current = llama4.Attention.__dict__.get("__call__")
    assert getattr(current, _PATCH_MARKER, False), (
        "llama4_attention patch did not take effect; mlx-lm's "
        "llama4.Attention shape changed (version drift)."
    )
