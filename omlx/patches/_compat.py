# SPDX-License-Identifier: Apache-2.0
"""Central guard for the pinned upstream versions the patches target.

Everything under ``omlx.patches`` monkey-patches mlx-lm / mlx-vlm internals by
name (module, class, function). A different runtime version can change those
internals so a patch silently no-ops -- or, worse, corrupts behavior. Keep the
expected versions in one place here and check them once when the patches are
imported, so a mismatch is loud instead of silent.

Bumping any pin in ``pyproject.toml`` must be paired with re-verifying the
patches it covers (run ``tests/test_patches_compat.py``).
"""

from __future__ import annotations

import importlib.metadata as _md
import logging

logger = logging.getLogger(__name__)

# Keep in lockstep with pyproject.toml [project].dependencies.
REQUIRED_VERSIONS: dict[str, str] = {
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-vlm": "0.7.1",
    "mlx-embeddings": "0.1.0",
}


def _installed_version(distribution: str) -> str | None:
    try:
        return _md.version(distribution)
    except _md.PackageNotFoundError:
        return None


def check_pins() -> dict[str, tuple[str | None, str]]:
    """Return ``{distribution: (installed, required)}`` for each mismatch."""
    mismatches: dict[str, tuple[str | None, str]] = {}
    for distribution, required in REQUIRED_VERSIONS.items():
        installed = _installed_version(distribution)
        if installed != required:
            mismatches[distribution] = (installed, required)
    return mismatches


def assert_pins_match(*, strict: bool = False) -> None:
    """Log -- or raise when ``strict`` -- if the runtime drifted from the pins."""
    mismatches = check_pins()
    if not mismatches:
        return
    details = ", ".join(
        f"{dist}={installed or 'missing'} (need {required})"
        for dist, (installed, required) in mismatches.items()
    )
    message = (
        "Upstream version drift: the post-load patches target pinned versions "
        f"but the runtime differs -- {details}. Patches may silently no-op; "
        "re-verify them (tests/test_patches_compat.py) or fix the environment."
    )
    if strict:
        raise RuntimeError(message)
    logger.warning(message)
