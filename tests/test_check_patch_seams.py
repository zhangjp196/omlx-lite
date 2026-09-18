# SPDX-License-Identifier: Apache-2.0
"""The patch-seam checker must actually flag a missing upstream attribute.

The checker (``scripts/check_patch_seams.py``) is report-only to avoid
false-positive CI failures, so this pins its detection logic: a probe that
reads a non-existent attribute off a real mlx-vlm module must be listed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_check_patch_seams", _ROOT / "scripts" / "check_patch_seams.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_flags_missing_upstream_attribute(tmp_path):
    checker = _load_checker()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from mlx_vlm.models.qwen3_5 import language as q35\n\n\n"
        "def run():\n"
        "    return q35._definitely_missing_seam_xyz()\n",
        encoding="utf-8",
    )
    hits = checker.check_file(probe, skip_names=set())
    assert hits, "checker failed to flag a missing upstream attribute"
    assert any("_definitely_missing_seam_xyz" in ref for _, ref, _ in hits)


def test_checker_ignores_guarded_reference(tmp_path):
    checker = _load_checker()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from mlx_vlm.models.qwen3_5 import language as q35\n\n\n"
        "def run():\n"
        '    fn = getattr(q35, "_target_verify_linear", None)\n'
        "    return fn\n",
        encoding="utf-8",
    )
    assert checker.check_file(probe, skip_names=set()) == []


def test_repo_has_no_unguarded_upstream_seam_references():
    """The patch tree must stay free of unguarded upstream-symbol reads.

    A patch that reads an attribute the pinned mlx-lm / mlx-vlm does not
    expose crashes the engine at model load or inference -- the class of bug
    behind the qwen3.5 seam crashes. Guard against a regression by running the
    checker over the whole tree.
    """
    checker = _load_checker()
    skip = checker._omlx_defined_names()
    hits = []
    for path in sorted((_ROOT / "omlx" / "patches").rglob("*.py")):
        if "vendor" in path.parts:
            continue
        for lineno, ref, upstream in checker.check_file(path, skip_names=skip):
            hits.append(
                f"{path.relative_to(_ROOT)}:{lineno}: {ref} (from {upstream})"
            )
    assert not hits, (
        "unguarded references to upstream symbols the pinned deps do not "
        "expose:\n" + "\n".join(hits)
    )
