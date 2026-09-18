# SPDX-License-Identifier: Apache-2.0
"""Report omlx/patches attribute accesses on upstream modules the installed
mlx-lm / mlx-vlm does not expose.

The qwen3.5 target-verify seam rename in mlx-vlm 0.7.1 silently disabled
several patches -- and crashed one (``Qwen3_5MoeSparseMoeBlock.__call__() got
an unexpected keyword argument 'target_verify'``). This is the standing
detector for that class of drift. It is **report-only** by default: upstream
introspection has benign false-positive categories, so it must not gate CI on
its own.

Filtered as benign:
- ``_omlx_*`` / dunder attributes: markers the patches inject themselves.
- an attribute named as a string literal anywhere in the file: treated as
  guarded (``hasattr`` / ``getattr`` / a ``needed`` tuple).
- assignment targets (``alias.attr = ...``): the patch is installing it.
- names oMLX itself defines (vendored / MTP classes): injected, not drift.
- attributes on the bare ``mlx_lm.models`` / ``mlx_vlm.models`` packages:
  dynamic submodule registration.
- aliases that do not resolve (vendored / optional upstream): skipped.

Usage::

    python scripts/check_patch_seams.py            # report, always exit 0
    python scripts/check_patch_seams.py --strict    # exit 1 when anything is listed
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCHES = ROOT / "omlx" / "patches"
_UPSTREAM = ("mlx_lm", "mlx_vlm")
_SUBMODULE_PACKAGES = {"mlx_lm.models", "mlx_vlm.models"}


def _aliases(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(_UPSTREAM):
                    found[a.asname or a.name.split(".")[0]] = a.name
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith(_UPSTREAM)
        ):
            for a in node.names:
                if a.name != "*":
                    found[a.asname or a.name] = f"{node.module}.{a.name}"
    return found


def _assignment_target_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for target in ast.walk(node):
                if isinstance(target, ast.Attribute):
                    ids.add(id(target))
    return ids


def _omlx_defined_names() -> set[str]:
    """Names oMLX defines anywhere (classes/functions/assignments).

    Vendored and MTP modules the patches inject register attributes such as
    ``MTPDecoderLayer`` on upstream modules; those are not drift.
    """
    names: set[str] = set()
    pattern = re.compile(r"^\s*(?:def|class)\s+(\w+)|^\s*(\w+)\s*=")
    for path in (ROOT / "omlx").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            m = pattern.match(line)
            if m:
                names.add(m.group(1) or m.group(2))
    return names


def _resolve(target: str):
    parent, _, leaf = target.rpartition(".")
    try:
        return getattr(importlib.import_module(parent), leaf, None)
    except Exception:
        return None


def check_file(path: pathlib.Path, skip_names: set[str]) -> list[tuple[int, str, str]]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    aliases = _aliases(tree)
    targets = _assignment_target_ids(tree)
    mentioned = set(re.findall(r'["\'](\w+)["\']', text))
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
            continue
        base = node.value.id
        upstream = aliases.get(base)
        if upstream is None or upstream in _SUBMODULE_PACKAGES:
            continue
        attr = node.attr
        if attr.startswith("_omlx_") or attr.startswith("__"):
            continue
        if id(node) in targets or attr in mentioned or attr in skip_names:
            continue
        obj = _resolve(upstream)
        if obj is None:
            continue  # alias itself not importable here
        if not hasattr(obj, attr):
            hits.append((node.lineno, f"{base}.{attr}", upstream))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    skip_names = _omlx_defined_names()
    total = 0
    for path in sorted(PATCHES.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        hits = check_file(path, skip_names)
        if hits:
            total += len(hits)
            print(f"\n{path.relative_to(ROOT)}:")
            for lineno, ref, upstream in hits:
                print(f"  L{lineno}: {ref}   (from {upstream})")
    print(f"\n{total} reference(s) to check against the pinned upstream")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
