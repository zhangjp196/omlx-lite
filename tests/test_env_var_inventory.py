# SPDX-License-Identifier: Apache-2.0
"""Guard the OMLX_* env-var inventory against drift (#10).

``docs/ENV_VARS.md`` is generated from ``omlx/`` by
``scripts/gen_env_docs.py``. This asserts it is current so a new env var (or a
removed one) cannot land without updating the index.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_gen_env_docs", _ROOT / "scripts" / "gen_env_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_env_var_inventory_is_current():
    gen = _load_generator()
    committed = (_ROOT / "docs" / "ENV_VARS.md").read_text(encoding="utf-8")
    assert gen.render(gen.scan()) == committed, (
        "docs/ENV_VARS.md is stale; run: python scripts/gen_env_docs.py"
    )
