# SPDX-License-Identifier: Apache-2.0
"""Machine-independent performance regression gates for per-token hot paths.

These assert SCALING (O(N) vs O(N^2)), not absolute wall-clock, so they stay
stable on shared CI runners. They guard the optimizations recorded in
``todo/performance-adjustments.md``:

- **#2** ``Request.append_output_text`` / ``output_text`` -- parts-append plus
  a join on read, instead of a per-token ``str +=`` (O(N^2)).
- **A1** ``get_iogpu_wired_limit_bytes`` -- the ~6ms ``sysctl`` subprocess read
  is cached, so repeated admin/telemetry reads do not fork per call.

Two source-shape tripwires cover the scheduler call sites (#1 and the #2
accumulator), which need a model to execute and therefore cannot be timed in a
unit test. They fail loudly if the quadratic form is reintroduced.

Run: ``python -m pytest -q tests/test_perf_regression.py``
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import omlx.process_memory_enforcer as pme
from omlx.request import Request

_SCHEDULER_SRC = (
    Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
).read_text(encoding="utf-8")

# Compare a workload at N and 4N. A linear path scales ~4x; a quadratic one
# ~16x. The gap is wide enough that a 9x threshold is noise-tolerant on CI.
_N_SMALL = 8_000
_N_LARGE = 32_000
_MAX_LINEAR_RATIO = 9.0
_REPEATS = 7


# --------------------------------------------------------------------------- #
# #1 output_token_ids: the full token list is only consumed on finish.
# --------------------------------------------------------------------------- #
def test_scheduler_defers_output_token_ids_materialization():
    """#1 tripwire: the per-step snapshot must be gated on ``is_finished``.

    An unconditional ``output_token_ids=list(request.output_token_ids)`` copies
    the whole list every step (O(N^2) per request, ~918ms over 32k tokens).
    """
    assert re.search(
        r"output_token_ids\s*=\s*list\(request\.output_token_ids\)"
        r"\s+if\s+is_finished\s+else\s+\[\]",
        _SCHEDULER_SRC,
    ), (
        "scheduler.py no longer gates the output_token_ids snapshot on "
        "is_finished; the per-step O(N) copy (O(N^2) per request) is back. "
        "See todo/performance-adjustments.md #1."
    )


# --------------------------------------------------------------------------- #
# #2 output_text: accumulate parts (O(1)/token), join on read.
# --------------------------------------------------------------------------- #
def test_scheduler_uses_append_output_text():
    """#2 tripwire: the scheduler must not do ``request.output_text += ...``.

    The attribute-state string has refcount >= 2, so CPython cannot extend it
    in place and each append copies the whole cumulative text (O(N^2)).
    """
    assert "append_output_text(" in _SCHEDULER_SRC, (
        "scheduler.py no longer calls request.append_output_text(); the "
        "streaming output text accumulator is gone. See "
        "todo/performance-adjustments.md #2."
    )
    assert not re.search(r"request\.output_text\s*\+=", _SCHEDULER_SRC), (
        "scheduler.py uses request.output_text += again (O(N^2) per request). "
        "Use request.append_output_text(). See "
        "todo/performance-adjustments.md #2."
    )


def _best_append_join(visible: str, n: int) -> float:
    """Fastest wall-clock over ``_REPEATS`` runs of append*n + one join."""
    best = None
    for _ in range(_REPEATS):
        request = Request(request_id="perf", prompt="x", sampling_params=None)
        t0 = time.perf_counter()
        for _ in range(n):
            request.append_output_text(visible)
        _ = request.output_text  # join materializes the cumulative text
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best


def test_output_text_append_scales_linearly():
    """#2: append_output_text + join must be O(N), not the old O(N^2) shape."""
    small = _best_append_join("token ", _N_SMALL)
    large = _best_append_join("token ", _N_LARGE)
    ratio = large / max(small, 1e-9)
    assert ratio < _MAX_LINEAR_RATIO, (
        f"output_text accumulation scaled {ratio:.1f}x for a 4x size increase "
        f"(linear~4, quadratic~16). Likely a per-token full-copy regression in "
        f"Request.append_output_text or the output_text getter."
    )


def test_output_text_join_matches_parts():
    """#2 correctness: the join view equals the concatenated parts."""
    request = Request(request_id="perf", prompt="x", sampling_params=None)
    for part in ("a", "bb", "ccc"):
        request.append_output_text(part)
    assert request.output_text == "abbccc"
    # The setter keeps the streaming contract for finalize/prefix rewrites.
    request.output_text = "reset"
    assert request.output_text == "reset"


# --------------------------------------------------------------------------- #
# A1 iogpu.wired_limit_mb sysctl cache.
# --------------------------------------------------------------------------- #
def test_iogpu_wired_limit_read_is_cached(monkeypatch):
    """A1: repeated sysctl reads are served from cache (one subprocess)."""
    calls = {"n": 0}
    pme._iogpu_wired_limit_cache = None

    def fake_read() -> int:
        calls["n"] += 1
        return 4 * 1024**3

    monkeypatch.setattr(pme, "_read_iogpu_wired_limit_bytes", fake_read)
    try:
        assert pme.get_iogpu_wired_limit_bytes() == 4 * 1024**3
        for _ in range(100):
            pme.get_iogpu_wired_limit_bytes()
        assert calls["n"] == 1, "iogpu.wired_limit_mb sysctl read is not cached"
    finally:
        pme._iogpu_wired_limit_cache = None
