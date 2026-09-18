# SPDX-License-Identifier: Apache-2.0
"""Synthetic operator-layer baseline for the Apple-silicon (Metal) CI gate.

Times a fixed set of shape-stable MLX ops that back LLM prefill/decode --
quantized matmul (qmv/qmm) and scaled-dot-product attention -- plus the MoE
sorted-gather custom path when it is importable. No model download: all weights
and activations are synthetic, so the suite runs on a `macos-14` runner.

Usage::

    python benchmarks/operator_baseline.py --write-baseline   # record
    python benchmarks/operator_baseline.py                    # compare

Comparison is ratio-based against ``benchmarks/operator_baseline.json``. Shared
CI runners are thermally noisy, so this is a SOFT gate: the default tolerance is
1.5x and the CI job sets ``continue-on-error: true``. It exists to catch
catastrophic (multi-x) regressions, not jitter.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

BASELINE_PATH = Path(__file__).resolve().parent / "operator_baseline.json"

_GROUP_SIZE = 64
_BITS = 4


def _time_ms(fn, iters: int, repeats: int = 3, warmup: int = 3) -> float:
    """Median ms/iter over ``repeats`` runs (min-of-medians reduces jitter)."""
    samples = []
    for _ in range(repeats):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        samples.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(samples)


def _quantized_dense(k: int = 4096, n: int = 4096):
    w = (mx.random.normal((n, k), key=mx.random.key(1)) * 0.02).astype(mx.bfloat16)
    wq = mx.quantize(w, group_size=_GROUP_SIZE, bits=_BITS)
    mx.eval(*wq)
    return wq


def _ops() -> dict[str, tuple]:
    """Return {name: (callable, iters)} for a fixed, shape-stable op set."""
    wq = _quantized_dense()
    x_dec = mx.random.normal((1, 4096), key=mx.random.key(2)).astype(mx.bfloat16)
    x_pre = mx.random.normal((4096, 4096), key=mx.random.key(3)).astype(mx.bfloat16)

    def qmv_decode():
        return mx.quantized_matmul(
            x_dec, *wq, transpose=True, group_size=_GROUP_SIZE, bits=_BITS
        )

    def qmm_prefill():
        return mx.quantized_matmul(
            x_pre, *wq, transpose=True, group_size=_GROUP_SIZE, bits=_BITS
        )

    # SDPA: B=1, H=32, D=128; decode attends 4096 cached keys.
    q_d = mx.random.normal((1, 32, 1, 128), key=mx.random.key(4)).astype(mx.bfloat16)
    q_p = mx.random.normal((1, 32, 4096, 128), key=mx.random.key(5)).astype(
        mx.bfloat16
    )
    k = mx.random.normal((1, 32, 4096, 128), key=mx.random.key(6)).astype(mx.bfloat16)
    v = mx.random.normal((1, 32, 4096, 128), key=mx.random.key(7)).astype(mx.bfloat16)
    mx.eval(q_d, q_p, k, v)

    def sdpa_decode():
        return mx.fast.scaled_dot_product_attention(q_d, k, v, scale=128**-0.5)

    def sdpa_prefill():
        return mx.fast.scaled_dot_product_attention(q_p, k, v, scale=128**-0.5)

    ops: dict[str, tuple] = {
        "qmv_decode": (qmv_decode, 50),
        "qmm_prefill": (qmm_prefill, 20),
        "sdpa_decode": (sdpa_decode, 50),
        "sdpa_prefill": (sdpa_prefill, 10),
    }

    # MoE sorted-gather custom path (oMLX reroute); skip if unavailable.
    try:
        import importlib.util

        import omlx.patches.m5_gather_qmm as reroute

        spec = importlib.util.spec_from_file_location(
            "bench_m5_sorted_gather_chunk",
            Path(__file__).resolve().parent / "bench_m5_sorted_gather_chunk.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        reroute.apply_m5_gather_qmm_workaround()
        gather = reroute._gather_qmm_rerouted
        gu, dn = mod._build(512, 2560, 640, _BITS, _GROUP_SIZE, mx.random.key(0))
        chunk, topk = 4096, 10
        kx, ki = mx.random.split(mx.random.key(99))
        xm = mx.random.normal((1, chunk, 2560), key=kx).astype(mx.bfloat16)
        inds = mx.random.randint(0, 512, (1, chunk, topk), key=ki).astype(mx.uint32)
        mx.eval(xm, inds)

        def moe_sorted_prefill():
            return mod._layer(
                xm, inds, gu, dn, _BITS, _GROUP_SIZE, True, gather
            )

        ops["moe_sorted_prefill"] = (moe_sorted_prefill, 5)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] moe_sorted_prefill unavailable: {exc}")

    return ops


def _measure() -> dict[str, float]:
    results = {}
    for name, (fn, iters) in _ops().items():
        results[name] = _time_ms(fn, iters)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    ap.add_argument("--tolerance", type=float, default=1.5)
    args = ap.parse_args()

    print(
        f"mlx {mx.__version__}  device "
        f"{mx.device_info().get('device_name')}  gpu "
        f"{mx.device_info().get('architecture')}"
    )
    current = _measure()

    if args.write_baseline:
        args.baseline.write_text(json.dumps(current, indent=2) + "\n")
        for name, ms in current.items():
            print(f"  {name:<20} {ms:8.3f} ms")
        print(f"\nwrote baseline -> {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"no baseline at {args.baseline}; run --write-baseline first")
        return 0

    baseline = json.loads(args.baseline.read_text())
    regressions = []
    print(f"{'op':<20} {'baseline':>10} {'current':>10} {'ratio':>8}")
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
            print(f"{name:<20} {'(none)':>10} {cur:>10.3f} {'-':>8}")
            continue
        ratio = cur / base
        flag = ""
        if ratio > args.tolerance:
            flag = "  <-- REGRESSION"
            regressions.append(name)
        print(f"{name:<20} {base:>10.3f} {cur:>10.3f} {ratio:>7.2f}x{flag}")

    if regressions:
        print(
            f"\n{len(regressions)} op(s) slower than {args.tolerance}x baseline: "
            f"{', '.join(regressions)}"
        )
        return 1
    print("\nall ops within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
