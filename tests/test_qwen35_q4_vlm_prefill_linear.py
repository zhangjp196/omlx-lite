# SPDX-License-Identifier: Apache-2.0
"""Parity for the 0.7.1 VLM prefill-linear wrappers (attention / GDN).

mlx-vlm 0.7.1 dropped the ``_target_verify_*`` seam the app routed prefill
projections through, so the VLM patch now wraps the module forwards directly.
The wrappers mirror the upstream forward verbatim; this proves the GDN one does
by routing through a plain linear (so the mirrored body runs without a native
kernel) and comparing with the stock forward.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

_CFG = dict(
    model_type="qwen3_5",
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    vocab_size=128,
    num_key_value_heads=2,
    max_position_embeddings=64,
    hidden_size=256,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
)


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_vlm_gdn_prefill_linear_wrapper_matches_stock(monkeypatch):
    import omlx.patches.qwen35_q4_mlp as q4
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet as GDN

    monkeypatch.setenv("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "1")
    monkeypatch.setattr(q4, "_has_native_qmm", lambda: True)
    monkeypatch.setattr(q4, "_can_route_affine_linear", lambda *a, **k: True)
    # Route through a plain linear so the mirrored body runs without a native
    # kernel; the wrapper must then equal the stock forward exactly.
    monkeypatch.setattr(q4, "_backend_or_qmm", lambda linear, x, variant: linear(x))
    monkeypatch.setattr(q4, "_LINEAR_PATCHED", False)
    if hasattr(GDN, "_omlx_q4_prefill_linear_original_call"):
        monkeypatch.setattr(GDN, "__call__", GDN._omlx_q4_prefill_linear_original_call)
    monkeypatch.setattr(GDN, "_omlx_q4_prefill_linear_patched", False, raising=False)

    cfg = TextConfig(**_CFG)
    layer = GDN(cfg)
    layer.set_dtype(mx.bfloat16)
    mx.eval(layer.parameters())
    mx.random.seed(0)
    x = (mx.random.normal((1, 4, 256)) * 0.02).astype(mx.bfloat16)

    def fresh_cache():
        cache = ArraysCache(size=2)
        cache[0] = mx.zeros((1, 3, layer.conv_dim), dtype=mx.bfloat16)
        cache[1] = mx.zeros(
            (1, layer.num_v_heads, layer.head_v_dim, layer.head_k_dim),
            dtype=mx.float32,
        )
        return cache

    stock_forward = GDN.__call__
    stock = stock_forward(layer, x, cache=fresh_cache())
    mx.eval(stock)

    assert q4.apply_qwen35_q4_prefill_linear_patch() is True
    assert GDN.__call__ is not stock_forward, "GDN forward was not wrapped"
    routed = GDN.__call__(layer, x, cache=fresh_cache())
    mx.eval(routed)

    assert routed.shape == stock.shape
    diff = mx.abs(stock.astype(mx.float32) - routed.astype(mx.float32)).max().item()
    assert diff == 0.0, f"mirrored GDN prefill wrapper diverged: {diff}"
