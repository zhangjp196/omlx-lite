# SPDX-License-Identifier: Apache-2.0
"""Regression: the VLM MoE-router patch must match the pinned mlx-vlm API.

mlx_vlm 0.7.1's ``Qwen3_5MoeSparseMoeBlock.__call__`` takes only ``x`` (the
target-verify MoE composition lives in the model's speculative verifier, not on
the block). An older oMLX VLM branch passed ``target_verify`` and raised
``TypeError`` on every prefill/chat request.
"""

from __future__ import annotations

import mlx.core as mx
import pytest


@pytest.fixture
def patched():
    if not mx.metal.is_available():
        pytest.skip("Metal required for the fused router patch")
    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    apply_qwen35_moe_router_patch()


class _FakeMoe:
    """Minimal stand-in exposing what the upstream __call__ reads."""

    num_experts = 64
    top_k = 2

    def gate(self, x):
        return mx.zeros((*x.shape[:-1], self.num_experts), dtype=x.dtype)

    def switch_mlp(self, x, inds):
        return mx.zeros((*x.shape[:-1], self.top_k, x.shape[-1]), dtype=x.dtype)

    def shared_expert(self, x):
        return mx.zeros(x.shape, dtype=x.dtype)

    def shared_expert_gate(self, x):
        return mx.zeros((*x.shape[:-1], 1), dtype=x.dtype)

    def _shared_expert_scale(self, x):
        return mx.sigmoid(self.shared_expert_gate(x))


def test_vlm_moe_router_fallback_accepts_plain_x(patched):
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    if not getattr(cls, "_omlx_router_fused", False):
        pytest.skip("VLM MoE router patch not applied in this environment")

    # 16 rows > the fused router's _MAX_ROWS, so this takes the fallback branch
    # that previously passed target_verify to the upstream __call__.
    x = mx.zeros((1, 16, 8), dtype=mx.bfloat16)
    out = cls.__call__(_FakeMoe(), x)
    mx.eval(out)
    assert out.shape == x.shape


def test_vlm_moe_router_fused_arm_returns_correct_shape(patched):
    """The eligible (decode-width) arm must not crash either."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    if not getattr(cls, "_omlx_router_fused", False):
        pytest.skip("VLM MoE router patch not applied in this environment")

    x = mx.zeros((1, 4, 8), dtype=mx.bfloat16)  # 4 rows -> eligible
    out = cls.__call__(_FakeMoe(), x)
    mx.eval(out)
    assert out.shape == x.shape
