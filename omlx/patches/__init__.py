# SPDX-License-Identifier: Apache-2.0
"""Post-load model patches for performance optimization and correctness."""

from ._compat import assert_pins_match

# Patch application targets exact upstream internals; surface version drift
# once here (warn only) so a mismatched mlx-lm / mlx-vlm does not silently
# disable the patches.
assert_pins_match()
