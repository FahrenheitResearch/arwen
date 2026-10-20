"""Independent frozen/total FP32 sums can put Morrison SR above one.

Negative sedimentation is fixed at its source; the former lower floor is
retired, so a negative value remains visible to the physics validator.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]


def test_apply_bounds_sr_after_the_scheme_and_before_returning_it():
    tree = ast.parse((ROOT / "gpuwm" / "core" / "morrison.py").read_text(
        encoding="utf-8"))
    apply = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "apply")
    calls = [node.func.id for node in ast.walk(apply)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "launch_morrison" in calls and "bound_snow_ratio" in calls
    order = [name for name in calls
             if name in ("launch_morrison", "bound_snow_ratio")]
    assert order == ["launch_morrison", "bound_snow_ratio"]


@pytest.mark.gpu
@requires_gpu
def test_bound_snow_ratio_moves_only_out_of_range_words():
    import cupy as cp

    from gpuwm.core.morrison import bound_snow_ratio

    f32 = np.float32
    values = np.asarray([
        -4.68609561e-11, -1.0, 0.0, 1e-30, 0.25, 1.0,
        np.nextafter(f32(1.0), f32(2.0)), 7.5, np.nan], dtype=f32)
    sr = cp.asarray(values)
    bound_snow_ratio(sr)
    got = cp.asnumpy(sr)
    want = np.asarray([-4.68609561e-11, -1.0, 0.0, 1e-30, 0.25, 1.0, 1.0, 1.0, np.nan],
                      dtype=f32)
    inside = (values >= 0) & (values <= 1)
    # In-range words are WRF's, bit for bit.
    assert np.array_equal(got[inside].view(np.uint32),
                          values[inside].view(np.uint32))
    np.testing.assert_array_equal(got, want)
    assert np.isnan(got[-1])           # left for the driver's refusal
