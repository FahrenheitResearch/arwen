"""The host oracle preserves the driver's in-place threshold semantics."""
import numpy as np
import pytest

from tools.thompson_real_column_parity.host_backend import _fake_cupy


def test_host_zero_out_keeps_threshold_equality_and_mutates_views():
    cp = _fake_cupy()
    floor = cp.ElementwiseKernel(
        "", "float32 x", "if (x < 0.0f) x = 0.0f;", "gpuwm_mp_floor_zero")
    below = cp.ElementwiseKernel(
        "float32 t", "float32 x", "if (x < t) x = 0.0f;",
        "gpuwm_mp_zero_below")
    fields = np.array([[-1, 2, -1, 3], [-1, 2, -1, 3]], dtype=np.float32)
    floor(fields[0, ::2])
    below(np.float32(2), fields[1])
    np.testing.assert_array_equal(fields, [[0, 2, 0, 3], [0, 2, 0, 3]])


def test_host_elementwise_kernel_refuses_an_unimplemented_operation():
    with pytest.raises(RuntimeError, match="no implementation"):
        _fake_cupy().ElementwiseKernel("", "float32 x", "x = 1.0f;", "other")
