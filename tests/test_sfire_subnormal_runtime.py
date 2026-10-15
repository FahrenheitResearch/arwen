"""Production SFIRE modules must retain arithmetic on positive subnormals."""
import numpy as np
import pytest
from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_core_boundary_extrapolation_keeps_subnormal_words():
    import cupy as cp
    from gpuwm.core.sfire_core import continue_at_boundary
    tiny = np.array([1], np.uint32).view(np.float32)[0]
    value = cp.full((5, 5), tiny, cp.float32)
    continue_at_boundary(value, bias=0., halo=1)
    assert np.all(cp.asnumpy(value).view(np.uint32) == 1)


@requires_gpu
@pytest.mark.gpu
def test_atmospheric_sum_keeps_subnormal_additions():
    import cupy as cp
    from gpuwm.core.sfire_atm import sum_fire_cells
    tiny = np.array([1], np.uint32).view(np.float32)[0]
    result = sum_fire_cells(cp.full((4, 4), tiny, cp.float32), 2, 2)
    assert np.all(cp.asnumpy(result).view(np.uint32) == 4)


@requires_gpu
@pytest.mark.gpu
def test_coupling_floor_keeps_positive_subnormal():
    import cupy as cp
    from gpuwm.core.sfire_coupler import _launch
    values = cp.asarray(np.array([1, 2, 3, 7], np.uint32).view(np.float32))
    _launch("sfire_plane_floor", values.size, (values, np.int32(values.size), np.float32(0)))
    np.testing.assert_array_equal(cp.asnumpy(values).view(np.uint32), [1, 2, 3, 7])
