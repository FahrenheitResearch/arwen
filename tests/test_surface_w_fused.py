"""Compare the fused surface launch with the retained eager oracle."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize('scalar_type', [float, np.float64])
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('bx,by', [(False, False), (True, False),
                                  (False, True), (True, True)])
def test_surface_words_match_eager_and_leave_upper_levels(mapped, bx, by, scalar_type):
    import cupy as cp
    from gpuwm.core.dycore import set_w_surface, _set_w_surface_eager
    rng = np.random.default_rng(3187)
    nz, ny, nx = 7, 13, 17
    state = SimpleNamespace(
        u=cp.asarray(rng.standard_normal((nz, ny, nx + 1)), dtype=cp.float32),
        v=cp.asarray(rng.standard_normal((nz, ny + 1, nx)), dtype=cp.float32),
        ht=cp.asarray(rng.uniform(-100, 4000, (ny, nx)), dtype=cp.float32),
        msft=cp.asarray(rng.uniform(.5, 1.5, (ny, nx)), dtype=cp.float32),
        w=cp.asarray(rng.standard_normal((nz + 1, ny, nx)), dtype=cp.float32),
        cf1=np.float32(1.7), cf2=np.float32(-.9), cf3=np.float32(.2),
        has_msf=mapped)
    expected = SimpleNamespace(**vars(state))
    expected.w = state.w.copy()
    upper = state.w[1:].copy()
    cfg = SimpleNamespace(dx=scalar_type(750.0), dy=scalar_type(612.5), open_x=bx, open_y=by,
                          specified=False, nested=False)
    _set_w_surface_eager(expected, cfg)
    set_w_surface(state, cfg)
    np.testing.assert_array_equal(state.w.view(cp.uint32).get(),
                                  expected.w.view(cp.uint32).get())
    np.testing.assert_array_equal(state.w[1:].view(cp.uint32).get(),
                                  upper.view(cp.uint32).get())
