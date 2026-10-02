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


@pytest.mark.gpu
@pytest.mark.parametrize('implementation', ['fused', 'eager'])
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('bx,by', [(False, False), (True, False),
                                  (False, True), (True, True)])
def test_sloped_edge_surface_matches_wrf_clamped_donors(implementation, mapped, bx, by):
    """An exterior terrain donor is the edge cell, not its interior neighbour.

    The independent cold-start reference follows module_bc_em.F:1246-1279
    by selecting donor indexes. Nonuniform winds and nonzero slopes expose
    every edge and corner; periodic controls retain the wrap.
    """
    import cupy as cp
    from gpuwm.core.dycore import set_w_surface, _set_w_surface_eager
    from test_wrfinput_cold_start import _wrf_set_w_surface

    nz, ny, nx = 5, 7, 9
    kk, jj, ii = np.indices((nz, ny, nx + 1))
    u = (2.0 + .125 * ii - .0625 * jj + .5 * kk).astype(np.float32)
    kk, jj, ii = np.indices((nz, ny + 1, nx))
    v = (-3.0 + .0625 * ii + .125 * jj - .25 * kk).astype(np.float32)
    jj, ii = np.indices((ny, nx))
    ht = (8 * ii + 16 * jj + 2 * ii * jj).astype(np.float32)
    # Dyadic maps isolate donor choice from map-factor multiplication order.
    msft = (np.power(2.0, (ii + jj) % 3 - 1) if mapped
            else np.ones((ny, nx))).astype(np.float32)
    cf1, cf2, cf3 = np.float32(1.5), np.float32(-.625), np.float32(.125)
    cfg = SimpleNamespace(dx=512.0, dy=1024.0, open_x=bx, open_y=by,
                          specified=False, nested=False)
    reference = _wrf_set_w_surface(
        u, v, ht, msft, msft, np.linspace(1, 0, nz + 1, dtype=np.float32),
        cf1, cf2, cf3, np.float32(1 / cfg.dx), np.float32(1 / cfg.dy),
        not bx, not by)[0]
    state = SimpleNamespace(
        u=cp.asarray(u), v=cp.asarray(v), ht=cp.asarray(ht), msft=cp.asarray(msft),
        w=cp.full((nz + 1, ny, nx), np.float32(7.25)),
        cf1=cf1, cf2=cf2, cf3=cf3, has_msf=mapped)
    upper = state.w[1:].copy()
    function = set_w_surface if implementation == 'fused' else _set_w_surface_eager
    function(state, cfg)
    np.testing.assert_array_equal(state.w[0].get().view(np.uint32),
                                  reference.view(np.uint32))
    np.testing.assert_array_equal(state.w[1:].get().view(np.uint32),
                                  upper.get().view(np.uint32))
