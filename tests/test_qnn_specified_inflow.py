"""Stock WRF QNN inflow uses the resolved CCN reservoir, including calm flow."""
import sys

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.ingest.lateral_bc import apply_flow_dependent_boundaries


def _stock_qnn(field, u, v, width, concentration):
    """Independent loops from module_bc.F:2502-2579, with 0-based indices."""
    result = field.copy()
    nz, ny, nx = field.shape
    for j in range(width):
        for k in range(nz):
            for i in range(j, nx-j):
                inner = min(max(i, width), nx-1-width)
                result[k,j,i] = (field[k,width,inner] if v[k,j,i] < 0
                                 else concentration)
    for j in range(ny-width, ny):
        distance = ny-1-j
        for k in range(nz):
            for i in range(distance, nx-distance):
                inner = min(max(i, width), nx-1-width)
                result[k,j,i] = (field[k,ny-1-width,inner] if v[k,j+1,i] > 0
                                 else concentration)
    for i in range(width):
        for k in range(nz):
            for j in range(i+1, ny-i-1):
                inner = min(max(j, width), ny-1-width)
                result[k,j,i] = (field[k,inner,width] if u[k,j,i] < 0
                                 else concentration)
    for i in range(nx-width, nx):
        distance = nx-1-i
        for k in range(nz):
            for j in range(distance+1, ny-distance-1):
                inner = min(max(j, width), ny-1-width)
                result[k,j,i] = (field[k,inner,nx-1-width] if u[k,j,i+1] > 0
                                 else concentration)
    return result


def _fixture(dtype):
    field = (1 + np.arange(3*9*11).reshape(3,9,11)).astype(dtype)
    # Every face sees inflow, outflow and zero velocities, including corners.
    u = (np.arange(3*9*12).reshape(3,9,12) % 3 - 1).astype(dtype)
    v = (np.arange(3*10*11).reshape(3,10,11) % 3 - 1).astype(dtype)
    return field, u, v


@pytest.mark.parametrize('concentration', [0., 1.0e8, 2.3456789e9, 408163264.])
def test_generic_inflow_matches_stock_on_every_edge_and_corner(monkeypatch, concentration):
    monkeypatch.setitem(sys.modules, 'cupy', np)
    field, u, v = _fixture(np.float64)
    expected = _stock_qnn(field, u, v, 2, concentration)
    apply_flow_dependent_boundaries((field,), u, v, 2, inflow_value=concentration)
    np.testing.assert_array_equal(field, expected)


@pytest.mark.gpu
@pytest.mark.parametrize('concentration', [0., 1.0e8, 2.3456789e9, 408163264.])
def test_cuda_inflow_matches_stock_float32_words(concentration):
    cp = pytest.importorskip('cupy')
    field, u, v = _fixture(np.float32)
    expected = _stock_qnn(field, u, v, 2, np.float32(concentration))
    actual = cp.asarray(field)
    apply_flow_dependent_boundaries((actual,), cp.asarray(u), cp.asarray(v),
                                    2, inflow_value=concentration)
    np.testing.assert_array_equal(cp.asnumpy(actual).view('u4'), expected.view('u4'))


@pytest.mark.gpu
@pytest.mark.parametrize('final', [False, True])
@pytest.mark.parametrize('mp,concentration', [(16, 2.3456789e9), (18, 408163264.)])
def test_actual_scalar_stage_routes_only_qnn_to_ccn_inflow(mp, concentration, final):
    cp = pytest.importorskip('cupy')
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.moist import advance_scalars_stage

    cfg = RunConfig(nx=13, ny=12, nz=6, dx=12000., dy=12000.,
                    ztop=10000., dt=1., run_seconds=120., moist=True,
                    mp_physics=mp, specified=True,
                    wdm6_ccn_conc=(concentration if mp == 16 else 1.0e8))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    name, ordinary = ('nn', 'nc') if mp == 16 else ('qnn', 'qndrop')
    for field, value in [('qv', .01), (name, 8.5e8), (ordinary, 5.5e5)]:
        getattr(state, field)[...] = value
        getattr(state, field+'0')[...] = value
    state.mup0[...] = state.mup
    # Zero is inflow under each stock strict inequality; it also eliminates
    # transport changes so the interior and ordinary fields are exact controls.
    advance_scalars_stage(state, cfg, cp.zeros_like(state.u),
                          cp.zeros_like(state.v), cp.zeros_like(state.w),
                          dt_eff=1., final=final, apply_relax=True)
    qnn = cp.asnumpy(getattr(state, name))
    number = cp.asnumpy(getattr(state, ordinary))
    edge = np.ones((cfg.ny, cfg.nx), dtype=bool)
    sz = cfg.spec_zone
    edge[sz:-sz, sz:-sz] = False
    np.testing.assert_array_equal(qnn[:, edge], np.float32(concentration))
    np.testing.assert_array_equal(number[:, edge], np.float32(0.))
    np.testing.assert_allclose(qnn[:, ~edge], 8.5e8, rtol=2e-7)
    np.testing.assert_allclose(number[:, ~edge], 5.5e5, rtol=2e-7)


@pytest.mark.gpu
@pytest.mark.parametrize('final', [False, True])
def test_wdm6_ccn_and_cloud_number_are_transported_in_the_interior(final):
    cp = pytest.importorskip('cupy')
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.moist import advance_scalars_stage, moist_species

    cfg = RunConfig(nx=13, ny=12, nz=6, dx=3000., dy=3000., ztop=10000.,
                    dt=1., run_seconds=0., moist=True, mp_physics=16)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), cfg.p_surf, cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    assert moist_species(state) == ('qv','qc','qr','qi','qs','qg','nn','nc','nr')
    initial = cp.broadcast_to(cp.arange(cfg.nx, dtype=cp.float32)[None,None,:]
                               * 1000. + 10000., state.qv.shape).copy()
    for name in ('nn', 'nc', 'nr'):
        getattr(state, name)[...] = initial
        getattr(state, name+'0')[...] = initial
    state.mup0[...] = state.mup
    advance_scalars_stage(state, cfg, cp.full_like(state.u, 500000.),
                          cp.zeros_like(state.v), cp.zeros_like(state.w),
                          dt_eff=1., final=final)
    # nr was already transported. Equal input profiles under identical
    # fluxes must produce equal fields, and that common field must move.
    assert bool(cp.any(state.nr[:, 3:-3, 3:-3] != initial[:, 3:-3, 3:-3]))
    for name in ('nn', 'nc'):
        np.testing.assert_array_equal(cp.asnumpy(getattr(state, name)).view('u4'),
                                      cp.asnumpy(state.nr).view('u4'))
