"""Regression coverage for the live chemistry/radiation boundary.

These synthetic tests cover the defect. They do not qualify an aerosol event.
"""
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("kind", ["sw", "lw"])
def test_aerosol_is_not_mcica_masked_and_mixes_before_delta(kind):
    import cupy as cp
    from gpuwm.core.chem_radiation import finalize
    from gpuwm.core.rrtmgp import GasOpticsResult, CloudOpticsResult
    shape = (3, 4, 6)
    bands = cp.asarray([0, 0, 1, 1, 2, 2], cp.int32)
    tables = SimpleNamespace(kind=kind, ngpt=6, nband=3,
        to_device=lambda: SimpleNamespace(gpoint_bands=bands))
    a = lambda x, s: cp.full(s, x, cp.float32)
    bs = (3, 4, 3)
    gas = GasOpticsResult(a(.1, shape), a(.2, shape))
    cloud = CloudOpticsResult(a(2., bs), a(.9, bs), a(.6, bs))
    aer = (a(.7, bs), a(.8, bs), a(.5, bs))
    dark = cp.zeros(shape, cp.bool_)
    clear = finalize(tables, gas, cloud, dark, aer, clear=True)
    masked = finalize(tables, gas, cloud, dark, aer)
    for x, y in zip(clear, masked):
        cp.testing.assert_array_equal(x, y)
    if kind == "lw":
        cp.testing.assert_allclose(clear[0], .1 + .7 * .2, rtol=1.e-6)
    else:
        tau = .1 + .7
        scatter = .1 * .2 + .7 * .8
        w = scatter / tau
        g = .7 * .8 * .5 / scatter
        f = g*g
        expected = (tau * (1-w*f), w*(1-f)/(1-w*f), (g-f)/(1-f))
        for x, value in zip(clear, expected):
            cp.testing.assert_allclose(x, value, rtol=1.e-6)
    lit = finalize(tables, gas, cloud, cp.ones(shape, cp.bool_), aer)
    assert bool(cp.all(lit[0] > clear[0]))


@pytest.mark.parametrize("feedback", [0, 1])
def test_history_optics_before_first_radiation_and_between_calls(feedback):
    import cupy as cp
    from test_chem_transport import _bubble_case
    from gpuwm.core.chem_history import history_fields
    cfg, state = _bubble_case(chem_sets="dust", aer_ra_feedback=feedback,
                              ra_lw_physics=4, ra_sw_physics=4)
    row = state.chem.table.rows_for("optics.gocart")[0]
    mass = getattr(state, row.state_attr)
    mass.fill(100.)
    fields = history_fields(state)
    initial = fields["AOD5502D"][0].copy()
    assert bool(cp.all(cp.isfinite(initial))) and float(initial.max()) > 0
    mass.fill(200.)
    # No chemistry/radiation call or clock advance: history must read mass.
    cp.testing.assert_allclose(history_fields(state)["AOD5502D"][0],
                               2 * initial, rtol=2.e-5)


def test_live_optics_before_first_call_and_above_model_zeros():
    import cupy as cp
    from test_chem_transport import _bubble_case
    from gpuwm.core.chem_radiation import prepare
    cfg, state = _bubble_case(chem_sets="dust", aer_ra_feedback=1,
                              ra_lw_physics=4, ra_sw_physics=4)
    row = state.chem.table.rows_for("optics.gocart")[0]
    getattr(state, "chem_" + row.name).fill(100.)
    optics = prepare(state, cfg)
    assert optics is prepare(state, cfg)
    aod = optics.ctx.diag["aod550"]
    assert bool(cp.all(cp.isfinite(aod))) and float(aod.max()) > 0
    nz = state.qv.shape[0]
    columns = cp.asarray([0, 3, 9], cp.int32)
    for kind in ("lw", "sw"):
        value = optics.legacy(columns, nz+3, kind)
        if kind == "sw":
            tau = value[0]
            assert tau.shape == (3, 14, nz+3)
            assert float(tau[:,:,:nz].sum()) > 0
            assert bool(cp.all(tau[:,:,nz:] == 0))
        else:
            assert value.shape == (3, nz+3, 16)
            assert float(value[:,:nz,:].sum()) > 0
            assert bool(cp.all(value[:,nz:,:] == 0))
        native = optics.modern(columns, nz+3, kind)
        assert float(native[0][:,:nz,:].sum()) > 0
        for x in native:
            assert bool(cp.all(cp.isfinite(x)))
            assert bool(cp.all(x[:,nz:,:] == 0))
    old = aod.copy()
    getattr(state, "chem_" + row.name).fill(200.)
    state.elapsed_seconds += cfg.dt
    updated = prepare(state, cfg)
    assert updated is not optics
    cp.testing.assert_allclose(updated.ctx.diag["aod550"], 2*old, rtol=2.e-5)


@pytest.mark.parametrize("variant,workspace", [("legacy", False), ("modern", False), ("modern", True)])
def test_live_feedback_reaches_surface_fluxes(variant, workspace):
    from datetime import datetime
    import cupy as cp
    from test_chem_transport import _bubble_case
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.core.rrtmgp import RRTMGPRadiation
    from gpuwm.core.rrtmg_legacy import RRTMGLegacyRadiation
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    cfg, state = _bubble_case(chem_sets="dust", aer_ra_feedback=1,
        ra_lw_physics=4, ra_sw_physics=4, mp_physics=1, icloud=1)
    shape = state.qv.shape[1:]
    fields = {name: cp.full(shape, value, cp.float32) for name, value in
              (("tsk", 290.), ("albedo", .15), ("emiss", .96),
               ("xland", 1.), ("xice", 0.), ("snow", 0.))}
    lat, lon = (cp.full(shape, value, cp.float32) for value in (35., -5.))
    cls = RRTMGLegacyRadiation if variant == "legacy" else RRTMGPRadiation
    opts = {"p_top": float(state.p_top)} if variant == "legacy" else {}
    rad = cls(datetime(2022, 3, 15, 12), lat, lon, column_chunk=97, **opts)
    if workspace:
        rad.chunk_workspace = SharedRRTMGPChunkWorkspace(
            nz=cfg.nz, column_chunk=97, p_top=float(state.p_top))
    atmosphere = _prepare_atmosphere(state)
    clean = rad(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    for row in state.chem.table.rows_for("optics.gocart"):
        getattr(state, "chem_" + row.name).fill(50.)
    state.chem.radiation_aerosols_time = None
    dusty = rad(atmosphere=atmosphere, fields=fields, state=state, cfg=cfg)
    assert bool(cp.all(cp.isfinite(dusty.swdown)))
    assert bool(cp.all(cp.isfinite(dusty.rthratensw)))
    assert float((clean.swdown - dusty.swdown).mean()) > 1.
    assert float(cp.abs(clean.glw - dusty.glw).mean()) > .001
    assert float(cp.abs(dusty.rthratensw - clean.rthratensw).max()) > 1.e-8
