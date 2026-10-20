"""Parent runoff step-depth contract and genuine bottom-drainage publication."""
from datetime import datetime

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.hydro.noahowp.coupling import PreparedCatchmentExchange


def _exchange(fields):
    return PreparedCatchmentExchange(
        offsets=cp.asarray([0, 2], dtype=cp.int64),
        indices=cp.asarray([0, 1], dtype=cp.int32),
        weights=cp.asarray([.25, .75], dtype=cp.float32),
        runsrf=fields["runsfxy"], runsub=fields["runsbxy"],
        soldrain=fields["soldrain"], sfcheadrt=fields["sfcheadrt"],
        smois=fields["smois"], sh2o=fields["sh2o"],
        ecan=fields["ecanxy"], edir=fields["edirxy"], etran=fields["etranxy"],
        dt_seconds=60., weights_sha256="test", fabric_sha256="test", numerics="strict")


def test_depths_use_lsm_interval_et_keeps_rate_and_head_keeps_state():
    fields = {key: cp.full((1, 2), value, cp.float32) for key, value in
              {"runsfxy": 60., "runsbxy": 120., "soldrain": 30.,
               "sfcheadrt": 37., "ecanxy": 1., "edirxy": 2., "etranxy": 3.}.items()}
    fields["smois"] = cp.full((4, 1, 2), .31, cp.float32)
    fields["sh2o"] = cp.full((4, 1, 2), .29, cp.float32)
    exchange = _exchange(fields)
    exchange.launch(60.)
    # 60 mm over a minute is 1 mm/s; 6 mm/s ET is already a rate.
    for name, value in {"surface_runoff_m_s": .001, "subsurface_runoff_m_s": .002,
                        "recharge_m_s": .0005, "evapotranspiration_m_s": .006}.items():
        cp.testing.assert_allclose(exchange.arrays[name], cp.float32(value), rtol=2e-7, atol=0)
    assert bool(cp.all(fields["sfcheadrt"] == cp.float32(37.)).item())
    for name in ("smois", "sh2o", "sfcheadrt", "soldrain"):
        assert exchange.arrays[name].data.ptr == fields[name].data.ptr
    assert exchange.arrays["infxsrt"].data.ptr == fields["runsfxy"].data.ptr
    assert int(exchange.arrays["status"].item()) == 0


def test_actual_parent_publishes_bottom_drainage_then_exchange_uses_depths():
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.diagnostics import update_diagnostics

    cfg = RunConfig(nx=2, ny=1, nz=12, dx=2000., dy=2000., ztop=8000., dt=10.,
                    run_seconds=0., time_step_sound=4, moist=True, mp_physics=0,
                    sf_sfclay_physics=1, sf_surface_physics=4, bl_pbl_physics=0, bldt=1.)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 298.+.004*np.asarray(z, np.float64),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base,
                               lambda z: .01*np.exp(-np.asarray(z, np.float64)/2400.))
    state.u.fill(6.)
    state.v.fill(.5)
    physics = initialize_physics(state, cfg, soil_temperature=285., soil_moisture=.31,
                                 liquid_moisture=.31, swdown=450., glw=310.,
                                 noahmp_start_time=datetime(2026, 7, 1, 18),
                                 noahmp_latitude=40., noahmp_longitude=-100.)
    fields = physics.fields
    fields["soldrain"] = cp.full_like(fields["runsfxy"], cp.nan)
    fields["sfcheadrt"] = cp.full_like(fields["runsfxy"], 37.)
    exchange = _exchange(fields)
    pointers = {name: a.data.ptr for name, a in exchange.arrays.items()}
    update_diagnostics(state)
    physics.compute(state, cfg)
    exchange.launch(60.)
    assert all(a.data.ptr == pointers[name] for name, a in exchange.arrays.items())
    assert bool(cp.all(cp.isfinite(fields["soldrain"]) & (fields["soldrain"] > 0)).item())
    assert bool(cp.all(fields["sfcheadrt"] == cp.float32(37.)).item())
    # Check mass units independently: rate * duration = weighted step volume / area.
    weights = cp.asarray([.25, .75], cp.float32)
    for output, source in (("surface_runoff_m_s", "runsfxy"),
                           ("subsurface_runoff_m_s", "runsbxy"),
                           ("recharge_m_s", "soldrain")):
        cp.testing.assert_allclose(exchange.arrays[output]*cp.float32(60000.),
                                   cp.sum(fields[source].reshape(-1)*weights), rtol=3e-7, atol=0)
    assert int(exchange.arrays["status"].item()) == 0
