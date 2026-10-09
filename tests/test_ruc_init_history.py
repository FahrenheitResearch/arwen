"""Native cold-start restoration must retain RUCLSMINIT's published words."""
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.ingest.wrfinput import initialize_wrfinput_physics
from gpuwm.io.history_layout import live_state_history_fields


def _case():
    path = Path(__file__).parents[1] / "gpuwm/data/ruc/oracle/init.csv"
    rows = list(csv.DictReader(path.open(encoding="ascii")))
    profile = lambda name: np.asarray([float(r[name]) for r in rows], np.float32).reshape(4, 9).T[:, None, :]
    row = lambda name, dtype=np.float32: np.asarray([r[name] for r in rows[::9]], dtype)[None, :]
    raw = dict(TSLB=profile("tslb"), SMOIS=profile("smois"),
               SH2O=np.full((9, 1, 4), .125, np.float32),
               LU_INDEX=row("ivgtyp", np.int32), ISLTYP=row("isltyp", np.int32),
               XICE=row("xice"), TSK=np.full((1, 4), 280., np.float32),
               LANDMASK=np.asarray([[1, 1, 0, 0]], np.float32))
    for name, value in dict(VEGFRA=60., TMN=280., SNOW=0., SNOWH=0., ALBBCK=.2, LAI=2., GLW=300.).items():
        raw[name] = np.full((1, 4), value, np.float32)
    cfg = RunConfig(nx=4, ny=1, nz=10, dx=3000., dy=3000., ztop=16000.,
                    dt=20., run_seconds=20., moist=True, mp_physics=8,
                    sf_sfclay_physics=1, sf_surface_physics=3, num_soil_layers=9,
                    bl_pbl_physics=1, ra_physics=0)
    return cfg, SimpleNamespace(raw=raw, global_attributes={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH"}), profile("sh2o")


def test_restore_keeps_initialized_soil_water_in_history(monkeypatch):
    """CPU seam check, including liquid=1 water and liquid=0 ice."""
    from gpuwm.core.ruc import ruc_initialize_cold_start
    from gpuwm.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    cfg, restored, expected = _case()
    r = restored.raw
    cold = ruc_initialize_cold_start(r["TSLB"], r["SMOIS"], r["ISLTYP"], r["LU_INDEX"], r["XICE"])
    owned = {n: getattr(cold, n).copy() for n in ("sh2o", "smfr3d", "mavail", "znt")}
    for name, value in owned.items():
        r[name.upper()] = np.full_like(value, .125)
    driver = SimpleNamespace(fields=dict(**owned,
        albbck=r["ALBBCK"].copy(), lai=r["LAI"].copy()), rainc=None,
        scheme_dispatch={"sf_surface_physics": PHYSICS_SLOT_DISPATCH["sf_surface_physics"][3]})
    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setitem(sys.modules, "gpuwm.core.physics", SimpleNamespace(
        physics_driver_required=lambda cfg: True,
        initialize_physics=lambda *a, **k: driver))
    initialize_wrfinput_physics(object(), restored, cfg)
    got = live_state_history_fields(SimpleNamespace(physics=driver))["SH2O"]
    np.testing.assert_array_equal(got.view("u4"), expected.view("u4"))
    for name in owned:
        np.testing.assert_array_equal(driver.fields[name].view("u4"), getattr(cold, name).view("u4"))


@pytest.mark.gpu
def test_native_driver_t0_history_matches_wrf_init():
    import cupy as cp
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    cfg, restored, expected = _case()
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), cfg.p_surf, cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: np.full_like(z, .005))
    state.physics = initialize_wrfinput_physics(state, restored, cfg)
    got = cp.asnumpy(live_state_history_fields(state)["SH2O"])
    np.testing.assert_array_equal(got.view("u4"), expected.view("u4"))
    from gpuwm.io.wrfout import state_frame, _device_state_frame
    for build in (state_frame, _device_state_frame):
        frame = build(state, include_diagnostic_pressure=False)
        np.testing.assert_array_equal(cp.asnumpy(frame["SH2O"]).view("u4"), expected.view("u4"))


@pytest.mark.gpu
def test_native_liquid_water_reaches_a_coupled_surface_step():
    from dataclasses import replace
    import cupy as cp
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.dycore import step, stability_report
    from gpuwm.io.wrfout import state_frame
    cfg, restored, expected = _case()
    cfg = replace(cfg, nz=20, dt=1., run_seconds=1.,
                  sf_sfclay_physics=5, bl_pbl_physics=5)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.), cfg.p_surf, cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: np.full_like(z, .005))
    state.u.fill(cp.float32(5))
    state.v.fill(cp.float32(1))
    state.physics = initialize_wrfinput_physics(state, restored, cfg)
    state.physics.set_forcing(gsw=0.0, swdown=0.0)
    frame = state_frame(state, include_diagnostic_pressure=False)
    np.testing.assert_array_equal(frame["SH2O"].view("u4"), expected.view("u4"))
    step(state, cfg)
    assert state.elapsed_seconds == cfg.dt and not stability_report(state, cfg)["nan"]
    assert bool(cp.all(cp.isfinite(state.physics.fields["sh2o"])))
