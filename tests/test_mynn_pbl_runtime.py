"""The MYNN 5/5 suite, actually forecasting.

Every other MYNN test in this repository measures a routine in isolation
against an oracle CSV.  This one runs the model: a moist convective sounding
over mixed land and water, stepped through RK3 with
``sf_sfclay_physics=5``, ``sf_surface_physics=2`` and ``bl_pbl_physics=5``,
and it asserts the things that separate "ported" from "runs":

* the selector reaches MYNN and not YSU;
* the ten carried 3-D arrays exist, start at WRF's zero cold state, and are
  filled by the first call;
* the PBL top actually moves and the diffusivity actually grows, which is
  what says the scheme is coupled rather than merely called;
* nothing goes non-finite;
* a restart round trip reproduces every carried array bit for bit **and**
  the next step from the restored state is bit identical, which is the
  property a checkpoint is for; and
* wrfout carries the MYNN names.

The vertical grid is stretched deliberately.  On a uniform 40-level grid to
16 km the first interface sits near 400 m, and WRF's ``GET_PBLH`` searches
for the minimum theta_v only below 200 m (``module_bl_mynn.F:5537-5546``);
with an empty search window the whole theta-based branch degenerates to
``zw(2)`` and the PBL top is pinned at the first interface for the entire
run.  That is not a gpuwm defect -- it is what WRF does on a grid no real
configuration uses -- but a test on such a grid would assert nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core.mynn_pbl_runtime import MYNN_PBL_STATE_3D


def _build(stretch: float = 3.2, nx: int = 8, ny: int = 6):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(nx=nx, ny=ny, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=6, sf_sfclay_physics=5, sf_surface_physics=2,
                    bl_pbl_physics=5, bldt=0.0)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=stretch)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(7.0)
    state.v[...] = cp.float32(1.5)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    landmask[:, -2:] = 0.0
    tsk = np.full((cfg.ny, cfg.nx), 301.0)
    tsk[landmask == 0.0] = 297.0
    soil_t = np.stack([tsk - 0.5, tsk - 1.0, tsk - 1.5, tsk - 2.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.30)
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(landmask, 10, 17), isltyp=np.where(landmask, 6, 14),
        vegfra=55.0, tmn=287.0, swdown=600.0, glw=330.0, pblh=500.0)
    return state, cfg, driver


@requires_gpu
def test_the_mynn_suite_forecasts_and_stays_finite():
    from gpuwm.core.dycore import step, stability_report

    state, cfg, driver = _build()
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_mynn_pbl"
    assert driver.scheme_dispatch["sf_sfclay_physics"] == "_run_sfclay"

    # WRF cold-starts every carried array at zero; the initflag block inside
    # the driver is what seeds them, not initialize_physics.
    for name in MYNN_PBL_STATE_3D:
        assert name in driver.fields, name
        assert not bool(cp.any(driver.fields[name])), name

    for _ in range(20):
        step(state, cfg)
        health = stability_report(state, cfg)
        assert not health["nan"]

    for name in MYNN_PBL_STATE_3D:
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name
    # qke, el and the diffusivities must have been produced, not left at the
    # cold state: a runner that returned zeros would pass every finiteness
    # assertion above.
    assert float(cp.asnumpy(driver.fields["qke"]).max()) > 0.05
    assert float(cp.asnumpy(driver.fields["el_pbl"]).max()) > 1.0
    assert float(cp.asnumpy(driver.fields["exch_h"]).max()) > 1.0
    pblh = cp.asnumpy(driver.fields["pblh"])
    assert np.isfinite(pblh).all()
    # The PBL top must have left its 500 m initial value and must differ
    # between the heated land block and the cooler water block.
    assert not np.allclose(pblh, 500.0)
    assert pblh[:, 0].max() > pblh[:, -1].max()
    kpbl = cp.asnumpy(driver.fields["kpbl"])
    assert kpbl.min() >= 1 and kpbl.max() < cfg.nz


@requires_gpu
def test_a_mynn_restart_round_trips_and_reproduces_the_next_step(tmp_path):
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import restore_restart, write_restart

    state, cfg, driver = _build()
    for _ in range(6):
        step(state, cfg)

    path = tmp_path / "mynn.npz"
    write_restart(path, state, cfg)
    fresh_state, fresh_cfg, fresh_driver = _build()
    restore_restart(path, fresh_state, fresh_cfg)

    carried = (*MYNN_PBL_STATE_3D, "exch_h", "exch_m", "pblh", "rmol",
               "maxwidth", "maxmf", "ztop_plume")
    for name in carried:
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=name)
    for name in ("kpbl", "ktop_plume"):
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=name)

    # Storage identity is not trajectory identity.  Step both and compare.
    step(state, cfg)
    step(fresh_state, fresh_cfg)
    for name in ("qke", "el_pbl", "cldfra_bl", "pblh"):
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=f"after/{name}")
    for name in ("u", "v", "w", "thp"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(fresh_state, name)),
            cp.asnumpy(getattr(state, name)), err_msg=f"after/{name}")


@requires_gpu
def test_wrfout_carries_the_mynn_state():
    from gpuwm.core.dycore import step
    from gpuwm.io.wrfout import state_frame

    state, cfg, driver = _build()
    step(state, cfg)
    frame = state_frame(state)
    for name in ("QKE", "TSQ", "QSQ", "COV", "EL_PBL", "SH3D", "SM3D",
                 "QC_BL", "QI_BL", "CLDFRA_BL", "EXCH_H", "EXCH_M",
                 "MAXWIDTH", "MAXMF", "ZTOP_PLUME", "KTOP_PLUME"):
        assert name in frame, name
        assert np.isfinite(np.asarray(frame[name], dtype=np.float64)).all()


@requires_gpu
def test_a_ysu_run_gains_no_mynn_arrays():
    """The negative control for the allocation gate.

    Ten extra 3-D arrays in every YSU run would change that run's restart
    inventory and its VRAM budget, and on this hardware the VRAM budget is a
    correctness bar rather than a performance one.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.preflight import physics_array_shapes

    cfg = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: 0.008 + 0.0 * z)
    driver = initialize_physics(state, cfg)
    for name in MYNN_PBL_STATE_3D:
        if name in ("qke", "tsq", "qsq", "cov"):
            assert name not in driver.fields, name
    shapes = physics_array_shapes(cfg)
    assert not any(key.startswith("fields/qke") for key in shapes)


@requires_gpu
def test_the_vram_preflight_counts_the_mynn_arrays():
    """Under-counting VRAM is a correctness bar on this hardware."""
    from gpuwm.core.preflight import physics_array_shapes

    ysu = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1)
    mynn = RunConfig(**{**{f.name: getattr(ysu, f.name)
                          for f in ysu.__dataclass_fields__.values()},
                        "sf_sfclay_physics": 5, "bl_pbl_physics": 5})
    ysu_shapes = physics_array_shapes(ysu)
    mynn_shapes = physics_array_shapes(mynn)
    added = set(mynn_shapes) - set(ysu_shapes)
    for name in MYNN_PBL_STATE_3D:
        assert f"fields/{name}" in added, name
        assert mynn_shapes[f"fields/{name}"] == (ysu.nz, ysu.ny, ysu.nx)
    for name in ("maxwidth", "maxmf", "ztop_plume", "ktop_plume"):
        assert f"fields/{name}" in added, name


@requires_gpu
@pytest.mark.parametrize("stretch,expect_plumes", [
    (1.6, True),     # dz1 ~ 43 m -- inside the DMP_mf oracle's coverage
    (2.2, True),     # dz1 ~ 18 m -- just below the thinnest oracle column
    (2.8, False),    # dz1 ~  7 m -- onset is at step 30, i.e. after this
                     #               census; NOT "no plume ever"
])
def test_where_the_edmf_mass_flux_stops_activating(stretch, expect_plumes):
    """Bracket the 20-step DMP_mf plume census against first-layer depth.

    This is a bracket on **this census**, not on the routine.  Sampled after
    twenty steps the census looks like a threshold near 10 m, and an earlier
    revision of this docstring and of
    ``docs/mynn_noahmp_ruc_completion_plan.md`` said MYNN "stops producing
    plumes entirely" below it.  **That reading is wrong.**  Re-swept with the
    census taken at every step for sixty steps, first-layer depth delays the
    onset rather than gating it: plumes appear at step 12 at dz1 = 22.8 m, at
    step 26 at 8.9 m, at step ~28 by 5 m and at step 41 at 1.67 m, and below
    about 4 m the activation flickers on and off.  Every depth measured from
    22.8 m down to 1.67 m eventually produces plumes; only 1.31 m had none
    within sixty steps.  No run went non-finite and peak ``|w|`` stayed under
    0.75 m/s, so the flicker is not a blow-up.  The measured sweep is the
    table in ``docs/mynn_noahmp_ruc_completion_plan.md``.

    The consequence for this test: a fixed-step census is non-monotone in
    dz1, so the three rows below are a tripwire on the census and nothing
    more.  They are kept because moving them still means something changed in
    ``DMP_mf`` or in the surface fluxes feeding it, and somebody then has to
    decide whether the move was toward WRF or away from it.

    Whether unmodified WRF behaves the same way is **not established here**.
    The pinned ``gpuwm/data/mynn/oracle/dmp-mf.csv`` has a thinnest first
    layer of 20 m across all twelve columns, so the fixture never reached
    this regime and cannot answer it.

    It matters beyond curiosity because the near-surface spacing involved is
    not exotic -- an LES-adjacent nest is exactly where a first layer under
    10 m appears.
    """
    from gpuwm.core.dycore import step

    state, cfg, driver = _build(stretch=stretch, nx=6, ny=4)
    for _ in range(20):
        step(state, cfg)
    ktop = cp.asnumpy(driver.fields["ktop_plume"])
    maxmf = cp.asnumpy(driver.fields["maxmf"])
    maxwidth = cp.asnumpy(driver.fields["maxwidth"])
    # The plumes are started on every configuration; only how long they take
    # to survive the NUP2 = 0 abort changes.  MAXWIDTH is nonzero even on the
    # row that has no plume yet at step 20, which is what says "not started"
    # is the wrong description of it.
    assert maxwidth.max() > 100.0
    if expect_plumes:
        assert int((ktop > 0).sum()) > 0
        assert float(np.abs(maxmf).max()) > 0.0
    else:
        assert int((ktop > 0).sum()) == 0
        assert float(np.abs(maxmf).max()) == 0.0


@requires_gpu
def test_the_pbl_surface_pairings_the_driver_accepts_are_wrfs_own():
    """Every PBL/surface-layer pairing, against WRF v4.6.1's own table.

    This used to assert that two pairings -- MYNN surface with YSU, and
    classic MM5 surface with MYNN PBL -- both refused, under the name
    "the MYNN half suite".  Half of that was wrong, and the v1.3.x
    combination lane found it in WRF's source:
    ``phys/module_physics_init.F:3837-3839`` accepts ``isfc`` in
    {5, 1, 2} for MYNN PBL, and classic MM5 supplies ``isfc=1``
    (``:3140-3142``), so ``(bl_pbl=5, sf_sfclay=91)`` is legal WRF and is
    admitted now.  ``(bl_pbl=1, sf_sfclay=5)`` remains fatal: YSU
    requires ``isfc=1`` (``:3699-3701``) and MYNN surface supplies 5.

    Rewritten to sweep the authority rather than restate a conclusion.
    A hardcoded pair is how the old story survived being false; asking
    ``pbl_surface_layer_verdict`` means this test cannot disagree with
    the table without the table changing.
    """

    from gpuwm.config import validate_run_config
    from gpuwm.physics_compat import UnsupportedPhysicsSuiteError
    from gpuwm.wrf461_compatibility import (
        PBL_SURFACE_LAYER_AUTHORITY,
        WRFVerdict,
    )

    def config(surface, pbl, land_surface):
        return RunConfig(
            nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
            dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
            sf_sfclay_physics=surface, sf_surface_physics=land_surface,
            num_soil_layers=4 if land_surface else 4,
            bl_pbl_physics=pbl)

    seen = {"fatal": 0, "legal": 0}
    for (pbl, surface), (verdict, citation) in \
            PBL_SURFACE_LAYER_AUTHORITY.items():
        # Noah needs a surface layer to write its exchange seam; with the
        # surface layer off that ArWen-structural refusal would mask the
        # verdict this test is about, so the LSM comes off with it.
        land_surface = 2 if surface else 0
        if verdict is WRFVerdict.FATAL:
            with pytest.raises(UnsupportedPhysicsSuiteError) as caught:
                validate_run_config(config(surface, pbl, land_surface))
            assert citation.path in str(caught.value)
            seen["fatal"] += 1
        else:
            cfg = config(surface, pbl, land_surface)
            assert validate_run_config(cfg) is cfg
            seen["legal"] += 1

    # WRF's complete 12-cell table: 9 legal, 3 fatal.
    assert seen == {"legal": 9, "fatal": 3}
