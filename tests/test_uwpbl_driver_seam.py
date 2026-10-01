"""The UW moist-turbulence PBL (bl_pbl_physics=9) through the physics driver.

On the card: ``initialize_physics`` allocates the scheme's fields, the
dycore's step calls ``_run_uwpbl``, and the tendencies are coupled.  Split
from tests/test_uwpbl_integration.py because the state helper below imports
cupy, which marks this whole module ``gpu`` (tests/conftest.py).
"""

from __future__ import annotations

import pytest

from gpuwm.config import UW_PBL_SCHEME, RunConfig


def _driver_cfg(**overrides) -> RunConfig:
    base = dict(nx=6, ny=5, nz=24, dx=3000.0, dy=3000.0, dt=6.0,
                ztop=15000.0, run_seconds=60.0, moist=True, mp_physics=8,
                sf_sfclay_physics=1, bl_pbl_physics=UW_PBL_SCHEME,
                sf_surface_physics=2, num_soil_layers=4, bldt=0.0,
                radt=0.0, cu_physics=0, km_opt=4, c_s=0.25)
    base.update(overrides)
    return RunConfig(**base)


def _driver_state(cfg):
    import cupy as cp
    import numpy as np

    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest

    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: 300.0 + 0.003 * np.asarray(z, np.float64),
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_at_rest(cfg, coord, base)
    state.u[...] = cp.asarray(np.full(tuple(state.u.shape), 6.0),
                              dtype=state.u.dtype)
    if state.qv is not None:
        state.qv[...] = 0.008
    return state


@pytest.mark.gpu
def test_the_driver_runs_the_scheme_and_couples_it():
    """Allocated, called, finite and coupled on a moist Thompson state.

    The bars separate "called" from "coupled": the diffusivities leave the
    cold state, the PBL top is a depth, TKE_PBL is published, and the
    scheme's fields are the ones the restart writer serializes.
    """
    import cupy as cp
    import numpy as np

    from gpuwm.core.dycore import run_steps
    from gpuwm.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)

    cfg = _driver_cfg()
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=310.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=600.0)
    for name in ("uw_kvm", "uw_kvh", "tke_pbl", "turbtype3d", "smaw3d"):
        assert driver.fields[name].shape == (cfg.nz + 1, cfg.ny, cfg.nx)
    assert float(driver.fields["tke_pbl"][0, 0, 0]) == pytest.approx(0.2)

    # The first step is the scheme's own reset (kvinit: the carried
    # diffusivities start from zero, module_bl_camuwpbl_driver.F:436-441),
    # and this weakly stable column can fall back below the critical
    # Richardson number on a later step, which zeroes the diffusivity
    # again; so the bar is that it left the cold state on SOME step.
    kvh_peak = 0.0
    for _ in range(10):
        run_steps(state, cfg, 1)
        kvh_peak = max(kvh_peak,
                       float(np.max(cp.asnumpy(driver.fields["uw_kvh"]))))

    assert driver.call_counts["ysu"] > 0          # the PBL-slot counter
    for name in ("uw_kvh", "uw_kvm", "tke_pbl", "pblh", "exch_h",
                 "tauresx2d", "tpert2d"):
        values = cp.asnumpy(driver.fields[name])
        assert np.all(np.isfinite(values)), name
    assert kvh_peak > 0.0
    assert float(np.max(cp.asnumpy(driver.fields["pblh"]))) > 0.0
    assert np.array_equal(cp.asnumpy(driver.fields["exch_h"]),
                          cp.asnumpy(driver.fields["uw_kvh"])[:cfg.nz])
    assert np.all(np.isfinite(cp.asnumpy(state.thp)))
    # Thompson carries WRF's P_QNI as state.ni: the scheme's ice-number
    # tendency reaches that species and no other.
    extra = getattr(driver.pbl_tendencies, "extra_scalars", None)
    assert extra is not None and set(extra) == {"ni"}


@pytest.mark.gpu
def test_a_scheme_without_ice_number_holds_no_extra_tendency():
    import cupy  # noqa: F401

    from gpuwm.core.dycore import run_steps
    from gpuwm.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)

    cfg = _driver_cfg(mp_physics=6)
    state = _driver_state(cfg)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=305.0,
                                glw=DECLARED_CONSTANT_GLW_WM2, swdown=0.0)
    run_steps(state, cfg, 2)
    assert getattr(driver.pbl_tendencies, "extra_scalars", None) is None


def _uw_restart_state(cp):
    """tests/test_restart.py's small full-physics state under the UW PBL.

    Morrison, analytic radiation every minute, KF, MM5 surface layer and
    Noah, as that module's restart gates use; the wind is given a shear
    profile and the surface a warm start so the scheme's turbulent layers
    are active at the checkpoint rather than a quiescent stable column,
    whose diffusivities are zero.
    """
    import numpy as np

    from test_restart import _physics_state

    state, cfg, driver = _physics_state(cp, bl_pbl_physics=UW_PBL_SCHEME)
    profile = np.linspace(2.0, 14.0, cfg.nz, dtype=np.float32)
    state.u[...] = cp.asarray(profile)[:, None, None]
    # A warm skin and soil, so the surface heats the column on the water
    # columns and the land columns alternate between heating and cooling.
    driver.fields["tsk"][...] = cp.float32(306.0)
    driver.fields["tslb"][...] = cp.float32(303.0)
    return state, cfg, driver


@pytest.mark.gpu
def test_a_uw_run_restarts_bit_identically(tmp_path):
    """20 steps + checkpoint + 20 steps == 40 steps, bit for bit.

    The scheme carries state from step to step -- the interface
    diffusivities it reads back as kvm_in/kvh_in, the residual surface
    stress, and the radiation step's held cloud fraction -- and a forecast
    that restarts must resume on exactly those words, not on the scheme's
    first-step reset.  The 20-step boundary is off the radiation calendar
    (due at steps 1, 7, 13, 19, ...), so the held cloud fraction crosses
    it as a stored field.  Morrison carries WRF's P_QNI, so the ice-number
    tendency path runs too (restart-exact at bldt=0, which the admission
    requires for it).
    """
    import cupy as cp
    import numpy as np

    from gpuwm.core.dycore import run_steps
    from gpuwm.io import restart
    from test_restart import _assert_restart_equal

    state_a, cfg_a, _ = _uw_restart_state(cp)
    run_steps(state_a, cfg_a, 40)
    reference = restart.write_restart(tmp_path / "reference.npz",
                                      state_a, cfg_a)

    state_b, cfg_b, driver_b = _uw_restart_state(cp)
    run_steps(state_b, cfg_b, 20)
    # The state this test is about: at the checkpoint the scheme is
    # carrying diffusivities, not its cold start.
    assert float(cp.abs(driver_b.fields["uw_kvh"]).max()) > 0.0
    mid = restart.write_restart(tmp_path / "mid.npz", state_b, cfg_b)

    state_c, cfg_c, _ = _uw_restart_state(cp)
    info = restart.restore_restart(mid, state_c, cfg_c)
    assert info.elapsed_seconds == 200.0
    run_steps(state_c, cfg_c, 20)
    resumed = restart.write_restart(tmp_path / "resumed.npz",
                                    state_c, cfg_c)
    _assert_restart_equal(resumed, reference)

    keys = set(restart.read_restart_header(reference)["array_manifest"])
    for name in ("uw_kvm", "uw_kvh", "tauresx2d", "tauresy2d",
                 "uw_cldfra", "tke_pbl"):
        assert f"fields/{name}" in keys, name
    assert np.isfinite(cp.asnumpy(state_c.thp)).all()
