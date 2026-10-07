"""bl_mynn_mixscalars=1 without number species is inert, bit for bit, on the card.

WRF v4.6.1 gates every MYNN scalar solve on the key AND the species flag
(phys/module_bl_mynn.F mynn_tendencies :4654/:4695/:4736/:4778/:4820) and
zeroes each column whose flag is false, so a microphysics scheme that
carries no number species at all (WSM6 here: Registry package
moist:qv,qc,qr,qi,qs,qg, no scalar) runs the key as a no-op.  gpuwm used
to refuse that configuration ("requires mp_physics=28") -- a refusal that
named no breakage, since WRF runs it.  The fix admits it as WRF's no-op:
the driver is handed the key-0 path (config.mynn_mixscalars_driver_value)
and nothing is staged.  This test is the measurement behind that claim:
two short runs on WSM6, key 0 and key 1, must leave every state array
bitwise identical, and the active case (mp=28, the whole family) must
still hand the driver 1.  Thompson (mp=8) is NOT an inert case: WRF mixes
its qni under the key, and the validator refuses it by name
(tests/test_mynn_options.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

try:
    import cupy as cp
except Exception:  # pragma: no cover - the marker skips
    cp = None

from gpuwm.config import (RunConfig, mynn_mixscalars_active,
                          mynn_mixscalars_driver_value, validate_run_config)

STEPS = 12


def _config(**overrides):
    values = dict(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=6, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0)
    values.update(overrides)
    cfg = RunConfig(**values)
    validate_run_config(cfg)
    return cfg


def _run(cfg):
    """Step a small balanced column set and return every state array
    plus the ``bl_mynn_mixscalars`` the MYNN driver was handed."""
    from gpuwm.core.dycore import step
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    import gpuwm.core.mynn_pbl_runtime as runtime_mod

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0135,
                        np.maximum(0.0135 - 6.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=1.6)
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
    assert driver.scheme_dispatch["bl_pbl_physics"] == "_run_mynn_pbl"

    handed: list[int] = []
    orig_drv = runtime_mod.mynn_bl_driver_cuda

    def drv_wrap(values, **kw):
        handed.append(int(kw.get("bl_mynn_mixscalars", 0)))
        return orig_drv(values, **kw)

    runtime_mod.mynn_bl_driver_cuda = drv_wrap
    try:
        for _ in range(STEPS):
            step(state, cfg)
    finally:
        runtime_mod.mynn_bl_driver_cuda = orig_drv
    arrays = {name: cp.asnumpy(value) for name, value in vars(state).items()
              if isinstance(value, cp.ndarray)}
    assert arrays, "the state exposes no device arrays to compare"
    return arrays, handed


@requires_gpu
def test_key_without_qn_family_is_bitwise_the_key_off_run():
    if cp is None:
        pytest.skip("no CUDA GPU / cupy")
    off_cfg = _config(bl_mynn_mixscalars=0)
    on_cfg = _config(bl_mynn_mixscalars=1)
    assert not mynn_mixscalars_active(on_cfg)
    assert mynn_mixscalars_driver_value(on_cfg) == 0
    off, handed_off = _run(off_cfg)
    on, handed_on = _run(on_cfg)
    assert handed_off and set(handed_off) == {0}
    assert handed_on and set(handed_on) == {0}, (
        "the inert key reached the driver as 1: the no-op is not WRF's")
    assert off.keys() == on.keys()
    for name in sorted(off):
        a, b = off[name], on[name]
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
            f"{name}: bl_mynn_mixscalars=1 under a microphysics with no "
            "qn family changed the trajectory; WRF's skipped solves do not")


@requires_gpu
def test_key_with_qn_family_reaches_the_driver_active():
    if cp is None:
        pytest.skip("no CUDA GPU / cupy")
    cfg = _config(mp_physics=28, bl_mynn_mixscalars=1)
    assert mynn_mixscalars_active(cfg)
    _, handed = _run(cfg)
    assert handed and set(handed) == {1}
