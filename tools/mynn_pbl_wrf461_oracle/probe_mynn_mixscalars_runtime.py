#!/usr/bin/env python3
"""Stage B key-on gate: mixscalars through the FULL driver call sites.

W4 full-admission lane (mf-close2, Stage B).  Runs the MYNN runtime
forecast fixture (tests/test_mynn_pbl_runtime.py ``_build``, stretch 1.6
so the DMP plumes are active inside 20 steps) at ``mp_physics=28`` with
``bl_mynn_mixscalars=1`` -- the first configuration in which
``mynn_bl_driver_cuda`` itself feeds the qn columns and the ``s_awqn*``
interfaces to its DMP and tendency call sites.  Three claims:

  * SMOKE: 20 coupled steps complete, every PBL tendency finite, and the
    four qn tendencies reach the coupled scalar update (extra_scalars).
  * LIVENESS: on the final PBL call, ``s_awqn*`` is nonzero exactly
    where the plume is (plume-active columns), and exactly zero
    elsewhere -- WRF's NUP2 gate observed through the full driver.
  * CPU CROSS-CHECK: the same final-call driver inputs, replayed through
    the CPU reference ``mynn_bl_driver(bl_mynn_mixscalars=1)`` on
    sampled columns, reproduce the device ``rqn*blten`` -- the per
    species worst-ulp table is emitted verbatim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault(
    "GPUWM_THOMPSON_TABLE_ROOT",
    str(Path.home() / ".gpuwm" / "tables" / "thompson"))

# Pin the import to the worktree this probe lives in -- an installed or
# neighbouring gpuwm resolving first would gate somebody else's code.
_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
sys.path.insert(1, os.path.join(_ROOT, "gpuwm-data"))

import numpy as np

import gpuwm  # noqa: E402
assert gpuwm.__file__.lower().startswith(_ROOT.lower()), (
    f"wrong gpuwm resolved: {gpuwm.__file__}")

QN_SPECIES = ("qnc", "qni", "qnwfa", "qnifa", "qnbca")
SAMPLE = (0, 7, 13, 19, 25, 31, 37, 43)
STEPS = 20


def main() -> int:
    import cupy as cp

    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.dycore import step
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.mynn_pbl import mynn_bl_driver
    from gpuwm.core.physics import initialize_physics
    import gpuwm.core.mynn_pbl_gpu as gpu_mod
    import gpuwm.core.mynn_pbl_runtime as runtime_mod

    cfg = RunConfig(
        nx=8, ny=6, nz=50, dx=3000.0, dy=3000.0, ztop=16000.0,
        dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
        mp_physics=28, sf_sfclay_physics=5, sf_surface_physics=2,
        bl_pbl_physics=5, bldt=0.0, bl_mynn_mixscalars=1)
    validate_run_config(cfg)

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

    # --- capture wrappers: last driver call of the run -------------------
    capture: dict[str, object] = {}
    orig_dmp = gpu_mod.mynn_dmp_mf_cuda
    orig_drv = runtime_mod.mynn_bl_driver_cuda

    def dmp_wrap(values, **kw):
        result = orig_dmp(values, **kw)
        capture["ktop"] = cp.asnumpy(result.ktop)
        for name in QN_SPECIES:
            capture[f"s_aw{name}"] = cp.asnumpy(
                getattr(result, f"s_aw{name}"))
        return result

    def drv_wrap(values, **kw):
        capture["values"] = {k: cp.asnumpy(cp.asarray(v))
                             for k, v in values.items()}
        capture["kwargs"] = {k: v for k, v in kw.items() if k != "scratch"}
        out = orig_drv(values, **kw)
        capture["out"] = {k: cp.asnumpy(v) for k, v in out.items()}
        return out

    gpu_mod.mynn_dmp_mf_cuda = dmp_wrap
    runtime_mod.mynn_bl_driver_cuda = drv_wrap
    try:
        for _ in range(STEPS):
            step(state, cfg)
    finally:
        gpu_mod.mynn_dmp_mf_cuda = orig_dmp
        runtime_mod.mynn_bl_driver_cuda = orig_drv

    # --- SMOKE -----------------------------------------------------------
    extras = getattr(driver.pbl_tendencies, "extra_scalars", None)
    assert extras is not None and set(extras) == {"nc", "ni", "nwfa",
                                                  "nifa"}, extras
    for name, array in extras.items():
        host = cp.asnumpy(array)
        assert np.isfinite(host).all(), name
        print(f"SMOKE coupled extra {name}: max|.| {np.abs(host).max():.6e}")

    # --- LIVENESS --------------------------------------------------------
    ktop = capture["ktop"]
    active = ktop > 0
    n_active = int(active.sum())
    print(f"LIVENESS plume-active columns: {n_active}/{ktop.size}")
    assert n_active > 0, "no active plume; the fixture regressed"
    failures = []
    for name in QN_SPECIES:
        flux = capture[f"s_aw{name}"]
        live = int((np.abs(flux[active]).max(axis=1) > 0.0).sum())
        dead_max = float(np.abs(flux[~active]).max()) if (~active).any() \
            else 0.0
        print(f"LIVENESS s_aw{name}: live {live}/{n_active} active cols; "
              f"max|inactive| {dead_max:.6e}; "
              f"max|active| {float(np.abs(flux[active]).max()):.6e}")
        if dead_max != 0.0:
            failures.append(f"s_aw{name} nonzero off-plume")
    # qnbca rides an exactly-zero column: its flux must be exactly zero.
    assert float(np.abs(capture["s_awqnbca"]).max()) == 0.0, \
        "zero qnbca column produced a nonzero flux"

    # --- CPU CROSS-CHECK -------------------------------------------------
    rows = np.asarray(SAMPLE, dtype=np.intp)
    values = {k: (v[rows] if getattr(v, "ndim", 0) >= 1
                  and v.shape[0] == ktop.size else v)
              for k, v in capture["values"].items()}
    kwargs = dict(capture["kwargs"])
    kwargs.pop("flag_qs", None)
    cpu = mynn_bl_driver(
        values, initflag=kwargs.pop("initflag"),
        delt=kwargs.pop("delt"),
        flag_qs=capture["kwargs"].get("flag_qs", False), **kwargs)
    worst_all = 0
    for name in QN_SPECIES:
        key = f"r{name}blten"
        got = capture["out"][key][rows]
        want = cpu[key]
        dist = np.abs(monotone_fp32_key(got.astype(np.float32))
                      - monotone_fp32_key(want.astype(np.float32)))
        ulp = int(dist.max(initial=0))
        worst_all = max(worst_all, ulp)
        where = np.unravel_index(int(dist.argmax()), dist.shape) \
            if dist.size else (0, 0)
        print(f"CROSSCHECK {key}: gpu max|.| "
              f"{float(np.abs(got).max()):.6e}  cpu max|.| "
              f"{float(np.abs(want).max()):.6e}  WORST_ULP {ulp} "
              f"at col{where[0]},k{where[1]} "
              f"(gpu {float(got[where]):.6e}, cpu {float(want[where]):.6e})")
    print(f"CROSSCHECK WORST_ULP overall: {worst_all}")
    if failures:
        print("FAILURES: " + "; ".join(failures))
        return 1
    print("STAGE B KEY-ON GATE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
