"""Exercise MP24 through the real model state, RK3 and physics dispatch."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu

@pytest.mark.gpu
@requires_gpu
def test_hail_survives_real_state_and_model_dispatch():
    import cupy as cp
    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.moist import extra_moist_species
    from gpuwm.verify.cases import moist_bubble
    from gpuwm.verify.cases.wk82 import wk82_sounding
    cfg = validate_run_config(replace(
        moist_bubble.default_config(), nx=32, ny=8, nz=40,
        dt=1.5, run_seconds=4.5, time_step_sound=4, mp_physics=24,
        km_opt=1, khdif=0., kvdif=0., diff_6th_opt=0, damp_opt=0))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = moist_bubble.build(cfg, coord, base)
    assert extra_moist_species(state) == ("qi", "qs", "qg", "qh")
    assert getattr(state, 'qnh', None) is None and state.qh0 is not None
    z = cp.asarray(state.height_half())[:, None, None]
    seed = cp.exp(-((z - np.float32(6000.)) / np.float32(1800.))**2)
    for name, amount in (("qc",2e-4),("qi",5e-5),("qr",8e-5),
                         ("qs",6e-5),("qg",2e-5),("qh",4e-5)):
        getattr(state, name)[...] += np.float32(amount) * seed
    initial_hail = state.qh.copy()
    run_steps(state, cfg, n=3)
    cp.cuda.runtime.deviceSynchronize()
    assert state.elapsed_seconds == pytest.approx(4.5)
    for name in ("qv", "qc", "qi", "qr", "qs", "qg", "qh"):
        field = getattr(state, name)
        assert bool(cp.isfinite(field).all()), name
        assert float(field.min()) >= -1.e-10, name
    assert float(cp.max(cp.abs(state.qh - initial_hail))) > 0.
    assert float(cp.max(cp.abs(state.h_diabatic))) > 0.
    assert state.scratch((cfg.ny,cfg.nx), "mp_hailnc") is not None
