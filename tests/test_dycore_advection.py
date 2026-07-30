# tests/test_dycore_advection.py
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig

pytestmark = pytest.mark.gpu


@requires_gpu
def test_theta_bubble_translates_in_uniform_wind():
    import cupy as cp
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.dycore import run_steps
    cfg = RunConfig(nx=100, ny=1, nz=32, dx=200.0, dy=200.0, ztop=6400.0,
                    dt=2.0, run_seconds=0.0)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: np.full_like(np.asarray(z, float), 300.0),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    def blob(x, z):
        out = np.zeros((cfg.nz, cfg.ny, cfg.nx))
        out[:] = 2.0 * np.exp(-((x[None, None, :] + 4000.0) / 1500.0) ** 2) \
                     * np.exp(-((z[:, None, None] - 3000.0) / 800.0) ** 2)
        return out
    s = init_theta_perturbation(cfg, vc, b, blob)
    s.u[:] = 10.0                       # uniform 10 m/s
    thp_initial = cp.asnumpy(s.thp).copy()
    n = int(400.0 / cfg.dt)             # 400 s at 10 m/s -> 4000 m -> exactly 20 cells
    run_steps(s, cfg, n, acoustic=False)
    got = cp.asnumpy(s.thp)
    expected = np.roll(thp_initial, 20, axis=2)
    assert np.max(np.abs(got - expected)) < 0.05 * 2.0   # <5% of amplitude
