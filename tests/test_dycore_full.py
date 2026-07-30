# tests/test_dycore_full.py
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig

pytestmark = pytest.mark.gpu


def _theta_centroid_height(s, b, vc):
    """theta'-weighted mean of the base-state half-level heights (m)."""
    import cupy as cp
    thp = np.clip(cp.asnumpy(s.thp).astype(np.float64), 0.0, None)
    z = (0.5 * (b.phb[:-1] + b.phb[1:]) / 9.81)[:, None, None]
    return float((thp * z).sum() / thp.sum())


def _rest_setup(nx=64, nz=32, N=0.01):
    # constant Brunt-Vaisala frequency N: theta(z) = 300*exp(N^2 z / g)
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=0.0)
    vc = make_vertical_coord(nz)
    th = lambda z: 300.0 * np.exp(N * N * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b


@requires_gpu
def test_at_rest_stays_at_rest():
    import cupy as cp
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.dycore import run_steps, stability_report
    cfg, vc, b = _rest_setup()
    s = init_at_rest(cfg, vc, b)
    run_steps(s, cfg, n=500)                     # 3000 s
    r = stability_report(s, cfg)
    assert not r["nan"]
    assert r["cfl"] is not None and r["cfl"] < 0.5   # real CFL value exercised
    assert r["w_max"] < 1e-3                     # FP32 round-off scale only
    assert float(cp.abs(s.thp).max()) < 1e-2


@requires_gpu
def test_warm_bubble_rises():
    import cupy as cp
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.dycore import run_steps
    cfg, vc, b = _rest_setup(nx=100, nz=50)
    cfg = RunConfig(**{**cfg.__dict__, "dx": 200.0, "dy": 200.0, "dt": 1.0})
    def bubble(x, z):
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((z[:, None, None] - 2000.0) / 2000.0) ** 2)
        return np.where(L < 1.0, 2.0 * np.cos(np.pi * L / 2) ** 2, 0.0) \
                 * np.ones((cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, bubble)
    zc0 = _theta_centroid_height(s, b, vc)
    run_steps(s, cfg, n=600)                     # 600 s
    zc1 = _theta_centroid_height(s, b, vc)
    assert zc1 > zc0 + 500.0                     # bubble clearly rising
    from gpuwm.core.dycore import stability_report
    assert not stability_report(s)["nan"]
