# tests/test_diffusion.py
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_diff2_matches_reference():
    import cupy as cp
    from gpuwm.core.diffusion import launch_add_diff2
    from gpuwm.verify.npref import np_add_diff2, random_field_setup
    f, meta = random_field_setup(nz=10, ny=3, nx=20, seed=5)
    tend = cp.zeros_like(cp.asarray(f, dtype=cp.float32))
    launch_add_diff2(cp.asarray(f, dtype=cp.float32), tend, kh=75.0, kv=75.0, **meta)
    ref = np_add_diff2(f, kh=75.0, kv=75.0, **meta)
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=1e-4, atol=1e-7)


@requires_gpu
def test_gaussian_decay_rate():
    # pure diffusion of sin(kx): amplitude must decay as exp(-K k^2 t) within 2%
    import cupy as cp
    from gpuwm.core.diffusion import diffuse_only_test
    nx, dx, K = 128, 100.0, 75.0
    kwave = 2 * np.pi / (nx * dx) * 8
    x = (np.arange(nx) + 0.5) * dx
    q0 = np.sin(kwave * x).astype(np.float32)
    t = 200.0
    qf = diffuse_only_test(q0, K=K, dx=dx, dt=1.0, t_end=t)
    expected = np.exp(-K * kwave ** 2 * t)
    measured = np.max(np.abs(qf)) / np.max(np.abs(q0))
    assert abs(measured - expected) / expected < 0.02


@requires_gpu
def test_rayleigh_damping_smoke():
    # damp_opt=3 analogue: implicit relaxation toward the at-rest base state,
    # active only above ztop - zdamp; damp_opt=0 must be an exact no-op.
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.diffusion import apply_rayleigh_damping
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import init_at_rest
    cfg = RunConfig(nx=8, ny=1, nz=16, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=0.0, damp_opt=3, zdamp=4000.0,
                    dampcoef=0.2)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: np.full_like(np.asarray(z, float), 300.0),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    s.w[...] = 1.0
    s.thp[...] = 1.0
    apply_rayleigh_damping(s, cfg)
    w = cp.asnumpy(s.w)
    assert np.all(np.isfinite(w))
    z_full = cp.asnumpy(s.phb).astype(np.float64) / 9.81
    below = z_full < cfg.ztop - cfg.zdamp - 1.0
    assert np.all(w[below] == 1.0)               # untouched below the layer
    # top w level: full strength, factor 1/(1 + dt*dampcoef)
    np.testing.assert_allclose(w[-1], 1.0 / (1.0 + cfg.dt * cfg.dampcoef),
                               rtol=2e-3)
    assert float(s.thp[-1, 0, 0]) < 1.0          # theta' relaxed toward base
    # damp_opt=0: exact no-op
    cfg0 = RunConfig(**{**cfg.__dict__, "damp_opt": 0})
    s.w[...] = 1.0
    apply_rayleigh_damping(s, cfg0)
    assert float(cp.abs(s.w - 1.0).max()) == 0.0
