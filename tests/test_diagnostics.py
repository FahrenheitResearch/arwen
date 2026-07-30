# tests/test_diagnostics.py
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.grid import make_vertical_coord, make_base_state

pytestmark = pytest.mark.gpu

def _setup(nx=32, nz=16):
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b

@requires_gpu
def test_eos_matches_numpy_reference():
    import cupy as cp
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg, vc, b = _setup()
    rng = np.random.default_rng(0)
    def thp_f(x, z):
        return rng.normal(0.0, 1.0, (cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, thp_f)
    update_diagnostics(s)
    p_ref, al_ref, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc)
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.al), al_ref, rtol=2e-5, atol=2e-5)  # al = alt - alb: atol matches alt's absolute FP32 slack

@requires_gpu
def test_rest_state_pressure_equals_base():
    import cupy as cp
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.diagnostics import update_diagnostics
    cfg, vc, b = _setup()
    s = init_at_rest(cfg, vc, b)
    update_diagnostics(s)
    p = cp.asnumpy(s.p)[:, 0, 0]
    np.testing.assert_allclose(p, b.pb, rtol=3e-4)  # FP32 + discrete-vs-analytic slack
