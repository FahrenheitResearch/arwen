# tests/test_state.py
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig

pytestmark = pytest.mark.gpu

CFG = dict(nx=16, ny=1, nz=8, dx=100.0, dy=100.0, ztop=6400.0,
           dt=0.5, run_seconds=10.0)

@requires_gpu
def test_shapes_dtypes_and_rest_state():
    import cupy as cp
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import init_at_rest
    cfg = RunConfig(**CFG)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: np.full_like(np.asarray(z, float), 300.0),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    assert s.u.shape == (8, 1, 17) and s.u.dtype == cp.float32
    assert s.v.shape == (8, 2, 16)
    assert s.w.shape == (9, 1, 16)
    assert float(cp.abs(s.thp).max()) == 0.0
    assert float(cp.abs(s.php).max()) == 0.0

@requires_gpu
def test_bubble_init_is_rebalanced():
    import cupy as cp
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import init_theta_perturbation
    cfg = RunConfig(**CFG)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: np.full_like(np.asarray(z, float), 300.0),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    def bubble(x, z):
        # Coordinate contract: 1-D domain-centered cell-center x, half-level z.
        assert x.shape == (cfg.nx,) and z.shape == (cfg.nz,)
        assert np.isclose(x[0], -0.5 * cfg.nx * cfg.dx + 0.5 * cfg.dx)   # -750.0
        assert np.isclose(x[-1], 0.5 * cfg.nx * cfg.dx - 0.5 * cfg.dx)   # +750.0
        assert np.all(np.diff(z) > 0) and 0.0 < z[0] and z[-1] < cfg.ztop
        out = np.zeros((cfg.nz, cfg.ny, cfg.nx), np.float64)
        out[2:4, :, 6:10] = -5.0
        return out
    s = init_theta_perturbation(cfg, vc, b, bubble)
    assert float(cp.abs(s.thp).max()) == 5.0
    assert float(cp.abs(s.php).max()) > 0.0   # rebalance moved geopotential
    assert float(cp.abs(s.mup).max()) == 0.0
