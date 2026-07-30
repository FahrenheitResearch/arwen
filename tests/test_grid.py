# tests/test_grid.py
import numpy as np
from gpuwm.core import constants as c
from gpuwm.core.grid import make_vertical_coord, make_base_state, rebalance_hydrostatic

def theta_const(z):
    return np.full_like(np.asarray(z, float), 300.0)

def test_coord_conventions():
    vc = make_vertical_coord(64)
    assert vc.znw[0] == 1.0 and vc.znw[-1] == 0.0
    assert np.all(vc.dnw < 0)
    np.testing.assert_allclose(vc.znu, 0.5 * (vc.znw[:-1] + vc.znw[1:]))

def test_base_state_discrete_balance():
    vc = make_vertical_coord(64)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    # the recurrence must hold to round-off by construction
    resid = (b.phb[1:] - b.phb[:-1]) + vc.dnw * b.mub * b.alb
    assert np.max(np.abs(resid)) < 1e-8
    # and heights must be physically sane for isentropic 300K atmosphere
    z_top_model = b.phb[-1] / c.G
    assert abs(z_top_model - 6400.0) < 50.0

def test_rebalance_matches_base_when_unperturbed():
    vc = make_vertical_coord(32)
    b = make_base_state(vc, theta_const, p_surf=1.0e5, ztop=6400.0)
    th3 = np.broadcast_to(b.thb[:, None, None], (32, 1, 8)).copy()
    ph3 = rebalance_hydrostatic(th3, b.mub, vc, p_surf=1.0e5)
    np.testing.assert_allclose(ph3[:, 0, 0], b.phb, atol=1e-6)
