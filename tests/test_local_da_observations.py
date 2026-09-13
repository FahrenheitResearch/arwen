"""Neutral-table adapter units that need no shared observation package.

The shared-table half of this adapter lives in
``tests/test_local_da_obs_tables.py``: it needs the pinned global package
for its row type, and a module-level ``importorskip`` here would have taken
these two with it.  A file that collects nothing is invisible to
``tools/battery/no_silent_deselection.py`` unless somebody excuses it by
name, so the dependency is split out rather than allowed to silence a whole
file.
"""
from types import SimpleNamespace
import numpy as np

from gpuwm.da.obs_point import _interp_logp, state_columns


def grid():
    return SimpleNamespace(nz=2, ny=3, nx=3, terrain_m=np.zeros((3, 3)),
        z_w=np.broadcast_to(np.array([0., 1000., 3000.])[:, None, None], (3, 3, 3)),
        mass_index=lambda lat, lon: (np.asarray(lon), np.asarray(lat)))


def test_log_pressure_extrapolation_and_nonmonotonic_column_refused():
    assert np.isnan(_interp_logp([90000., 70000.], [290., 270.], 50000.))
    assert np.isnan(_interp_logp([70000., 90000.], [270., 290.], 80000.))
    assert np.isnan(_interp_logp([90000., 90000.], [290., 270.], 80000.))


def test_native_columns_convert_mixing_ratio_and_rotate_staggered_winds():
    g = grid()
    g.lon = np.zeros((3, 3))
    g.projection = SimpleNamespace(rotation=lambda l: (np.ones_like(l), np.zeros_like(l)))
    state = dict(p=np.full((2, 3, 3), 90000.), thp=np.zeros((2, 3, 3)), qv=np.full((2, 3, 3), .01),
                 u=np.ones((2, 3, 4)), v=np.zeros((2, 4, 3)), w=np.zeros((3, 3, 3)))
    col = state_columns([state], {'thb': np.full((2, 3, 3), 300.)}, g)[0]
    np.testing.assert_allclose(col['q'], .01 / 1.01)
    np.testing.assert_allclose(col['u'], 0.)
    assert np.all(abs(col['v']) == 1.)
