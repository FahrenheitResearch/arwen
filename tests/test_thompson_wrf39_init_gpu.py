"""Start emission against the unmodified fork's thompson_init.

The fixture is emitted by tools/thompson_fork_oracle/start_emission.F90,
linked to the pinned fork module. It covers rectangular cells on both
sides of the 20 km scale cap, zero CCN and five positive analysed numbers.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

from gpuwm.config import RunConfig
from gpuwm.core.microphysics_aerosol import thompson_aerosol_init_fill

_NUMBERS = np.array([0., 1.e6, 1.e7, 1.e8, 1.e9, 9.999e9], np.float32)
_ORACLE = np.loadtxt(Path(__file__).parent / "data" /
                     "thompson_wrf39_start_emission.txt")


def _state():
    shape = (3, 2, 6)
    return SimpleNamespace(
        p=cp.ones(shape, cp.float32), phb=cp.zeros(4, cp.float32),
        php=cp.zeros((4, 2, 6), cp.float32),
        nwfa=cp.array(np.broadcast_to(_NUMBERS, shape).copy()),
        nifa=cp.full(shape, 5.e5, cp.float32),
        nwfa2d=cp.full((2, 6), 123., cp.float32),
        scratch=lambda shape, slot: cp.empty(shape, cp.float32))


@pytest.mark.parametrize("row", _ORACLE)
def test_fork_start_emission_matches_its_own_fortran(row):
    state = _state()
    cfg = RunConfig(nx=6, ny=2, nz=3, dx=float(row[0]), dy=float(row[1]),
                    dt=20., ztop=16000., run_seconds=0.,
                    mp_physics=28, thompson_version="wrf_39_noaa")
    original = state.nwfa.copy()
    assert thompson_aerosol_init_fill(state, cfg) == {"ccn": False, "in": False}
    np.testing.assert_allclose(cp.asnumpy(state.nwfa2d),
                               np.broadcast_to(row[2:], (2, 6)), rtol=3.e-6)
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), cp.asnumpy(original))
    # A second domain initialization recomputes from the current analysis.
    state.nwfa2d.fill(-1.)
    thompson_aerosol_init_fill(state, cfg)
    np.testing.assert_allclose(cp.asnumpy(state.nwfa2d[0]), row[2:], rtol=3.e-6)


def test_v461_initialization_preserves_analysed_surface_emission_bitwise():
    state = _state()
    before = {name: cp.asnumpy(getattr(state, name)).copy()
              for name in ("nwfa", "nifa", "nwfa2d")}
    cfg = RunConfig(nx=6, ny=2, nz=3, dx=3000., dy=3000., mp_physics=28,
                    dt=20., ztop=16000., run_seconds=0.)
    assert thompson_aerosol_init_fill(state, cfg) == {"ccn": False, "in": False}
    for name, expected in before.items():
        np.testing.assert_array_equal(cp.asnumpy(getattr(state, name)), expected)
