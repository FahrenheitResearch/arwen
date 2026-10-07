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


def _fortran_repair(field):
    """module_mp_thompson.F:499-512 as written, column by column."""
    out = field.copy()
    nz = out.shape[0]
    for j in range(out.shape[1]):
        for i in range(out.shape[2]):
            for k in range(nz):
                if out[k, j, i] < 0.0:
                    out[k, j, i] = max(out[max(k - 1, 0), j, i],
                                       out[min(k + 1, nz - 1), j, i])
                    out[k, j, i] = max(np.float32(0.0), out[k, j, i])
    return out


def test_fork_repairs_negative_analysed_aerosol_as_its_thompson_init():
    """Audit T25: negative QNWFA/QNIFA in an analysed profile take the
    larger neighbour (the one below already repaired), floored at zero,
    before the start emission reads the lowest level."""
    from gpuwm.core.microphysics_aerosol import wrf39_repair_negative_aerosol
    rng = np.random.default_rng(7)
    field = rng.uniform(1.e6, 1.e9, (9, 3, 5)).astype(np.float32)
    field[0, 0, 0] = -5.      # bottom: neighbour above
    field[8, 0, 1] = -1.      # top: neighbour below
    field[3:6, 1, 2] = -2.    # a run: repaired upward from the level below
    field[:, 2, 3] = -1.      # a whole negative column floors at zero
    field[4, 2, 4] = -3.e-30  # a tiny negative
    expected = _fortran_repair(field)
    device = cp.asarray(field)
    assert wrf39_repair_negative_aerosol(device) == int((field < 0).sum())
    np.testing.assert_array_equal(cp.asnumpy(device).view(np.uint32),
                                  expected.view(np.uint32))
    assert wrf39_repair_negative_aerosol(device) == 0


def test_fork_start_repairs_before_emission_and_v461_does_not_repair():
    state = _state()
    state.nwfa[0, 0, 0] = -1.e6
    state.nifa[1, 1, 1] = -4.
    fork = RunConfig(nx=6, ny=2, nz=3, dx=3000., dy=3000., dt=20.,
                     ztop=16000., run_seconds=0., mp_physics=28,
                     thompson_version="wrf_39_noaa")
    assert thompson_aerosol_init_fill(state, fork) == {"ccn": False,
                                                       "in": False}
    assert float(state.nwfa[0, 0, 0]) == float(state.nwfa[1, 0, 0])
    assert float(state.nifa[1, 1, 1]) == 5.e5
    assert float(cp.min(state.nwfa2d)) >= 0.0
    v461 = _state()
    v461.nwfa[0, 0, 0] = -1.e6
    before = cp.asnumpy(v461.nwfa).copy()
    generic = RunConfig(nx=6, ny=2, nz=3, dx=3000., dy=3000., mp_physics=28,
                        dt=20., ztop=16000., run_seconds=0.)
    thompson_aerosol_init_fill(v461, generic)
    np.testing.assert_array_equal(cp.asnumpy(v461.nwfa), before)
