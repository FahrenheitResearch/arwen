"""The shared microphysics finish clamps at WRF's REAL product mp_tend_lim*dt.

The breakage this guards: WRF's ``moist_physics_finish_em`` bounds the
microphysics theta increment by ``mp_tend_lim*dt``, two REALs multiplied
(dyn_em/module_big_step_utilities_em.F:5706-5707), so the bound is ONE
float32 rounding of two float32 operands.  gpuwm's generic finish
(gpuwm.core.microphysics.moist_physics_finish, which Kessler, WSM6,
Milbrandt-Yau, Morrison, WDM6, NSSL, aerosol-aware Thompson and P3 all
finish through) and its float64 mirror (gpuwm.verify.npref.
np_moist_physics_finish) formed the double product and rounded that
instead.  For the pairs below the two words differ by one float32 unit --
HRRR's 0.07 K/s at dt = 12 s among them -- so every clamped cell's thp and
h_diabatic sat one unit off WRF's.  Classic Thompson's fused finish already
formed the REAL product (gpuwm/core/thompson.py launch_adapter_finish).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core import microphysics
from gpuwm.verify.npref import np_moist_physics_finish

f32 = np.float32

#: (mp_tend_lim K/s, dt s) pairs whose double product rounds to a different
#: float32 word than WRF's REAL product: up for the first two, down for the
#: third.
PAIRS = [(0.07, 12.0), (0.001, 20.0), (0.02, 15.0)]


def _wrf_lim(mp_tend_lim, dt):
    return f32(f32(mp_tend_lim) * f32(dt))


def _bits(value):
    return int(np.asarray(value, dtype=f32).view(np.uint32))


@pytest.mark.parametrize("mp_tend_lim,dt", PAIRS)
def test_the_pair_separates_the_two_roundings(mp_tend_lim, dt):
    assert _bits(f32(mp_tend_lim * dt)) != _bits(_wrf_lim(mp_tend_lim, dt))


@pytest.mark.parametrize("mp_tend_lim,dt", PAIRS)
def test_production_finish_clamps_at_wrf_real_product(mp_tend_lim, dt,
                                                      monkeypatch):
    monkeypatch.setattr(microphysics, "cp", np)
    wrf = _wrf_lim(mp_tend_lim, dt)
    saved = np.full((1, 1, 3), f32(300.0), dtype=f32)
    # increments far beyond the clamp, and one well inside it
    th_phy = saved + np.array([5.0, -5.0, 0.0], dtype=f32).reshape(1, 1, 3)
    state = SimpleNamespace(h_diabatic=saved.copy(),
                            thp=np.zeros((1, 1, 3), dtype=f32))
    cfg = SimpleNamespace(no_mp_heating=0, mp_tend_lim=mp_tend_lim)
    microphysics.moist_physics_finish(state, cfg, th_phy, dt)
    # thp started at 0, so it now holds the clamped increment word itself
    assert _bits(state.thp[0, 0, 0]) == _bits(wrf)
    assert _bits(state.thp[0, 0, 1]) == _bits(-wrf)
    assert state.thp[0, 0, 2] == 0.0
    assert _bits(state.thp[0, 0, 0]) != _bits(f32(mp_tend_lim * dt))
    np.testing.assert_array_equal(
        state.h_diabatic.ravel(),
        np.array([wrf / f32(dt), -wrf / f32(dt), 0.0], dtype=f32))


@pytest.mark.parametrize("mp_tend_lim,dt", PAIRS)
def test_npref_mirror_clamps_at_wrf_real_product(mp_tend_lim, dt):
    wrf = float(_wrf_lim(mp_tend_lim, dt))
    th_saved = np.zeros(3)
    th_after = np.array([5.0, -5.0, 0.0])
    th_new, hd = np_moist_physics_finish(th_after, th_saved, dt,
                                         mp_tend_lim=mp_tend_lim)
    np.testing.assert_array_equal(th_new, np.array([wrf, -wrf, 0.0]))
    np.testing.assert_array_equal(hd, np.array([wrf / dt, -wrf / dt, 0.0]))
    assert th_new[0] != float(f32(mp_tend_lim * dt))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("mp_tend_lim,dt", PAIRS)
def test_production_finish_on_the_card_matches_wrf_real_product(mp_tend_lim,
                                                                 dt):
    """The same finish on device arrays, through the real CuPy module."""
    import cupy as cp
    wrf = _wrf_lim(mp_tend_lim, dt)
    saved = cp.full((1, 1, 3), f32(300.0), dtype=cp.float32)
    th_phy = saved + cp.asarray([5.0, -5.0, 0.0],
                                dtype=cp.float32).reshape(1, 1, 3)
    state = SimpleNamespace(h_diabatic=saved.copy(),
                            thp=cp.zeros((1, 1, 3), dtype=cp.float32))
    cfg = SimpleNamespace(no_mp_heating=0, mp_tend_lim=mp_tend_lim)
    microphysics.moist_physics_finish(state, cfg, th_phy, dt)
    thp = state.thp.get().ravel()
    assert [_bits(v) for v in thp[:2]] == [_bits(wrf), _bits(-wrf)]
    assert thp[2] == 0.0
    np.testing.assert_array_equal(
        state.h_diabatic.get().ravel(),
        np.array([wrf / f32(dt), -wrf / f32(dt), 0.0], dtype=f32))
