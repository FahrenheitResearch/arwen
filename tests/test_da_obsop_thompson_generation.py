"""H_Z(x) reads the Thompson GENERATION the run integrates (CPU, no device).

``RunConfig.thompson_version`` selects which aerosol-aware Thompson runs:
WRF v4.6.1 (``wrf_461``), whose calc_refl10cm reads the graupel number the
scheme evolved from its entry diagnosis (graupel content alone, N0 between
1e2 and 1e6), or the operational WRF 3.9 fork (``wrf_39_noaa``), whose
calc_refl10cm diagnoses its own intercept from graupel content and
supercooled rain, non-increasing downward, between 1e4 and 3e6.  The
mp=28 adapter writes REFL_10CM through whichever the run selected.  Until
this fix the observation operator always derived the v4.6.1 moment, so a
fork run's DA inverted dBZ through a graupel PSD the model never wrote:
on the Iowa 2024-05-21 18Z f01 crop the two intercepts differ by 1..5 dBZ
over a quarter to a half of the 30..50 dBZ cells (box S measurement,
WOOF-FIX-PROGRAM-2026-10-06/dbz/REPORT.md).

These tests pin the routing with recorded launchers; the device-side
bit-for-bit pin against the product kernels is
``test_da_obsop_thompson_gpu.py::
test_mp28_under_the_fork_derives_the_forks_reflectivity_intercept``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import obsop


def test_the_default_generation_is_v461_and_classic_mp8_has_only_one():
    assert obsop.thompson_generation(SimpleNamespace(mp_physics=28)) == "wrf_461"
    assert obsop.thompson_generation(
        SimpleNamespace(mp_physics=28, thompson_version="wrf_461")) == "wrf_461"
    assert obsop.thompson_generation(
        SimpleNamespace(mp_physics=28, thompson_version="wrf_39_noaa")) == "wrf_39_noaa"
    # mp=8 carries one generation whatever a config object says: RunConfig
    # already refuses the fork's name on mp=8 (gpuwm/config.py).
    assert obsop.thompson_generation(
        SimpleNamespace(mp_physics=8, thompson_version="wrf_39_noaa")) == "wrf_461"


def test_an_unknown_generation_is_refused_by_name():
    with pytest.raises(ValueError, match="thompson_version='wrf_50'"):
        obsop.thompson_generation(
            SimpleNamespace(mp_physics=28, thompson_version="wrf_50"))


class _DuckState:
    def __init__(self, shape=(3, 2, 2)):
        self._slots = {}
        self.p = np.full(shape, 8.0e4, dtype=np.float32)
        self.qv = np.full(shape, 1.0e-2, dtype=np.float32)
        self.qr = np.full(shape, 1.0e-3, dtype=np.float32)
        self.nr = np.full(shape, 1.0e4, dtype=np.float32)
        self.qg = np.full(shape, 1.0e-3, dtype=np.float32)

    def scratch(self, shape, slot, dtype=None):
        buf = self._slots.get(slot)
        if buf is None:
            buf = np.zeros(shape, dtype=np.float32)
            self._slots[slot] = buf
        return buf


def _recorders(monkeypatch):
    """Replace the three device launchers with recorders; the function under
    test then touches no device (temperature and pressure are supplied, so
    it never forms them with CuPy either)."""
    calls = []

    def classic_init(qg, t, p, qv, shadow):
        calls.append(("v461_init", shadow))

    def classic_finalize(qg, t, p, qv, shadow):
        calls.append(("v461_finalize", shadow))

    def fork_intercept(qg, qr, nr, t, p, qv, out, *, mode):
        from gpuwm.core.thompson_aerosol_launch import active_thompson_version
        from gpuwm.core.thompson_aerosol_state import (
            WRF39_INTERCEPT_REFLECTIVITY)

        assert mode == WRF39_INTERCEPT_REFLECTIVITY
        # The fork's kernels compile under the fork define only inside the
        # version scope; the operator must open it as the adapter does.
        assert active_thompson_version() == "wrf_39_noaa"
        calls.append(("fork_reflectivity_intercept", out))

    import gpuwm.core.thompson as thompson
    import gpuwm.core.thompson_aerosol_state as fork

    monkeypatch.setattr(thompson, "launch_classic_graupel_number_init",
                        classic_init)
    monkeypatch.setattr(thompson, "launch_classic_graupel_number_finalize",
                        classic_finalize)
    monkeypatch.setattr(fork, "launch_wrf39_graupel_intercept",
                        fork_intercept)
    return calls


def test_the_v461_generation_derives_the_wrappers_diagnosis(monkeypatch):
    calls = _recorders(monkeypatch)
    state = _DuckState()
    t = np.full(state.p.shape, 280.0, dtype=np.float32)
    _, _, shadow = obsop._thompson_graupel_number(
        state, t, state.p, version="wrf_461")
    assert [name for name, _ in calls] == ["v461_init", "v461_finalize"]
    assert all(buf is shadow for _, buf in calls)
    assert shadow is state.scratch(
        state.p.shape, "mp_thompson_graupel_number_shadow")


def test_the_fork_generation_derives_the_forks_reflectivity_intercept(
        monkeypatch):
    calls = _recorders(monkeypatch)
    state = _DuckState()
    t = np.full(state.p.shape, 280.0, dtype=np.float32)
    returned_t, returned_p, shadow = obsop._thompson_graupel_number(
        state, t, state.p, version="wrf_39_noaa")
    assert [name for name, _ in calls] == ["fork_reflectivity_intercept"]
    assert calls[0][1] is shadow
    # The pair the moment was formed at is the pair handed on.
    assert returned_t is t and returned_p is state.p
    assert shadow is state.scratch(
        state.p.shape, "mp_thompson_graupel_number_shadow")


def test_the_generation_scope_is_closed_again_after_the_fork_derivation(
        monkeypatch):
    from gpuwm.core.thompson_aerosol_launch import active_thompson_version

    _recorders(monkeypatch)
    state = _DuckState()
    t = np.full(state.p.shape, 280.0, dtype=np.float32)
    obsop._thompson_graupel_number(state, t, state.p, version="wrf_39_noaa")
    assert active_thompson_version() == "wrf_461"
