"""Reflectivity echo conditioning (gpuwm.da.reflectivity_echo).

The breakage these pin, named: on the CONUS 9 km first-light analysis
(box E, 2026-10-01 19Z, 32 members, mp=28) returns from -15 dBZ were
differenced against the -35 dBZ H(x) floor (mean innovation +36 dBZ against
a 4.2 dBZ spread), the filter extrapolated past the members (16.8 K, 57 m/s,
12 g/kg), and one hour later the members held 2.8 times MRMS's 35 dBZ area.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da.letkf import GriddedObs
from gpuwm.da.reflectivity_echo import (
    DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS, DEFAULT_REFLECTIVITY_FLOOR_DBZ,
    ReflectivityEchoError, check_floor, check_sigmas,
    condition_reflectivity_batch)


def _batch(values, sim, err=3.0):
    values = np.asarray(values, dtype=np.float64).reshape(1, 1, -1)
    sim = np.asarray(sim, dtype=np.float64).reshape(
        len(sim), 1, 1, -1)
    mask = np.ones(values.shape, dtype=bool)
    errors = np.full(values.shape, err)
    return GriddedObs(name="z", values=values, errors=errors,
                      simulated=sim, mask=mask)


def test_defaults_are_on():
    assert DEFAULT_REFLECTIVITY_FLOOR_DBZ == 15.0
    assert DEFAULT_REFLECTIVITY_OUTLIER_SIGMAS == 3.0


def test_a_weak_return_against_a_clear_ensemble_carries_no_innovation():
    # obs 0 dBZ, members at the -35 floor with one trace member at -20:
    # raw innovation ~ +33 dB; floored both sides, zero.
    sim = [[-35.0], [-35.0], [-20.0], [-35.0]]
    batch, receipt = condition_reflectivity_batch(
        _batch([0.0], sim), floor_dbz=15.0, outlier_sigmas=None)
    assert receipt["before"]["innovation_mean"] > 30.0
    d = batch.values[0, 0, 0] - batch.simulated[:, 0, 0, 0].mean()
    assert d == 0.0
    assert batch.simulated[:, 0, 0, 0].std() == 0.0
    assert receipt["observations_raised_to_floor"] == 1


def test_a_weak_return_still_pulls_spurious_echo_down():
    sim = [[40.0], [10.0], [-35.0], [30.0]]
    batch, _ = condition_reflectivity_batch(
        _batch([5.0], sim), floor_dbz=15.0, outlier_sigmas=None)
    assert batch.values[0, 0, 0] == 15.0
    hx = batch.simulated[:, 0, 0, 0]
    assert hx.tolist() == [40.0, 15.0, 15.0, 30.0]
    assert batch.values[0, 0, 0] - hx.mean() < 0.0


def test_the_shared_h_of_x_is_never_written():
    sim = np.array([[-35.0], [-20.0], [20.0]])
    raw = _batch([0.0], sim)
    before = raw.simulated.copy()
    condition_reflectivity_batch(raw, floor_dbz=15.0, outlier_sigmas=3.0)
    assert np.array_equal(raw.simulated, before)


def test_outliers_are_tempered_to_k_sigma_and_consistent_obs_untouched():
    # consistent: d = 2, spread^2 = 6, err 3 -> untouched
    sim_ok = [[38.0], [41.0], [44.0], [41.0]]
    b, r = condition_reflectivity_batch(
        _batch([43.0], sim_ok), floor_dbz=15.0, outlier_sigmas=3.0)
    assert b.errors[0, 0, 0] == pytest.approx(3.0)
    assert r["observations_error_inflated"] == 0
    # 5 sigma but inside 3 sigma? d = 8, sqrt(6 + 9) = 3.87 -> 2.07 sigma
    b, r = condition_reflectivity_batch(
        _batch([49.0], sim_ok), floor_dbz=15.0, outlier_sigmas=3.0)
    assert r["observations_error_inflated"] == 0
    # unreachable: d = 40, spread ~ 0.8 -> sigma^2 = 40^2/9 - s^2
    sim_far = [[15.0], [16.0], [17.0], [16.0]]
    b, r = condition_reflectivity_batch(
        _batch([56.0], sim_far), floor_dbz=15.0, outlier_sigmas=3.0)
    s2 = np.var([15.0, 16.0, 17.0, 16.0], ddof=1)
    eff = b.errors[0, 0, 0]
    assert eff == pytest.approx(np.sqrt(40.0 ** 2 / 9.0 - s2))
    assert 40.0 / np.sqrt(s2 + eff ** 2) == pytest.approx(3.0)
    assert r["observations_error_inflated"] == 1


def test_off_returns_the_same_batch():
    raw = _batch([0.0], [[-35.0], [-20.0]])
    out, receipt = condition_reflectivity_batch(
        raw, floor_dbz=None, outlier_sigmas=None)
    assert out is raw
    assert receipt["applied"] is False


def test_floor_must_be_finite():
    with pytest.raises(ReflectivityEchoError):
        check_floor(float("nan"))
    assert check_floor(None) is None
    with pytest.raises(ReflectivityEchoError):
        check_sigmas(0.0)
    assert check_sigmas(None) is None


def test_the_config_carries_the_defaults_and_refuses_a_bad_floor():
    from gpuwm.da.letkf import Localization
    from gpuwm.da.radar_assimilation import (RadarAssimilationConfig,
                                             RadarAssimilationError)

    loc = Localization(horizontal_m=36000.0, vertical_m=6000.0)
    cfg = RadarAssimilationConfig(localization=loc, rtps_alpha=0.9)
    assert cfg.reflectivity_floor_dbz == 15.0
    assert cfg.reflectivity_outlier_sigmas == 3.0
    with pytest.raises(RadarAssimilationError):
        RadarAssimilationConfig(localization=loc, rtps_alpha=0.9,
                                reflectivity_floor_dbz=float("inf"))


def test_the_cycle_door_parses_the_flags():
    import importlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    door = importlib.import_module("tools.da_cycle_prepared")
    floor = door._reflectivity_floor_argument
    assert floor("none") is None
    assert floor("10") == 10.0
