"""Radar observation classes (gpuwm.da.radar_classes, design A1).

Breakage pinned, named: on the 2026-10-01 19Z CONUS snapshot the max
superob sat 6.6 dB above the in-cell mean and put 8.1x MRMS's 35 dBZ area
into the observations, weak returns had no class of their own, and an
observation floor of -15 dBZ against an H(x) floor of -35 pumped vapour.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da import radar_classes as rc
from gpuwm.da.letkf import GriddedObs


def _batch(values, sim, err=3.0, name="z"):
    values = np.asarray(values, dtype=np.float64).reshape(1, 1, -1)
    sim = np.asarray(sim, dtype=np.float64).reshape(len(sim), 1, 1, -1)
    return GriddedObs(name=name, values=values,
                      errors=np.full(values.shape, err), simulated=sim,
                      mask=np.ones(values.shape, dtype=bool))


def test_mean_is_the_default_reduction():
    from gpuwm.da.letkf import Localization
    from gpuwm.da.obs_radar import Z_SOURCES
    from gpuwm.da.radar_assimilation import RadarAssimilationConfig

    assert Z_SOURCES[0] == "z_mean"
    cfg = RadarAssimilationConfig(
        localization=Localization(horizontal_m=36e3, vertical_m=6e3),
        rtps_alpha=0.9)
    assert cfg.z_source == "z_mean"
    assert cfg.reflectivity_clear_floor_dbz == 0.0
    assert cfg.reflectivity_dead_band is True


def test_three_classes_share_one_floor_with_hx():
    # precip 30, dead band 8, clear -10; members trace/storm
    # members x obs: obs 0 (30 dBZ), obs 1 (8 dBZ), obs 2 (-10 dBZ)
    sim = [[-35.0, -35.0, -20.0], [-30.0, 12.0, 3.0], [20.0, -35.0, -35.0]]
    b, floor, r = rc.classify_echo_batch(
        _batch([30.0, 8.0, -10.0], sim), clear_floor=0.0, dead_band=True,
        echo_error=None, clear_error=5.0, keep=None)
    assert r["classes"]["precipitation"] == 1
    assert r["classes"]["dead_band"] == 1
    assert r["classes"]["clear"] == 1
    mask = np.asarray(b.mask).ravel()
    assert mask.tolist() == [True, False, True]
    assert b.values.ravel()[2] == 0.0 and b.errors.ravel()[2] == 5.0
    b = rc.apply_floor(b, floor)
    rc.assert_shared_floor(b, floor)
    hx = b.simulated[:, 0, 0, :]
    assert hx[:, 0].min() == 15.0          # precip floor
    assert hx[:, 2].tolist() == [0.0, 3.0, 0.0]  # clear floor at 0


def test_clear_over_clear_gives_zero_innovation():
    b, floor, _ = rc.classify_echo_batch(
        _batch([-12.0], [[-35.0], [-20.0]]), clear_floor=5.0,
        dead_band=True, echo_error=None, clear_error=5.0, keep=None)
    b = rc.apply_floor(b, floor)
    d = b.values.ravel()[0] - b.simulated[:, 0, 0, 0].mean()
    assert d == 0.0


def test_dead_band_off_keeps_band_as_no_significant_echo():
    b, floor, r = rc.classify_echo_batch(
        _batch([8.0], [[-35.0], [20.0]]), clear_floor=0.0, dead_band=False,
        echo_error=None, clear_error=5.0, keep=None)
    assert bool(np.asarray(b.mask).ravel()[0])
    assert b.values.ravel()[0] == 15.0 and floor.ravel()[0] == 15.0


def test_the_shared_hx_is_not_written():
    raw = _batch([-12.0], [[-35.0], [-20.0]])
    before = raw.simulated.copy()
    b, floor, _ = rc.classify_echo_batch(
        raw, clear_floor=0.0, dead_band=True, echo_error=None,
        clear_error=5.0, keep=None)
    rc.apply_floor(b, floor)
    assert np.array_equal(raw.simulated, before)


def test_level_stride_and_top():
    p = np.broadcast_to(np.linspace(100000, 10000, 10)[:, None, None],
                        (10, 2, 2))
    keep = rc.level_keep(p, 2, rc.DEFAULT_TOP_PA)
    kept_levels = np.flatnonzero(keep[:, 0, 0])
    assert kept_levels.tolist() == [0, 2, 4, 6]   # 8 is above 11 km
    assert np.all(p[kept_levels, 0, 0] >= rc.DEFAULT_TOP_PA)


def test_huber_inflates_only_beyond_c():
    sim = [[20.0], [21.0], [22.0], [21.0]]
    b, r = rc.huber_errors(_batch([22.0], sim), 2.0)
    assert r["inflated"] == 0 and b.errors.ravel()[0] == 3.0
    b, r = rc.huber_errors(_batch([60.0], sim), 2.0)
    s = np.sqrt(np.var([20, 21, 22, 21], ddof=1) + 9.0)
    assert r["inflated"] == 1
    assert b.errors.ravel()[0] == pytest.approx(3.0 * np.sqrt(39.0 / (2 * s)))


def test_settings_are_checked():
    with pytest.raises(rc.RadarClassError):
        rc.check_settings(clear_floor=20.0, echo_error=None, clear_error=5.0,
                          echo_stride=2, clear_stride=4, top_pa=None,
                          huber_c=None)
    with pytest.raises(rc.RadarClassError):
        rc.check_settings(clear_floor=0.0, echo_error=None, clear_error=5.0,
                          echo_stride=0, clear_stride=4, top_pa=None,
                          huber_c=None)
