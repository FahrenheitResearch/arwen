"""Calibration windows: a calibration period may be several windows around the validation month
(calibrate December + February, validate January). Checks the window helpers of the gauge-calibration
door on the CPU with numpy standing in for cupy (the helpers take the array module as an argument)."""
import numpy as np
import pytest

from gpuwm.hydro import calibrate as c


def test_single_window_spec_is_unchanged():
    assert c._windows([72, 744]) == [(72, 744)]
    assert c._windows([[72, 744], [1488, 2160]]) == [(72, 744), (1488, 2160)]
    with pytest.raises(ValueError):
        c._windows([744, 72])


def test_blocks_concatenate_each_window():
    rng = np.random.default_rng(1)
    x = rng.uniform(1.0, 2.0, size=(40, 3, 2))
    one = c._blocks(np, x, [(4, 16)], 6)
    assert np.array_equal(one, np.log(x[4:16].reshape(2, 6, 3, 2).mean(1)))
    two = c._blocks(np, x, [(4, 16), (28, 40)], 6)
    assert np.array_equal(two, np.concatenate([one, np.log(x[28:40].reshape(2, 6, 3, 2).mean(1))]))


def test_scores_over_two_windows_ignore_the_gap_and_report_real_hours():
    T = 60
    obs = np.ones((1, T))
    sim = np.ones((T, 1, 1))
    obs[0, 50] = 5.0        # observed peak in the second window
    sim[47, 0, 0] = 5.0     # model peak 3 h earlier
    sim[25, 0, 0] = 99.0    # a huge value in the gap (validation hours) must not count
    s = c._scores(np, sim, obs, [(0, 20), (40, 60)])
    assert s["peak_timing_h"][0, 0] == -3.0
    single = c._scores(np, sim, obs, [(40, 60)])
    assert single["peak_timing_h"][0, 0] == -3.0
    assert np.isfinite(s["nse"]).all() and s["nse"][0, 0] < 1.0
