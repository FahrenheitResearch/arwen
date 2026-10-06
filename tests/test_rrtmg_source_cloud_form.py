"""Public operational wrapper rules on resolved and unsized cloudy layers."""
import numpy as np
import pytest

from gpuwm.core import rrtmg_legacy_prep as prep

F = np.float32


def radii(land, fallback, *, batch=False):
    # Below, above and exactly on the source 2.5 um threshold; one clear layer.
    cloud = np.array([1, 1, 1, 0], dtype='f4')
    rc = np.array([2.49, 12, 2.5, 2.49], dtype='f4') * F(1e-6)
    zeros = np.zeros(4, 'f4')
    temperature = np.full(4, 270, 'f4')
    args = (1, 1, 1, 1, temperature, cloud, F(land), rc,
            np.full(4, F(20e-6)), np.full(4, F(50e-6)), zeros, zeros, zeros)
    if not batch:
        return prep._effective_radii(4, *args, sw_cloud_fallback=fallback)
    return prep._effective_radii_b(
        np, *args[:4], temperature[None], cloud[None], np.array([land], 'f4'),
        rc[None], args[8][None], args[9][None], zeros[None], zeros[None],
        zeros[None], sw_cloud_fallback=fallback)


@pytest.mark.parametrize('land,stock,fork', [(1, 7.5, 5.4), (2, 10.5, 9.6)])
def test_public_sw_fallback_preserves_sized_and_clear_layers(land, stock, fork):
    # v4.1.21 SW:10464-10474, LW:12032-12039. No LW radius change.
    for batch in (False, True):
        before = radii(land, False, batch=batch)[2].reshape(-1)
        after = radii(land, True, batch=batch)[2].reshape(-1)
        assert before[0] == F(stock) and after[0] == F(fork)
        np.testing.assert_array_equal(before[[1, 3]].view('u4'),
                                      after[[1, 3]].view('u4'))
        assert after[2] == F(fork)


def test_snow_path_uses_all_snow_mass_below_130_microns():
    # v4.1.21 LW:12345-12360 / SW:10742-10758. Layer pressure thickness
    # 100 hPa, g=10, qs=1e-4, cloud fraction=.5 gives 200 g/m2 in-cloud.
    z = np.zeros(2, 'f4')
    args = (2, 5, 5, z, z, np.full(2, F(1e-4)), np.full(2, F(.5)),
            np.full(2, F(100)), np.full(2, F(260)), F(10), F(1), F(0), F(0),
            np.full(2, F(10)), np.full(2, F(20)), np.array([50, 260], 'f4'))
    before = prep._cloud_properties(*args)
    after = prep._cloud_properties(*args, cloud_form='noaa_wrf39')
    assert abs(float(after[2][0]) - 200) < 2e-5
    assert abs(float(before[2][0]) - 198) < 2e-5
    # The source geometric area factor (130/260)^2=.25 dominates both.
    np.testing.assert_array_equal(before[2][1:].view('u4'), after[2][1:].view('u4'))
    for index in (0, 1, 3, 4):
        np.testing.assert_array_equal(before[index].view('u4'), after[index].view('u4'))


def test_form_typo_cannot_silently_select_stock_optics():
    with pytest.raises(ValueError, match='rrtmg_cloud_optics_form'):
        prep.cloud_form_is_fork('noaa_wrf93')
