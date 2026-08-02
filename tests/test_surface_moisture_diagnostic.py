"""The 2 m mixing ratio must stay a mixing ratio, even over Antarctic snow.

WRF's flux inversion ``Q2 = QSFC - QFX/(RHO*CQS2)``
(phys/module_sf_sfcdiags.F:56) has no lower bound, and neither does Noah's
own ``QSFC = Q1/(1-Q1)`` with ``Q1 = qv(k=1) + QFX/(RHO*CHS)``
(phys/module_sf_noahlsm.F:795,801).  Under downward moisture flux over very
cold snow the correction exceeds the whole vapour content of the lowest
model level and both go negative.

WRF authored the remedy and commented it out at
phys/module_sf_noahdrv.F:1276-1282 -- "over snow cover ... the cqs2 value
also becomes irrelevant / by setting cqs2=chs in this situation the 2m q
should become just qv(k=1)".  With ``CQS2 = CHS`` the two corrections cancel
exactly and ``Q2 = qv(k=1)``; that identity is what the guard publishes, and
only where the flux inversion has left the physical range.
"""
import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.core import physics as physics_module
from gpuwm.core import constants as c


# Column state read out of a real McMurdo run at its worst point
# (wrfout f001, `--point -77.85,166.7`, 12 km, GFS 2026-07-28T00).
MCMURDO = {
    "qsfc": -2.400e-4,
    "qfx": -5.548e-7,
    "cqs2": 3.600e-3,
    "chs2": 3.600e-3,
    "psfc": 80000.0,
    "tsk": 231.6,
    "hfx": -3.0,
    "qv1": 7.671e-5,
}
TEMPERATE = {
    "qsfc": 1.480e-2,
    "qfx": 8.400e-5,
    "cqs2": 1.100e-2,
    "chs2": 1.100e-2,
    "psfc": 97000.0,
    "tsk": 298.0,
    "hfx": 120.0,
    "qv1": 1.290e-2,
}


def _wrf_sfcdiags_q2(column):
    """phys/module_sf_sfcdiags.F:56, verbatim, in float64."""
    rho = column["psfc"] / (c.RD * column["tsk"])
    return column["qsfc"] - column["qfx"] / (rho * column["cqs2"])


def test_negative_control_wrfs_own_formula_goes_negative_here():
    """Without a guard the published mixing ratio is negative."""
    assert _wrf_sfcdiags_q2(MCMURDO) < 0.0
    assert MCMURDO["qv1"] > 0.0
    assert _wrf_sfcdiags_q2(TEMPERATE) > 0.0


def test_the_cancellation_identity_wrf_names_is_exact():
    """CQS2 = CHS makes the two corrections cancel, leaving qv(k=1).

    Noah sets QSFC from qv(k=1) + QFX/(RHO*CHS); SFCDIAGS subtracts
    QFX/(RHO*CQS2).  Equal coefficients therefore return qv(k=1) exactly,
    which is what module_sf_noahdrv.F:1278 says the remedy produces.
    """
    column = dict(MCMURDO)
    rho = column["psfc"] / (c.RD * column["tsk"])
    chs = 2.063e-3
    q1 = column["qv1"] + column["qfx"] / (rho * chs)
    qsfc_from_noah = q1 / (1.0 - q1)
    # The real path: CQS2 != CHS, and the surface value is already negative.
    assert qsfc_from_noah < 0.0
    # WRF's remedy: CQS2 = CHS collapses the diagnostic onto level 1.
    collapsed = q1 - column["qfx"] / (rho * chs)
    assert collapsed == pytest.approx(column["qv1"], rel=1e-12)


def _driver_with(column):
    cp = physics_module.cp
    dtype = physics_module.DTYPE
    driver = physics_module.PhysicsDriver.__new__(physics_module.PhysicsDriver)
    shape = (2, 3)
    def full(value):
        return cp.full(shape, dtype(value), dtype=dtype)
    driver.fields = {
        "qsfc": full(column["qsfc"]), "qfx": full(column["qfx"]),
        "cqs2": full(column["cqs2"]), "chs2": full(column["chs2"]),
        "psfc": full(column["psfc"]), "tsk": full(column["tsk"]),
        "hfx": full(column["hfx"]),
        "q2": full(0.0), "t2": full(0.0), "th2": full(0.0),
    }
    atmosphere = {"qv": cp.full((1,) + shape, dtype(column["qv1"]),
                                dtype=dtype)}
    return driver, atmosphere


@requires_gpu
def test_antarctic_column_publishes_the_lowest_level_value():
    driver, atmosphere = _driver_with(MCMURDO)
    driver._refresh_surface_diagnostics(atmosphere)
    q2 = np.asarray(driver.fields["q2"].get())
    assert (q2 > 0.0).all()
    np.testing.assert_allclose(q2, MCMURDO["qv1"], rtol=1e-6)


@requires_gpu
def test_a_representable_column_is_bit_for_bit_wrfs_arithmetic():
    """No drift: where the inversion is meaningful nothing is touched."""
    driver, atmosphere = _driver_with(TEMPERATE)
    driver._refresh_surface_diagnostics(atmosphere)
    q2 = np.asarray(driver.fields["q2"].get())
    cp = physics_module.cp
    dtype = physics_module.DTYPE
    f = driver.fields
    rho = f["psfc"] / (dtype(c.RD) * f["tsk"])
    expected = np.asarray(
        (f["qsfc"] - f["qfx"] / (rho * f["cqs2"])).get())
    np.testing.assert_array_equal(q2, expected)


@requires_gpu
def test_a_negative_surface_endpoint_with_a_representable_result_is_kept():
    """The divergence is scoped to the declared-range violation only.

    A column whose QSFC is negative but whose 2 m value lands inside the
    range a mixing ratio can occupy keeps WRF's arithmetic bit for bit.
    """
    column = dict(MCMURDO)
    column["qsfc"] = -1.0e-6
    column["qfx"] = -5.548e-7
    assert _wrf_sfcdiags_q2(column) > 0.0
    driver, atmosphere = _driver_with(column)
    driver._refresh_surface_diagnostics(atmosphere)
    q2 = np.asarray(driver.fields["q2"].get())
    f = driver.fields
    dtype = physics_module.DTYPE
    rho = f["psfc"] / (dtype(c.RD) * f["tsk"])
    expected = np.asarray((f["qsfc"] - f["qfx"] / (rho * f["cqs2"])).get())
    np.testing.assert_array_equal(q2, expected)


@requires_gpu
def test_the_guard_fires_only_on_the_unrepresentable_points():
    cp = physics_module.cp
    dtype = physics_module.DTYPE
    driver, atmosphere = _driver_with(TEMPERATE)
    # One point of the tile carries the Antarctic column.
    for name, value in MCMURDO.items():
        if name == "qv1":
            continue
        driver.fields[name][0, 1] = dtype(value)
    atmosphere["qv"][0, 0, 1] = dtype(MCMURDO["qv1"])
    driver._refresh_surface_diagnostics(atmosphere)
    q2 = np.asarray(driver.fields["q2"].get())
    assert (q2 > 0.0).all()
    assert q2[0, 1] == pytest.approx(MCMURDO["qv1"], rel=1e-6)
    others = np.delete(q2.ravel(), 1)
    assert np.allclose(others, _wrf_sfcdiags_q2(TEMPERATE), rtol=1e-5)


@requires_gpu
def test_temperature_diagnostic_is_untouched_by_the_moisture_guard():
    driver, atmosphere = _driver_with(MCMURDO)
    driver._refresh_surface_diagnostics(atmosphere)
    t2 = np.asarray(driver.fields["t2"].get())
    rho = MCMURDO["psfc"] / (c.RD * MCMURDO["tsk"])
    expected = (MCMURDO["tsk"]
                - MCMURDO["hfx"] / (rho * c.CP * MCMURDO["chs2"]))
    np.testing.assert_allclose(t2, expected, rtol=1e-5)
