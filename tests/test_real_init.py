"""Phase 3 Task 8: WRF real-data vertical interpolation and initialization."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.grid import make_vertical_coord
from gpuwm.ingest.real import (
    _WRF_QV_MIN_VALUE,
    _cap_stratospheric_qv,
    _floor_flag_sh_surface_mixing_ratio,
    _mixing_ratio_to_relative_humidity,
    _saturation_mixing_ratio,
    _specific_humidity_to_mixing_ratio,
    _wrf_flag_sh_surface_specific_humidity,
    hydrostatic_residual,
    initialize_real,
    surface_pressure_from_surface,
)
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.soil import (
    preprocess_noah_soil as _preprocess_noah_soil)


# These fixtures hand the soil router the raw SST/SKINTEMP pair on purpose:
# what they pin is WRF's OWN per-cell water-skin fallback
# (module_initialize_real.F:2844-2866), which is exactly what
# `water_temperature_policy = "wrf_compat"` names.  The router refuses that
# pair when nobody declares a decision, so the declaration is made once here
# instead of at every call site below, and the tests keep asserting the
# historical numbers they were written for.
def preprocess_noah_soil(fields, **kwargs):
    kwargs.setdefault("water_temperature_policy", "wrf_compat")
    return _preprocess_noah_soil(fields, **kwargs)

from gpuwm.ingest.vert import interpolate_logp_gpu
from gpuwm.verify.npref import np_vertical_interpolate_logp


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
MET_EM = BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc"
requires_bundle = pytest.mark.skipif(
    not MET_EM.is_file(), reason="WRF_1974_MP55 reference bundle not present"
)


def _soil_type(fields):
    return np.where(np.asarray(fields["LANDSEA"]) >= 0.5, 6, 14)


def test_explicit_wrf_eta_levels_are_preserved():
    eta = np.array([1.0, 0.92, 0.73, 0.48, 0.21, 0.0], dtype=np.float64)
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    np.testing.assert_array_equal(coord.znw, eta)
    assert np.all(coord.dnw < 0.0)


def test_initialize_real_rejects_declared_and_catalog_orography_conflict():
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    fields = {
        "TT": np.full((2, 1, 1), 270.0),
        "RH": np.full((2, 1, 1), 50.0),
        "GHT": np.array([[[5500.0]], [[100.0]]]),
        "UU": np.zeros((2, 1, 2)),
        "VV": np.zeros((2, 2, 1)),
        "PSFC": np.full((1, 1), 100000.0),
        "T2": np.full((1, 1), 280.0),
        "D2": np.full((1, 1), 275.0),
        "U10": np.zeros((1, 2)),
        "V10": np.zeros((2, 1)),
        "SOURCE_OROGRAPHY": np.full((1, 1), 123.0, dtype=np.float64),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(1999, 5, 3, 12), levels_hpa=levels,
        fields=fields)
    eta = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    coord = make_vertical_coord(2, eta_levels=eta)
    cfg = RunConfig(
        nx=1, ny=1, nz=2, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=30.0, run_seconds=30.0, moist=True)
    with pytest.raises(ValueError) as caught:
        initialize_real(
            snapshot, cfg, coord, np.zeros((1, 1)),
            source_orography=np.full((1, 1), 456.0))
    message = str(caught.value)
    assert "declared source_orography" in message
    assert "forcing catalog SOURCE_OROGRAPHY" in message


def test_pressure_level_surface_humidity_requires_exactly_d2_or_rh2():
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    fields = {
        "TT": np.full((2, 1, 1), 270.0),
        "RH": np.full((2, 1, 1), 50.0),
        "GHT": np.array([[[5500.0]], [[100.0]]]),
        "UU": np.zeros((2, 1, 2)),
        "VV": np.zeros((2, 2, 1)),
        "PSFC": np.full((1, 1), 100000.0),
        "T2": np.full((1, 1), 280.0),
        "U10": np.zeros((1, 2)),
        "V10": np.zeros((2, 1)),
    }
    cfg = RunConfig(
        nx=1, ny=1, nz=2, dx=12000.0, dy=12000.0, ztop=16000.0,
        dt=30.0, run_seconds=30.0, moist=True)
    terrain = np.zeros((1, 1))

    for extras in ({}, {"D2": np.full((1, 1), 275.0),
                          "RH2": np.full((1, 1), 60.0)}):
        snapshot = HorizontalSnapshot(
            valid_time=datetime(1999, 5, 3, 12), levels_hpa=levels,
            fields={**fields, **extras})
        with pytest.raises(KeyError, match="exactly one of D2 or RH2"):
            initialize_real(
                snapshot, cfg, make_vertical_coord(2, eta_levels=[1.0, 0.5, 0.0]),
                terrain, source_orography=terrain)


def test_logp_mirror_interpolation_and_wrf_extrapolation():
    source_p = np.array([100000.0, 80000.0, 50000.0, 20000.0])[:, None, None]
    scalar = 4.0 + 2.5 * np.log(source_p)
    target_p = np.array([110000.0, 90000.0, 65000.0, 20000.0])[:, None, None]

    got = np_vertical_interpolate_logp(
        scalar, source_p, target_p, below="constant", above="error"
    )
    assert got[0, 0, 0] == scalar[0, 0, 0]
    np.testing.assert_allclose(got[1:3, 0, 0],
                               4.0 + 2.5 * np.log(target_p[1:3, 0, 0]),
                               rtol=0.0, atol=2.0e-14)
    assert got[-1, 0, 0] == scalar[-1, 0, 0]

    theta = np.full_like(source_p, 300.0)
    standard = np_vertical_interpolate_logp(
        theta, source_p, target_p[:1], below="temperature", above="error"
    )[0, 0, 0]
    p1, pt = source_p[0, 0, 0], target_p[0, 0, 0]
    t1 = 300.0 * (p1 / c.P0) ** c.RCP
    dp = pt - p1
    pavg = 0.5 * (pt + p1)
    dhdp = 11880.516 * 0.1902632 * (pavg / 100.0) ** (0.1902632 - 1.0)
    expected = (t1 + dhdp * (dp / 100.0) * 0.0065) * (c.P0 / pt) ** c.RCP
    assert standard == pytest.approx(expected, rel=0.0, abs=2.0e-12)


def test_logp_mirror_above_source_top_is_fatal():
    source_p = np.array([100000.0, 70000.0, 20000.0])[:, None, None]
    field = np.array([290.0, 270.0, 220.0])[:, None, None]
    target_p = np.array([10000.0])[:, None, None]

    with pytest.raises(ValueError, match="above source top"):
        np_vertical_interpolate_logp(
            field, source_p, target_p, below="constant", above="error"
        )


@requires_gpu
@pytest.mark.gpu
def test_logp_gpu_kernel_matches_float64_mirror():
    import cupy as cp

    rng = np.random.default_rng(23)
    nsrc, nz, ny, nx = 9, 7, 4, 6
    source_1d = np.geomspace(102000.0, 7000.0, nsrc)
    source_p = source_1d[:, None, None] * (
        1.0 + rng.normal(0.0, 0.002, (nsrc, ny, nx)))
    field = rng.normal(280.0, 7.0, (nsrc, ny, nx))
    target_p = np.geomspace(108000.0, 7500.0, nz)[:, None, None] * np.ones(
        (1, ny, nx))
    ref = np_vertical_interpolate_logp(
        field, source_p, target_p, below="temperature", above="error"
    )
    got = interpolate_logp_gpu(
        cp.asarray(field, cp.float32), cp.asarray(source_p, cp.float32),
        cp.asarray(target_p, cp.float32), below="temperature", above="error"
    )
    np.testing.assert_allclose(cp.asnumpy(got), ref, rtol=3.0e-5, atol=2.0e-3)

    with pytest.raises(ValueError, match="above source top"):
        interpolate_logp_gpu(
            cp.asarray(field, cp.float32), cp.asarray(source_p, cp.float32),
            cp.full((1, ny, nx), 5000.0, cp.float32),
            below="temperature", above="error",
        )


def test_qv_construction_uses_wrf_floor_and_invalid_guard():
    temperature = np.array([250.0, 300.0, 300.0, 0.0])
    pressure = np.array([100000.0, 100000.0, 1000.0, 100000.0])
    rh = np.array([0.0, 50.0, 100.0, 100.0])

    got = _saturation_mixing_ratio(temperature, pressure, rh)
    np.testing.assert_array_equal(got[[0, 2, 3]], 1.0e-6)

    es_hpa = 0.5 * (10.0 * c.SVP1) * np.exp(
        c.SVP2 * (temperature[1] - c.SVPT0)
        / (temperature[1] - c.SVP3)
    )
    # rh_to_mxrat1's own EPS = 0.622 (module_initialize_real.F:7379),
    # not module ep_2 -- the ingest lane review's parity residual.
    expected = 0.622 * es_hpa / (pressure[1] / 100.0 - es_hpa)
    assert got[1] == pytest.approx(expected, rel=0.0, abs=1.0e-15)
    assert got[1] > 1.0e-6


def test_hrrr_specific_humidity_uses_wrf_dry_air_conversion_without_floor():
    specific = np.array([0.0, 1.0e-7, 0.001, 0.02])
    got = _specific_humidity_to_mixing_ratio(specific)
    np.testing.assert_array_equal(got, specific / (1.0 - specific))
    assert got[0] == 0.0
    with pytest.raises(ValueError, match=r"finite in \[0, 1\)"):
        _specific_humidity_to_mixing_ratio([0.0, 1.0])
    with pytest.raises(ValueError, match=r"finite in \[0, 1\)"):
        _specific_humidity_to_mixing_ratio([np.nan])

    undershoot = np.array([-3.6e-5, 0.0, 0.01])
    np.testing.assert_allclose(
        _specific_humidity_to_mixing_ratio(
            undershoot, allow_wps_undershoot=True),
        undershoot / (1.0 - undershoot), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match=r"finite in \[-0.028126, 1\)"):
        _specific_humidity_to_mixing_ratio(
            [-0.028127], allow_wps_undershoot=True)


def test_hrrr_relative_humidity_is_inverse_of_wrf_mixing_ratio_relation():
    temperature = np.array([220.0, 260.0, 290.0, 305.0])
    pressure = np.array([15000.0, 50000.0, 85000.0, 100000.0])
    expected_rh = np.array([3.0, 35.0, 78.0, 102.0])
    qv = _saturation_mixing_ratio(temperature, pressure, expected_rh)
    # Avoid the 1e-6 floor in the inverse identity's deliberately dry point.
    active = qv > 1.0e-6
    got = _mixing_ratio_to_relative_humidity(
        temperature[active], pressure[active], qv[active])
    np.testing.assert_allclose(
        got, np.clip(expected_rh[active], 0.0, 100.0),
        rtol=0.0, atol=2.0e-13)
    with pytest.raises(ValueError, match="mixing ratio non-negative"):
        _mixing_ratio_to_relative_humidity(280.0, 90000.0, -1.0e-6)
    mapped_specific_humidity = np.array([-3.6e-5, 0.0, 0.01])
    mapped_qv = _specific_humidity_to_mixing_ratio(
        mapped_specific_humidity, allow_wps_undershoot=True)
    mapped_rh = _mixing_ratio_to_relative_humidity(
        np.full(3, 280.0), np.full(3, 90000.0), mapped_qv,
        allow_wps_undershoot=True)
    assert mapped_rh[0] < 0.0
    assert mapped_rh[1] == 0.0
    assert mapped_rh[2] > 0.0


def test_humidity_elementwise_workers_are_byte_identical():
    """Setup-only row/level workers cannot alter per-element arithmetic."""
    rng = np.random.default_rng(7416)
    shape = (11, 7, 9)
    temperature = rng.uniform(215.0, 310.0, shape)
    pressure = rng.uniform(5000.0, 102000.0, shape)
    specific = rng.uniform(0.0, 0.025, shape)
    specific[0, 0, 0] = -3.6e-5

    expected_qv = _specific_humidity_to_mixing_ratio(
        specific, allow_wps_undershoot=True, column_workers=1)
    expected_rh = _mixing_ratio_to_relative_humidity(
        temperature, pressure, expected_qv,
        allow_wps_undershoot=True, column_workers=1)
    expected_cap = _cap_stratospheric_qv(
        expected_qv, pressure, column_workers=1)
    for workers in (2, 3, 8, 32):
        qv = _specific_humidity_to_mixing_ratio(
            specific, allow_wps_undershoot=True,
            column_workers=workers)
        rh = _mixing_ratio_to_relative_humidity(
            temperature, pressure, qv, allow_wps_undershoot=True,
            column_workers=workers)
        cap = _cap_stratospheric_qv(
            qv, pressure, column_workers=workers)
        np.testing.assert_array_equal(qv, expected_qv)
        np.testing.assert_array_equal(rh, expected_rh)
        np.testing.assert_array_equal(cap, expected_cap)


def test_thermodynamic_elementwise_workers_are_byte_identical():
    from gpuwm.ingest.real import (
        _moist_specific_volume,
        _potential_temperature_from_temperature,
        _temperature_from_potential_temperature,
    )

    rng = np.random.default_rng(7434)
    shape = (13, 7, 11)
    temperature = rng.uniform(205.0, 315.0, shape)
    pressure = rng.uniform(9000.0, 103000.0, shape)
    qv = rng.uniform(1.0e-6, 0.025, shape)
    expected_theta = _potential_temperature_from_temperature(
        temperature, pressure, column_workers=1)
    expected_temperature = _temperature_from_potential_temperature(
        expected_theta, pressure, column_workers=1)
    expected_alpha = _moist_specific_volume(
        expected_theta, qv, pressure, column_workers=1)
    for workers in (2, 3, 8, 32):
        theta = _potential_temperature_from_temperature(
            temperature, pressure, column_workers=workers)
        diagnosed_temperature = _temperature_from_potential_temperature(
            theta, pressure, column_workers=workers)
        alpha = _moist_specific_volume(
            theta, qv, pressure, column_workers=workers)
        np.testing.assert_array_equal(theta, expected_theta)
        np.testing.assert_array_equal(
            diagnosed_temperature, expected_temperature)
        np.testing.assert_array_equal(alpha, expected_alpha)


def test_flag_sh_surface_fallback_matches_wrf_whole_domain_decision():
    pressure = np.broadcast_to(
        np.array([95000.0, 70000.0, 30000.0])[:, None, None],
        (3, 2, 3)).copy()
    spfh = np.arange(18, dtype=np.float64).reshape(3, 2, 3) * 1.0e-4
    q2 = np.full((2, 3), 0.006)
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(q2, spfh, pressure), q2)

    q2[0, 0] = -1.0e-5
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(q2, spfh, pressure), spfh[0])
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            q2, spfh, pressure, force_fallback=False), q2)
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            np.full_like(q2, 0.006), spfh, pressure, force_fallback=True),
        spfh[0])
    # Reversed source-level order selects the opposite endpoint but preserves
    # the same nearest-to-surface physical level.
    np.testing.assert_array_equal(
        _wrf_flag_sh_surface_specific_humidity(
            q2, spfh[::-1], pressure[::-1]), spfh[0])


def test_noah_soil_layer_interpolation_and_surface_consistency():
    land = np.array([[1.0, 0.0], [1.0, 0.0]])
    fields = {
        "LANDSEA": land,
        "SKINTEMP": np.array([[290.0, 280.0], [292.0, 281.0]]),
        "SST": np.array([[0.0, 285.0], [0.0, 286.0]]),
        "ST000007": np.full((2, 2), 289.0),
        "ST007028": np.full((2, 2), 287.0),
        "ST028100": np.full((2, 2), 284.0),
        "ST100289": np.full((2, 2), 281.0),
        "TMN": np.full((2, 2), 279.0),
        "SM000007": np.full((2, 2), 0.12),
        "SM007028": np.full((2, 2), 0.18),
        "SM028100": np.full((2, 2), 0.24),
        "SM100289": np.full((2, 2), 0.30),
        "SNOW_EC": np.array([[0.012, 0.0], [0.001, 0.0]]),
    }
    soil = preprocess_noah_soil(fields, soil_type=_soil_type(fields))
    np.testing.assert_array_equal(soil.landmask, land)
    np.testing.assert_array_equal(soil.xice, 0.0)
    np.testing.assert_allclose(soil.tsk[:, 1], [285.0, 286.0])
    np.testing.assert_array_equal(soil.soil_moisture[:, :, 1], 1.0)
    np.testing.assert_allclose(soil.snow_water[:, 0], [12.0, 1.0])
    np.testing.assert_allclose(soil.snow_depth[:, 0], [0.06, 0.005])
    assert soil.soil_temperature.shape == (4, 2, 2)
    assert np.all((soil.soil_temperature >= 170.0)
                  & (soil.soil_temperature <= 400.0))
    assert np.all((soil.soil_moisture >= 0.0) & (soil.soil_moisture <= 1.0))

    fields_without_tmn = dict(fields)
    fields_without_tmn.pop("TMN")
    with pytest.raises(KeyError, match="TMN"):
        preprocess_noah_soil(
            fields_without_tmn, soil_type=_soil_type(fields_without_tmn))


def test_hrrr_soil_depth_nodes_are_interpolated_to_noah_midpoints():
    depths = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60,
                       1.0, 1.6, 3.0])
    fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), 289.0),
        "TMN": np.full((1, 1), 281.0),
        "SOILT": (280.0 + 2.0 * depths)[:, None, None],
        "SOILW": (0.10 + 0.05 * depths)[:, None, None],
    }
    soil = preprocess_noah_soil(fields, soil_type=np.full((1, 1), 6))
    target = np.array([0.05, 0.25, 0.70, 1.50])
    np.testing.assert_allclose(
        soil.soil_temperature[:, 0, 0], 280.0 + 2.0 * target,
        rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        soil.soil_moisture[:, 0, 0], 0.10 + 0.05 * target,
        rtol=0.0, atol=1.0e-13)
    assert soil.deep_soil_temperature[0, 0] == 281.0

    with pytest.raises(KeyError, match="SOILT and SOILW together"):
        preprocess_noah_soil(
            {key: value for key, value in fields.items() if key != "SOILW"},
            soil_type=np.full((1, 1), 6))


def test_gfs_exact_noah_layers_are_copied_without_era5_interpolation():
    fields = {
        "LANDSEA": np.ones((1, 1)),
        "SKINTEMP": np.full((1, 1), 290.0),
        "TMN": np.full((1, 1), 280.0),
    }
    temperature_names = (
        "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200")
    moisture_names = (
        "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200")
    for layer, name in enumerate(temperature_names):
        fields[name] = np.full((1, 1), 289.0 - layer)
    for layer, name in enumerate(moisture_names):
        fields[name] = np.full((1, 1), 0.10 + 0.05 * layer)

    soil = preprocess_noah_soil(fields, soil_type=np.full((1, 1), 6))
    np.testing.assert_array_equal(
        soil.soil_temperature[:, 0, 0], [289.0, 288.0, 287.0, 286.0])
    np.testing.assert_allclose(
        soil.soil_moisture[:, 0, 0], [0.10, 0.15, 0.20, 0.25],
        rtol=0.0, atol=1.0e-15)

    with pytest.raises(KeyError, match="all four temperature and moisture"):
        preprocess_noah_soil(
            {key: value for key, value in fields.items()
             if key != "GFS_ST100200"},
            soil_type=np.full((1, 1), 6))

    with pytest.raises(ValueError, match="cannot be mixed"):
        preprocess_noah_soil(
            {**fields,
             "SOILT": np.full((9, 1, 1), 285.0),
             "SOILW": np.full((9, 1, 1), 0.2)},
            soil_type=np.full((1, 1), 6))


def test_noah_exports_wrf_repaired_deep_soil_temperature():
    fields = {
        "LANDSEA": np.array([[1.0, 0.0]]),
        "SKINTEMP": np.array([[289.0, 285.0]]),
        "SST": np.array([[0.0, 286.0]]),
        "TMN": np.array([[100.0, 500.0]]),
    }
    for name in ("ST000007", "ST007028", "ST028100", "ST100289"):
        fields[name] = np.full((1, 2), 285.0)
    for name in ("SM000007", "SM007028", "SM028100", "SM100289"):
        fields[name] = np.full((1, 2), 0.2)

    soil = preprocess_noah_soil(
        fields, soil_type=np.array([[6, 14]]))
    np.testing.assert_array_equal(
        soil.deep_soil_temperature, [[289.0, 286.0]])


def test_noah_snow_state_follows_wrf_real_presence_invariants():
    """module_initialize_real.F:517-543, including all flag pairings."""
    base = {
        "LANDSEA": np.ones((1, 2)),
        "SKINTEMP": np.full((1, 2), 270.0),
        "ST000007": np.full((1, 2), 269.0),
        "ST007028": np.full((1, 2), 270.0),
        "ST028100": np.full((1, 2), 271.0),
        "ST100289": np.full((1, 2), 272.0),
        "TMN": np.full((1, 2), 273.0),
        "SM000007": np.full((1, 2), 0.25),
        "SM007028": np.full((1, 2), 0.25),
        "SM028100": np.full((1, 2), 0.25),
        "SM100289": np.full((1, 2), 0.25),
    }

    soil_type = _soil_type(base)
    neither = preprocess_noah_soil(base, soil_type=soil_type)
    np.testing.assert_array_equal(neither.snow_water, 0.0)
    np.testing.assert_array_equal(neither.snow_depth, 0.0)

    depth_only = preprocess_noah_soil(
        {**base, "SNOWH": [[0.35, 0.10]]}, soil_type=soil_type)
    np.testing.assert_allclose(depth_only.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(depth_only.snow_depth, [[0.35, 0.10]])

    swe_only = preprocess_noah_soil(
        {**base, "SNOW": [[70.0, 20.0]]}, soil_type=soil_type)
    np.testing.assert_allclose(swe_only.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(swe_only.snow_depth, [[0.35, 0.10]])

    both = preprocess_noah_soil({
        **base, "SNOW": [[70.0, 20.0]], "SNOWH": [[0.50, 0.25]],
    }, soil_type=soil_type)
    np.testing.assert_allclose(both.snow_water, [[70.0, 20.0]])
    np.testing.assert_allclose(both.snow_depth, [[0.50, 0.25]])


@requires_bundle
def test_met_em_soil_oracle_is_remapped_at_noah_midpoints():
    import netCDF4

    names = ("LANDSEA", "SKINTEMP", "SST", "ST000007", "ST007028",
             "ST028100", "ST100289", "SM000007", "SM007028",
             "SM028100", "SM100289", "SNOW", "SOILTEMP", "HGT_M")
    with netCDF4.Dataset(MET_EM) as ds:
        fields = {name: np.asarray(ds.variables[name][0], dtype=np.float64)
                  for name in names}
    # SNOW is already kg m-2 in met_em; the public API accepts that spelling.
    land = fields["LANDSEA"] >= 0.5
    fields["TMN"] = np.where(
        land, fields["SOILTEMP"] - 0.0065 * fields["HGT_M"],
        fields["SOILTEMP"],
    )
    soil = preprocess_noah_soil(fields, soil_type=_soil_type(fields))
    target = np.array([0.05, 0.25, 0.70, 1.50])
    # WRF nodes are the integer-cm layer midpoints (char2int2: (0+7)/2=3,
    # (7+28)/2=17, (28+100)/2=64, (100+289)/2=194 cm), TSK at 0, TMN at 3 m.
    source = np.array([0.0, 0.03, 0.17, 0.64, 1.94, 3.0])
    valid_tmn = ((fields["TMN"] >= 170.0) & (fields["TMN"] <= 400.0)
                 & np.isfinite(fields["TMN"]))
    wrf_tmn = np.where(land & valid_tmn, fields["TMN"], soil.tsk)
    temp_nodes = np.stack([
        soil.tsk, fields["ST000007"], fields["ST007028"],
        fields["ST028100"], fields["ST100289"], wrf_tmn,
    ])
    moisture_nodes = np.stack([
        fields["SM000007"], fields["SM000007"], fields["SM007028"],
        fields["SM028100"], fields["SM100289"], fields["SM100289"],
    ])
    expected_t = np.stack([
        temp_nodes[k] + (temp_nodes[k + 1] - temp_nodes[k])
        * ((z - source[k]) / (source[k + 1] - source[k]))
        for z in target for k in [np.searchsorted(source, z) - 1]
    ])
    expected_m = np.stack([
        moisture_nodes[k] + (moisture_nodes[k + 1] - moisture_nodes[k])
        * ((z - source[k]) / (source[k + 1] - source[k]))
        for z in target for k in [np.searchsorted(source, z) - 1]
    ])
    expected_t[:, ~land] = soil.tsk[~land]
    expected_m[:, ~land] = 1.0
    t_rmse = np.sqrt(np.mean((soil.soil_temperature - expected_t) ** 2))
    m_rmse = np.sqrt(np.mean((soil.soil_moisture - expected_m) ** 2))
    assert t_rmse < 1.0e-12
    assert m_rmse < 1.0e-12


def test_stratospheric_qv_cap_pins_wrf_thresholds():
    """rh_to_mxrat1 caps (module_initialize_real.F:7490-7506).

    Both comparisons are strict: p < qv_max_p_safe (10000 Pa) and
    qv > qv_max_flag (1e-5) force qv_max_value (3e-6)
    (Registry.EM_COMMON:2306-2308).
    """
    from gpuwm.ingest.real import _cap_stratospheric_qv

    pressure = np.array([9999.9, 10000.0, 5000.0, 5000.0, 5000.0])
    qv = np.array([2.0e-5, 2.0e-5, 1.0e-5, 1.00001e-5, 5.0e-6])
    got = _cap_stratospheric_qv(qv, pressure)
    np.testing.assert_array_equal(
        got, [3.0e-6, 2.0e-5, 1.0e-5, 3.0e-6, 5.0e-6])


def _stratospheric_snapshot(cp, ny, nx):
    """Synthetic snapshot with levels above 100 hPa carrying wet qv."""
    from gpuwm.ingest.horiz import HorizontalSnapshot

    levels = np.array([50.0, 100.0, 200.0, 300.0, 500.0, 700.0, 850.0,
                       1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = np.broadcast_to(
        215.0 + 72.0 * (pressure / 100000.0) ** 0.20, shape).copy()
    height = np.broadcast_to(
        -7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 50.0 * (pressure / 100000.0), shape).copy()
    u = np.broadcast_to(12.0 + 0.8 * np.log(100000.0 / pressure),
                        (levels.size, ny, nx + 1)).copy()
    v = np.broadcast_to(-3.0 + 0.4 * np.log(100000.0 / pressure),
                        (levels.size, ny + 1, nx)).copy()
    fields = {
        "TT": temperature, "GHT": height, "RH": rh, "UU": u, "VV": v,
        "PSFC": np.full((ny, nx), 96000.0),
        "T2": np.full((ny, nx), 286.0), "D2": np.full((ny, nx), 279.0),
        "U10": np.full((ny, nx + 1), 11.0),
        "V10": np.full((ny + 1, nx), -2.0),
    }
    return HorizontalSnapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels,
        fields={name: cp.asarray(value, cp.float32)
                for name, value in fields.items()},
    )


def test_vectorized_moisture_integral_is_byte_identical_to_scalar_oracle():
    from gpuwm.ingest.real import (
        _integrate_moisture, _integrate_moisture_scalar_reference)

    rng = np.random.default_rng(7405)
    nlev, ny, nx = 9, 5, 7
    pressure_column = np.array(
        [100000.0, 92500.0, 85000.0, 70000.0, 50000.0,
         30000.0, 20000.0, 10000.0, 5000.0])
    pressure = np.broadcast_to(
        pressure_column[:, None, None], (nlev, ny, nx)).copy()
    pressure += rng.uniform(-40.0, 40.0, pressure.shape)
    temperature = rng.uniform(210.0, 305.0, pressure.shape)
    qv = rng.uniform(1.0e-6, 0.025, pressure.shape)
    base_height = np.array(
        [80.0, 700.0, 1450.0, 3000.0, 5600.0,
         8900.0, 11100.0, 14500.0, 19000.0])
    height = np.broadcast_to(
        base_height[:, None, None], pressure.shape).copy()
    height += rng.uniform(-25.0, 25.0, pressure.shape)
    # Exercise WRF's non-increasing-height branch in selected upper columns.
    height[6, 1, 2] = height[5, 1, 2] - 5.0
    height[4, 3, 5] = height[3, 3, 5]
    psfc = rng.uniform(62000.0, 101000.0, (ny, nx))
    tsfc = rng.uniform(265.0, 310.0, (ny, nx))
    qsfc = rng.uniform(1.0e-5, 0.025, (ny, nx))
    surface_height = rng.uniform(0.0, 2400.0, (ny, nx))

    expected = _integrate_moisture_scalar_reference(
        qv, pressure, temperature, height, psfc, tsfc, qsfc,
        surface_height)
    for workers in (1, 2, 3, 8):
        actual = _integrate_moisture(
            qv, pressure, temperature, height, psfc, tsfc, qsfc,
            surface_height, column_workers=workers)
        for observed, reference in zip(actual, expected):
            np.testing.assert_array_equal(observed, reference)


@pytest.mark.parametrize("hypsometric_opt", [1, 2])
def test_base_and_rebalance_workers_are_byte_identical(hypsometric_opt):
    from gpuwm.ingest.real import (
        _make_real_base, _rebalance_moist_pressure)

    rng = np.random.default_rng(7427 + hypsometric_opt)
    ny, nx, nz = 11, 13, 9
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2)
    terrain = rng.uniform(0.0, 2400.0, (ny, nx))
    expected_base = _make_real_base(
        coord, terrain, 10000.0, 290.0,
        hypsometric_opt=hypsometric_opt, column_workers=1)
    dry_mass = expected_base.mub + rng.uniform(-500.0, 500.0, (ny, nx))
    qv = rng.uniform(1.0e-6, 0.025, (nz, ny, nx))
    expected_pressure = _rebalance_moist_pressure(
        expected_base.pb, qv, dry_mass, expected_base, coord,
        column_workers=1)

    for workers in (2, 3, 8, 32):
        base = _make_real_base(
            coord, terrain, 10000.0, 290.0,
            hypsometric_opt=hypsometric_opt, column_workers=workers)
        for name in ("mub", "pb", "alb", "thb", "phb", "terrain_z"):
            np.testing.assert_array_equal(
                getattr(base, name), getattr(expected_base, name))
        pressure = _rebalance_moist_pressure(
            np.full_like(expected_base.pb, -999.0), qv, dry_mass, base,
            coord, column_workers=workers)
        np.testing.assert_array_equal(pressure, expected_pressure)


@requires_gpu
@pytest.mark.gpu
def test_moisture_integral_uses_unadjusted_psfc_with_capped_source_qv():
    """integ_moist parity (module_initialize_real.F:1457, 7022, 1116).

    WRF integrates moisture with the ORIGINAL met surface pressure
    (integ_moist's psfc = p_gc level 1) and qv_gc already carrying the
    rh_to_mxrat1 stratospheric caps; only p_dts (:1482) pairs the
    sfcprs2-adjusted psfc with the resulting intq.  The float64 oracle
    below reproduces exactly that pairing.
    """
    import cupy as cp
    from gpuwm.ingest.real import (_cap_stratospheric_qv,
                                   _integrate_moisture, initialize_real)

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    snapshot = _stratospheric_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]
    result = initialize_real(snapshot, cfg, coord, terrain,
                             source_orography=source_orography,
                             p_top=5000.0, sfcp_to_sfcp=True)

    fields = {name: np.asarray(cp.asnumpy(value), dtype=np.float64)
              for name, value in snapshot.fields.items()}
    pressure = np.broadcast_to(
        snapshot.levels_hpa[:, None, None] * 100.0,
        fields["TT"].shape).copy()
    source_qv = _saturation_mixing_ratio(fields["TT"], pressure,
                                         fields["RH"])
    assert float(source_qv[0].min()) > 1.0e-5  # the 50 hPa cap must bite
    source_qv = _cap_stratospheric_qv(source_qv, pressure)
    np.testing.assert_array_equal(source_qv[0], 3.0e-6)
    expected_pd, expected_intq, _ = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"], fields["PSFC"],
        fields["T2"], result.surface_qv, source_orography)
    np.testing.assert_array_equal(
        result.integrated_moisture_pressure, expected_intq)
    np.testing.assert_array_equal(
        result.dry_mass, result.surface_pressure - expected_intq - 5000.0)
    # The adjusted-psfc pairing the audit flagged is measurably different.
    _, adjusted_intq, _ = _integrate_moisture(
        source_qv, pressure, fields["TT"], fields["GHT"],
        result.surface_pressure, fields["T2"], result.surface_qv,
        source_orography)
    assert float(np.max(np.abs(adjusted_intq - expected_intq))) > 0.0
    # WRF calls rh_to_mxrat1 after eta interpolation and again after its
    # final hydrostatic pressure diagnosis.  The uploaded target state must
    # therefore retain the strict stratospheric cap, not just the source
    # pressure-level column used by integ_moist.
    final_qv = cp.asnumpy(result.state.qv)
    upper = result.total_pressure < 10000.0
    assert np.any(upper)
    assert np.any(final_qv[upper] == np.float32(3.0e-6))
    assert np.all(final_qv[upper] <= np.float32(1.0e-5))


def _synthetic_horizontal_snapshot(cp, ny, nx):
    from gpuwm.ingest.horiz import HorizontalSnapshot

    levels = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = 215.0 + 72.0 * (pressure / 100000.0) ** 0.20
    temperature = np.broadcast_to(temperature, shape).copy()
    height = np.broadcast_to(-7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(35.0 + 50.0 * (pressure / 100000.0), shape).copy()
    u = np.broadcast_to(12.0 + 0.8 * np.log(100000.0 / pressure),
                        (levels.size, ny, nx + 1)).copy()
    v = np.broadcast_to(-3.0 + 0.4 * np.log(100000.0 / pressure),
                        (levels.size, ny + 1, nx)).copy()
    fields = {
        "TT": temperature, "GHT": height, "RH": rh, "UU": u, "VV": v,
        "PSFC": np.full((ny, nx), 96000.0),
        "T2": np.full((ny, nx), 286.0), "D2": np.full((ny, nx), 279.0),
        "U10": np.full((ny, nx + 1), 11.0),
        "V10": np.full((ny + 1, nx), -2.0),
    }
    return HorizontalSnapshot(
        valid_time=datetime(1974, 4, 3, 12), levels_hpa=levels,
        fields={name: cp.asarray(value, cp.float32) for name, value in fields.items()},
    )


def test_cpu_real_state_and_lbc_materialization_do_not_call_cuda(monkeypatch):
    """The public CPU preprocessing path must work with no CUDA device.

    This pins the production failure found by the genuine GFS d01-d06 gate:
    CPU interpolation previously ended by allocating ``DomainState`` and LBC
    storage through CuPy, raising ``cudaErrorNoDevice`` before WRF export.
    """
    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.ingest.lateral_bc import (
        attach_lateral_boundaries,
        build_state_lateral_boundaries,
    )
    from gpuwm.verify.npref import np_wrf_real_vert_interp

    class Plan:
        def __init__(self, source, surface, target):
            self.source = np.asarray(source, dtype=np.float32)
            self.surface = np.asarray(surface, dtype=np.float32)
            self.target = np.asarray(target, dtype=np.float32)

        def apply(self, field, surface_value, **options):
            options.pop("values_are_finite", None)
            return np.asarray(np_wrf_real_vert_interp(
                field, surface_value, self.source, self.surface,
                self.target, **options), dtype=np.float32)

    class Backend:
        name = "cpu-test"
        array_module = np

        @staticmethod
        def float32(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def prepare_wrf_vertical(source, surface, target):
            return Plan(source, surface, target)

        @staticmethod
        def regular_plan(*_args, **_kwargs):  # pragma: no cover - contract
            raise AssertionError("unused")

        masked_nearest = regular_plan
        rotate_earth_to_grid = regular_plan
        era5_rh_to_water = regular_plan

        @staticmethod
        def receipt():
            return {"backend": "cpu-test"}

    ny, nx = 12, 13
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=10800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0, specified=True)
    snapshot = _synthetic_horizontal_snapshot(np, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 50.0 * np.sin(np.arange(nx))[None, :]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU preprocessing attempted a CuPy allocation")

    monkeypatch.setattr(cp, "zeros", forbidden)
    monkeypatch.setattr(cp, "ones", forbidden)
    monkeypatch.setattr(cp, "asarray", forbidden)

    def initialize():
        # The synthetic source column tops at 100 hPa exactly, so the
        # model top is pinned there; the DEFAULT (50 hPa) is pinned by
        # tests/test_ptop_default.py and would sit above this source.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0,
            preprocess_backend=Backend(), state_backend="cpu")

    first = initialize()
    second = initialize()
    for result in (first, second):
        assert isinstance(result.state.u, np.ndarray)
        update_diagnostics(result.state, cfg.hypsometric_opt)
        assert np.isfinite(result.state.p).all()
    times = (snapshot.valid_time,
             snapshot.valid_time + timedelta(hours=3))
    boundaries = build_state_lateral_boundaries(
        [first.state, second.state], times,
        spec_bdy_width=cfg.spec_bdy_width,
        spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
    attach_lateral_boundaries(first.state, boundaries)
    assert first.state.lateral_boundaries is boundaries
    assert isinstance(first.state._scratch["lbc_forcing_tables"], np.ndarray)


@requires_gpu
@pytest.mark.gpu
def test_hrrr_defaults_to_rh_vertical_path_unless_use_sh_qv_is_explicit():
    import cupy as cp

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0, mp_physics=6)
    base_snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    fields = dict(base_snapshot.fields)
    levels = base_snapshot.levels_hpa
    fields["PRES"] = cp.asarray(
        np.broadcast_to(levels[:, None, None] * 100.0,
                        (levels.size, ny, nx)), cp.float32)
    fields["SPFH"] = cp.zeros((levels.size, ny, nx), cp.float32)
    fields["Q2"] = cp.zeros((ny, nx), cp.float32)
    # FLAG_SH must make a separately supplied RH field unnecessary: real.exe
    # overwrites rh_gc from the horizontally mapped SPECHUMD/TT/PRES values.
    fields.pop("RH")
    for name in ("QC", "QR", "QI", "QS", "QG"):
        fields[name] = cp.full(
            (levels.size, ny, nx), 2.5e-4, cp.float32)
    fields.pop("D2")
    snapshot = HorizontalSnapshot(
        valid_time=base_snapshot.valid_time, levels_hpa=levels, fields=fields)
    # Byte-equal source/target terrain exercises WRF's defined sfcprs2
    # no-op (exp(0) = 1); case-level wrfinput gates own the provenance
    # tripwire for regressed source-orography data.
    source_orography = np.full((ny, nx), 300.0)
    terrain = source_orography.copy()

    # p_top pinned to the synthetic column's own 100 hPa top; the flag
    # under test is use_sh_qv, not the model top.
    default_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography, p_top=10000.0)
    direct_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography, p_top=10000.0,
        use_sh_qv=True)
    default_qv = cp.asnumpy(default_result.state.qv)
    direct_qv = cp.asnumpy(direct_result.state.qv)
    np.testing.assert_array_equal(
        default_qv, np.full(default_qv.shape, np.float32(1.0e-6)))
    np.testing.assert_array_equal(
        direct_qv, np.zeros(direct_qv.shape, dtype=np.float32))
    for state in (default_result.state, direct_result.state):
        for name in ("qc", "qr", "qi", "qs", "qg"):
            value = cp.asnumpy(getattr(state, name))
            assert np.isfinite(value).all()
            assert float(value.min()) >= 0.0
            assert float(value.max()) <= 2.5e-4 * (1.0 + 2.0e-7)
            assert np.any(value > 0.0)

    bad_fields = dict(fields)
    bad_fields["QG"] = cp.full(
        (levels.size, ny, nx), -1.0e-7, cp.float32)
    bad_snapshot = HorizontalSnapshot(
        valid_time=base_snapshot.valid_time, levels_hpa=levels,
        fields=bad_fields)
    with pytest.raises(ValueError, match="non-finite or negative"):
        initialize_real(
            bad_snapshot, cfg,
            make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                                eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0)


@requires_gpu
@pytest.mark.gpu
def test_real_init_builds_nonnegative_balanced_fp32_domain_state():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    coord = make_vertical_coord(eta.size - 1, hybrid_opt=2, etac=0.2,
                                eta_levels=eta)
    cfg = RunConfig(nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
                    ztop=16000.0, dt=30.0, run_seconds=1800.0,
                    hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
                    base_temp=290.0)
    snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]
    result = initialize_real(snapshot, cfg, coord, terrain,
                             source_orography=source_orography,
                             p_top=10000.0, sfcp_to_sfcp=True)
    state = result.state

    assert state.thp.dtype == cp.float32 and state.qv.dtype == cp.float32
    assert bool(cp.isfinite(state.thp).all()) and bool(cp.isfinite(state.php).all())
    assert float(state.qv.min()) >= 1.0e-6
    residual = hydrostatic_residual(result)
    assert residual.shape == (cfg.ny, cfg.nx)
    assert residual.max() < 2.0e-2

    expected_psfc = surface_pressure_from_surface(
        np.full((ny, nx), 96000.0), source_orography, terrain,
        np.full((ny, nx), 286.0), result.surface_qv,
    )
    np.testing.assert_allclose(result.surface_pressure, expected_psfc,
                               rtol=0.0, atol=2.0e-9)
    update_diagnostics(state)
    np.testing.assert_allclose(cp.asnumpy(state.p), result.total_pressure,
                               rtol=4.0e-4, atol=4.0)

    baseline = residual[0, 0]
    state.php[1, 0, 0] += cp.float32(1.0)
    assert hydrostatic_residual(result)[0, 0] > baseline + 0.5


@requires_gpu
@pytest.mark.gpu
def test_real_init_parallel_cpu_backend_is_worker_stable_and_matches_cuda():
    import cupy as cp

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=1800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0)
    snapshot = _synthetic_horizontal_snapshot(cp, ny, nx)
    source_orography = np.linspace(200.0, 800.0, ny * nx).reshape(ny, nx)
    terrain = source_orography + 120.0 * np.sin(np.arange(nx))[None, :]

    def initialize(**options):
        # p_top pinned to the synthetic column's own 100 hPa top.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, p_top=10000.0,
            **options)

    cuda = initialize(preprocess_backend="cuda")
    try:
        cpu_serial = initialize(
            preprocess_backend="cpu", preprocess_workers=1)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")
    cpu_parallel = initialize(
        preprocess_backend="cpu", preprocess_workers=8)

    setup_names = (
        "surface_pressure", "surface_qv", "dry_mass", "dry_pressure",
        "total_pressure", "total_geopotential", "total_specific_volume",
        "integrated_moisture_pressure",
    )
    state_names = ("mup", "thp", "php", "qv", "u", "v", "w")
    for name in setup_names:
        np.testing.assert_array_equal(
            getattr(cpu_parallel, name), getattr(cpu_serial, name))
    for name in state_names:
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(cpu_parallel.state, name)),
            cp.asnumpy(getattr(cpu_serial.state, name)))

    for name in ("mup", "thp", "qv", "u", "v", "w"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(cpu_parallel.state, name)),
            cp.asnumpy(getattr(cuda.state, name)),
            rtol=3.0e-5, atol=5.0e-3)
    np.testing.assert_allclose(
        cp.asnumpy(cpu_parallel.state.php), cp.asnumpy(cuda.state.php),
        rtol=3.0e-5, atol=2.0e-2)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("mp_physics", (6, 8))
def test_real_init_cpu_backend_handles_hrrr_hydrometeor_inventory(
        mp_physics, monkeypatch, tmp_path):
    import cupy as cp

    if mp_physics == 8:
        monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
        monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(tmp_path))

    ny, nx = 4, 7
    eta = np.array([1.0, 0.92, 0.78, 0.60, 0.40, 0.22, 0.09, 0.0])
    cfg = RunConfig(
        nx=nx, ny=ny, nz=eta.size - 1, dx=12000.0, dy=12000.0,
        ztop=16000.0, dt=30.0, run_seconds=1800.0,
        hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        base_temp=290.0, mp_physics=mp_physics,
        moist_cq=mp_physics == 8, top_lid=mp_physics != 8)
    base = _synthetic_horizontal_snapshot(cp, ny, nx)
    fields = dict(base.fields)
    fields["PRES"] = cp.asarray(
        np.broadcast_to(
            base.levels_hpa[:, None, None] * 100.0,
            (base.levels_hpa.size, ny, nx)), dtype=cp.float32)
    fields["SPFH"] = cp.zeros(
        (base.levels_hpa.size, ny, nx), dtype=cp.float32)
    fields["Q2"] = cp.zeros((ny, nx), dtype=cp.float32)
    fields.pop("RH")
    fields.pop("D2")
    for index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        fields[name] = cp.full(
            (base.levels_hpa.size, ny, nx), index * 2.5e-5,
            dtype=cp.float32)
    snapshot = HorizontalSnapshot(
        valid_time=base.valid_time, levels_hpa=base.levels_hpa,
        fields=fields)
    terrain = np.full((ny, nx), 300.0)
    # Byte-equal source/target terrain is WRF's defined sfcprs2 no-op.
    source_orography = terrain

    def initialize(backend):
        # p_top pinned to the synthetic column's own 100 hPa top.
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, use_sh_qv=True,
            p_top=10000.0,
            preprocess_backend=backend,
            preprocess_workers=8 if backend == "cpu" else None)

    cuda = initialize("cuda")
    try:
        cpu = initialize("cpu")
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"native CPU bridge is not built: {exc}")
    for name in ("qc", "qr", "qi", "qs", "qg"):
        actual = cp.asnumpy(getattr(cpu.state, name))
        expected = cp.asnumpy(getattr(cuda.state, name))
        assert np.isfinite(actual).all()
        assert np.all(actual >= 0.0)
        np.testing.assert_allclose(
            actual, expected, rtol=3.0e-5, atol=5.0e-8)
    if mp_physics == 8:
        for state in (cpu.state, cuda.state):
            np.testing.assert_array_equal(
                cp.asnumpy(state.ni), np.zeros(state.ni.shape, np.float32))
            np.testing.assert_array_equal(
                cp.asnumpy(state.nr), np.zeros(state.nr.shape, np.float32))


def test_real74_d01_30min_gate_admits_complete_no_pbl_operator():
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases.real74_d01 import config, phase3_config

    legacy = validate_run_config(config(run_seconds=1800.0))
    assert legacy.km_opt == 4
    assert legacy.bl_pbl_physics == 0

    supported = validate_run_config(phase3_config(run_seconds=1800.0))
    assert supported.km_opt == 4
    assert supported.bl_pbl_physics == 1


# ---------------------------------------------------------------------------
# mp_physics=28 (Thompson aerosol-aware) real-data ingest.
#
# Before this lane existed, gpuwm/ingest/real.py named mp_physics 1, 6, 8, 10
# and 18 at three separate sites and 28 at none of them.  The consequence was
# not an error: an mp=28 run from a cloudy HRRR analysis silently produced a
# condensate-free initial state, because the five analyzed species were never
# declared required (:1106), never shape-checked (:1150) and never vertically
# interpolated (:1390).  Each of these tests fails against that tree.
#
# WRF authority, v4.6.1 commit d66e442fccc04111067e29274c9f9eaccc3cef28:
#   Registry/Registry.EM_COMMON:3036   thompsonaero package membership
#   Registry/Registry.EM_COMMON:3024   thompson, for the moist-list identity
#   dyn_em/module_initialize_real.F:2332-2345   aer_init_opt=0, 3-D aerosol
#   dyn_em/module_initialize_real.F:4501-4510   aer_init_opt=0, 2-D emission
#   dyn_em/module_initialize_real.F:2735-2736   the FATAL ArWen deviates from
#   phys/module_mp_thompson.F:493,:531          thompson_init's MAXVAL tests
# ---------------------------------------------------------------------------


class _ReferenceVerticalPlan:
    """WRF-real vertical interpolation through the NumPy reference."""

    def __init__(self, source, surface, target):
        self.source = np.asarray(source, dtype=np.float32)
        self.surface = np.asarray(surface, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def apply(self, field, surface_value, **options):
        from gpuwm.verify.npref import np_wrf_real_vert_interp

        options.pop("values_are_finite", None)
        return np.asarray(np_wrf_real_vert_interp(
            field, surface_value, self.source, self.surface, self.target,
            **options), dtype=np.float32)


class _ReferencePreprocessBackend:
    """Independent CPU implementation of the preprocessing ABI.

    Deliberately not the packaged Rust CPU bridge: these tests are about the
    scheme-selection logic in :func:`initialize_real`, and must not skip on a
    machine where the bridge is unbuilt.
    """

    name = "cpu-reference-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def regular_plan(*args, **kwargs):
        raise AssertionError("horizontal preprocessing is unused here")

    masked_nearest = regular_plan
    rotate_earth_to_grid = regular_plan
    era5_rh_to_water = regular_plan

    @staticmethod
    def prepare_wrf_vertical(source, surface, target):
        return _ReferenceVerticalPlan(source, surface, target)

    @staticmethod
    def receipt():
        return {"backend": "cpu-reference-test"}


def _analyzed_hrrr_real_init(
        mp_physics, *, state_backend="cpu", terrain_m=0.0,
        drop=(), reshape=None, **config_overrides):
    """One decoded-native-HRRR real initialization, never a fabricated state.

    ``drop`` removes analyzed species from the decoded snapshot and
    ``reshape`` truncates one of them, so the required-field and shape gates
    can be exercised for real rather than asserted about.
    """

    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    height += terrain_m
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    row_index = np.arange(ny, dtype=np.float32)[None, :, None]
    column_index = np.arange(nx, dtype=np.float32)[None, None, :]
    analyzed = {}
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        value = np.asarray(
            species_index * 1.0e-6
            * (1.0 + level_index + 0.25 * row_index
               + 0.125 * column_index), dtype=np.float32)
        # One exact zero so a nonzero-mask fingerprint is a real discriminator.
        value[0, 0, 0] = np.float32(0.0)
        analyzed[name] = value
    for name in drop:
        analyzed.pop(name)
    if reshape is not None:
        analyzed[reshape] = analyzed[reshape][:, :, :-1].copy()
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
        **analyzed,
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0,
        dt=30.0, run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True,
        terrain_opt=1, mp_physics=mp_physics, **config_overrides)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.full((ny, nx), float(terrain_m), dtype=np.float64)
    result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0, use_sh_qv=True,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend=state_backend)
    return result, cfg


def _host_array(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def test_mp28_real_ingest_retains_every_analyzed_hydrometeor():
    """mp=28's Registry moist list is character for character mp=8's.

    Registry.EM_COMMON:3024 gives ``thompson`` moist:qv,qc,qr,qi,qs,qg and
    :3036 gives ``thompsonaero`` the same six; the aerosol-aware scheme adds
    only ``scalar:`` members.  So all five decoded HRRR mass species must
    survive to the state, exactly as for mp=8.  Against the tree that omitted
    28 from the three mp tuples this state was identically zero and nothing
    raised.
    """
    result, _ = _analyzed_hrrr_real_init(28)
    state = result.state

    evidence = result.hydrometeor_initialization
    # v2 since the 1.4.1 merge: the HRRR vertical-disposition work on the
    # release line moved the correspondence schema while the port was off
    # the line, and gpuwm/ingest/real.py now emits v2 for every scheme.  The
    # mp=28 claim this test makes -- that a 28 run retains every analyzed
    # hydrometeor -- is unchanged; only the schema stamp moved.
    assert evidence["schema"] == "gpuwm-real-hydrometeor-correspondence-v2"
    assert evidence["mp_physics"] == 28
    assert set(evidence["retained_correspondence"]) == {
        "QC", "QR", "QI", "QS", "QG"}
    assert evidence["discarded_source_species"] == {}
    for source_name, state_name in sorted(
            evidence["retained_correspondence"].items()):
        live = _host_array(getattr(state, state_name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert np.isfinite(live).all()
        assert live.min() >= 0.0
        assert np.count_nonzero(live) > 0, (
            f"{source_name}->{state_name} carried no analyzed mass")
        assert live.max() <= float(
            evidence["decoded_source_species"][source_name]["maximum"]
        ) * (1.0 + 4.0e-7)

    # An mp=8 initialization of the same snapshot is the control: the two
    # must agree bit for bit on the mass species, because the mass path is
    # the same code and the only difference is which scalars exist.
    control, _ = _analyzed_hrrr_real_init(8)
    for name in ("qc", "qr", "qi", "qs", "qg"):
        np.testing.assert_array_equal(
            _host_array(getattr(state, name)),
            _host_array(getattr(control.state, name)))


def test_mp28_real_ingest_requires_the_analyzed_inventory_by_name():
    """The :1106 required-field gate, exercised rather than asserted."""
    with pytest.raises(KeyError, match=r"missing real-data field\(s\).*QI"):
        _analyzed_hrrr_real_init(28, drop=("QI",))


def test_mp28_real_ingest_shape_checks_the_analyzed_inventory():
    """The :1150 mass-shape gate.

    Without 28 in that tuple a truncated QG reached the vertical plan and
    failed later, inside the interpolation, with a message about neither the
    field nor the scheme.
    """
    with pytest.raises(ValueError, match="mass-field shapes do not match"):
        _analyzed_hrrr_real_init(28, reshape="QG")


def test_mp28_real_ingest_zeroes_the_source_absent_number_moments():
    """Registry scalars the analysis does not carry begin at exact zero.

    mp=8's arm zeroes ni and nr.  mp=28 adds nc, because the aerosol-aware
    scheme promotes cloud droplet number from the constant Nt_c to a
    prognostic Registry scalar (QNCLOUD, Registry.EM_COMMON:3036).  Exact
    FP32 zero is required, not "small": real.exe initializes absent package
    members to 0.0 and the scheme owns their first physical update.
    """
    result, _ = _analyzed_hrrr_real_init(28)
    state = result.state
    for name in ("nc", "nr", "ni"):
        live = _host_array(getattr(state, name))
        assert live.dtype == np.float32
        assert live.shape == (8, 2, 3)
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} is not exact FP32 zero")


def test_mp28_real_ingest_leaves_the_aerosols_for_the_init_hook():
    """The aerosol fields must arrive EXACTLY zero, and say so.

    Zero here is not "unset": ``thompson_init`` decides whether to install
    its synthetic CCN/IN profile by testing MAXVAL(nwfa) < eps
    (phys/module_mp_thompson.F:493) and MAXVAL(nifa) < eps (:531).  Any
    nonzero placeholder written by the ingest would flip those tests and
    permanently suppress the profile, leaving the aerosol-aware physics inert
    with no error anywhere.  WRF's own initializer writes exactly 0.0 for
    both 3-D fields (dyn_em/module_initialize_real.F:2332-2345) and both 2-D
    emissions (:4501-4510) under aer_init_opt=0.
    """
    result, cfg = _analyzed_hrrr_real_init(28)
    state = result.state
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        live = _host_array(getattr(state, name))
        assert live.dtype == np.float32
        assert int(live.view(np.uint32).max()) == 0, (
            f"state.{name} is not exact FP32 zero")

    receipt = result.aerosol_initialization
    assert receipt["policy"] == (
        "aer-init-opt-0-zero-then-thompson-init-synthetic-profile")
    assert receipt["registry_citation"] == (
        "Registry/Registry.EM_COMMON:3036")
    assert receipt["real_citation"] == (
        "dyn_em/module_initialize_real.F:2332-2345")
    assert receipt["wrf_real_refuses_this_configuration"] == (
        "dyn_em/module_initialize_real.F:2735-2736")
    assert receipt["deferred_to"] == (
        "gpuwm.core.physics.initialize_physics -> "
        "gpuwm.core.microphysics.microphysics_init")
    assert receipt["awaiting_profile_fill"] is True
    assert receipt["aer_init_opt"] == 0 and receipt["wif_input_opt"] == 0
    assert set(receipt["not_initialized_here"]) == {
        "nwfa", "nifa", "nwfa2d", "nifa2d"}
    fingerprints = receipt["source_absent_state_fields"]
    assert set(fingerprints) == {
        "nc", "nr", "ni", "nwfa", "nifa", "nwfa2d", "nifa2d"}
    assert all(item["nonzero_count"] == 0 for item in fingerprints.values())

    # Every other scheme carries an empty aerosol receipt, so the field can
    # never be read as "this run has aerosol provenance" when it does not.
    control, _ = _analyzed_hrrr_real_init(8)
    assert control.aerosol_initialization == {}
    assert cfg.mp_physics == 28


def test_mp28_real_ingest_refuses_an_aerosol_source_it_cannot_read():
    """WIF selectors fail closed inside the ingest, not only in the config.

    ``initialize_real`` is reachable with a RunConfig that never went through
    ``validate_run_config``; accepting wif_input_opt=1 there would promise a
    metgrid WIF stream this module cannot read and then hand the microphysics
    an all-zero aerosol field as if it were the requested climatology.
    """
    from gpuwm.config import MP28_AEROSOL_SOURCE_DEVIATION

    for overrides, name in (
            ({"wif_input_opt": 1}, "wif_input_opt"),
            ({"wif_input_opt": 2}, "wif_input_opt"),
            ({"aer_init_opt": 1}, "aer_init_opt"),
            ({"aer_init_opt": 2}, "aer_init_opt")):
        with pytest.raises(NotImplementedError) as caught:
            _analyzed_hrrr_real_init(28, **overrides)
        message = str(caught.value)
        assert name in message
        assert MP28_AEROSOL_SOURCE_DEVIATION in message

    # Not a blanket refusal: the same selectors are inert under mp=8 and the
    # ingest must not start policing another scheme's namelist.
    result, _ = _analyzed_hrrr_real_init(8)
    assert result.aerosol_initialization == {}


def test_mp28_real_ingest_does_not_call_the_profile_fill_itself():
    """The ingest is not allowed to be the caller of ``microphysics_init``.

    Proven structurally rather than by comment: the module source contains no
    call, and the state it returns is empty of aerosol.  The fill belongs to
    ``gpuwm.core.physics.initialize_physics``, which BOTH production
    real-data front doors reach with the state this function returns
    (``gpuwm/ingest/hrrr_physics.py``), so a call here would be the second
    one.
    """
    import inspect
    import re

    from gpuwm.ingest import hrrr_physics
    import gpuwm.ingest.real as real_module

    source = inspect.getsource(real_module)
    assert not re.search(r"\bmicrophysics_init\s*\(", source), (
        "gpuwm/ingest/real.py must not call microphysics_init")
    assert not re.search(r"\bthompson_aerosol_init_fill\s*\(", source)

    # The named successor really is on this path.
    physics_source = inspect.getsource(hrrr_physics)
    assert "initialize_physics(" in physics_source


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_then_microphysics_init_fills_exactly_once():
    """End to end: real ingest -> physics init -> WRF's synthetic profile.

    This is the ownership proof the ingest lane owes.  The ingest leaves the
    aerosol at exact zero; the FIRST ``microphysics_init`` performs both
    fills and reports ``{'ccn': True, 'in': True}``; a SECOND call reports
    ``{'ccn': False, 'in': False}`` and changes not one bit, because
    ``thompson_init``'s own MAXVAL guard (module_mp_thompson.F:493/:531) now
    sees a populated field.  So the fill happens exactly once on this path,
    and an ingest that had populated the aerosol itself would be preserved
    rather than overwritten.

    The values are checked against WRF's own parameters rather than merely
    for being nonzero: naCCN1=50.0E6 and naCCN0=300.0E6 (:96-97) bound nwfa
    to [5.0e7, 3.5e8], naIN1=0.5E6 and naIN0=1.5E6 (:94-95) bound nifa to
    [5.0e5, 2.0e6], and both profiles decrease monotonically upward from a
    terrain below the 1000 m ``h_01`` breakpoint (:500-506).
    """
    import cupy as cp

    from gpuwm.core.microphysics import microphysics_init

    result, cfg = _analyzed_hrrr_real_init(
        28, state_backend="cuda", terrain_m=250.0)
    state = result.state
    assert isinstance(state.nwfa, cp.ndarray)
    for name in ("nwfa", "nifa", "nwfa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0

    first = microphysics_init(state, cfg)
    assert first == {"thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    nwfa2d = cp.asnumpy(state.nwfa2d)

    assert nwfa.dtype == np.float32 and nwfa.shape == (8, 2, 3)
    assert nifa.dtype == np.float32 and nifa.shape == (8, 2, 3)
    assert nwfa2d.dtype == np.float32 and nwfa2d.shape == (2, 3)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert 0.5e6 <= nifa.min() and nifa.max() <= 2.0e6
    assert nwfa2d.min() > 0.0 and np.isfinite(nwfa2d).all()
    # nifa2d is never assigned anywhere in module_mp_thompson.F; it is not
    # even a thompson_init dummy argument.  Zero is WRF's behaviour.
    assert int(cp.asnumpy(state.nifa2d).view(np.uint32).max()) == 0
    for column in ((0, 0), (1, 2)):
        profile = nwfa[:, column[0], column[1]]
        assert np.all(np.diff(profile) <= 0.0)
        assert np.all(np.diff(nifa[:, column[0], column[1]]) <= 0.0)

    second = microphysics_init(state, cfg)
    assert second == {"thompson_aerosol_profile": {"ccn": False, "in": False}}
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), nwfa)
    np.testing.assert_array_equal(cp.asnumpy(state.nifa), nifa)
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa2d), nwfa2d)


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_through_initialize_physics_fills_exactly_once():
    """The whole production chain, not the hook in isolation.

    ``initialize_real`` -> ``gpuwm.core.physics.initialize_physics`` is the
    path every real-data front door takes
    (``gpuwm/ingest/hrrr_physics.py::initialize_prepared_physics`` calls
    ``initialize_physics`` on ``result.state``, and
    ``::initialize_hrrr_physics`` delegates to it), and it is the reason this
    module must not perform the fill itself.  WRF's own structure is the same
    one: ``phys/module_physics_init.F:1635`` calls ``mp_init`` as the last
    physics initializer, and ``mp_init``'s THOMPSONAERO arm calls
    ``thompson_init``; ``dyn_em/module_initialize_real.F`` never does.

    ANSWER TO THE OWNERSHIP QUESTION, measured: the physics init path is
    responsible.  The ingest leaves exact zero, the first driver's
    ``microphysics_init_receipt`` reports both fills ran, and a second
    ``initialize_physics`` on the same state reports neither ran -- because
    ``thompson_init``'s own MAXVAL presence tests now see a populated field.
    So a duplicate call is idempotent rather than destructive, and an ingest
    that DID carry aerosol would survive the physics init untouched; the
    reason the ingest still must not call it is structural (WRF's split, and
    the receipt this function publishes), not a race.
    """
    import cupy as cp

    from gpuwm.core.physics import initialize_physics

    result, cfg = _analyzed_hrrr_real_init(
        28, state_backend="cuda", terrain_m=250.0)
    state = result.state
    assert result.aerosol_initialization["awaiting_profile_fill"] is True
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0

    driver = initialize_physics(state, cfg)
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    nwfa2d = cp.asnumpy(state.nwfa2d)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert 0.5e6 <= nifa.min() and nifa.max() <= 2.0e6
    assert nwfa2d.min() > 0.0

    second = initialize_physics(state, cfg)
    assert second.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": False, "in": False}}
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), nwfa)
    np.testing.assert_array_equal(cp.asnumpy(state.nifa), nifa)
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa2d), nwfa2d)

    # The mass species the ingest DID initialize are untouched by the fill.
    for name in ("qc", "qr", "qi", "qs", "qg"):
        live = cp.asnumpy(getattr(state, name))
        assert np.count_nonzero(live) > 0
        assert np.isfinite(live).all()


def test_mp28_real_ingest_receipt_agrees_with_the_published_deviation():
    """The ingest receipt and the user-facing deviation cannot drift apart.

    ``gpuwm.config.MP28_AEROSOL_SOURCE_DEVIATION`` is the sentence a user
    sees in the namelist importer's printed receipt.  It asserts three
    things this lane is now the production evidence for: the aerosol initial
    state comes from thompson_init's synthetic profile, ArWen has no
    QNWFA/QNIFA ingest lane, and WRF's real.exe FATALs this configuration at
    dyn_em/module_initialize_real.F:2735-2736.  If the ingest ever grew an
    aerosol source, that sentence would become false somewhere no test looks
    -- so it is bound here, to the receipt the ingest actually emits.
    """
    from gpuwm.config import (
        MP28_AEROSOL_SOURCE_DEVIATION, MP28_AEROSOL_SOURCE_OPTIONS)
    from gpuwm.ingest.real import WRF_REAL_MP28_AEROSOL_SOURCE_POLICY

    result, _ = _analyzed_hrrr_real_init(28)
    receipt = result.aerosol_initialization

    assert receipt["wrf_real_refuses_this_configuration"] in (
        MP28_AEROSOL_SOURCE_DEVIATION)
    assert "module_mp_thompson.F" in receipt["microphysics_citation"]
    assert receipt["awaiting_profile_fill"] is True
    assert receipt["not_initialized_here"] == (
        WRF_REAL_MP28_AEROSOL_SOURCE_POLICY["not_initialized_here"])
    # The receipt reports the selectors it actually ran under, and the only
    # values it can ever report are the ones the port implements.
    for name, (only, _citation, _why) in MP28_AEROSOL_SOURCE_OPTIONS.items():
        assert receipt[name] == only
    assert set(MP28_AEROSOL_SOURCE_OPTIONS) == {
        "aer_init_opt", "wif_input_opt"}


def _pressure_level_real_init(mp_physics, **config_overrides):
    """The other production lane: pressure-level TT/RH forcing (ERA5, GFS).

    No analyzed hydrometeors exist on this lane for ANY scheme, which is
    exactly why the mp=28 aerosol policy cannot live inside the native-HRRR
    ``if hydrometeors:`` branch -- a user arriving with ERA5 must still get
    the exact-zero aerosol state thompson_init's presence test needs.
    """
    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    fields = {
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "RH": np.full(pressure.shape, 60.0, dtype=np.float32),
        "D2": np.full((ny, nx), 283.0, dtype=np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
    }
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0,
        dt=30.0, run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True,
        terrain_opt=1, mp_physics=mp_physics, **config_overrides)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.zeros((ny, nx), dtype=np.float64)
    return initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0,
        preprocess_backend=_ReferencePreprocessBackend(),
        state_backend="cpu"), cfg


def test_mp28_pressure_level_lane_also_publishes_the_aerosol_policy():
    """ERA5/GFS forcing reaches mp=28 too, and gets the same policy."""
    result, _ = _pressure_level_real_init(28)
    state = result.state

    # No analyzed condensate exists on this lane; qc/qr are explicitly zeroed
    # and every other species is at its allocation zero.  That is the same
    # for mp=8 and is not an aerosol question.
    for name in ("qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni"):
        live = _host_array(getattr(state, name))
        assert int(live.view(np.uint32).max()) == 0, name
    assert result.hydrometeor_initialization == {}

    receipt = result.aerosol_initialization
    assert receipt["policy"] == (
        "aer-init-opt-0-zero-then-thompson-init-synthetic-profile")
    assert receipt["awaiting_profile_fill"] is True
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(_host_array(getattr(state, name)).view(
            np.uint32).max()) == 0

    control, _ = _pressure_level_real_init(8)
    assert control.aerosol_initialization == {}


def test_mp28_pressure_level_lane_refuses_an_unreadable_aerosol_source():
    with pytest.raises(NotImplementedError, match="wif_input_opt=1"):
        _pressure_level_real_init(28, wif_input_opt=1)


@requires_gpu
@pytest.mark.gpu
def test_mp28_real_ingest_runs_on_the_production_cuda_preprocessing():
    """The shipped backend, not only the NumPy reference.

    Everything above selects an independent CPU reference implementation of
    the preprocessing ABI so the scheme-selection logic can be tested without
    a GPU.  This one runs the production CUDA vertical interpolation and the
    device state, then the real physics init, so the mp=28 real-data lane is
    proven on the code path a user actually gets.
    """
    import cupy as cp

    from gpuwm.core.physics import initialize_physics
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

    ny, nx, nz = 2, 3, 8
    levels = np.array(
        [100.0, 300.0, 500.0, 700.0, 850.0, 1000.0], dtype=np.float64)
    pressure = np.broadcast_to(
        levels[:, None, None] * 100.0, (levels.size, ny, nx)).copy()
    temperature = np.broadcast_to(
        215.0 + 75.0 * (pressure / 100000.0) ** 0.22, pressure.shape).copy()
    height = 250.0 + np.broadcast_to(
        -7900.0 * np.log(pressure / 100000.0), pressure.shape).copy()
    level_index = np.arange(levels.size, dtype=np.float32)[:, None, None]
    fields = {
        "PRES": pressure.astype(np.float32),
        "SPFH": np.full(pressure.shape, 0.004, dtype=np.float32),
        "TT": temperature.astype(np.float32),
        "GHT": height.astype(np.float32),
        "UU": np.full((levels.size, ny, nx + 1), 8.0, dtype=np.float32),
        "VV": np.full((levels.size, ny + 1, nx), -2.0, dtype=np.float32),
        "PSFC": np.full((ny, nx), 100000.0, dtype=np.float32),
        "T2": np.full((ny, nx), 289.0, dtype=np.float32),
        "Q2": np.full((ny, nx), 0.004, dtype=np.float32),
        "U10": np.full((ny, nx + 1), 7.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -1.0, dtype=np.float32),
    }
    for species_index, name in enumerate(("QC", "QR", "QI", "QS", "QG"), 1):
        fields[name] = np.asarray(
            species_index * 1.0e-6 * (1.0 + level_index)
            * np.ones((1, ny, nx), dtype=np.float32), dtype=np.float32)
    snapshot = HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20, 6), levels_hpa=levels, fields=fields)
    cfg = RunConfig(
        nx=nx, ny=ny, nz=nz, dx=12000.0, dy=12000.0, ztop=18000.0, dt=30.0,
        run_seconds=60.0, hybrid_opt=2, etac=0.2, moist=True, terrain_opt=1,
        mp_physics=28)
    eta = np.linspace(1.0, 0.0, nz + 1)
    terrain = np.full((ny, nx), 250.0, dtype=np.float64)

    backend = resolve_preprocess_backend("cuda")
    assert backend.receipt()["backend"] != "cpu-reference-test"
    result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
        terrain, source_orography=terrain, p_top=10000.0, use_sh_qv=True,
        preprocess_backend="cuda", state_backend="cuda")
    state = result.state

    for name in ("qc", "qr", "qi", "qs", "qg"):
        live = cp.asnumpy(getattr(state, name))
        assert np.isfinite(live).all() and live.min() >= 0.0
        assert np.count_nonzero(live) == live.size
    for name in ("nc", "nr", "ni", "nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert int(cp.asnumpy(getattr(state, name)).view(np.uint32).max()) == 0
    assert result.aerosol_initialization["awaiting_profile_fill"] is True

    driver = initialize_physics(state, cfg)
    assert driver.microphysics_init_receipt == {
        "thompson_aerosol_profile": {"ccn": True, "in": True}}
    nwfa = cp.asnumpy(state.nwfa)
    assert 50.0e6 <= nwfa.min() and nwfa.max() <= 350.0e6
    assert cp.asnumpy(state.nwfa2d).min() > 0.0


def test_mp28_real_ingest_state_is_shaped_typed_and_physical():
    """Task-level acceptance: shapes, dtypes and ranges of the mp=28 state.

    The scheme is only reachable from real initial conditions if the state it
    receives is a real one.  Every field mp=28 adds over mp=8 is checked for
    shape, dtype and value, and the shared thermodynamic state is graded by
    the same discrete moist-hydrostatic residual mp=8 is graded by -- and
    required to be IDENTICAL to it, because the aerosol scheme changes no
    part of the mass/thermodynamic setup.
    """
    result, cfg = _analyzed_hrrr_real_init(28)
    control, _ = _analyzed_hrrr_real_init(8)
    state = result.state
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx

    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                 "nwfa", "nifa"):
        live = _host_array(getattr(state, name))
        assert live.shape == (nz, ny, nx), name
        assert live.dtype == np.float32, name
        assert np.isfinite(live).all(), name
        assert live.min() >= 0.0, name
    for name in ("nwfa2d", "nifa2d"):
        live = _host_array(getattr(state, name))
        assert live.shape == (ny, nx), name
        assert live.dtype == np.float32, name

    # WRF's effective-radius cold start, shared with mp=8
    # (module_model_constants.F RE_QC_BG/RE_QI_BG/RE_QS_BG).
    assert float(_host_array(state.effc).max()) == pytest.approx(2.49)
    assert float(_host_array(state.effi).max()) == pytest.approx(4.99)
    assert float(_host_array(state.effs).max()) == pytest.approx(9.99)

    # Vapour is a real analysis, not a placeholder.
    qv = _host_array(state.qv)
    assert 0.0 < qv.min() and qv.max() < 0.05

    residual = hydrostatic_residual(result)
    control_residual = hydrostatic_residual(control)
    assert np.isfinite(residual).all()
    np.testing.assert_array_equal(residual, control_residual)


#: What the refusing d02 actually held, and what its healthy neighbours
#: held, from the gpuwm 1.8.4 nested HRRR tree at 39.0,-103.0 on cycle
#: 2026-08-08T09.  216x272 cells; exactly two below zero.
_SAN_JUAN_SURFACE_QV = (-1.785645637e-05, -1.478747851e-05)
_SAN_JUAN_SURFACE_QV_MAX = 0.0176926
#: The same tree's ROOT, which passed: its minimum is small but positive,
#: so the floor must leave it alone.  And the flat-terrain Oklahoma d02
#: from the release's completing cell, three orders of magnitude clear.
_COLORADO_ROOT_SURFACE_QV_MIN = 3.043969e-05
_OKLAHOMA_CHILD_SURFACE_QV_MIN = 0.00871458


def test_flag_sh_surface_qv_floor_lifts_the_refusing_colorado_cells():
    """The two real negative cells become WRF's qv_min_value, nothing else.

    These are the values that produced "prepared near-surface surface_qv
    is outside the physical range 0.0..0.2" on the 1.8.4 nested-HRRR tree
    over eastern Colorado.  The floor is WRF's own ``qv_min_value``
    (Registry default 1e-6, module_initialize_real.F:7499-7503) -- the
    same constant :func:`_saturation_mixing_ratio` already applies to the
    RH lane's surface value -- so after it the guard's 0.0 bound cannot be
    crossed by this lane.
    """
    surface_qv = np.full((4, 4), 0.004)
    surface_qv[1, 2] = _SAN_JUAN_SURFACE_QV[0]
    surface_qv[2, 1] = _SAN_JUAN_SURFACE_QV[1]
    surface_qv[0, 0] = _SAN_JUAN_SURFACE_QV_MAX
    psfc = np.full((4, 4), 68_000.0)

    floored, receipt = _floor_flag_sh_surface_mixing_ratio(surface_qv, psfc)

    assert floored.min() == pytest.approx(_WRF_QV_MIN_VALUE)
    assert floored[1, 2] == _WRF_QV_MIN_VALUE
    assert floored[2, 1] == _WRF_QV_MIN_VALUE
    # Every healthy cell is byte-identical, including the domain maximum.
    untouched = np.ones((4, 4), dtype=bool)
    untouched[1, 2] = untouched[2, 1] = False
    np.testing.assert_array_equal(floored[untouched], surface_qv[untouched])
    assert receipt["floored_cells"] == 2
    assert receipt["negative_cells"] == 2
    assert receipt["min_pre_floor"] == pytest.approx(
        min(_SAN_JUAN_SURFACE_QV))
    assert receipt["qv_min_value"] == _WRF_QV_MIN_VALUE


@pytest.mark.parametrize(
    "minimum",
    [_COLORADO_ROOT_SURFACE_QV_MIN, _OKLAHOMA_CHILD_SURFACE_QV_MIN])
def test_flag_sh_surface_qv_floor_is_a_no_op_above_wrf_qv_min(minimum):
    """Shapes that already passed keep every bit and report nothing.

    The Colorado ROOT on the very cycle whose child refused, and the
    Oklahoma child that completed.  Both sit above WRF's floor, so the
    fix cannot move a single existing artifact -- which is the whole
    reason it is safe to apply on a release line.
    """
    surface_qv = np.linspace(minimum, 0.02, 12).reshape(3, 4)
    psfc = np.full((3, 4), 84_000.0)

    floored, receipt = _floor_flag_sh_surface_mixing_ratio(surface_qv, psfc)

    np.testing.assert_array_equal(floored, surface_qv)
    assert receipt == {}


def test_flag_sh_surface_qv_floor_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        _floor_flag_sh_surface_mixing_ratio(
            np.zeros((2, 3)), np.zeros((3, 2)))


def test_rh_lane_surface_qv_already_carries_the_same_floor():
    """The divergence this fix closed, stated as a test.

    GFS/ERA5 build surface_qv through :func:`_saturation_mixing_ratio`,
    which floors at WRF's qv_min_value inline (real.exe rh_to_mxrat1:7379)
    -- so that lane could never present a negative to the prepared
    near-surface guard, and a nested GFS run over the same eastern
    Colorado placement completed while the HRRR one refused.  The FLAG_SH
    lane had no floor at all; now both share this constant.
    """
    bone_dry = _saturation_mixing_ratio(
        np.full((2, 2), 250.0), np.full((2, 2), 68_000.0),
        np.zeros((2, 2)))

    assert np.all(bone_dry == _WRF_QV_MIN_VALUE)
