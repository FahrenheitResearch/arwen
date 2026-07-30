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
    _cap_stratospheric_qv,
    _mixing_ratio_to_relative_humidity,
    _saturation_mixing_ratio,
    _specific_humidity_to_mixing_ratio,
    _wrf_flag_sh_surface_specific_humidity,
    hydrostatic_residual,
    initialize_real,
    surface_pressure_from_surface,
)
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.soil import preprocess_noah_soil
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
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography,
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

    default_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography)
    direct_result = initialize_real(
        snapshot, cfg,
        make_vertical_coord(cfg.nz, hybrid_opt=2, etac=0.2,
                            eta_levels=eta),
        terrain, source_orography=source_orography, use_sh_qv=True)
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
            terrain, source_orography=source_orography)


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
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, **options)

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
        return initialize_real(
            snapshot, cfg,
            make_vertical_coord(
                cfg.nz, hybrid_opt=2, etac=0.2, eta_levels=eta),
            terrain, source_orography=source_orography, use_sh_qv=True,
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


def test_retired_real74_d01_30min_gate_rejects_incomplete_no_pbl_operator():
    """The old dynamics-only acceptance path cannot claim WRF km_opt=4.

    That path disabled PBL physics and therefore lacks WRF's required
    ``vertical_diffusion_2`` contribution.  Keep the former 30-minute entry
    point as an explicit rejection contract; the phase3 configuration is the
    supported real74 production path and retains PBL physics.
    """
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases.real74_d01 import (build_forcing,
                                               phase3_config)

    with pytest.raises(NotImplementedError,
                       match=r"km_opt=4.*bl_pbl_physics=0.*vertical"):
        build_forcing(run_seconds=1800.0)

    supported = validate_run_config(phase3_config(run_seconds=1800.0))
    assert supported.km_opt == 4
    assert supported.bl_pbl_physics == 1
