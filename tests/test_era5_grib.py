"""Phase 3 Task 6: ERA5 GRIB1 decode and Vtable-exact field mapping.

Authority is the bundle ``Vtable.ERA5_CDO`` and its combined/per-time GRIB1
files.  The bundled CDO-produced NetCDF files are an independent value and
orientation oracle for the all-Rust bridge.
"""
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.grib import (
    Era5Snapshot,
    build_rust_bridge,
    decode_era5_grib,
    parse_vtable,
)


BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
ERA5 = BUNDLE / "era5_grib"
COMBINED = ERA5 / "era5_19740403.grb"
VTABLE = ERA5 / "Vtable.ERA5_CDO"
NC_DIR = ERA5 / "nc"

requires_bundle = pytest.mark.skipif(
    not COMBINED.is_file() or not VTABLE.is_file(),
    reason="WRF_1974_MP55 ERA5 reference bundle not present",
)

# The source contains every parameter-coded Vtable field below except HGT,
# SOILGEO, SOILHGT, PMSL, and SEAICE.  Blank-param rows (derived 2-m RH and
# SNOW) are not decode inventory.  The CDO-only parameter 2 ``utc_date`` is
# deliberately excluded because it is not in the Vtable.
EXPECTED_FIELDS = frozenset(
    {
        "Z", "T", "U", "V", "RH",
        "U10", "V10", "T2", "D2", "LANDSEA", "PSFC",
        "SKINTEMP", "SST", "SNOW_EC",
        "ST000007", "ST007028", "ST028100", "ST100289",
        "SM000007", "SM007028", "SM028100", "SM100289",
    }
)

ORACLE_NAMES = {
    "Z": "Z",
    "T": "T",
    "U": "U",
    "V": "V",
    "RH": "R",
    "PSFC": "SP",
    "SNOW_EC": "SD",
    "U10": "VAR_10U",
    "V10": "VAR_10V",
    "T2": "VAR_2T",
    "D2": "VAR_2D",
    "ST000007": "STL1",
    "ST007028": "STL2",
    "ST028100": "STL3",
    "ST100289": "STL4",
    "SM000007": "SWVL1",
    "SM007028": "SWVL2",
    "SM028100": "SWVL3",
    "SM100289": "SWVL4",
    "SKINTEMP": "SKT",
    "SST": "SSTK",
    "LANDSEA": "LSM",
}

# CDO's NetCDF writer applies a field-wide float32 scale/offset transform, so
# decoding the original GRIB simple packing is slightly more accurate than the
# oracle arrays.  These bounds are at most a few float32 quantization steps and
# many orders tighter than downstream metgrid interpolation gates.
ORACLE_ATOL = {
    "Z": 0.05,
    "PSFC": 0.05,
    "T": 2.5e-4,
    "T2": 2.5e-4,
    "D2": 2.5e-4,
    "ST000007": 2.5e-4,
    "ST007028": 2.5e-4,
    "ST028100": 2.5e-4,
    "ST100289": 2.5e-4,
    "SKINTEMP": 2.5e-4,
    "SST": 2.5e-4,
    "RH": 1.0e-5,
    "SNOW_EC": 1.0e-5,
    "SM000007": 4.0e-8,
    "SM007028": 4.0e-8,
    "SM028100": 4.0e-8,
    "SM100289": 4.0e-8,
    "LANDSEA": 2.0e-5,
    "U": 1.0e-7,
    "V": 1.0e-7,
    "U10": 1.0e-7,
    "V10": 1.0e-7,
}


def test_parse_vtable_preserves_grib1_keys_and_names(tmp_path):
    table = tmp_path / "Vtable"
    table.write_text(
        "GRIB1| Level| From | To | metgrid | metgrid | Description |GRIB2|\n"
        "Param| Type |Level1|Level2| Name   | Units   |             |Discp|\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n"
        " 129 | 100  |   *  |      | GEOPT  | m2 s-2  | geopotential|  0  |\n"
        " 165 |  1   |   0  |      | UU     | m s-1   | 10 m U     |  0  |\n"
        "     |  1   |   0  |      | RH     | %       | derived     |  0  |\n"
        "-----+------+------+------+--------+---------+-------------+-----+\n",
        encoding="utf-8",
    )
    entries = parse_vtable(table)
    assert len(entries) == 3
    assert (entries[0].parameter, entries[0].level_type) == (129, 100)
    assert (entries[0].level1, entries[0].name, entries[0].units) == (
        "*", "GEOPT", "m2 s-2"
    )
    assert entries[1].name == "UU"
    assert entries[2].parameter is None and entries[2].name == "RH"


@requires_bundle
def test_bundle_vtable_contains_required_parameter_mappings():
    by_key = {
        (e.parameter, e.level_type): e.name
        for e in parse_vtable(VTABLE)
        if e.parameter is not None
    }
    expected = {
        (129, 100): "GEOPT", (130, 100): "TT", (131, 100): "UU",
        (132, 100): "VV", (157, 100): "RH", (165, 1): "UU",
        (166, 1): "VV", (167, 1): "TT", (168, 1): "DEWPT",
        (172, 1): "LANDSEA", (134, 1): "PSFC", (235, 1): "SKINTEMP",
        (34, 1): "SST", (141, 1): "SNOW_EC", (139, 1): "ST000007",
        (170, 1): "ST007028", (183, 1): "ST028100",
        (236, 1): "ST100289", (39, 1): "SM000007",
        (40, 1): "SM007028", (41, 1): "SM028100",
        (42, 1): "SM100289",
    }
    assert expected.items() <= by_key.items()


def _synthetic_snapshot():
    levels = np.array([1000.0, 850.0], dtype=np.float64)
    latitude = np.array([25.0, 25.25], dtype=np.float64)
    longitude = np.array([250.0, 250.25, 250.5], dtype=np.float64)
    return Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12),
        levels_hpa=levels,
        latitude=latitude,
        longitude=longitude,
        fields={
            "T": np.arange(12, dtype=np.float64).reshape(2, 2, 3) + 250.0,
            "PSFC": np.arange(6, dtype=np.float64).reshape(2, 3) + 90000.0,
        },
    )


def test_snapshot_npz_round_trip_without_pickle(tmp_path):
    original = _synthetic_snapshot()
    path = tmp_path / "snapshot.npz"
    original.save_npz(path)
    restored = Era5Snapshot.load_npz(path)
    assert restored.valid_time == original.valid_time
    np.testing.assert_array_equal(restored.levels_hpa, original.levels_hpa)
    np.testing.assert_array_equal(restored.latitude, original.latitude)
    np.testing.assert_array_equal(restored.longitude, original.longitude)
    assert restored.fields.keys() == original.fields.keys()
    for name in original.fields:
        np.testing.assert_array_equal(restored.fields[name], original.fields[name])
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "valid_time", "levels_hpa", "latitude", "longitude",
            "field_names", "field__T", "field__PSFC",
        }


def test_snapshot_rejects_inconsistent_shapes_and_non_float64():
    snap = _synthetic_snapshot()
    with pytest.raises(ValueError, match="T.*shape"):
        Era5Snapshot(
            snap.valid_time, snap.levels_hpa, snap.latitude, snap.longitude,
            {"T": np.zeros((2, 3, 2), dtype=np.float64)},
        )
    with pytest.raises(TypeError, match="float64"):
        Era5Snapshot(
            snap.valid_time, snap.levels_hpa, snap.latitude, snap.longitude,
            {"PSFC": np.zeros((2, 3), dtype=np.float32)},
        )


@lru_cache(maxsize=1)
def _decode_bundle_cached():
    bridge = build_rust_bridge(release=True)
    return decode_era5_grib(COMBINED, VTABLE, bridge=bridge)


@requires_bundle
def test_rust_bridge_decodes_grib1_and_discovers_all_times():
    snapshots = _decode_bundle_cached()
    assert tuple(s.valid_time for s in snapshots) == (
        datetime(1974, 4, 3, 12),
        datetime(1974, 4, 3, 18),
        datetime(1974, 4, 4, 0),
    )
    for snapshot in snapshots:
        assert snapshot.fields.keys() == EXPECTED_FIELDS


@requires_bundle
def test_bundle_grid_levels_shapes_and_ranges():
    snapshots = _decode_bundle_cached()
    expected_levels = np.array(
        [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
         200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
         750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000],
        dtype=np.float64,
    )
    for snapshot in snapshots:
        np.testing.assert_array_equal(snapshot.levels_hpa, expected_levels)
        np.testing.assert_allclose(snapshot.latitude, np.arange(25, 55.01, 0.25),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(snapshot.longitude, np.arange(250, 300.01, 0.25),
                                   rtol=0.0, atol=1e-12)
        for name in ("Z", "T", "U", "V", "RH"):
            assert snapshot.fields[name].shape == (37, 121, 201)
        for name in EXPECTED_FIELDS - {"Z", "T", "U", "V", "RH"}:
            assert snapshot.fields[name].shape == (121, 201)
        assert 180.0 <= np.nanmin(snapshot.fields["T"])
        assert np.nanmax(snapshot.fields["T"]) <= 330.0
        assert 60000.0 <= np.nanmin(snapshot.fields["PSFC"])
        assert np.nanmax(snapshot.fields["PSFC"]) <= 110000.0
        assert np.nanmin(snapshot.fields["LANDSEA"]) >= 0.0
        assert np.nanmax(snapshot.fields["LANDSEA"]) <= 1.0


def _oracle(path):
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as ds:
        out = {}
        for canonical, nc_name in ORACLE_NAMES.items():
            values = np.ma.asarray(ds.variables[nc_name][:]).squeeze(axis=0)
            out[canonical] = np.asarray(
                np.ma.filled(values, np.nan), dtype=np.float32
            )
        return out


@requires_bundle
@pytest.mark.parametrize(
    "index,name",
    [(0, "era5_19740403_12.nc"), (1, "era5_19740403_18.nc"),
     (2, "era5_19740404_00.nc")],
)
def test_rust_decode_matches_cdo_netcdf_oracle_within_quantization(index, name):
    snapshot = _decode_bundle_cached()[index]
    oracle = _oracle(NC_DIR / name)
    for field, expected in oracle.items():
        actual = snapshot.fields[field]
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected),
                                      err_msg=f"{field} missing mask")
        np.testing.assert_allclose(actual, expected, rtol=0.0,
                                   atol=ORACLE_ATOL[field], equal_nan=True,
                                   err_msg=field)


@requires_bundle
def test_real_snapshot_npz_round_trip(tmp_path):
    original = _decode_bundle_cached()[0]
    path = tmp_path / "era5_19740403_12.npz"
    original.save_npz(path)
    restored = Era5Snapshot.load_npz(path)
    assert restored.valid_time == original.valid_time
    assert restored.fields.keys() == original.fields.keys()
    for name in original.fields:
        np.testing.assert_array_equal(restored.fields[name], original.fields[name])


@requires_bundle
@pytest.mark.parametrize(
    "index,name",
    [(0, "era5_19740403_12.grb"), (1, "era5_19740403_18.grb"),
     (2, "era5_19740404_00.grb")],
)
def test_per_time_grib_files_match_combined_inventory_and_values(index, name):
    bridge = build_rust_bridge(release=True)
    (single,) = decode_era5_grib(ERA5 / "grb" / name, VTABLE, bridge=bridge)
    combined = _decode_bundle_cached()[index]
    assert single.valid_time == combined.valid_time
    assert single.fields.keys() == EXPECTED_FIELDS
    for field in EXPECTED_FIELDS:
        np.testing.assert_array_equal(single.fields[field], combined.fields[field])
