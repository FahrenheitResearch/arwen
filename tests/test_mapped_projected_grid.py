"""The mapped route's declared projected-source capability.

A projected regular source (HRRR's Lambert CONUS grid; RAP and NAM are the
same family) becomes TABLE WORK: the mapping's ``grid`` block declares the
family and parameters, the decoder cross-checks every observed GRIB grid
octet against the declaration, winds rotate to the earth basis at decode
time, and the target side projects into the source plane through one
transform.  These tests pin that engine against the CERTIFIED native HRRR
route's own projection and rotation math, so the generic path and the
native path cannot silently disagree about where a point is.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm import mapped_source as ms
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import source_coordinate_transform


HRRR_PARAMETERS = {
    "latin1": 38.5, "latin2": 38.5, "lov": 262.5,
    "lat1": 21.138123, "lon1": 237.280472,
    "dx_m": 3000.0, "dy_m": 3000.0, "nx": 1799, "ny": 1059,
    "earth_radius_m": 6371229.0, "shape_of_earth": 6,
}


def _declaration(**overrides):
    parameters = dict(HRRR_PARAMETERS)
    parameters.update(overrides.pop("parameters", {}))
    raw = {
        "family": "lambert_conformal",
        "wind_basis": "grid_relative_with_rotation",
        "parameters": parameters,
    }
    raw.update(overrides)
    return ms._validate_grid_declaration(raw, "grib2")


def test_grid_declaration_normalizes_and_fails_closed():
    declared = _declaration()
    assert declared["family"] == "lambert_conformal"
    assert declared["wind_basis"] == "grid_relative_with_rotation"

    with pytest.raises(ValueError, match="unsupported mapping.grid.family"):
        ms._validate_grid_declaration({"family": "icosahedral"}, "grib2")
    with pytest.raises(ValueError, match="GRIB2 sources only"):
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "wind_basis": "earth_relative",
             "parameters": dict(HRRR_PARAMETERS)},
            "netcdf",
        )
    with pytest.raises(ValueError, match="wind_basis"):
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "parameters": dict(HRRR_PARAMETERS)},
            "grib2",
        )
    with pytest.raises(ValueError, match="missing required key"):
        parameters = dict(HRRR_PARAMETERS)
        del parameters["earth_radius_m"]
        ms._validate_grid_declaration(
            {"family": "lambert_conformal",
             "wind_basis": "earth_relative",
             "parameters": parameters},
            "grib2",
        )
    with pytest.raises(ValueError, match="parameters is not used"):
        ms._validate_grid_declaration(
            {"family": "regular_latitude_longitude",
             "parameters": dict(HRRR_PARAMETERS)},
            "grib2",
        )


def test_declared_lambert_grid_matches_the_native_hrrr_source_grid():
    """The generic table-built grid IS the native route's grid."""

    from gpuwm.ingest.hrrr import hrrr_source_grid

    declared = ms.declared_lambert_source_grid(HRRR_PARAMETERS)
    native = hrrr_source_grid()
    assert declared.stand_lon == native.stand_lon
    assert declared.truelat1 == native.truelat1
    assert declared.ref_lat == native.ref_lat
    assert declared.dx == pytest.approx(native.dx, abs=1e-9)
    lat = np.array([[25.0, 30.0, 35.0], [40.0, 45.0, 47.0]])
    lon = np.array([[-120.0, -110.0, -100.0], [-95.0, -85.0, -75.0]])
    np.testing.assert_allclose(
        declared.latlon_to_ij(lat, lon), native.latlon_to_ij(lat, lon),
        rtol=0.0, atol=1e-8,
    )


def test_declared_rotation_matches_the_native_window_rotation():
    from gpuwm.ingest import hrrr as native

    class _Window:
        i_start = 0
        j_start = 0
        nx = HRRR_PARAMETERS["nx"]
        ny = HRRR_PARAMETERS["ny"]

    sina, cosa = ms._declared_grid_rotation(HRRR_PARAMETERS)
    native_sina, native_cosa = native._source_window_rotation(_Window())
    np.testing.assert_allclose(sina, native_sina, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(cosa, native_cosa, rtol=0.0, atol=1e-12)


def test_observed_grid_cross_check_names_both_vocabularies():
    declared = _declaration()
    row = {
        "index": "7", "nx": "1799", "ny": "1059",
        "lat1": "21.138123", "lon1": "237.280472",
        "dx": "3000", "dy": "3000",
        "latin1": "38.5", "latin2": "38.5", "lov": "262.5",
        "shape_of_earth": "6", "resolution_flags": "0x08",
    }
    ms._require_declared_grib2_grid(row, declared, Path("hrrr.grib2"))

    drifted = dict(row, lov="265.0")
    with pytest.raises(ValueError) as error:
        ms._require_declared_grib2_grid(drifted, declared, Path("hrrr.grib2"))
    assert "declared" in str(error.value) and "observed" in str(error.value)
    assert "265.0" in str(error.value) and "262.5" in str(error.value)

    earth_relative = dict(row, resolution_flags="0x00")
    with pytest.raises(ValueError, match="resolution_flags bit 0x08"):
        ms._require_declared_grib2_grid(
            earth_relative, declared, Path("hrrr.grib2"))


def test_projected_axes_are_regular_far_below_one_period():
    y, x = ms._projected_axes(HRRR_PARAMETERS)
    assert y.size == 1059 and x.size == 1799
    assert x[1] - x[0] == pytest.approx(0.03)
    assert float(x[-1]) < 180.0 and float(y[-1]) < 90.0
    from gpuwm.ingest.horiz import global_longitude_period_columns

    assert global_longitude_period_columns(x) is None


def _lambert_mapping(nx=4, ny=3):
    parameters = dict(
        HRRR_PARAMETERS, nx=nx, ny=ny,
    )
    declared = ms._validate_grid_declaration(
        {
            "family": "lambert_conformal",
            "wind_basis": "grid_relative_with_rotation",
            "parameters": parameters,
        },
        "grib2",
    )
    field = {
        "units": {"source": "m s-1", "target": "m s-1"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "missing": {"kind": "reject"},
    }
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "projected-test",
        "format": "grib2",
        "grid": declared,
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "levels": [100000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": {
            "eastward_wind_10m": {
                **field,
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 2,
                    "parameter": 2, "level_type": 103, "level_value": 10.0,
                }],
            },
            "northward_wind_10m": {
                **field,
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 2,
                    "parameter": 3, "level_type": 103, "level_value": 10.0,
                }],
            },
        },
        "derivations": [],
        "target": {"required_fields": [], "soil_layer_count": None},
    }


def _record(index, parameter, values, latitude, longitude):
    instant = datetime(2026, 8, 15, 0)
    return ms._GribRecord(
        source=Path("hrrr.grib2"), index=index,
        reference_time=instant, valid_time=instant,
        member=None, parameter=parameter, level_type=103, level_value=10.0,
        table_version=None, center=7, subcenter=0,
        master_table_version=2, local_table_version=1,
        discipline=0, category=2,
        second_level_type=255, second_level_value=0.0,
        process_identity=(2, 83), time_semantics=(0,),
        values=values, latitude=latitude, longitude=longitude,
        grid_fingerprint="one-grid",
    )


def test_grid_relative_winds_rotate_to_the_earth_basis_at_decode():
    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    shape = (parameters["ny"], parameters["nx"])
    u_grid = np.full(shape, 5.0)
    v_grid = np.zeros(shape)
    records = [
        _record(0, 2, u_grid, latitude, longitude),
        _record(1, 3, v_grid, latitude, longitude),
    ]
    collection = ms._assemble_grib(mapping, records)
    key_u = (datetime(2026, 8, 15, 0), None, "eastward_wind_10m")
    key_v = (datetime(2026, 8, 15, 0), None, "northward_wind_10m")
    sina, cosa = ms._declared_grid_rotation(parameters)
    np.testing.assert_allclose(
        collection.direct[key_u].values, u_grid * cosa, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        collection.direct[key_v].values, u_grid * sina, rtol=0.0, atol=1e-12)


def test_rotation_refuses_half_a_vector():
    mapping = _lambert_mapping()
    parameters = mapping["grid"]["parameters"]
    latitude, longitude = ms._projected_axes(parameters)
    shape = (parameters["ny"], parameters["nx"])
    records = [_record(0, 2, np.full(shape, 5.0), latitude, longitude)]
    del mapping["fields"]["northward_wind_10m"]
    with pytest.raises(ValueError, match="both components"):
        ms._assemble_grib(mapping, records)


def test_target_transform_round_trips_the_source_grid_points():
    """Source grid points transform to exactly their own axis coordinates."""

    parameters = dict(HRRR_PARAMETERS)
    grid = ms.declared_lambert_source_grid(parameters)
    i = np.array([1.0, 400.0, 1799.0])
    j = np.array([1.0, 500.0, 1059.0])
    lat, lon = grid.ij_to_latlon(*np.meshgrid(i, j))
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=ms._projected_axes(parameters)[0],
        longitude=ms._projected_axes(parameters)[1],
        fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
            },
        },
    )
    transform, projected = source_coordinate_transform(snapshot)
    assert projected
    y_units, x_units = transform(lat, lon)
    expected_x = (i - 1.0) * 3000.0 / ms.PROJECTED_AXIS_UNIT_M
    expected_y = (j - 1.0) * 3000.0 / ms.PROJECTED_AXIS_UNIT_M
    np.testing.assert_allclose(
        x_units, np.meshgrid(expected_x, expected_y)[0], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        y_units, np.meshgrid(expected_x, expected_y)[1], rtol=0.0, atol=1e-8)

    plain = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=np.array([30.0, 31.0]),
        longitude=np.array([-100.0, -99.0]),
        fields={},
    )
    identity, projected = source_coordinate_transform(plain)
    assert not projected
    same_lat, same_lon = identity(lat, lon)
    assert same_lat is lat and same_lon is lon


def test_projected_snapshot_refuses_the_projectionless_npz_archive(tmp_path):
    parameters = dict(HRRR_PARAMETERS)
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 8, 15, 0),
        levels_hpa=np.array([1000.0]),
        latitude=ms._projected_axes(parameters)[0],
        longitude=ms._projected_axes(parameters)[1],
        fields={},
        projection={
            "family": "lambert_conformal",
            "parameters": {
                **parameters, "axis_unit_m": ms.PROJECTED_AXIS_UNIT_M,
            },
        },
    )
    with pytest.raises(ValueError, match="projection"):
        snapshot.save_npz(tmp_path / "snapshot.npz")


def test_declared_level_subset_admits_extra_levels_and_refuses_missing():
    """The GRIB twin of the NetCDF rule: declared levels select by value."""

    mapping = {
        "schema": "rw-wps.mapping.v1", "name": "level-subset",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "levels": [100000.0, 85000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": {
            "air_temperature": {
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 0,
                    "parameter": 0, "level_type": 100,
                }],
                "units": {"source": "K", "target": "K"},
                "source_axes": ["vertical", "y", "x"],
                "target_axes": ["vertical", "y", "x"],
                "location": "mass", "missing": {"kind": "reject"},
            },
        },
        "derivations": [],
        "target": {"required_fields": [], "soil_layer_count": None},
    }
    instant = datetime(2026, 8, 15, 0)
    latitude = np.array([0.0, 1.0])
    longitude = np.array([10.0, 11.0])

    def level_record(index, level_pa):
        return ms._GribRecord(
            source=Path("hrrr.grib2"), index=index,
            reference_time=instant, valid_time=instant,
            member=None, parameter=0, level_type=100,
            level_value=float(level_pa), table_version=None,
            center=7, subcenter=0, master_table_version=2,
            local_table_version=1, discipline=0, category=0,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 83), time_semantics=(0,),
            values=np.full((2, 2), 280.0),
            latitude=latitude, longitude=longitude,
            grid_fingerprint="one-grid",
        )

    # The producer's extra 1013.2 hPa standard-atmosphere record is
    # admitted-and-ignored; both declared levels arrive.
    collection = ms._assemble_grib(mapping, [
        level_record(0, 100000.0), level_record(1, 85000.0),
        level_record(2, 101320.0),
    ])
    key = (instant, None, "air_temperature")
    assert collection.direct[key].values.shape == (2, 2, 2)

    with pytest.raises(ValueError, match="missing"):
        ms._assemble_grib(mapping, [
            level_record(0, 100000.0), level_record(2, 101320.0),
        ])


def test_pdt_selector_key_separates_the_accumulation_twin():
    selector = {
        "format": "grib2", "discipline": 0, "category": 1, "parameter": 13,
        "level_type": 1, "pdt": 0,
    }
    instant = datetime(2026, 8, 15, 0)

    def twin(index, pdt):
        return ms._GribRecord(
            source=Path("hrrr.grib2"), index=index,
            reference_time=instant, valid_time=instant,
            member=None, parameter=13, level_type=1, level_value=0.0,
            table_version=None, center=7, subcenter=0,
            master_table_version=2, local_table_version=1,
            discipline=0, category=1,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 83), time_semantics=(pdt,),
            values=np.zeros((2, 2)),
            latitude=np.array([0.0, 1.0]), longitude=np.array([10.0, 11.0]),
            grid_fingerprint="one-grid",
        )

    assert ms._selector_matches_record(selector, twin(0, 0), "grib2")
    assert not ms._selector_matches_record(selector, twin(1, 8), "grib2")


def test_load_mapping_accepts_and_normalizes_the_grid_block(tmp_path):
    import json

    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (root / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json")
        .read_text(encoding="utf-8"))
    raw["grid"] = {
        "family": "lambert_conformal",
        "wind_basis": "grid_relative_with_rotation",
        "parameters": dict(HRRR_PARAMETERS),
    }
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = ms.load_mapping(path)
    assert loaded["grid"]["family"] == "lambert_conformal"
    assert loaded["grid"]["parameters"]["nx"] == 1799
    assert loaded["grid"]["wind_basis"] == "grid_relative_with_rotation"

    raw["grid"] = {"family": "regular_latitude_longitude"}
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = ms.load_mapping(path)
    assert loaded["grid"]["family"] == "regular_latitude_longitude"
    assert loaded["grid"]["wind_basis"] == "earth_relative"
