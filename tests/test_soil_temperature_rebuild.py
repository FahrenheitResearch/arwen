"""real.exe's TSLB reasonableness rebuild, on every soil route.

HRRRv2 analyses (January 2017, western snowpack) carry land soil
temperatures of 60 to 168 K beside a skin near 270 K.  Both HRRR routes
refused every preparation whose domain reached them: the pressure-level
mapped route with "declarative mapped soil temperature is missing or
outside 170..400 K on land", the native ``--source hrrr`` route at its
source admission.  real.exe rebuilds such a land column linear in depth
from TSK at 0 m to TMN at 3 m and keeps its moisture
(dyn_em/module_initialize_real.F:3536-3595); gpuwm now does the same,
with WRF's ``tmn*(0-zs)`` sign corrected, and still refuses a land column
with a missing sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.ingest.hrrr import (
    _record_soil_field_stats, _require_source_physical_ranges)
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.ingest.soil import (
    HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M, preprocess_noah_soil,
    tsk_tmn_soil_profile, unreasonable_land_soil_columns)
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)

_SHAPE = (3, 4)
_LAND = np.array([[1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0]])
#: The measured shape: a snowpack land column whose top node reads 64 K.
_BAD = (1, 2)


def _selector(name, depth):
    temperature = name == "soil_temperature"
    return {
        "format": "grib2", "discipline": 2, "category": 3 if temperature else 0,
        "parameter": 18 if temperature else 192, "center": 7, "subcenter": 0,
        "master_table_version": 2, "local_table_version": 1,
        "level_type": 106, "level_value": depth,
        "second_level_type": 106, "second_level_value": depth,
    }


def _contract():
    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {"depth": float(depth), "selectors": {
                "soil_temperature": _selector("soil_temperature", float(depth)),
                "volumetric_soil_moisture": _selector(
                    "volumetric_soil_moisture", float(depth))}}
            for depth in HRRR_SOIL_NODE_DEPTHS_M],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0}],
        "remap": {"kind": "linear_node_samples",
                  "source_value_location": "level_node",
                  "target_value_location": "layer_midpoint"},
        "missing": {"land": "reject", "ocean": {
            "stage": "after_horizontal_interpolation",
            "temperature": "skin_temperature", "moisture": 1.0}},
    }


def _nodes(*, bad=True):
    temperature = np.broadcast_to(
        np.linspace(268.0, 276.0, 9)[:, None, None], (9,) + _SHAPE).copy()
    moisture = np.full((9,) + _SHAPE, 0.25)
    moisture[:, _BAD[0], _BAD[1]] = np.linspace(0.20, 0.32, 9)
    if bad:
        temperature[0, _BAD[0], _BAD[1]] = 64.0
        temperature[1, _BAD[0], _BAD[1]] = 150.0
    return temperature, moisture


def _surface():
    return {
        "LANDSEA": _LAND,
        "SKINTEMP": np.full(_SHAPE, 271.0),
        "TMN": np.full(_SHAPE, 277.0),
    }


def _mapped_fields(*, bad=True):
    temperature, moisture = _nodes(bad=bad)
    return {**_surface(), MAPPED_SOIL_TEMPERATURE: temperature,
            MAPPED_SOIL_MOISTURE: moisture}


def _native_fields(*, bad=True):
    temperature, moisture = _nodes(bad=bad)
    return {**_surface(), "SOILT": temperature, "SOILW": moisture}


def _prepare(fields, scheme=2, **kwargs):
    contract = _contract() if MAPPED_SOIL_TEMPERATURE in fields else None
    return preprocess_land_surface_soil(
        fields, sf_surface_physics=scheme, soil_type=np.full(_SHAPE, 6),
        soil_layer_contract=contract, landmask=_LAND, **kwargs)


def test_columns_are_land_only_any_sample_and_never_a_missing_one():
    temperature = np.full((3, 2, 2), 280.0)
    temperature[2, 0, 0] = 150.0      # a deep sample only, on land: rebuilt
    temperature[0, 0, 1] = 60.0       # on land, but a sample is missing:
    temperature[1, 0, 1] = np.nan     # the refusal keeps it
    temperature[0, 1, 1] = 50.0       # water
    land = np.array([[True, True], [True, False]])
    columns = unreasonable_land_soil_columns(temperature, land)
    assert columns.tolist() == [[True, False], [False, False]]


def test_profile_runs_linear_from_tsk_at_0_m_to_tmn_at_3_m():
    profile = tsk_tmn_soil_profile(
        [0.0, 1.5, 3.0], np.array([[270.0]]), np.array([[282.0]]))
    np.testing.assert_allclose(profile[:, 0, 0], [270.0, 276.0, 282.0])


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_an_unreasonable_land_column_is_rebuilt_on_noah_layers(route, capsys):
    fields = _mapped_fields() if route == "mapped" else _native_fields()
    state = _prepare(fields)
    expected = tsk_tmn_soil_profile(
        NOAH_LAYER_MIDPOINTS_M, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]], expected,
        rtol=0.0, atol=1.0e-12)
    receipt = state.soil_temperature_repair
    assert receipt["repaired_land_columns"] == 1
    assert receipt["land_cells"] == int(np.count_nonzero(_LAND))
    assert receipt["samples_outside_band"] == 2
    assert receipt["pre_repair_min_k"] == 64.0
    assert receipt["bounding_box"] == {"rows": [1, 1], "columns": [2, 2]}
    # Its moisture is kept: the ordinary node interpolation of its own
    # samples, as on every other column.
    healthy = _prepare(
        _mapped_fields(bad=False) if route == "mapped"
        else _native_fields(bad=False))
    np.testing.assert_array_equal(
        state.soil_moisture, healthy.soil_moisture)
    # Every other column is untouched.
    others = np.ones(_SHAPE, dtype=bool)
    others[_BAD] = False
    np.testing.assert_array_equal(
        state.soil_temperature[:, others], healthy.soil_temperature[:, others])
    err = capsys.readouterr().err
    assert ("soil temperature rebuild: 1 of 10 land column(s) carried a "
            "source soil temperature outside 170..400 K (64..276 K, rows "
            "1..1 and columns 2..2 of the 3x4 grid)") in err
    assert ("following WRF real.exe's rebuild with its deep-temperature "
            "sign corrected") in err


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_a_healthy_source_carries_no_receipt_and_says_nothing(route, capsys):
    fields = (_mapped_fields(bad=False) if route == "mapped"
              else _native_fields(bad=False))
    state = _prepare(fields)
    assert state.soil_temperature_repair == {}
    assert "soil temperature rebuild" not in capsys.readouterr().err


@pytest.mark.parametrize("route", ["mapped", "native"])
def test_an_unreasonable_land_column_is_rebuilt_on_ruc_levels(route):
    fields = _mapped_fields() if route == "mapped" else _native_fields()
    state = _prepare(fields, scheme=3)
    expected = tsk_tmn_soil_profile(
        state.level_depths, np.full(_SHAPE, 271.0),
        np.full(_SHAPE, 277.0))[:, _BAD[0], _BAD[1]]
    np.testing.assert_allclose(
        state.soil_temperature[:, _BAD[0], _BAD[1]],
        expected.astype(state.soil_temperature.dtype), rtol=0.0, atol=0.0)
    assert state.soil_temperature_repair["repaired_land_columns"] == 1
    assert float(np.min(state.soil_temperature)) > 170.0


def test_a_missing_land_soil_temperature_is_still_refused():
    fields = _mapped_fields()
    fields[MAPPED_SOIL_TEMPERATURE][3, _BAD[0], _BAD[1]] = np.nan
    with pytest.raises(ValueError, match="missing or outside 170..400 K"):
        _prepare(fields)
    fields = _native_fields(bad=False)
    fields["SOILT"][3, 0, 0] = np.nan
    with pytest.raises(ValueError, match="HRRR SOILT nodes are non-finite"):
        _prepare(fields)


def test_the_native_admission_leaves_the_band_to_the_soil_initializer():
    source = {
        "LANDSEA": np.ones((2, 2)), "SOILW": np.full((9, 2, 2), 0.3),
        "SPFH": np.full((3, 2, 2), 0.005), "Q2": np.full((2, 2), 0.005),
        "SOILT": np.full((9, 2, 2), 275.0),
    }
    source["SOILT"][0, 1, 1] = 64.0
    _require_source_physical_ranges(source)
    # Off land the band still refuses: nothing rebuilds a water column.
    source["LANDSEA"][1, 1] = 0.0
    with pytest.raises(ValueError, match="outside 170..400 K on 1 source open-water"):
        _require_source_physical_ranges(source)
    source["LANDSEA"][1, 1] = 1.0
    source["SOILT"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="SOILT is non-finite"):
        _require_source_physical_ranges(source)


def test_the_proof_receipt_adds_the_box_in_degrees_and_is_absent_when_healthy():
    from types import SimpleNamespace

    from gpuwm.ingest.soil import (
        soil_temperature_repair_proof, soil_temperature_repair_receipt)

    temperature, _ = _nodes()
    land = _LAND.astype(bool)
    columns = unreasonable_land_soil_columns(temperature, land)
    latitude = np.broadcast_to(np.array([43.0, 44.0, 45.0])[:, None], _SHAPE)
    longitude = np.broadcast_to(
        np.array([-118.0, -117.0, -116.0, -115.0])[None, :], _SHAPE)
    grid = SimpleNamespace(latlon_mass=lambda: (latitude, longitude))
    receipt = soil_temperature_repair_proof(SimpleNamespace(
        soil_temperature_repair=soil_temperature_repair_receipt(
            temperature, columns, land)), grid)
    assert receipt["bounding_box"] == {
        "rows": [1, 1], "columns": [2, 2],
        "latitude": [44.0, 44.0], "longitude": [-116.0, -116.0]}
    assert soil_temperature_repair_proof(
        SimpleNamespace(soil_temperature_repair={}), grid) is None


def test_the_native_mapping_admits_land_outside_the_band_and_counts_it():
    source = np.full((9, 3, 3), 275.0)
    source[0, 1, 1] = 64.0
    source_land = np.ones((3, 3), dtype=bool)
    candidate = np.full((9, 2, 2), 275.0)
    candidate[0, 0, 0] = 120.0
    target_land = np.array([[True, True], [True, False]])
    report = {}
    _record_soil_field_stats(
        report, "SOILT", source, candidate, source_land, target_land,
        (170.0, 400.0), land_columns_rebuilt=True)
    assert report["SOILT"][
        "target_land_columns_outside_limits_rebuilt_tsk_to_tmn"] == 1
    # Off land the band still refuses, and so does a field that has not
    # opted in (soil moisture).
    candidate[0, 1, 1] = 120.0
    with pytest.raises(ValueError, match="non-finite or outside 170.0..400.0"):
        _record_soil_field_stats(
            {}, "SOILT", source, candidate, source_land, target_land,
            (170.0, 400.0), land_columns_rebuilt=True)
    candidate[0, 1, 1] = 275.0
    with pytest.raises(ValueError, match="non-finite or outside 170.0..400.0"):
        _record_soil_field_stats(
            {}, "SOILT", source, candidate, source_land, target_land,
            (170.0, 400.0))
    healthy = {}
    candidate[0, 0, 0] = 275.0
    _record_soil_field_stats(
        healthy, "SOILT", source, candidate, source_land, target_land,
        (170.0, 400.0), land_columns_rebuilt=True)
    assert not any("rebuilt" in key for key in healthy["SOILT"])
