"""Geopotential height on a pressure-level source that leaves it out.

A pressure-level source that publishes no geopotential height on some or
all of its levels used to refuse the whole preparation with ``mapped
frame at <time> lacks required fields ['geopotential_height']``.  Two
things now hold for every pressure-level mapping, with no per-source
code:

* a field may list a second publication of the same quantity after its
  first selector (geopotential beside geopotential height, with the
  ``scale`` that takes it to metres), and a record of that selector is
  used only at a level the first one did not publish;
* where neither is published, the frame integrates the hypsometric
  equation from its own temperature, humidity, surface pressure and
  terrain height, anchored on the nearest level of the same column that
  carries the source's own height, and says how many values it derived.

The Python engine is driven directly on decoded records here.  The Rust
engine runs the same operation, and its unit tests in ``derive.rs`` pin
the same isothermal answers.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gpuwm import mapped_source as ms


T0 = datetime(2025, 3, 15, 0)
LEVELS = (100000.0, 85000.0, 50000.0)
GRAVITY = 9.80665
RD = 287.06
VIRTUAL = 0.609133
TEMPERATURE = 250.0
HUMIDITY = 0.004
SURFACE_PRESSURE = 95000.0
TERRAIN = 500.0

HEIGHT_SELECTOR = {"format": "grib2", "discipline": 0, "category": 3,
                   "parameter": 5, "level_type": 100}
GEOPOTENTIAL_SELECTOR = {"format": "grib2", "discipline": 0, "category": 3,
                         "parameter": 4, "level_type": 100,
                         "scale": 1.0 / GRAVITY}
#: The rest of the initialization state, constant and beside the point
#: here: (name, axes, location, level type), each under its own local
#: parameter so no record can answer another field's selector.
OTHER_FIELDS = (
    ("eastward_wind", ["vertical", "y", "x"], "mass", 100),
    ("northward_wind", ["vertical", "y", "x"], "mass", 100),
    ("skin_temperature", ["y", "x"], "surface", 1),
    ("air_temperature_2m", ["y", "x"], "surface", 1),
    ("specific_humidity_2m", ["y", "x"], "surface", 1),
    ("eastward_wind_10m", ["y", "x"], "surface", 1),
    ("northward_wind_10m", ["y", "x"], "surface", 1),
    ("land_fraction", ["y", "x"], "surface", 1),
    ("soil_temperature", ["soil", "y", "x"], "soil", 106),
    ("volumetric_soil_moisture", ["soil", "y", "x"], "soil", 106),
)


def _isothermal_height(pressure: float) -> float:
    """The exact height of ``pressure`` in an isothermal column."""

    virtual = TEMPERATURE * (1.0 + VIRTUAL * HUMIDITY)
    return TERRAIN + RD / GRAVITY * virtual * math.log(SURFACE_PRESSURE / pressure)


def _field(selectors, units, axes, location):
    return {
        "selectors": selectors,
        "units": {"source": units, "target": units},
        "source_axes": axes, "target_axes": axes,
        "location": location, "staggering": "none",
        "missing": {"kind": "reject"},
    }


def _mapping(*, geopotential: bool = False) -> dict:
    column = ["vertical", "y", "x"]
    plane = ["y", "x"]
    height = [HEIGHT_SELECTOR] + ([GEOPOTENTIAL_SELECTOR] if geopotential else [])
    fields = {
        "geopotential_height": _field(height, "m", column, "mass"),
        "air_temperature": _field(
            [{"format": "grib2", "discipline": 0, "category": 0,
              "parameter": 0, "level_type": 100}], "K", column, "mass"),
        "specific_humidity": _field(
            [{"format": "grib2", "discipline": 0, "category": 1,
              "parameter": 0, "level_type": 100}], "kg kg-1", column, "mass"),
        "surface_pressure": _field(
            [{"format": "grib2", "discipline": 0, "category": 3,
              "parameter": 0, "level_type": 1}], "Pa", plane, "surface"),
        "terrain_height": _field(
            [{"format": "grib2", "discipline": 0, "category": 3,
              "parameter": 4, "level_type": 1}], "m", plane, "surface"),
        "air_pressure": {
            **_field([], "Pa", column, "mass"),
            "derivation": "pressure-from-coordinate",
        },
    }
    for parameter, (name, axes, location, level_type) in enumerate(OTHER_FIELDS):
        fields[name] = _field(
            [{"format": "grib2", "discipline": 0, "category": 200,
              "parameter": parameter, "level_type": level_type}],
            "1", axes, location)
    return {
        "schema": "rw-wps.mapping.v1", "name": "pressure-height-probe",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "positive": "down", "levels": list(LEVELS)},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": [{"name": "pressure-from-coordinate",
                         "operation": "pressure_from_vertical_coordinate"}],
        "target": {"required_fields": [
            {"name": name, "axes": field["target_axes"],
             "location": field["location"],
             "target_units": field["units"]["target"]}
            for name, field in fields.items() if name != "air_pressure"
        ], "soil_layer_count": 1, "initialization_policies": {
            name: "explicit_zero_with_adapter_validation"
            for name in (
                "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
                "cloud_ice_mixing_ratio", "snow_mixing_ratio",
                "graupel_or_hail_mixing_ratio", "vertical_velocity",
                "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
            )
        }},
    }


class _Records:
    def __init__(self):
        self.items: list[ms._GribRecord] = []

    def add(self, category, parameter, level_type, value, level=0.0):
        self.items.append(ms._GribRecord(
            source=Path("probe.grib2"), index=len(self.items),
            reference_time=T0, valid_time=T0, member=None,
            parameter=parameter, level_type=level_type, level_value=level,
            table_version=None, center=98, subcenter=0,
            master_table_version=36, local_table_version=0,
            discipline=0, category=category,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 5), time_semantics=(0,),
            values=np.full((2, 2), float(value)),
            latitude=np.array([0.0, 1.0]), longitude=np.array([10.0, 11.0]),
            grid_fingerprint="one-grid",
        ))


def _state(heights=(), geopotentials=()) -> _Records:
    """An isothermal column with heights published at the given levels."""

    records = _Records()
    for level in LEVELS:
        records.add(0, 0, 100, TEMPERATURE, level)
        records.add(1, 0, 100, HUMIDITY, level)
    records.add(3, 0, 1, SURFACE_PRESSURE)
    records.add(3, 4, 1, TERRAIN)
    for parameter, (_name, axes, _location, level_type) in enumerate(OTHER_FIELDS):
        for level in (LEVELS if "vertical" in axes else (0.0,)):
            records.add(200, parameter, level_type, 0.5, level)
    for level, value in heights:
        records.add(3, 5, 100, value, level)
    for level, value in geopotentials:
        records.add(3, 4, 100, value, level)
    return records


def _frame(mapping, records):
    collection = ms._assemble_grib(mapping, records.items)
    frames = ms._materialize_frames(
        mapping, collection, mapping_sha256="0" * 64, input_sha256={})
    assert len(frames) == 1
    return frames


def _heights(frames) -> np.ndarray:
    field = frames[0].fields["geopotential_height"]
    assert field.axes == ("vertical", "y", "x")
    return field.values


def test_a_source_with_no_height_is_completed_from_its_own_column(capsys):
    frames = _frame(_mapping(), _state())

    heights = _heights(frames)
    for index, level in enumerate(LEVELS):
        np.testing.assert_allclose(
            heights[index], _isothermal_height(level), rtol=0, atol=1e-9)
    field = frames[0].fields["geopotential_height"]
    assert field.missing_count == 0
    assert field.source_references[0] == "@completed.hypsometric:12"
    # Every operand the integration read is named after the completion.
    assert "probe.grib2:0" in field.source_references

    summary = ms.completed_field_summary(frames)
    assert summary == {"geopotential_height": {
        "method": "hypsometric", "frames": 1, "frame_count": 1,
        "values_derived": 12, "values_total": 12,
    }}
    ms.warn_completed_fields(summary, subject="the source")
    warning = capsys.readouterr().err
    assert "did not carry geopotential height for 12 of 12 values" in warning


def test_published_heights_are_kept_and_anchor_the_missing_levels():
    published = 5600.0
    frames = _frame(_mapping(), _state(heights=[(50000.0, published)]))

    heights = _heights(frames)
    np.testing.assert_array_equal(heights[2], published)
    # Both missing levels are placed from the source's own 500 hPa height,
    # not from the surface, so the column keeps the source's offset.
    offset = published - _isothermal_height(50000.0)
    for index in (0, 1):
        np.testing.assert_allclose(
            heights[index], _isothermal_height(LEVELS[index]) + offset,
            rtol=0, atol=1e-9)
    field = frames[0].fields["geopotential_height"]
    assert field.source_references[0] == "@completed.hypsometric:8"
    assert ms.completed_field_summary(frames)["geopotential_height"][
        "values_derived"] == 8


def test_a_source_that_publishes_every_height_derives_nothing():
    frames = _frame(_mapping(), _state(
        heights=[(level, 1000.0 + index) for index, level in enumerate(LEVELS)]))

    heights = _heights(frames)
    for index in range(len(LEVELS)):
        np.testing.assert_array_equal(heights[index], 1000.0 + index)
    field = frames[0].fields["geopotential_height"]
    assert not any(reference.startswith("@completed.")
                   for reference in field.source_references)
    assert ms.completed_field_summary(frames) == {}


def test_published_geopotential_stands_in_for_height_scaled_to_metres():
    records = _state(geopotentials=[(level, GRAVITY * (2000.0 + index))
                                    for index, level in enumerate(LEVELS)])
    frames = _frame(_mapping(geopotential=True), records)

    heights = _heights(frames)
    for index in range(len(LEVELS)):
        np.testing.assert_allclose(heights[index], 2000.0 + index,
                                   rtol=0, atol=1e-9)
    assert ms.completed_field_summary(frames) == {}
    references = frames[0].fields["geopotential_height"].source_references
    assert references == tuple(
        f"probe.grib2:{record.index}" for record in records.items
        if (record.category, record.parameter, record.level_type) == (3, 4, 100))


def test_height_is_preferred_to_geopotential_at_a_level_that_has_both():
    frames = _frame(_mapping(geopotential=True), _state(
        heights=[(50000.0, 5555.0)],
        geopotentials=[(level, GRAVITY * 3000.0) for level in LEVELS]))

    heights = _heights(frames)
    np.testing.assert_array_equal(heights[2], 5555.0)
    np.testing.assert_allclose(heights[:2], 3000.0, rtol=0, atol=1e-9)
    assert ms.completed_field_summary(frames) == {}


def test_a_column_without_the_surface_operands_keeps_the_refusal():
    mapping = _mapping()
    records = _state()
    records.items = [record for record in records.items
                     if not (record.category == 3 and record.parameter == 0)]
    mapping["fields"].pop("surface_pressure")
    mapping["target"]["required_fields"] = [
        item for item in mapping["target"]["required_fields"]
        if item["name"] != "surface_pressure"]

    with pytest.raises(ValueError, match=r"lacks required fields \['geopotential_height'\]"):
        _frame(mapping, records)


def test_the_packaged_aifs_mapping_lists_geopotential_after_height():
    path = (Path(ms.__file__).parent / "authorities"
            / "rw-wps-aifs-single-grib2.mapping.json")
    selectors = json.loads(path.read_text(encoding="utf-8"))[
        "fields"]["geopotential_height"]["selectors"]
    assert [selector["parameter"] for selector in selectors] == [5, 4]
    assert selectors[1]["scale"] == pytest.approx(1.0 / GRAVITY, rel=1e-15)
    ms.load_mapping(path)


def test_a_zero_selector_scale_is_refused(tmp_path):
    path = (Path(ms.__file__).parent / "authorities"
            / "rw-wps-aifs-single-grib2.mapping.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fields"]["geopotential_height"]["selectors"][1]["scale"] = 0.0
    broken = tmp_path / "zero-scale.mapping.json"
    broken.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="scale must be finite and nonzero"):
        ms.load_mapping(broken)


def test_the_forecast_runner_reads_the_receipt_count_it_is_given():
    from gpuwm.prepared_single_domain_forecast import _validate_completed_fields

    entry = {"method": "hypsometric", "frames": 2, "frame_count": 2,
             "values_derived": 24, "values_total": 24}
    _validate_completed_fields({"geopotential_height": entry}, 2, "probe")
    for broken in (
        {},
        {"geopotential_height": {**entry, "frame_count": 3}},
        {"geopotential_height": {**entry, "values_derived": 0}},
        {"geopotential_height": {**entry, "values_derived": 25}},
        {"geopotential_height": {**entry, "extra": 1}},
    ):
        with pytest.raises(ValueError, match="completed_fields"):
            _validate_completed_fields(broken, 2, "probe")
