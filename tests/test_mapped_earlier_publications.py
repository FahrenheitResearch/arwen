"""A product's earlier publication prepares on what its files carry.

ECMWF's open-data IFS files have changed shape between releases of one
product.  Earlier files spell their four soil layers as depth-below-land
layers on WMO and local parameters instead of ordinal soil levels, and
some carry no surface geopotential at all.  The packaged mapping reads
the earlier soil spellings through ``record_aliases`` and derives terrain
from each column's pressure-level geopotential height at its surface
pressure through ``fields.terrain_height.when_absent``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.mapped_source import (
    _DecodedCollection,
    _DirectValue,
    _GribRecord,
    _alias_record,
    _assemble_grib,
    _grib2_wanted_indices,
    _height_at_surface_pressure,
    load_mapping,
)
from gpuwm.source_authorities import packaged_authorities

IFS_PROFILE = "ecmwf-open-data-oper-grib2-v1"
CYCLE = datetime(2024, 5, 17, 0)
NY, NX = 2, 3
RD_OVER_G = 287.06 / 9.80665

#: How the earlier files spell each soil layer (discipline, category,
#: parameter, top, bottom), decoded from one of them.  The bottom of the
#: deepest layer is written as the all-ones missing value, and every
#: layer after the first is written in centimetres on a metre surface.
EARLIER_SOIL_TEMPERATURE = (
    (2, 0, 2, 0.0, 0.07),
    (192, 128, 170, 7.0, 28.0),
    (192, 128, 183, 28.0, 100.0),
    (192, 128, 236, 100.0, 4294967295.0),
)
EARLIER_SOIL_MOISTURE = (
    (192, 128, 39, 0.0, 7.0),
    (192, 128, 40, 7.0, 28.0),
    (192, 128, 41, 28.0, 100.0),
    (192, 128, 42, 100.0, 4294967295.0),
)


def _mapping_raw() -> dict:
    return json.loads(
        packaged_authorities(IFS_PROFILE)["mapping"].read_text(encoding="utf-8"))


def _load(tmp_path: Path, raw: dict):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_mapping(path)


def _soil_record(index: int, spelling, value: float) -> _GribRecord:
    discipline, category, parameter, top, bottom = spelling
    return _GribRecord(
        source=Path("/staged/earlier.grib2"), index=index,
        reference_time=CYCLE, valid_time=CYCLE, member=None,
        parameter=parameter, level_type=106, level_value=top,
        table_version=None, center=98, subcenter=0,
        master_table_version=27, local_table_version=0,
        discipline=discipline, category=category,
        second_level_type=106, second_level_value=bottom,
        process_identity=None, time_semantics=(0,),
        values=np.full((NY, NX), value),
        latitude=np.linspace(40.0, 39.0, NY),
        longitude=np.linspace(-100.0, -99.0, NX),
        grid_fingerprint="grid",
    )


def _soil_only_mapping(raw: dict) -> dict:
    """The packaged mapping reduced to its soil fields and aliases."""

    keep = {"soil_temperature", "volumetric_soil_moisture"}
    reduced = dict(raw)
    reduced["fields"] = {
        name: field for name, field in raw["fields"].items() if name in keep}
    return reduced


def test_the_earlier_soil_spellings_stack_as_the_four_ordinal_layers():
    mapping = load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])
    aliases = mapping["record_aliases"]
    records = []
    for layer, spelling in enumerate(EARLIER_SOIL_TEMPERATURE):
        records.append(_soil_record(len(records), spelling, 280.0 + layer))
    for layer, spelling in enumerate(EARLIER_SOIL_MOISTURE):
        records.append(_soil_record(len(records), spelling, 0.1 * (layer + 1)))
    renamed = [_alias_record(record, aliases) for record in records]
    assert all(aliased for _record, aliased in renamed)
    soil_only = _soil_only_mapping(mapping)
    collection = _assemble_grib(soil_only, [record for record, _ in renamed])
    temperature = next(value for value in collection.direct.values()
                       if value.name == "soil_temperature")
    moisture = next(value for value in collection.direct.values()
                    if value.name == "volumetric_soil_moisture")
    assert temperature.values.shape == (4, NY, NX)
    assert temperature.values[:, 0, 0].tolist() == [280.0, 281.0, 282.0, 283.0]
    assert moisture.values[:, 0, 0].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_the_inventory_selects_the_earlier_spellings_it_will_read():
    mapping = load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])
    rows = []
    for spelling in (*EARLIER_SOIL_TEMPERATURE, *EARLIER_SOIL_MOISTURE,
                     # A layer no selector or alias reads.
                     (192, 128, 231, 0.0, 7.0)):
        discipline, category, parameter, top, bottom = spelling
        rows.append({
            "index": str(len(rows)), "member": "-",
            "parameter": str(parameter), "level_type": "106",
            "level_value": repr(top), "center": "98", "subcenter": "0",
            "master_table_version": "27", "local_table_version": "0",
            "discipline": str(discipline), "category": str(category),
            "second_level_type": "106", "second_level_value": repr(bottom),
            "pdt": "0",
        })
    assert _grib2_wanted_indices(mapping, rows) == set(range(8))


def test_a_current_record_is_never_renamed():
    mapping = load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])
    current = _soil_record(0, (2, 3, 18, 1.0, 2.0), 281.0)
    current = _GribRecord(**{**current.__dict__, "level_type": 151,
                             "second_level_type": 151})
    record, aliased = _alias_record(current, mapping["record_aliases"])
    assert not aliased and record is current


@pytest.mark.parametrize(("change", "message"), [
    (lambda raw: raw["record_aliases"].append(dict(raw["record_aliases"][1])),
     r"can name the same record"),
    (lambda raw: raw["record_aliases"].append({
        "record": {"discipline": 0, "category": 0, "parameter": 0,
                   "level_type": 100},
        "reads_as": {"discipline": 2, "category": 3, "parameter": 18,
                     "level_type": 151, "level_value": 0,
                     "second_level_type": 151, "second_level_value": 1}}),
     r"is read as spelled by fields\.air_temperature"),
    (lambda raw: raw["record_aliases"].append({
        "record": {"discipline": 192, "category": 128, "parameter": 7},
        "reads_as": {"discipline": 2, "category": 3, "parameter": 18,
                     "level_type": 151, "level_value": 9,
                     "second_level_type": 151, "second_level_value": 10}}),
     r"is read by 0 mapped fields"),
    (lambda raw: raw["record_aliases"][0]["reads_as"].pop("second_level_value"),
     r"second_level_type and second_level_value together"),
    (lambda raw: raw.update(record_aliases=[]),
     r"record_aliases must be a non-empty list"),
])
def test_record_aliases_are_held_to_what_they_would_break(tmp_path, change, message):
    raw = _mapping_raw()
    change(raw)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, raw)


def test_record_aliases_on_another_format_are_refused_as_unread():
    from gpuwm.mapped_source import _validate_record_aliases

    raw = _mapping_raw()
    with pytest.raises(ValueError, match="read by the GRIB2 decoders only"):
        _validate_record_aliases(raw["record_aliases"], raw["fields"], "netcdf")


@pytest.mark.parametrize(("change", "message"), [
    (lambda fields: fields["skin_temperature"].update(
        when_absent=fields["terrain_height"]["when_absent"]),
     r"read only for terrain_height"),
    (lambda fields: fields["terrain_height"]["when_absent"].update(
        operation="nearest_neighbour"),
     r"operation must be one of \['height_at_surface_pressure'\]"),
    (lambda fields: fields["terrain_height"]["when_absent"].update(
        temperature="air_temperature_2m"),
     r"whose axes \('y', 'x'\) are not \('vertical', 'y', 'x'\)"),
    (lambda fields: fields["terrain_height"]["when_absent"].update(
        surface_pressure="air_pressure"),
     r"is not a directly decoded field"),
    (lambda fields: fields["terrain_height"]["when_absent"].pop(
        "surface_dewpoint"),
     r"missing"),
])
def test_when_absent_is_held_to_what_it_reads(tmp_path, change, message):
    raw = _mapping_raw()
    change(raw["fields"])
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, raw)


def _isothermal(levels, surface_pressure, t=250.0, dewpoint=100.0):
    levels = np.asarray(levels, dtype=np.float64)
    surface = np.asarray(surface_pressure, dtype=np.float64).reshape(1, -1)
    cells = surface.shape[1]
    heights = RD_OVER_G * t * np.log(100_000.0 / levels)
    column = np.repeat(heights[:, None, None], cells, axis=2)
    return (
        levels, column, np.full_like(column, t), np.zeros_like(column),
        surface, np.full(surface.shape, t), np.full(surface.shape, dewpoint),
    )


def test_surface_height_is_the_column_height_at_the_surface_pressure():
    levels = [5000.0, 100_000.0, 85_000.0, 50_000.0, 92_500.0, 70_000.0]
    surface = [80_000.0, 92_500.0, 103_000.0]
    height, counts = _height_at_surface_pressure(*_isothermal(levels, surface))
    expected = RD_OVER_G * 250.0 * np.log(100_000.0 / np.asarray(surface))
    np.testing.assert_allclose(height[0], expected, atol=1e-6)
    assert counts["cells"] == 3
    assert counts["cells_below_lowest_level"] == 1


def test_a_surface_above_the_highest_level_is_refused_by_count():
    with pytest.raises(ValueError, match=r"1 of 2 cells .* highest level \(85000 Pa\)"):
        _height_at_surface_pressure(
            *_isothermal([85_000.0, 100_000.0], [90_000.0, 60_000.0]))


def test_moist_air_below_the_ladder_lies_lower_than_dry_air():
    dry, _ = _height_at_surface_pressure(
        *_isothermal([85_000.0, 100_000.0], [102_000.0], t=290.0, dewpoint=100.0))
    moist, _ = _height_at_surface_pressure(
        *_isothermal([85_000.0, 100_000.0], [102_000.0], t=290.0, dewpoint=288.0))
    assert moist[0, 0] < dry[0, 0] < 0.0


def _primary(levels, surface):
    levels, heights, temperature, humidity, pressure, t2, d2 = _isothermal(
        levels, surface)
    cells = pressure.shape[1]
    first, later = CYCLE, datetime(2024, 5, 17, 3)
    direct = {}
    for when in (first, later):
        for name, values, axes in (
            ("geopotential_height", heights, ("vertical", "y", "x")),
            ("air_temperature", temperature, ("vertical", "y", "x")),
            ("specific_humidity", humidity, ("vertical", "y", "x")),
            ("surface_pressure", pressure, ("y", "x")),
            ("air_temperature_2m", t2, ("y", "x")),
            ("dewpoint_2m", d2, ("y", "x")),
        ):
            direct[(when, None, name)] = _DirectValue(
                name=name, valid_time=when, member=None, source_cycle=CYCLE,
                axes=axes, values=values if when == first else values + 1.0,
                missing_count=0, references=(f"/staged/f.grib2:{name}",))
    return _DecodedCollection(
        latitude=np.array([40.0]), longitude=np.linspace(-100.0, -99.0, cells),
        vertical_values=levels, direct=direct,
        source_cycles={(first, None): CYCLE, (later, None): CYCLE},
        grid_fingerprint="grid")


def test_absent_terrain_is_derived_once_from_the_first_valid_time():
    from gpuwm.mapped_composition import _derive_absent_terrain

    mapping = load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])
    surface = [95_000.0, 101_000.0]
    collection, receipt = _derive_absent_terrain(
        mapping, mapping["fields"]["terrain_height"]["when_absent"],
        _primary([5000.0, 50_000.0, 85_000.0, 100_000.0], surface))
    (key, terrain), = collection.direct.items()
    assert key == (CYCLE, None, "terrain_height")
    assert terrain.axes == ("y", "x")
    np.testing.assert_allclose(
        terrain.values[0],
        RD_OVER_G * 250.0 * np.log(100_000.0 / np.asarray(surface)), atol=1e-6)
    assert collection.source_cycles == {(CYCLE, None): CYCLE}
    assert receipt["operation"] == "height_at_surface_pressure"
    assert receipt["valid_time"] == CYCLE.isoformat()
    assert receipt["cells"] == 2 and receipt["cells_below_lowest_level"] == 1
    assert receipt["fields"] == [
        "geopotential_height", "air_temperature", "specific_humidity",
        "surface_pressure", "air_temperature_2m", "dewpoint_2m"]


def test_one_warning_says_the_terrain_was_derived(capsys):
    from gpuwm.mapped_direct import _announce_derived_terrain

    _announce_derived_terrain({
        "cells": 1038240, "cells_below_lowest_level": 615612,
        "minimum_m": -41.2, "maximum_m": 6123.9,
    })
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("warning:")]
    assert len(lines) == 1
    assert "carry no surface geopotential" in lines[0]
    assert "1038240" in lines[0] and "615612" in lines[0]



def test_the_forecast_runner_takes_a_receipt_that_counts_aliased_records():
    from gpuwm.prepared_single_domain_forecast import (
        MAPPED_COMPOSITION_RECEIPT_KEYS, mapped_composition_receipt_keys)

    receipt = {key: None for key in MAPPED_COMPOSITION_RECEIPT_KEYS}
    assert mapped_composition_receipt_keys(
        receipt, declared_bindings=False, source="ecmwf-open-data")         == set(MAPPED_COMPOSITION_RECEIPT_KEYS)
    receipt["record_aliases"] = {
        "soil_temperature": 8, "volumetric_soil_moisture": 8}
    assert mapped_composition_receipt_keys(
        receipt, declared_bindings=False, source="ecmwf-open-data")         == set(MAPPED_COMPOSITION_RECEIPT_KEYS) | {"record_aliases"}
    for malformed in ({}, {"soil_temperature": 0}, {"soil_temperature": True},
                      ["soil_temperature"]):
        receipt["record_aliases"] = malformed
        with pytest.raises(ValueError, match="composition receipt is malformed"):
            mapped_composition_receipt_keys(
                receipt, declared_bindings=False, source="ecmwf-open-data")
