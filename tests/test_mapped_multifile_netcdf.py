"""One source, many files: the NetCDF shapes real producers actually ship.

The mapped NetCDF decoder used to assume one file carrying every field on
exactly the declared levels.  No large archive is shaped that way.  NOAA
PSL publishes 20CRv3 as one variable per file per year, reuses the same
variable NAME for the same quantity at different levels (``air`` on
pressure levels and ``air`` at 2 m), gives its humidity 21 pressure
levels where its temperature has 28, and stacks its four soil layers
inside ONE variable with its own layer dimension.

Each of those is a general property of scientific archives, so each is
answered generally -- by the mapping table, never by a per-source branch.
This is the test for all four, written against synthetic files so the
shapes are exact and the failures are legible.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.mapped_source import decode_mapped_source, load_mapping


NY, NX = 4, 5
#: What the wind/height/temperature files carry, top-down like a real
#: archive; the humidity file carries only the middle five.
LEVELS_WIDE = (1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 100.0)
LEVELS_NARROW = (1000.0, 925.0, 850.0, 700.0, 500.0)
#: What the mapping declares: the common set, bottom-up.
LEVELS_DECLARED = (500.0, 700.0, 850.0, 925.0, 1000.0)
SOIL_CM = (0.0, 10.0, 40.0, 100.0)
TIMES = (0.0, 3.0)

PRESSURE = "Pressure Levels"


def _selector(name, **extra):
    return {"format": "netcdf", "name": name, **extra}


def _layer(name, value, **extra):
    return _selector(
        name, layer_dimension="level", layer_value=value, layer_units="cm",
        attributes={"level_desc": "Multiple Subsurface Levels"}, **extra)


def _field(selectors, units, source_axes, target_axes, location, **extra):
    return {
        "selectors": list(selectors),
        "units": {"source": units, "target": units},
        "source_axes": list(source_axes),
        "target_axes": list(target_axes),
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
        **extra,
    }


def _mass(name, units, level_desc=PRESSURE):
    return _field(
        [_selector([name], attributes={"level_desc": level_desc})], units,
        ["time", "vertical", "y", "x"], ["vertical", "y", "x"], "mass")


def _surface(name, units, level_desc):
    return _field(
        [_selector([name], attributes={"level_desc": level_desc})], units,
        ["time", "y", "x"], ["y", "x"], "surface")


#: ``canonical name -> (variable, units, level_desc)`` for the surface
#: half, which is where one producer's name reuse actually bites.
SURFACE_TABLE = {
    "surface_pressure": ("pres", "Pa", "Surface"),
    "terrain_height": ("orog", "m", "Surface"),
    "skin_temperature": ("skt", "K", "Surface"),
    "land_fraction": ("land", "1", "Surface"),
    "air_temperature_2m": ("air", "K", "2 m"),
    "specific_humidity_2m": ("shum", "kg kg-1", "2 m"),
    "eastward_wind_10m": ("uwnd", "m s-1", "10 m"),
    "northward_wind_10m": ("vwnd", "m s-1", "10 m"),
}
MASS_TABLE = {
    "air_temperature": ("air", "K"),
    "specific_humidity": ("shum", "kg kg-1"),
    "eastward_wind": ("uwnd", "m s-1"),
    "northward_wind": ("vwnd", "m s-1"),
    "geopotential_height": ("hgt", "m"),
}
SOIL_TABLE = {
    "soil_temperature": ("tsoil", "K"),
    "volumetric_soil_moisture": ("soilw", "m3 m-3"),
}
POLICY = [
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio", "vertical_velocity",
    "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
]


def _mapping() -> dict:
    """A mapping shaped like the packaged 20CRv3 NetCDF profile."""

    fields = {
        name: _mass(variable, units)
        for name, (variable, units) in MASS_TABLE.items()
    }
    fields["air_pressure"] = {
        "selectors": [], "derivation": "pressure-from-coordinate",
        "units": {"source": "millibar", "target": "Pa",
                  "scale": 100.0, "offset": 0.0},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass", "staggering": "none",
        "missing": {"kind": "reject"},
    }
    for name, (variable, units, level_desc) in SURFACE_TABLE.items():
        fields[name] = _surface(variable, units, level_desc)
    for name, (variable, units) in SOIL_TABLE.items():
        fields[name] = _field(
            [_layer([variable], value) for value in SOIL_CM], units,
            ["time", "soil", "y", "x"], ["soil", "y", "x"], "soil",
            selector_stack_axis="soil")

    required = [
        {"name": name, "axes": ["vertical", "y", "x"], "location": "mass",
         "target_units": units}
        for name, (_variable, units) in MASS_TABLE.items()
    ]
    required.append({
        "name": "air_pressure", "axes": ["vertical", "y", "x"],
        "location": "mass", "target_units": "Pa"})
    required.extend(
        {"name": name, "axes": ["y", "x"], "location": "surface",
         "target_units": units}
        for name, (_variable, units, _desc) in SURFACE_TABLE.items())
    required.extend(
        {"name": name, "axes": ["soil", "y", "x"], "location": "soil",
         "target_units": units}
        for name, (_variable, units) in SOIL_TABLE.items())

    return {
        "schema": "rw-wps.mapping.v1",
        "name": "multifile-netcdf-probe",
        "format": "netcdf",
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": _selector(["lat"]),
                "longitude": _selector(["lon"]),
            },
            "vertical": {
                "kind": "pressure", "selector": _selector(["level"]),
                "units": "millibar", "positive": "down",
                "levels": list(LEVELS_DECLARED),
            },
            "time": {
                "kind": "dimension", "selector": {"name": ["time"]},
                "units": "hours since 1974-04-03 18:00:00",
                "calendar": "standard",
            },
        },
        "fields": fields,
        "derivations": [{
            "name": "pressure-from-coordinate",
            "operation": "pressure_from_vertical_coordinate",
        }],
        "target": {
            "name": "probe", "physics_suite": "none", "max_dom": 1,
            "require_lateral_boundaries": True,
            "target_vertical_levels": 49, "soil_layer_count": 4,
            "boundary_interval_seconds": 10800,
            "required_fields": required,
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": POLICY,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation"
                for name in POLICY
            },
        },
    }


def _coordinates(dataset, *, levels=None, soil=False):
    dataset.createDimension("time", len(TIMES))
    dataset.createDimension("lat", NY)
    dataset.createDimension("lon", NX)
    time = dataset.createVariable("time", "f8", ("time",))
    time.units = "hours since 1974-04-03 18:00:00"
    time.calendar = "standard"
    time[:] = TIMES
    latitude = dataset.createVariable("lat", "f8", ("lat",))
    latitude.units = "degrees_north"
    latitude[:] = np.linspace(35.0, 41.0, NY)
    longitude = dataset.createVariable("lon", "f8", ("lon",))
    longitude.units = "degrees_east"
    longitude[:] = np.linspace(260.0, 268.0, NX)
    if levels is not None:
        dataset.createDimension("level", len(levels))
        level = dataset.createVariable("level", "f8", ("level",))
        level.units = "millibar"
        level[:] = levels
    if soil:
        dataset.createDimension("level", len(SOIL_CM))
        level = dataset.createVariable("level", "f8", ("level",))
        level.units = "cm"
        level[:] = SOIL_CM


def _mass_file(path, entries, levels):
    with netCDF4.Dataset(path, "w") as dataset:
        _coordinates(dataset, levels=levels)
        for name, units, base in entries:
            value = dataset.createVariable(
                name, "f8", ("time", "level", "lat", "lon"))
            value.units = units
            value.level_desc = PRESSURE
            shape = (len(TIMES), len(levels), NY, NX)
            value[:] = base + np.arange(
                np.prod(shape), dtype=np.float64).reshape(shape) * 1e-4


def _surface_file(path, entries):
    with netCDF4.Dataset(path, "w") as dataset:
        _coordinates(dataset)
        for name, units, level_desc, base in entries:
            value = dataset.createVariable(name, "f8", ("time", "lat", "lon"))
            value.units = units
            value.level_desc = level_desc
            value[:] = base


def _soil_file(path):
    with netCDF4.Dataset(path, "w") as dataset:
        _coordinates(dataset, soil=True)
        for name, units, layers in (
            ("tsoil", "K", (281.0, 282.0, 283.0, 284.0)),
            ("soilw", "m3 m-3", (0.21, 0.22, 0.23, 0.24)),
        ):
            value = dataset.createVariable(
                name, "f8", ("time", "level", "lat", "lon"))
            value.units = units
            value.level_desc = "Multiple Subsurface Levels"
            value[:] = np.broadcast_to(
                np.asarray(layers)[None, :, None, None],
                (len(TIMES), 4, NY, NX))


def _corpus(root: Path) -> list[Path]:
    """Eight files, the way an archive publishes a reanalysis."""

    root.mkdir(parents=True, exist_ok=True)
    _mass_file(root / "air.nc", [("air", "K", 250.0)], LEVELS_WIDE)
    _mass_file(root / "hgt.nc", [("hgt", "m", 1000.0)], LEVELS_WIDE)
    _mass_file(
        root / "wind.nc",
        [("uwnd", "m s-1", 5.0), ("vwnd", "m s-1", 2.0)], LEVELS_WIDE)
    # Published on FEWER levels than its siblings, exactly like 20CRv3's.
    _mass_file(
        root / "shum.nc", [("shum", "kg kg-1", 0.004)], LEVELS_NARROW)
    _surface_file(root / "sfc.nc", [
        ("pres", "Pa", "Surface", 98000.0),
        ("skt", "K", "Surface", 286.0),
        ("orog", "m", "Surface", 250.0),
        ("land", "1", "Surface", 1.0),
    ])
    _surface_file(root / "2m.nc", [
        ("air", "K", "2 m", 288.0), ("shum", "kg kg-1", "2 m", 0.006),
    ])
    _surface_file(root / "10m.nc", [
        ("uwnd", "m s-1", "10 m", 4.0), ("vwnd", "m s-1", "10 m", 1.0),
    ])
    _soil_file(root / "soil.nc")
    return [root / name for name in (
        "air.nc", "hgt.nc", "wind.nc", "shum.nc", "sfc.nc", "2m.nc",
        "10m.nc", "soil.nc")]


def _write(path: Path, mapping: dict) -> Path:
    path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_one_source_spread_over_eight_files_decodes_as_one_collection(
    tmp_path,
):
    """The headline: no merge step, no per-source reader."""

    files = _corpus(tmp_path / "corpus")
    mapping = _write(tmp_path / "mapping.json", _mapping())
    frames = decode_mapped_source(mapping, files)

    assert [frame.valid_time.isoformat() for frame in frames] == [
        "1974-04-03T18:00:00", "1974-04-03T21:00:00"]
    first = frames[0]
    assert set(first.fields) == (
        set(MASS_TABLE) | set(SURFACE_TABLE) | set(SOIL_TABLE)
        | {"air_pressure"})
    # The declared levels, in the MAPPING's order, not the file's.
    assert list(first.vertical_values) == list(LEVELS_DECLARED)
    assert first.fields["air_temperature"].values.shape == (
        len(LEVELS_DECLARED), NY, NX)
    assert first.fields["specific_humidity"].values.shape == (
        len(LEVELS_DECLARED), NY, NX)
    assert first.fields["air_temperature_2m"].values.shape == (NY, NX)
    assert first.fields["soil_temperature"].values.shape == (4, NY, NX)
    # The pressure derivation reads the DECLARED ladder, so it is bottom-up
    # in pascals whatever order the files were written in.
    assert list(first.fields["air_pressure"].values[:, 0, 0]) == [
        value * 100.0 for value in LEVELS_DECLARED]


def test_the_same_variable_name_in_two_files_is_told_apart_by_attribute(
    tmp_path,
):
    """``air`` is temperature twice; the file says which one it is."""

    files = _corpus(tmp_path / "corpus")
    mapping = _write(tmp_path / "mapping.json", _mapping())
    frame = decode_mapped_source(mapping, files)[0]
    assert np.allclose(frame.fields["air_temperature_2m"].values, 288.0)
    assert not np.allclose(frame.fields["air_temperature"].values, 288.0)

    # Drop the discriminator and one variable now answers to two fields.
    # That is refused by name, in the file where it happens, rather than
    # resolved by preferring one -- a preference would be a guess about
    # which quantity the producer meant.
    ambiguous = _mapping()
    ambiguous["fields"]["air_temperature_2m"]["selectors"] = [
        _selector(["air"])]
    path = _write(tmp_path / "ambiguous.json", ambiguous)
    with pytest.raises(
        ValueError,
        match="'air' directly provides both .* derive aliases explicitly",
    ):
        decode_mapped_source(path, files)


def test_a_field_absent_from_every_file_names_the_whole_inventory(tmp_path):
    """Absent HERE is normal; absent EVERYWHERE is the error."""

    files = _corpus(tmp_path / "corpus")
    mapping = _mapping()
    mapping["fields"]["skin_temperature"]["selectors"] = [
        _selector(["not_published"], attributes={"level_desc": "Surface"})]
    path = _write(tmp_path / "missing.json", mapping)
    with pytest.raises(ValueError) as refusal:
        decode_mapped_source(path, files)
    message = str(refusal.value)
    assert "skin_temperature" in message
    assert "no matching variable in any of the 8 supplied" in message
    assert "air.nc" in message and "soil.nc" in message


def test_a_field_that_needs_the_vertical_coordinate_gets_its_error(tmp_path):
    """Surface-only files need no vertical coordinate; a 3-D field does."""

    files = _corpus(tmp_path / "corpus")
    # Every surface file in this corpus lacks `level` entirely and decodes
    # anyway -- that is the first half, proved by the headline test.  Here
    # a 3-D field is pointed at a file with no vertical coordinate.
    mapping = _mapping()
    mapping["fields"]["air_temperature"]["selectors"] = [
        _selector(["air"], attributes={"level_desc": "2 m"})]
    # The 2 m field steps aside so the two selectors stay distinct; what
    # is under test is the 3-D field meeting a coordinate-less file.
    mapping["fields"]["air_temperature_2m"]["selectors"] = [
        _selector(["air"], attributes={"level_desc": "2 metres"})]
    path = _write(tmp_path / "vertical.json", mapping)
    with pytest.raises(ValueError, match="declares a vertical axis"):
        decode_mapped_source(path, files)


def test_a_declared_level_the_file_does_not_carry_is_refused(tmp_path):
    """Subsetting is by VALUE, so a missing level cannot slide by."""

    files = _corpus(tmp_path / "corpus")
    mapping = _mapping()
    mapping["coordinates"]["vertical"]["levels"] = [400.0, 500.0, 700.0]
    path = _write(tmp_path / "levels.json", mapping)
    with pytest.raises(ValueError, match="declared level 400.0 appears 0"):
        decode_mapped_source(path, files)


def test_a_soil_layer_is_addressed_by_its_own_coordinate_value(tmp_path):
    """Four selectors, one variable, four depths -- bound by value."""

    files = _corpus(tmp_path / "corpus")
    mapping = _write(tmp_path / "mapping.json", _mapping())
    frame = decode_mapped_source(mapping, files)[0]
    soil = frame.fields["soil_temperature"].values
    assert [float(soil[index, 0, 0]) for index in range(4)] == [
        281.0, 282.0, 283.0, 284.0]

    reordered = _mapping()
    reordered["fields"]["soil_temperature"]["selectors"] = [
        _layer(["tsoil"], value) for value in (100.0, 40.0, 10.0, 0.0)]
    path = _write(tmp_path / "reordered.json", reordered)
    frame = decode_mapped_source(path, files)[0]
    soil = frame.fields["soil_temperature"].values
    assert [float(soil[index, 0, 0]) for index in range(4)] == [
        284.0, 283.0, 282.0, 281.0]


def test_a_layer_value_the_file_does_not_have_is_refused(tmp_path):
    files = _corpus(tmp_path / "corpus")
    mapping = _mapping()
    mapping["fields"]["soil_temperature"]["selectors"][2] = _layer(
        ["tsoil"], 55.0)
    path = _write(tmp_path / "layer.json", mapping)
    with pytest.raises(ValueError, match="layer_value=55.0 matches 0 entries"):
        decode_mapped_source(path, files)


def test_the_layer_slice_keys_are_atomic(tmp_path):
    """A selector cannot half-declare a slice."""

    mapping = _mapping()
    mapping["fields"]["soil_temperature"]["selectors"][0] = {
        "format": "netcdf", "name": ["tsoil"], "layer_value": 0.0}
    path = _write(tmp_path / "atomic.json", mapping)
    with pytest.raises(ValueError, match="layer slice needs layer_dimension"):
        load_mapping(path)

    mapping = _mapping()
    mapping["fields"]["soil_temperature"]["selectors"][0] = _layer(
        ["tsoil"], 0.0)
    mapping["fields"]["soil_temperature"]["selectors"][0][
        "layer_units"] = "furlongs"
    path = _write(tmp_path / "units.json", mapping)
    with pytest.raises(ValueError, match="layer_units='furlongs'"):
        load_mapping(path)
