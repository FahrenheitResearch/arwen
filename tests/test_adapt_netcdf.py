"""`gpuwm adapt` must author an arbitrary NetCDF source, and refuse silently-zero ones.

The header layout used here is not invented.  It is the layout of an actual
Copernicus CDS delivery fetched on 2026-08-14 -- ``valid_time`` in
``seconds since 1970-01-01``, ``pressure_level`` in hPa, lower-case short-name
variables, and ECMWF's ``m**2 s**-2`` / ``m s**-1`` / ``kg kg**-1`` unit
spellings -- because a fixture written to match our own reader proves only
that the reader agrees with itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.adapt import author_adapter, verify_netcdf_inputs
from gpuwm.mapped_authoring import compile_mapping_descriptor


NY, NX, NZ = 4, 5, 4
LEVELS = [1000.0, 850.0, 500.0, 250.0]
#: (canonical name, variable, source units, target units)
_UPPER = (
    ("geopotential", "z", "m**2 s**-2", "m2 s-2"),
    ("air_temperature", "t", "K", "K"),
    ("specific_humidity", "q", "kg kg**-1", "kg kg-1"),
    ("eastward_wind", "u", "m s**-1", "m s-1"),
    ("northward_wind", "v", "m s**-1", "m s-1"),
)
_SURFACE = (
    ("surface_pressure", "sp", "Pa", "Pa"),
    ("skin_temperature", "skt", "K", "K"),
    ("air_temperature_2m", "t2m", "K", "K"),
    ("dewpoint_2m", "d2m", "K", "K"),
    ("eastward_wind_10m", "u10", "m s**-1", "m s-1"),
    ("northward_wind_10m", "v10", "m s**-1", "m s-1"),
    ("land_fraction", "lsm", "(0 - 1)", "1"),
)
_TV = ["vertical", "y", "x"]
_TS = ["y", "x"]
_POLICY = [
    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio", "vertical_velocity",
    "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
]


def _write_source(path: Path) -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.Conventions = "CF-1.7"
        dataset.createDimension("valid_time", 1)
        dataset.createDimension("pressure_level", NZ)
        dataset.createDimension("latitude", NY)
        dataset.createDimension("longitude", NX)
        time = dataset.createVariable("valid_time", "i8", ("valid_time",))
        time.units = "seconds since 1970-01-01"
        time.standard_name = "time"
        time[:] = [1590969600]
        level = dataset.createVariable("pressure_level", "f8", ("pressure_level",))
        level.units = "hPa"
        level.standard_name = "air_pressure"
        level[:] = LEVELS
        lat = dataset.createVariable("latitude", "f8", ("latitude",))
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat[:] = np.linspace(42.0, 40.0, NY)
        lon = dataset.createVariable("longitude", "f8", ("longitude",))
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon[:] = np.linspace(-90.0, -88.0, NX)

        upper_dims = ("valid_time", "pressure_level", "latitude", "longitude")
        surface_dims = ("valid_time", "latitude", "longitude")
        seeds = {"t": 280.0, "q": 0.005, "u": 4.0, "v": -3.0, "z": 50000.0}
        for _, variable, units, _target in _UPPER:
            value = dataset.createVariable(variable, "f4", upper_dims)
            value.units = units
            value[:] = seeds[variable]
        surface_seeds = {
            "sp": 98000.0, "skt": 288.0, "t2m": 287.0, "d2m": 283.0,
            "u10": 3.0, "v10": -2.0, "lsm": 1.0, "z_surface": 2000.0,
        }
        for _, variable, units, _target in _SURFACE:
            value = dataset.createVariable(variable, "f4", surface_dims)
            value.units = units
            value[:] = surface_seeds[variable]
        terrain = dataset.createVariable("z_surface", "f4", surface_dims)
        terrain.units = "m**2 s**-2"
        terrain[:] = surface_seeds["z_surface"]
        for index in range(1, 5):
            soil = dataset.createVariable(f"stl{index}", "f4", surface_dims)
            soil.units = "K"
            soil[:] = 285.0
            moisture = dataset.createVariable(f"swvl{index}", "f4", surface_dims)
            moisture.units = "m**3 m**-3"
            moisture[:] = 0.25


def _field(variable, source, target, axes, target_axes, location, stack=None):
    field = {
        "selectors": (
            [{"format": "netcdf", "name": variable}]
            if isinstance(variable, str)
            else [{"format": "netcdf", "name": name} for name in variable]
        ),
        "units": {"source": source, "target": target},
        "source_axes": axes, "target_axes": target_axes,
        "location": location, "staggering": "none",
        "missing": {"kind": "reject"},
    }
    if stack:
        field["selector_stack_axis"] = stack
    return field


def _derived(name, source, target, axes, location):
    return {
        "derivation": name, "units": {"source": source, "target": target},
        "source_axes": axes, "target_axes": axes, "location": location,
        "staggering": "none", "missing": {"kind": "reject"},
    }


def _descriptor() -> dict[str, object]:
    upper = ["time", "vertical", "y", "x"]
    surface = ["time", "y", "x"]
    fields: dict[str, object] = {}
    for canonical, variable, source, target in _UPPER:
        fields[canonical] = _field(variable, source, target, upper, _TV, "mass")
    for canonical, variable, source, target in _SURFACE:
        fields[canonical] = _field(
            variable, source, target, surface, _TS, "surface")
    fields["terrain_height"] = _field(
        "z_surface", "m**2 s**-2", "m", surface, _TS, "surface")
    fields["terrain_height"]["units"]["scale"] = 1.0 / 9.80665
    fields["geopotential_height"] = _derived(
        "height-from-geopotential", "m", "m", _TV, "mass")
    fields["air_pressure"] = _derived(
        "pressure-from-coordinate", "hPa", "Pa", _TV, "mass")
    # `pressure-from-coordinate` emits the vertical coordinate's OWN
    # values, and this file's coordinate is hPa, so reaching Pa is a real
    # conversion and has to say so.  Without it the compiler defaulted the
    # factor to 1.0 and the fixture asserted against pressures a hundred
    # times too small -- the same omission the shipped ERA5 mapping avoids
    # by carrying "scale": 100.0 on this exact field.
    fields["air_pressure"]["units"]["scale"] = 100.0
    fields["specific_humidity_2m"] = _derived(
        "humidity-2m-from-dewpoint", "kg kg-1", "kg kg-1", _TS, "surface")
    fields["soil_temperature"] = _field(
        [f"stl{index}" for index in range(1, 5)], "K", "K",
        ["time", "soil", "y", "x"], ["soil", "y", "x"], "soil", stack="soil")
    fields["volumetric_soil_moisture"] = _field(
        [f"swvl{index}" for index in range(1, 5)], "m**3 m**-3", "m3 m-3",
        ["time", "soil", "y", "x"], ["soil", "y", "x"], "soil", stack="soil")

    canonical_contract = {
        "air_temperature": (_TV, "mass", "K"),
        "specific_humidity": (_TV, "mass", "kg kg-1"),
        "eastward_wind": (_TV, "mass", "m s-1"),
        "northward_wind": (_TV, "mass", "m s-1"),
        "geopotential_height": (_TV, "mass", "m"),
        "air_pressure": (_TV, "mass", "Pa"),
        "surface_pressure": (_TS, "surface", "Pa"),
        "terrain_height": (_TS, "surface", "m"),
        "skin_temperature": (_TS, "surface", "K"),
        "air_temperature_2m": (_TS, "surface", "K"),
        "specific_humidity_2m": (_TS, "surface", "kg kg-1"),
        "eastward_wind_10m": (_TS, "surface", "m s-1"),
        "northward_wind_10m": (_TS, "surface", "m s-1"),
        "land_fraction": (_TS, "surface", "1"),
        "soil_temperature": (["soil", "y", "x"], "soil", "K"),
        "volumetric_soil_moisture": (["soil", "y", "x"], "soil", "m3 m-3"),
    }
    return {
        "schema": "rw-wps.descriptor.v1",
        "name": "cds-shaped-netcdf-under-test",
        "format": "netcdf",
        "coordinates": {
            "horizontal": {
                "kind": "variables",
                "latitude": {"format": "netcdf", "name": "latitude"},
                "longitude": {"format": "netcdf", "name": "longitude"},
            },
            "vertical": {
                "kind": "pressure",
                "selector": {"format": "netcdf", "name": "pressure_level"},
                "units": "hPa", "positive": "down", "levels": LEVELS,
            },
            "time": {
                "kind": "dimension",
                "selector": {"name": "valid_time"},
                "units": "seconds since 1970-01-01",
                "calendar": "proleptic_gregorian",
            },
        },
        "fields": fields,
        "derivations": [
            {"name": "height-from-geopotential",
             "operation": "geopotential_height",
             "geopotential": "geopotential", "gravity_m_s2": 9.80665},
            {"name": "pressure-from-coordinate",
             "operation": "pressure_from_vertical_coordinate"},
            {"name": "humidity-2m-from-dewpoint",
             "operation": "specific_humidity_from_dewpoint",
             "dewpoint": "dewpoint_2m", "temperature": "air_temperature_2m",
             "pressure": "surface_pressure"},
        ],
        "target": {
            "name": "arbitrary NetCDF under test",
            "physics_suite": "source-independent", "max_dom": 1,
            "require_lateral_boundaries": False, "target_vertical_levels": NZ,
            "soil_layer_count": 4, "boundary_interval_seconds": 21600,
            "pressure_requirement": "air_pressure",
            "required_fields": [
                {"name": name, "axes": axes, "location": location,
                 "target_units": units}
                for name, (axes, location, units) in canonical_contract.items()
            ],
            "policy_controlled_fields": _POLICY,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation"
                for name in _POLICY
            },
        },
        "adapt": {
            "model_top_pa": 25000.0,
            "soil_policy": {
                "kind": "conservative_layer_means",
                "source_layers_m": [[0.0, 0.07], [0.07, 0.28],
                                    [0.28, 1.0], [1.0, 2.89]],
                "target_layers_m": [[0.0, 0.1], [0.1, 0.4],
                                    [0.4, 1.0], [1.0, 2.0]],
            },
        },
    }


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source.nc"
    _write_source(source)
    descriptor = tmp_path / "source.descriptor.json"
    descriptor.write_text(json.dumps(_descriptor()), encoding="utf-8")
    return descriptor, source


def test_adapt_authors_a_runnable_adapter_from_arbitrary_netcdf(prepared, tmp_path):
    descriptor, source = prepared
    result = author_adapter(
        descriptor_path=descriptor,
        input_files=[source],
        output_dir=tmp_path / "adapter",
    )
    assert result["runnable"] is True
    assert result["stock_wrf_certified"] is False
    assert result["runtime_bindings"]["source_format"] == "netcdf"
    # NetCDF needs no external decoder; a GRIB2 decoder pair here would mean
    # the NetCDF route had quietly taken the GRIB path.
    assert result["runtime_bindings"]["decoders"] == {}
    inventory = result["battery"]["record_inventory"]
    # Positive evidence of work, not merely a PASS: a battery that resolved
    # nothing must never report success.
    assert inventory["declared_selector_count"] > 0
    assert inventory["resolved_selector_count"] == inventory["declared_selector_count"]
    for role in ("mapping", "composition", "provenance"):
        assert Path(result[role]["path"]).is_file()


def test_adapt_refuses_a_vtable_for_netcdf(prepared, tmp_path):
    descriptor, source = prepared
    with pytest.raises(ValueError) as error:
        author_adapter(
            vtable_path=tmp_path / "Vtable.ANY",
            descriptor_path=descriptor,
            input_files=[source],
            output_dir=tmp_path / "adapter",
        )
    assert "--vtable" in str(error.value)


def test_a_selector_matching_nothing_names_both_vocabularies(prepared, tmp_path):
    """The silent zero: a mapping that matches nothing must not pass."""

    descriptor, source = prepared
    document = json.loads(descriptor.read_text())
    # The exact shape of real producer drift: a renamed variable.
    document["fields"]["air_temperature"]["selectors"] = [
        {"format": "netcdf", "name": "TMP"}
    ]
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    mapping, _ = compile_mapping_descriptor(descriptor)
    with pytest.raises(ValueError) as error:
        verify_netcdf_inputs(mapping, [source])
    message = str(error.value)
    assert "air_temperature" in message and "TMP" in message   # asked for
    assert "skt" in message and "swvl1" in message             # file offers
    assert "matched no variable" in message


def test_a_battery_that_resolves_nothing_is_a_failure_not_a_pass(prepared, tmp_path):
    descriptor, source = prepared
    document = json.loads(descriptor.read_text())
    for name, field in document["fields"].items():
        if field.get("selectors"):
            field["selectors"] = [
                {"format": "netcdf", "name": f"absent_{name}_{index}"}
                for index in range(len(field["selectors"]))
            ]
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    mapping, _ = compile_mapping_descriptor(descriptor)
    with pytest.raises(ValueError):
        verify_netcdf_inputs(mapping, [source])


def test_netcdf_soil_geometry_must_be_declared_not_inferred(prepared, tmp_path):
    """Variable-name selectors carry no depth; silence is refused."""

    descriptor, source = prepared
    document = json.loads(descriptor.read_text())
    del document["adapt"]["soil_policy"]["source_layers_m"]
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        author_adapter(
            descriptor_path=descriptor,
            input_files=[source],
            output_dir=tmp_path / "adapter",
        )
    assert "source_layers_m" in str(error.value)


def test_hpa_vertical_coordinate_is_compared_against_the_model_top_in_pa(
    prepared, tmp_path
):
    """250 hPa is 25000 Pa, not 250 Pa; the wrong one passes any model top."""

    descriptor, source = prepared
    document = json.loads(descriptor.read_text())
    document["adapt"]["model_top_pa"] = 5000.0   # 50 hPa, above the source top
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        author_adapter(
            descriptor_path=descriptor,
            input_files=[source],
            output_dir=tmp_path / "adapter",
        )
    assert "vertical-coverage check failed" in str(error.value)
    assert "25000" in str(error.value)
