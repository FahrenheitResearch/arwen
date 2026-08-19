"""An atmosphere-only source declares its missing surfaces; nothing fakes them.

Some producers publish a complete atmospheric state and NO land surface at
all -- no soil, no mask, no orography, no skin temperature.  Such a source
can be decoded and inspected, but it cannot initialize a WRF-like model
until a cross-source composition supplies the missing state.  The grammar
these tests hold:

* ``target.pending_composition_requirements`` names the CANONICAL fields
  the source does not publish.  Every name must be canonical, unmapped,
  and absent from ``required_fields`` -- a pending field that is also
  mapped or required is a contradiction, refused by name.
* ``soil_layer_count: 0`` is legal exactly when both soil canonicals are
  pending; any other combination refuses.
* A canonical field that is neither mapped nor declared pending still
  refuses exactly as before: this grammar narrows nothing silently.
* A composition document may ship as an explicit PENDING declaration
  (``gpuwm-cross-source-composition-pending-v1``); loading it refuses by
  naming the missing state it declares, so no init can be built on it.
* When ZERO GRIB messages match a mapping whose selectors pin producer
  identity octets, the refusal names the pinned-vs-observed identity --
  the case of two products published under one filename, separable only
  by ``subcenter``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm.mapped_composition import (
    PENDING_COMPOSITION_SCHEMA,
    load_composition,
)
from gpuwm.mapped_source import (
    _selector_identity_refusal,
    load_mapping,
)


THREE_D = {
    "air_temperature": "K",
    "specific_humidity": "kg kg-1",
    "eastward_wind": "m s-1",
    "northward_wind": "m s-1",
    "geopotential_height": "m",
}

SURFACE = {
    "surface_pressure": ("Pa", "surface"),
    "terrain_height": ("m", "surface"),
    "skin_temperature": ("K", "surface"),
    "air_temperature_2m": ("K", "surface"),
    "specific_humidity_2m": ("kg kg-1", "surface"),
    "eastward_wind_10m": ("m s-1", "surface"),
    "northward_wind_10m": ("m s-1", "surface"),
    "land_fraction": ("1", "surface"),
    "soil_temperature": ("K", "soil"),
    "volumetric_soil_moisture": ("m3 m-3", "soil"),
}

#: The canonical state an atmosphere-only producer does not publish.
ATMOSPHERE_ONLY_PENDING = [
    "land_fraction",
    "skin_temperature",
    "soil_temperature",
    "specific_humidity_2m",
    "surface_pressure",
    "terrain_height",
    "volumetric_soil_moisture",
]

POLICY = [
    "cloud_water_mixing_ratio",
    "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "vertical_velocity",
    "snow_water_equivalent",
    "snow_depth",
    "sea_ice_fraction",
]


def _grib2_selector(
    discipline: int, category: int, parameter: int,
    level_type: int, level_value: int | None = None,
) -> dict[str, object]:
    selector: dict[str, object] = {
        "format": "grib2",
        "discipline": discipline,
        "category": category,
        "parameter": parameter,
        "center": 7,
        "subcenter": 0,
        "level_type": level_type,
    }
    if level_value is not None:
        selector["level_value"] = level_value
    return selector


def _atmosphere_only_mapping(
    *,
    pending: list[str] | None = None,
    soil_layer_count: int = 0,
) -> dict[str, object]:
    """A synthetic GRIB2 mapping publishing only the atmospheric state."""

    pending = ATMOSPHERE_ONLY_PENDING if pending is None else pending
    three_d_selectors = {
        "air_temperature": _grib2_selector(0, 0, 0, 100),
        "specific_humidity": _grib2_selector(0, 1, 0, 100),
        "eastward_wind": _grib2_selector(0, 2, 2, 100),
        "northward_wind": _grib2_selector(0, 2, 3, 100),
        "geopotential_height": _grib2_selector(0, 3, 5, 100),
    }
    fields: dict[str, object] = {}
    for name, units in THREE_D.items():
        fields[name] = {
            "selectors": [three_d_selectors[name]],
            "units": {"source": units, "target": units},
            "source_axes": ["vertical", "y", "x"],
            "target_axes": ["vertical", "y", "x"],
            "location": "mass",
            "staggering": "none",
            "missing": {"kind": "reject"},
        }
    fields["air_pressure"] = {
        "selectors": [],
        "derivation": "pressure-from-coordinate",
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    surface_selectors = {
        "surface_pressure": _grib2_selector(0, 3, 0, 1),
        "terrain_height": _grib2_selector(0, 3, 4, 1),
        "skin_temperature": _grib2_selector(0, 0, 17, 1),
        "air_temperature_2m": _grib2_selector(0, 0, 0, 103, 2),
        "specific_humidity_2m": _grib2_selector(0, 1, 0, 103, 2),
        "eastward_wind_10m": _grib2_selector(0, 2, 2, 103, 10),
        "northward_wind_10m": _grib2_selector(0, 2, 3, 103, 10),
        "land_fraction": _grib2_selector(2, 0, 0, 1),
        "soil_temperature": _grib2_selector(2, 3, 18, 151),
        "volumetric_soil_moisture": _grib2_selector(2, 0, 25, 151),
    }
    for name, (units, location) in SURFACE.items():
        if name in pending:
            continue
        axes = ["soil", "y", "x"] if location == "soil" else ["y", "x"]
        fields[name] = {
            "selectors": [surface_selectors[name]],
            "units": {"source": units, "target": units},
            "source_axes": axes,
            "target_axes": axes,
            "location": location,
            "staggering": "none",
            "missing": {"kind": "reject"},
        }
    required = [
        {
            "name": name, "axes": ["vertical", "y", "x"],
            "location": "mass", "target_units": units,
        }
        for name, units in THREE_D.items()
    ]
    required.append({
        "name": "air_pressure", "axes": ["vertical", "y", "x"],
        "location": "mass", "target_units": "Pa",
    })
    for name, (units, location) in SURFACE.items():
        if name in pending:
            continue
        required.append({
            "name": name,
            "axes": ["soil", "y", "x"] if location == "soil" else ["y", "x"],
            "location": location, "target_units": units,
        })
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "synthetic-atmosphere-only-grib2",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {
                "kind": "pressure", "units": "Pa", "positive": "down",
                "levels": [50000.0, 85000.0, 100000.0],
            },
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": [
            {
                "name": "pressure-from-coordinate",
                "operation": "pressure_from_vertical_coordinate",
            },
        ],
        "target": {
            "name": "gpuwm/wrf-real initialization",
            "physics_suite": "WSM6+YSU+Noah",
            "max_dom": 1,
            "require_lateral_boundaries": True,
            "target_vertical_levels": 49,
            "soil_layer_count": soil_layer_count,
            "boundary_interval_seconds": 21600,
            "required_fields": required,
            "pending_composition_requirements": pending,
            "pressure_requirement": "air_pressure",
            "policy_controlled_fields": POLICY,
            "initialization_policies": {
                name: "unavailable_pending_cross_source_composition"
                for name in POLICY
            },
        },
    }


def _load(tmp_path: Path, mapping: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return load_mapping(path)


def test_a_target_may_declare_canonical_state_pending_composition(tmp_path):
    mapping = _load(tmp_path, _atmosphere_only_mapping())
    assert sorted(
        mapping["target"]["pending_composition_requirements"]
    ) == sorted(ATMOSPHERE_ONLY_PENDING)


def test_pending_names_must_be_canonical(tmp_path):
    raw = _atmosphere_only_mapping()
    raw["target"]["pending_composition_requirements"] = (
        ATMOSPHERE_ONLY_PENDING + ["convective_inhibition"]
    )
    with pytest.raises(ValueError, match="convective_inhibition"):
        _load(tmp_path, raw)


def test_a_pending_field_must_not_be_mapped(tmp_path):
    raw = _atmosphere_only_mapping(
        pending=[
            name for name in ATMOSPHERE_ONLY_PENDING
            if name != "land_fraction"
        ],
    )
    # land_fraction is now mapped and required; declaring it pending too
    # is the contradiction under test.
    raw["target"]["pending_composition_requirements"] = (
        list(raw["target"]["pending_composition_requirements"])
        + ["land_fraction"]
    )
    with pytest.raises(ValueError, match="land_fraction"):
        _load(tmp_path, raw)


def test_a_pending_field_must_not_be_required(tmp_path):
    raw = _atmosphere_only_mapping()
    raw["target"]["required_fields"] = list(
        raw["target"]["required_fields"]
    ) + [{
        "name": "land_fraction", "axes": ["y", "x"],
        "location": "surface", "target_units": "1",
    }]
    with pytest.raises(ValueError, match="land_fraction"):
        _load(tmp_path, raw)


def test_zero_soil_layers_requires_both_soil_canonicals_pending(tmp_path):
    partial = [
        name for name in ATMOSPHERE_ONLY_PENDING
        if name != "soil_temperature"
    ]
    with pytest.raises(ValueError, match="soil_layer_count"):
        _load(tmp_path, _atmosphere_only_mapping(pending=partial))


def test_pending_soil_refuses_a_nonzero_layer_count(tmp_path):
    with pytest.raises(ValueError, match="soil_layer_count"):
        _load(
            tmp_path,
            _atmosphere_only_mapping(soil_layer_count=2),
        )


def test_absent_canonicals_without_pending_still_refuse(tmp_path):
    """The grammar narrows nothing: an undeclared absence is still fatal."""

    raw = _atmosphere_only_mapping()
    del raw["target"]["pending_composition_requirements"]
    raw["target"]["soil_layer_count"] = 4
    with pytest.raises(ValueError, match="canonical requirement"):
        _load(tmp_path, raw)


def test_a_pending_composition_document_refuses_by_name(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(_atmosphere_only_mapping(), indent=2) + "\n",
        encoding="utf-8",
    )
    composition_path = tmp_path / "composition.json"
    composition_path.write_text(json.dumps({
        "schema": PENDING_COMPOSITION_SCHEMA,
        "name": "synthetic-atmosphere-only-pending",
        "pending": {
            "missing_canonical_state": ATMOSPHERE_ONLY_PENDING,
            "reason": (
                "the producer publishes no land surface, terrain, mask, "
                "skin or surface-pressure state in any file"
            ),
            "supply_route": (
                "a cross-source composition must borrow the missing state "
                "from a producer that publishes it"
            ),
        },
    }, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as refusal:
        load_composition(composition_path, mapping_path)
    message = str(refusal.value)
    for name in ATMOSPHERE_ONLY_PENDING:
        assert name in message
    assert "no land surface" in message


def test_zero_selector_matches_name_the_pinned_identity_octets(tmp_path):
    """Two products under one filename separate only by subcenter."""

    mapping = _load(tmp_path, _atmosphere_only_mapping())
    observed_rows = [
        {"center": "7", "subcenter": "2",
         "master_table_version": "2", "local_table_version": "1"},
    ] * 5
    message = _selector_identity_refusal(
        mapping, observed_rows, [Path("imposter.grib2")],
    )
    assert "subcenter" in message
    assert "pin" in message and "0" in message and "2" in message
    # center agrees on both sides, so it must NOT be blamed.
    assert "center=7" not in message.replace("subcenter", "")


def test_matching_identity_still_gets_a_useful_zero_match_message(tmp_path):
    mapping = _load(tmp_path, _atmosphere_only_mapping())
    observed_rows = [
        {"center": "7", "subcenter": "0",
         "master_table_version": "2", "local_table_version": "1"},
    ] * 3
    message = _selector_identity_refusal(
        mapping, observed_rows, [Path("empty.grib2")],
    )
    assert "3" in message and "match" in message
