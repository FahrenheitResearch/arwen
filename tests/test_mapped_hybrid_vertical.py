"""hybrid_sigma_pressure is a consumed vertical kind, not a declared one.

The L137 proof lane (gpuwm-cases favor-2026-05-30/l137-mapping/PROOF.md)
exhausted the mapping grammar on real ERA5 model-level bytes and found the
kind validated everywhere and consumed nowhere: both engines emitted empty
A/B coefficient tuples, no derivation could materialize pressure from the
coordinate, and the regular-snapshot packer indexed ``air_pressure``
unconditionally.  These tests pin the closure of each named gap:

* G1 — the A/B coefficient entry: GRIB2 pv coordinate octets as the
  primary channel, inline ``vertical.hybrid_a``/``hybrid_b`` literal
  arrays as the declared-data fallback, counts held to N+1 (half-level
  interfaces) or N (full levels) against the declared ladder.
* G2 — hybrid pressure materialization: ``pressure_from_vertical_coordinate``
  accepts the hybrid kind and produces p = A + B*ps half levels, full
  levels as the mean of their bounding interfaces.
* G3 — 3-D geopotential height built hydrostatically from surface
  geopotential and virtual temperature up the hybrid ladder
  (``geopotential_height_hydrostatic``), since ECMWF does not archive z
  on all 137 model levels.
* The packer names its pressure requirement instead of raising a bare
  ``KeyError``.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import gpuwm.mapped_source as mapped_source
from gpuwm.mapped_source import (
    CanonicalField,
    _assemble_grib,
    _DecodedCollection,
    _evaluate_derivation,
    _GribRecord,
    _materialize_frames,
    load_mapping,
    mapped_frames_to_regular_snapshots,
)
from gpuwm.source_frame import VerticalDescriptor, _validate_vertical


# One small hybrid ladder, top of atmosphere first (the ECMWF order the
# pv octets use): N = 3 full levels, N + 1 = 4 half-level interfaces.
NLEV = 3
A_HALF = (0.0, 6000.0, 4000.0, 0.0)
B_HALF = (0.0, 0.24, 0.7, 1.0)
PS = 100000.0
# p_half = A + B*ps = [0, 30000, 74000, 100000] Pa
P_HALF = tuple(a + b * PS for a, b in zip(A_HALF, B_HALF))
# full level = mean of its bounding interfaces
P_FULL = tuple(
    0.5 * (P_HALF[k] + P_HALF[k + 1]) for k in range(NLEV)
)
NY, NX = 4, 5


def _grib2_selector(discipline: int, category: int, parameter: int) -> dict:
    return {
        "format": "grib2", "discipline": discipline, "category": category,
        "parameter": parameter, "level_type": 105, "center": 98,
    }


def _hybrid_field(
    name: str, units: str, axes: list[str], location: str,
    selector: dict | None,
) -> dict:
    field = {
        "selectors": [] if selector is None else [selector],
        "units": {"source": units, "target": units, "scale": 1.0, "offset": 0.0},
        "source_axes": axes,
        "target_axes": axes,
        "location": location,
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    return field


THREE_D_IDENTITY = {
    "air_temperature": ("K", (0, 0, 0)),
    "specific_humidity": ("kg kg-1", (0, 1, 0)),
    "eastward_wind": ("m s-1", (0, 2, 2)),
    "northward_wind": ("m s-1", (0, 2, 3)),
}
SURFACE_IDENTITY = {
    "surface_pressure": ("Pa", (0, 3, 0)),
    "terrain_height": ("m", (0, 3, 5)),
    "skin_temperature": ("K", (0, 0, 17)),
    "air_temperature_2m": ("K", (0, 0, 18)),
    "specific_humidity_2m": ("kg kg-1", (0, 1, 1)),
    "eastward_wind_10m": ("m s-1", (0, 2, 4)),
    "northward_wind_10m": ("m s-1", (0, 2, 5)),
    "land_fraction": ("1", (2, 0, 0)),
}
SOIL_IDENTITY = {
    "soil_temperature": ("K", (2, 3, 18)),
    "volumetric_soil_moisture": ("m3 m-3", (2, 3, 25)),
}
SOIL_DEPTHS = (0.05, 0.25, 0.7, 1.5)


def _hybrid_mapping(
    *,
    hybrid_a: tuple[float, ...] | None = A_HALF,
    hybrid_b: tuple[float, ...] | None = B_HALF,
    legacy_fields: bool = False,
    surface_pressure_field: str = "surface_pressure",
    with_air_pressure: bool = True,
) -> dict:
    """A self-contained grib2 hybrid mapping: every canonical field direct."""

    fields = {
        name: _hybrid_field(
            name, units, ["vertical", "y", "x"], "mass",
            _grib2_selector(*identity),
        )
        for name, (units, identity) in THREE_D_IDENTITY.items()
    }
    for name, (units, (discipline, category, parameter)) in SURFACE_IDENTITY.items():
        fields[name] = _hybrid_field(
            name, units, ["y", "x"], "surface",
            {"format": "grib2", "discipline": discipline, "category": category,
             "parameter": parameter, "level_type": 1, "center": 98},
        )
    for name, (units, (discipline, category, parameter)) in SOIL_IDENTITY.items():
        field = _hybrid_field(name, units, ["soil", "y", "x"], "soil", None)
        field["selectors"] = [
            {"format": "grib2", "discipline": discipline, "category": category,
             "parameter": parameter, "level_type": 106, "level_value": depth,
             "center": 98}
            for depth in SOIL_DEPTHS
        ]
        fields[name] = field
    if with_air_pressure:
        fields["air_pressure"] = {
            "selectors": [],
            "derivation": "pressure-from-hybrid",
            "units": {"source": "Pa", "target": "Pa", "scale": 1.0, "offset": 0.0},
            "source_axes": ["vertical", "y", "x"],
            "target_axes": ["vertical", "y", "x"],
            "location": "mass",
            "staggering": "none",
            "missing": {"kind": "reject"},
        }
    derivations = [
        {"name": "pressure-from-hybrid",
         "operation": "pressure_from_vertical_coordinate"},
        {"name": "height-hydrostatic",
         "operation": "geopotential_height_hydrostatic",
         "temperature": "air_temperature",
         "specific_humidity": "specific_humidity",
         "surface_geopotential_height": "terrain_height"},
    ]
    fields["geopotential_height"] = {
        "selectors": [],
        "derivation": "height-hydrostatic",
        "units": {"source": "m", "target": "m", "scale": 1.0, "offset": 0.0},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass",
        "staggering": "none",
        "missing": {"kind": "reject"},
    }
    vertical: dict[str, object] = {
        "kind": "hybrid_sigma_pressure", "units": "1", "positive": "down",
        "levels": [1.0, 2.0, 3.0],
        "surface_pressure_field": surface_pressure_field,
    }
    if hybrid_a is not None:
        vertical["hybrid_a"] = list(hybrid_a)
    if hybrid_b is not None:
        vertical["hybrid_b"] = list(hybrid_b)
    if legacy_fields:
        vertical["hybrid_a_field"] = "hybrid_a"
        vertical["hybrid_b_field"] = "hybrid_b"
    required = [
        {"name": name, "axes": ["vertical", "y", "x"], "location": "mass",
         "target_units": units}
        for name, (units, _identity) in THREE_D_IDENTITY.items()
    ]
    required.append({
        "name": "geopotential_height", "axes": ["vertical", "y", "x"],
        "location": "mass", "target_units": "m",
    })
    required += [
        {"name": name, "axes": ["y", "x"], "location": "surface",
         "target_units": units}
        for name, (units, _identity) in SURFACE_IDENTITY.items()
    ] + [
        {"name": name, "axes": ["soil", "y", "x"], "location": "soil",
         "target_units": units}
        for name, (units, _identity) in SOIL_IDENTITY.items()
    ]
    policy = [
        "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
        "cloud_ice_mixing_ratio", "snow_mixing_ratio",
        "graupel_or_hail_mixing_ratio", "vertical_velocity",
        "snow_water_equivalent", "snow_depth", "sea_ice_fraction",
    ]
    return {
        "schema": "rw-wps.mapping.v1",
        "name": "synthetic-hybrid-model-level",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": vertical,
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": derivations,
        "target": {
            "name": "gpuwm/wrf-real initialization",
            "physics_suite": "WSM6+YSU+Noah",
            "max_dom": 1,
            "require_lateral_boundaries": False,
            "target_vertical_levels": 49,
            "soil_layer_count": len(SOIL_DEPTHS),
            "boundary_interval_seconds": 3600,
            "required_fields": required,
            "pressure_requirement": "hybrid_coordinate",
            "policy_controlled_fields": policy,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation" for name in policy
            },
        },
    }


def _load(tmp_path: Path, mapping: dict) -> dict:
    path = tmp_path / "hybrid.mapping.json"
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return load_mapping(path)


# --------------------------------------------------------------------------
# G1 schema: the coefficient entries
# --------------------------------------------------------------------------


def test_inline_hybrid_coefficient_literals_load(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping())
    vertical = mapping["coordinates"]["vertical"]
    assert vertical["hybrid_a"] == list(A_HALF)
    assert vertical["hybrid_b"] == list(B_HALF)


def test_legacy_hybrid_coefficient_field_names_are_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="pv coordinate octets"):
        _load(tmp_path, _hybrid_mapping(legacy_fields=True))


def test_hybrid_literals_must_be_declared_together(tmp_path):
    with pytest.raises(ValueError, match="declared together"):
        _load(tmp_path, _hybrid_mapping(hybrid_b=None))


def test_hybrid_literal_count_is_held_to_the_declared_ladder(tmp_path):
    bad = _hybrid_mapping(
        hybrid_a=(0.0, 1.0, 2.0, 3.0, 4.0), hybrid_b=(0.0, 0.1, 0.2, 0.5, 1.0),
    )
    with pytest.raises(ValueError) as error:
        _load(tmp_path, bad)
    message = str(error.value)
    assert "5" in message and "4" in message and "3" in message


def test_full_level_literal_count_is_accepted(tmp_path):
    mapping = _hybrid_mapping(
        hybrid_a=(5000.0, 2000.0, 0.0), hybrid_b=(0.1, 0.5, 0.99),
    )
    loaded = _load(tmp_path, mapping)
    assert len(loaded["coordinates"]["vertical"]["hybrid_a"]) == NLEV


def test_hybrid_b_literals_are_held_to_the_unit_interval(tmp_path):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _load(tmp_path, _hybrid_mapping(hybrid_b=(0.0, 0.24, 0.7, 1.5)))


def test_surface_pressure_field_must_name_a_declared_field(tmp_path):
    with pytest.raises(ValueError, match="does not declare"):
        _load(
            tmp_path,
            _hybrid_mapping(surface_pressure_field="no_such_field"),
        )


def test_hybrid_without_surface_pressure_field_is_incomplete(tmp_path):
    mapping = _hybrid_mapping()
    del mapping["coordinates"]["vertical"]["surface_pressure_field"]
    with pytest.raises(ValueError, match="surface_pressure_field"):
        _load(tmp_path, mapping)


def test_pressure_derivation_is_legal_on_a_hybrid_vertical(tmp_path):
    # The proof lane's exact refusal: "pressure_from_vertical_coordinate
    # requires a pressure vertical coordinate" (mapped_source.py:1102-1106
    # at tip 11b269223).  The kind is now consumed, so the mapping loads.
    mapping = _load(tmp_path, _hybrid_mapping())
    assert mapping["fields"]["air_pressure"]["derivation"] == "pressure-from-hybrid"


def test_hydrostatic_height_derivation_is_in_the_catalog(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping())
    names = {item["name"] for item in mapping["derivations"]}
    assert "height-hydrostatic" in names


def test_hydrostatic_height_requires_a_hybrid_vertical(tmp_path):
    mapping = _hybrid_mapping()
    mapping["coordinates"]["vertical"] = {
        "kind": "pressure", "units": "hPa", "positive": "down",
        "levels": [1000.0, 850.0, 700.0],
    }
    with pytest.raises(ValueError, match="hybrid_sigma_pressure"):
        _load(tmp_path, mapping)


# --------------------------------------------------------------------------
# G1 frame contract: coefficient counts on the header
# --------------------------------------------------------------------------


def test_hybrid_descriptor_accepts_interface_coefficients():
    _validate_vertical("atmosphere", VerticalDescriptor(
        coordinate="hybrid", level_count=NLEV,
        level_values=(1.0, 2.0, 3.0),
        a_coefficients=A_HALF, b_coefficients=B_HALF,
        positive="down", units="1",
    ))


def test_hybrid_descriptor_accepts_full_level_coefficients():
    _validate_vertical("atmosphere", VerticalDescriptor(
        coordinate="hybrid", level_count=NLEV,
        level_values=(1.0, 2.0, 3.0),
        a_coefficients=(5000.0, 2000.0, 0.0),
        b_coefficients=(0.1, 0.5, 0.99),
        positive="down", units="1",
    ))


def test_hybrid_descriptor_still_refuses_empty_coefficients():
    # The proof lane's probe 5: what both engines emitted at the frozen
    # tip.  The refusal survives and names both accepted counts.
    with pytest.raises(ValueError) as error:
        _validate_vertical("atmosphere", VerticalDescriptor(
            coordinate="hybrid", level_count=NLEV,
            level_values=(1.0, 2.0, 3.0),
            positive="down", units="1",
        ))
    message = str(error.value)
    assert "4" in message and "3" in message


# --------------------------------------------------------------------------
# G1 decode: pv harvest into the collection
# --------------------------------------------------------------------------


def _record(
    field: str, index: int, level: float,
    *,
    pv: tuple[float, ...] | None,
    base: float,
) -> _GribRecord:
    if field in THREE_D_IDENTITY:
        discipline, category, parameter = THREE_D_IDENTITY[field][1]
        level_type = 105
    elif field in SURFACE_IDENTITY:
        discipline, category, parameter = SURFACE_IDENTITY[field][1]
        level_type = 1
    else:
        discipline, category, parameter = SOIL_IDENTITY[field][1]
        level_type = 106
    return _GribRecord(
        source=Path("synthetic.grib2"),
        index=index,
        reference_time=datetime(2026, 5, 30, 0),
        valid_time=datetime(2026, 5, 30, 0),
        member=None,
        parameter=parameter,
        level_type=level_type,
        level_value=level,
        table_version=None,
        center=98,
        subcenter=0,
        master_table_version=5,
        local_table_version=0,
        discipline=discipline,
        category=category,
        second_level_type=255,
        second_level_value=0.0,
        process_identity=(0, 153),
        time_semantics=(0,),
        coordinate_values=() if pv is None else pv,
        values=np.full((NY, NX), base, dtype=np.float64),
        latitude=np.linspace(48.0, 51.0, NY),
        longitude=np.linspace(16.0, 20.0, NX),
        grid_fingerprint="synthetic-grid",
    )


def _records(*, pv: tuple[float, ...] | None) -> list[_GribRecord]:
    records = []
    index = 0
    for field, base in (
        ("air_temperature", 260.0),
        ("specific_humidity", 0.004),
        ("eastward_wind", 8.0),
        ("northward_wind", -2.0),
    ):
        for level in (1.0, 2.0, 3.0):
            records.append(_record(
                field, index, level, pv=pv, base=base + level,
            ))
            index += 1
    for field, base in (
        ("surface_pressure", PS),
        ("terrain_height", 200.0),
        ("skin_temperature", 288.0),
        ("air_temperature_2m", 287.0),
        ("specific_humidity_2m", 0.006),
        ("eastward_wind_10m", 4.0),
        ("northward_wind_10m", 1.0),
        ("land_fraction", 1.0),
    ):
        records.append(_record(field, index, 0.0, pv=None, base=base))
        index += 1
    for field, base in (
        ("soil_temperature", 285.0),
        ("volumetric_soil_moisture", 0.3),
    ):
        for depth in SOIL_DEPTHS:
            records.append(_record(field, index, depth, pv=None, base=base))
            index += 1
    return records


PV = tuple(A_HALF) + tuple(B_HALF)


def test_pv_octets_are_harvested_into_the_collection(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(hybrid_a=None, hybrid_b=None))
    collection = _assemble_grib(mapping, _records(pv=PV))
    assert collection.hybrid_a is not None
    np.testing.assert_array_equal(collection.hybrid_a, np.asarray(A_HALF))
    np.testing.assert_array_equal(collection.hybrid_b, np.asarray(B_HALF))


def test_disagreeing_pv_octets_refuse(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(hybrid_a=None, hybrid_b=None))
    records = _records(pv=PV)
    tampered = PV[:1] + (7777.0,) + PV[2:]
    records[5] = _record(
        "specific_humidity", 5, 3.0, pv=tampered, base=0.004 + 3.0,
    )
    with pytest.raises(ValueError, match="do not share one pv"):
        _assemble_grib(mapping, records)


def test_no_pv_and_no_literals_refuses_naming_both_channels(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(hybrid_a=None, hybrid_b=None))
    with pytest.raises(ValueError) as error:
        _assemble_grib(mapping, _records(pv=None))
    message = str(error.value)
    assert "pv" in message and "hybrid_a" in message


def test_literals_disagreeing_with_pv_refuse_naming_the_index(tmp_path):
    shifted = (0.0, 6100.0) + A_HALF[2:]
    mapping = _load(tmp_path, _hybrid_mapping(hybrid_a=shifted))
    with pytest.raises(ValueError, match="disagree"):
        _assemble_grib(mapping, _records(pv=PV))


def test_pv_half_count_is_held_to_the_declared_ladder(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(hybrid_a=None, hybrid_b=None))
    long_pv = tuple(float(value) for value in range(12))
    with pytest.raises(ValueError) as error:
        _assemble_grib(mapping, _records(pv=long_pv))
    message = str(error.value)
    assert "6" in message and "4" in message and "3" in message


def test_literals_supply_the_ladder_when_the_bytes_carry_no_pv(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping())
    collection = _assemble_grib(mapping, _records(pv=None))
    np.testing.assert_array_equal(collection.hybrid_a, np.asarray(A_HALF))
    np.testing.assert_array_equal(collection.hybrid_b, np.asarray(B_HALF))


# --------------------------------------------------------------------------
# G2: hybrid pressure materialization
# --------------------------------------------------------------------------


def _collection_with_fields(tmp_path, mapping=None) -> tuple[dict, _DecodedCollection]:
    mapping = mapping or _load(tmp_path, _hybrid_mapping())
    collection = _assemble_grib(mapping, _records(pv=PV))
    return mapping, collection


def _available(collection) -> dict[str, CanonicalField]:
    available = {}
    for (_time, _member, name), value in collection.direct.items():
        available[name] = CanonicalField(
            name=name, units="", axes=value.axes, location="mass",
            staggering="none", values=value.values,
            missing_count=value.missing_count,
            source_references=value.references,
        )
    return available


def test_hybrid_pressure_is_the_mean_of_its_bounding_half_levels(tmp_path):
    mapping, collection = _collection_with_fields(tmp_path)
    available = _available(collection)
    operation = {"name": "pressure-from-hybrid",
                 "operation": "pressure_from_vertical_coordinate"}
    values, axes, references = _evaluate_derivation(
        operation, available, collection,
        mapping["fields"]["air_pressure"], "air_pressure",
        vertical=mapping["coordinates"]["vertical"],
    )
    assert axes == ("vertical", "y", "x")
    for k in range(NLEV):
        np.testing.assert_allclose(values[k], P_FULL[k], rtol=1e-13)
    # strictly monotonic: pressure decreases with level index toward the top
    assert np.all(np.diff(values, axis=0) > 0.0)
    assert any("@coordinate.vertical" in item for item in references)


def test_full_level_coefficients_materialize_pressure_directly(tmp_path):
    a_full = (5000.0, 2000.0, 0.0)
    b_full = (0.1, 0.5, 0.99)
    mapping = _load(tmp_path, _hybrid_mapping(
        hybrid_a=a_full, hybrid_b=b_full,
    ))
    collection = _assemble_grib(mapping, _records(pv=None))
    available = _available(collection)
    operation = {"name": "pressure-from-hybrid",
                 "operation": "pressure_from_vertical_coordinate"}
    values, _axes, _references = _evaluate_derivation(
        operation, available, collection,
        mapping["fields"]["air_pressure"], "air_pressure",
        vertical=mapping["coordinates"]["vertical"],
    )
    for k in range(NLEV):
        np.testing.assert_allclose(values[k], a_full[k] + b_full[k] * PS)


def test_a_non_monotonic_ladder_refuses_by_name(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(
        hybrid_a=(0.0, 4000.0, 6000.0, 0.0),
        hybrid_b=(0.0, 0.7, 0.24, 1.0),
    ))
    collection = _assemble_grib(mapping, _records(pv=None))
    available = _available(collection)
    operation = {"name": "pressure-from-hybrid",
                 "operation": "pressure_from_vertical_coordinate"}
    with pytest.raises(ValueError, match="strictly"):
        _evaluate_derivation(
            operation, available, collection,
            mapping["fields"]["air_pressure"], "air_pressure",
            vertical=mapping["coordinates"]["vertical"],
        )


# --------------------------------------------------------------------------
# G3: hydrostatic geopotential height
# --------------------------------------------------------------------------

RD = 287.06
VIRTUAL = 0.609133
GRAVITY = 9.80665


def _height_operation() -> dict:
    return {
        "name": "height-hydrostatic",
        "operation": "geopotential_height_hydrostatic",
        "temperature": "air_temperature",
        "specific_humidity": "specific_humidity",
        "surface_geopotential_height": "terrain_height",
    }


def _isothermal_available(collection) -> dict[str, CanonicalField]:
    available = _available(collection)

    def constant(name: str, value: float, three_d: bool) -> CanonicalField:
        shape = (NLEV, NY, NX) if three_d else (NY, NX)
        axes = ("vertical", "y", "x") if three_d else ("y", "x")
        return CanonicalField(
            name=name, units="", axes=axes, location="mass",
            staggering="none",
            values=np.full(shape, value, dtype=np.float64),
            missing_count=0, source_references=(f"@test.{name}",),
        )

    available["air_temperature"] = constant("air_temperature", 280.0, True)
    available["specific_humidity"] = constant("specific_humidity", 0.0, True)
    available["terrain_height"] = constant("terrain_height", 100.0, False)
    available["surface_pressure"] = constant("surface_pressure", PS, False)
    return available


def test_hydrostatic_height_matches_the_isothermal_analytic_answer(tmp_path):
    # For constant virtual temperature the half-level accumulation
    # telescopes to the exact analytic z(p) = z_s + (Rd Tv / g) ln(ps/p),
    # independent of how the integration is coded.
    mapping, collection = _collection_with_fields(tmp_path)
    available = _isothermal_available(collection)
    values, axes, references = _evaluate_derivation(
        _height_operation(), available, collection,
        mapping["fields"]["geopotential_height"], "geopotential_height",
        vertical=mapping["coordinates"]["vertical"],
    )
    assert axes == ("vertical", "y", "x")
    tv = 280.0
    # Full levels are bounded by their interfaces; the interface heights
    # are analytic.  ECMWF's alpha places the full level between them.
    z_half = [
        100.0 + (RD * tv / GRAVITY) * math.log(P_HALF[NLEV] / P_HALF[k])
        for k in range(1, NLEV + 1)
    ]
    # z_half above full level k is z_half[k-1] (k=1..N-1); the top full
    # level (k=0) has no finite upper interface.
    for k in range(1, NLEV):
        assert np.all(values[k] > z_half[k]), f"level {k} below its lower interface"
        assert np.all(values[k] < z_half[k - 1]), f"level {k} above its upper interface"
    assert np.all(values[0] > z_half[0])
    # ECMWF's own full-level rule: z_f = z_h(below) + Rd Tv alpha / g.
    ln = math.log
    alpha_bottom = 1.0 - (P_HALF[2] / (P_HALF[3] - P_HALF[2])) * ln(P_HALF[3] / P_HALF[2])
    expected_bottom = 100.0 + (RD * tv / GRAVITY) * alpha_bottom
    np.testing.assert_allclose(values[NLEV - 1], expected_bottom, rtol=1e-12)
    alpha_top = ln(2.0)
    expected_top = z_half[0] + (RD * tv / GRAVITY) * alpha_top
    np.testing.assert_allclose(values[0], expected_top, rtol=1e-12)
    # heights increase upward (toward smaller level index)
    assert np.all(np.diff(values, axis=0) < 0.0)
    assert references[0] == "@derived.hydrostatic"


def test_moisture_raises_the_hydrostatic_column(tmp_path):
    mapping, collection = _collection_with_fields(tmp_path)
    dry = _isothermal_available(collection)
    moist = _isothermal_available(collection)
    moist["specific_humidity"] = CanonicalField(
        name="specific_humidity", units="", axes=("vertical", "y", "x"),
        location="mass", staggering="none",
        values=np.full((NLEV, NY, NX), 0.01, dtype=np.float64),
        missing_count=0, source_references=("@test.q",),
    )
    dry_z, _axes, _refs = _evaluate_derivation(
        _height_operation(), dry, collection,
        mapping["fields"]["geopotential_height"], "geopotential_height",
        vertical=mapping["coordinates"]["vertical"],
    )
    moist_z, _axes, _refs = _evaluate_derivation(
        _height_operation(), moist, collection,
        mapping["fields"]["geopotential_height"], "geopotential_height",
        vertical=mapping["coordinates"]["vertical"],
    )
    assert np.all(moist_z > dry_z)


def test_hydrostatic_height_requires_interface_coefficients(tmp_path):
    mapping = _load(tmp_path, _hybrid_mapping(
        hybrid_a=(5000.0, 2000.0, 0.0), hybrid_b=(0.1, 0.5, 0.99),
    ))
    collection = _assemble_grib(mapping, _records(pv=None))
    available = _isothermal_available(collection)
    with pytest.raises(ValueError, match="interface"):
        _evaluate_derivation(
            _height_operation(), available, collection,
            mapping["fields"]["geopotential_height"], "geopotential_height",
            vertical=mapping["coordinates"]["vertical"],
        )


# --------------------------------------------------------------------------
# The packer names its pressure requirement
# --------------------------------------------------------------------------


def test_full_hybrid_frames_materialize_and_carry_coefficients(tmp_path):
    mapping, collection = _collection_with_fields(tmp_path)
    frames = _materialize_frames(
        mapping, collection, mapping_sha256="0" * 64, input_sha256={},
    )
    assert len(frames) == 1
    frame = frames[0]
    assert "air_pressure" in frame.fields
    assert "geopotential_height" in frame.fields
    descriptor = frame.header.vertical_coordinates["atmosphere"]
    assert descriptor.a_coefficients == A_HALF
    assert descriptor.b_coefficients == B_HALF
    pressure = frame.fields["air_pressure"].values
    for k in range(NLEV):
        np.testing.assert_allclose(pressure[k], P_FULL[k], rtol=1e-13)


def test_packer_names_the_missing_pressure_instead_of_a_keyerror(tmp_path):
    # The proof lane's probe 6: a coefficient-complete hybrid frame died
    # at the join with a bare KeyError 'air_pressure'.  Absent pressure
    # now refuses with the fix in the message.
    mapping = _load(tmp_path, _hybrid_mapping(with_air_pressure=False))
    collection = _assemble_grib(mapping, _records(pv=PV))
    frames = _materialize_frames(
        mapping, collection, mapping_sha256="0" * 64, input_sha256={},
    )
    with pytest.raises(ValueError, match="pressure_from_vertical_coordinate"):
        mapped_frames_to_regular_snapshots(frames)
