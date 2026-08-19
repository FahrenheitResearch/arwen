"""Generic mapped-route capabilities for analysis-step-invariant sources.

Two properties of real feeds, both TABLE-declarable and model-free:

* GDT-0 regular grids arrive in BOTH row orders.  Scan mode 0x40
  (south-to-north) was the only admitted order; 0x00 (north-to-south,
  the majority convention for global producers) is now admitted by row
  reversal, and every other scan mode still refuses by name.
* Some producers publish their invariant surface fields (land mask,
  surface geopotential) at the analysis step ONLY, not at every forecast
  step.  A mapping field may declare ``time_binding: cycle_invariant``
  and a composition terrain supplement may declare
  ``time_alignment: cycle_invariant_broadcast``; the decoded record is
  then broadcast to every frame of the one source cycle, with the
  invariance and single-cycle constraints enforced rather than assumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import gpuwm.mapped_source as ms
from gpuwm.mapped_composition import _compose_terrain, load_composition
from gpuwm.mapped_source import load_mapping


ROOT = Path(__file__).parents[1]
ERA5_MAPPING = ROOT / "configs" / "rw-wps-era5-1974-probe.mapping.json"
ERA5_COMPOSITION = ROOT / "configs" / "rw-wps-era5-1974-terrain.composition.json"


# ---------------------------------------------------------------------------
# GDT-0 row order
# ---------------------------------------------------------------------------


def _scan_row(scan_mode: str, *, lat1: str, ny: int, nx: int = 2) -> dict:
    return {
        "gdt": "0", "scan_mode": scan_mode, "nx": str(nx), "ny": str(ny),
        "lat1": lat1, "lon1": "0", "dx": "0.25", "dy": "0.25",
    }


def test_scan_mode_0x40_row_orientation_is_byte_identical_to_history():
    raw = np.arange(8, dtype=np.float64)
    latitude, _longitude, values = ms._regular_latlon_frame(
        _scan_row("0x40", lat1="30.0", ny=4), raw.copy())
    np.testing.assert_array_equal(latitude, [30.0, 30.25, 30.5, 30.75])
    np.testing.assert_array_equal(values, raw.reshape(4, 2))


def test_scan_mode_0x00_rows_reverse_to_an_ascending_latitude_axis():
    raw = np.arange(721 * 2, dtype=np.float64)
    latitude, _longitude, values = ms._regular_latlon_frame(
        _scan_row("0x00", lat1="90.0", ny=721), raw.copy())
    assert latitude[0] == -90.0 and latitude[-1] == 90.0
    assert np.all(np.diff(latitude) > 0)
    np.testing.assert_array_equal(values, raw.reshape(721, 2)[::-1, :])


def test_other_scan_modes_refuse_naming_both_admitted_orders():
    raw = np.zeros(20, dtype=np.float64)
    with pytest.raises(ValueError, match="0x40.*0x00|0x00.*0x40"):
        ms._regular_latlon_frame(_scan_row("0x80", lat1="90.0", ny=10), raw)
    with pytest.raises(ValueError, match="0x20"):
        ms._regular_latlon_frame(_scan_row("0x20", lat1="90.0", ny=10), raw)


# ---------------------------------------------------------------------------
# mapping field time_binding
# ---------------------------------------------------------------------------


TWENTYCR_MAPPING = ROOT / "gpuwm" / "authorities" / \
    "rw-wps-20crv3-member-grib2.mapping.json"
TWENTYCR_NETCDF_MAPPING = ROOT / "gpuwm" / "authorities" / \
    "rw-wps-20crv3-netcdf.mapping.json"


def _grib2_mapping() -> dict:
    """A raw assembly-level mapping, mirroring the level-subset fixture."""

    return {
        "schema": "rw-wps.mapping.v1",
        "name": "invariant-probe",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "positive": "down", "levels": [100000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": {
            "surface_pressure": {
                "selectors": [{
                    "format": "grib2", "discipline": 0, "category": 3,
                    "parameter": 0, "level_type": 1,
                }],
                "units": {"source": "Pa", "target": "Pa"},
                "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                "location": "surface", "staggering": "none",
                "missing": {"kind": "reject"},
            },
            "land_fraction": {
                "selectors": [{
                    "format": "grib2", "discipline": 2, "category": 0,
                    "parameter": 0, "level_type": 1,
                }],
                "units": {"source": "1", "target": "1"},
                "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                "location": "surface", "staggering": "none",
                "missing": {"kind": "reject"},
                "time_binding": "cycle_invariant",
            },
        },
        "derivations": [],
        "target": {
            "required_fields": [
                {"name": "surface_pressure", "axes": ["y", "x"],
                 "location": "surface", "target_units": "Pa"},
                {"name": "land_fraction", "axes": ["y", "x"],
                 "location": "surface", "target_units": "1"},
            ],
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_time_binding_cycle_invariant_loads_on_a_grib2_mapping(tmp_path):
    payload = json.loads(TWENTYCR_MAPPING.read_text(encoding="utf-8"))
    payload["fields"]["land_fraction"]["time_binding"] = "cycle_invariant"
    mapping = load_mapping(_write(tmp_path / "ok.json", payload))
    assert mapping["fields"]["land_fraction"]["time_binding"] == "cycle_invariant"
    assert "time_binding" not in mapping["fields"]["surface_pressure"]


def test_time_binding_refuses_unknown_values(tmp_path):
    payload = json.loads(TWENTYCR_MAPPING.read_text(encoding="utf-8"))
    payload["fields"]["land_fraction"]["time_binding"] = "sometimes"
    with pytest.raises(ValueError, match="time_binding"):
        load_mapping(_write(tmp_path / "bad-value.json", payload))


def test_time_binding_refuses_derived_fields(tmp_path):
    payload = json.loads(TWENTYCR_MAPPING.read_text(encoding="utf-8"))
    derived = payload["fields"]["specific_humidity_2m"]
    assert derived.get("derivation")
    derived["time_binding"] = "cycle_invariant"
    with pytest.raises(ValueError, match="time_binding.*direct"):
        load_mapping(_write(tmp_path / "derived.json", payload))


def test_time_binding_refuses_netcdf_mappings_by_name(tmp_path):
    payload = json.loads(TWENTYCR_NETCDF_MAPPING.read_text(encoding="utf-8"))
    assert payload["format"] == "netcdf"
    name, field = next(
        (name, field) for name, field in payload["fields"].items()
        if field.get("selectors")
    )
    field["time_binding"] = "cycle_invariant"
    with pytest.raises(ValueError, match="NetCDF"):
        load_mapping(_write(tmp_path / "netcdf.json", payload))


# ---------------------------------------------------------------------------
# frame assembly broadcast
# ---------------------------------------------------------------------------


T0 = datetime(2026, 8, 17, 0)
T6 = T0 + timedelta(hours=6)


def _record(index, *, valid, discipline, category, parameter, level_type,
            values, reference=T0, level_value=0.0):
    latitude = np.array([0.0, 1.0])
    longitude = np.array([10.0, 11.0])
    return ms._GribRecord(
        source=Path("probe.grib2"), index=index,
        reference_time=reference, valid_time=valid,
        member=None, parameter=parameter, level_type=level_type,
        level_value=level_value, table_version=None,
        center=98, subcenter=0, master_table_version=36,
        local_table_version=0, discipline=discipline, category=category,
        second_level_type=255, second_level_value=0.0,
        process_identity=(2, 5), time_semantics=(0,),
        values=np.asarray(values, dtype=np.float64),
        latitude=latitude, longitude=longitude,
        grid_fingerprint="one-grid",
    )


def _frames_for(records, mapping_payload, tmp_path):
    # Raw assembly-level mapping, the same technique as the declared-level
    # subset test: schema-level time_binding validation is covered above.
    collection = ms._assemble_grib(mapping_payload, records)
    return ms._materialize_frames(
        mapping_payload, collection, mapping_sha256="0" * 64, input_sha256={},
    )


def _full_mapping_and_records():
    """Every canonical WRF-initialization field, on a 2x2 grid, two times."""

    threed = ["air_temperature", "specific_humidity", "eastward_wind",
              "northward_wind", "geopotential_height"]
    surface = ["surface_pressure", "terrain_height", "skin_temperature",
               "air_temperature_2m", "specific_humidity_2m",
               "eastward_wind_10m", "northward_wind_10m", "land_fraction"]
    soil = ["soil_temperature", "volumetric_soil_moisture"]
    fields: dict[str, dict] = {}
    identities: dict[str, tuple[int, int, int, int]] = {}
    for index, name in enumerate([*threed, *surface, *soil]):
        if name in threed:
            level_type, axes = 100, ["vertical", "y", "x"]
        elif name in soil:
            level_type, axes = 106, ["soil", "y", "x"]
        else:
            level_type, axes = 1, ["y", "x"]
        identities[name] = (0, 200, index, level_type)
        selector = {
            "format": "grib2", "discipline": 0, "category": 200,
            "parameter": index, "level_type": level_type,
        }
        fields[name] = {
            "selectors": [selector],
            "units": {"source": "1", "target": "1"},
            "source_axes": axes, "target_axes": axes,
            "location": ("mass" if name in threed
                         else "soil" if name in soil else "surface"),
            "staggering": "none",
            "missing": {"kind": "reject"},
        }
    fields["land_fraction"]["time_binding"] = "cycle_invariant"
    fields["terrain_height"]["time_binding"] = "cycle_invariant"
    fields["air_pressure"] = {
        "selectors": [], "derivation": "pressure-from-coordinate",
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass", "staggering": "none",
        "missing": {"kind": "reject"},
    }
    mapping = {
        "schema": "rw-wps.mapping.v1", "name": "invariant-full-probe",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "positive": "down", "levels": [100000.0]},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": [{
            "name": "pressure-from-coordinate",
            "operation": "pressure_from_vertical_coordinate",
        }],
        "target": {
            "required_fields": [
                {"name": name, "axes": field["target_axes"],
                 "location": field["location"], "target_units": "1"}
                for name, field in fields.items()
                if name != "air_pressure"
            ],
            "soil_layer_count": 1,
            "initialization_policies": {
                name: "explicit_zero_with_adapter_validation"
                for name in (
                    "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
                    "cloud_ice_mixing_ratio", "snow_mixing_ratio",
                    "graupel_or_hail_mixing_ratio", "vertical_velocity",
                    "snow_water_equivalent", "snow_depth",
                    "sea_ice_fraction",
                )
            },
        },
    }
    land = np.array([[0.0, 1.0], [1.0, 0.25]])
    records = []
    index = 0
    for valid in (T0, T6):
        for name, (disc, cat, par, level_type) in identities.items():
            if name in {"land_fraction", "terrain_height"} and valid != T0:
                continue  # analysis-step-only invariants
            values = land if name == "land_fraction" \
                else np.full((2, 2), 1.0 + par)
            records.append(_record(
                index, valid=valid, discipline=disc, category=cat,
                parameter=par, level_type=level_type, values=values,
                level_value=100000.0 if level_type == 100 else 0.0,
            ))
            index += 1
    return mapping, records, land


def test_cycle_invariant_field_broadcasts_to_every_frame_of_one_cycle(tmp_path):
    mapping, records, land = _full_mapping_and_records()
    frames = _frames_for(records, mapping, tmp_path)
    assert [frame.valid_time for frame in frames] == [T0, T6]
    for frame in frames:
        np.testing.assert_array_equal(
            frame.fields["land_fraction"].values, land,
        )
        np.testing.assert_array_equal(
            frame.fields["terrain_height"].values,
            frames[0].fields["terrain_height"].values,
        )


def test_cycle_invariant_field_refuses_when_its_bytes_change(tmp_path):
    land = np.array([[0.0, 1.0], [1.0, 0.25]])
    with pytest.raises(ValueError, match="cycle-invariant.*land_fraction.*chang"):
        _frames_for([
            _record(0, valid=T0, discipline=0, category=3, parameter=0,
                    level_type=1, values=np.full((2, 2), 100000.0)),
            _record(1, valid=T6, discipline=0, category=3, parameter=0,
                    level_type=1, values=np.full((2, 2), 100100.0)),
            _record(2, valid=T0, discipline=2, category=0, parameter=0,
                    level_type=1, values=land),
            _record(3, valid=T6, discipline=2, category=0, parameter=0,
                    level_type=1, values=land + 0.5),
        ], _grib2_mapping(), tmp_path)


def test_cycle_invariant_field_refuses_mixed_source_cycles(tmp_path):
    land = np.array([[0.0, 1.0], [1.0, 0.25]])
    with pytest.raises(ValueError, match="cycle-invariant.*mixed source cycles"):
        _frames_for([
            _record(0, valid=T0, discipline=0, category=3, parameter=0,
                    level_type=1, values=np.full((2, 2), 100000.0)),
            _record(1, valid=T6, discipline=0, category=3, parameter=0,
                    level_type=1, values=np.full((2, 2), 100100.0),
                    reference=T6),
            _record(2, valid=T0, discipline=2, category=0, parameter=0,
                    level_type=1, values=land),
        ], _grib2_mapping(), tmp_path)


def test_absent_invariant_field_still_refuses_as_missing(tmp_path):
    with pytest.raises(ValueError, match="lacks required fields.*land_fraction"):
        _frames_for([
            _record(0, valid=T0, discipline=0, category=3, parameter=0,
                    level_type=1, values=np.full((2, 2), 100000.0)),
        ], _grib2_mapping(), tmp_path)


# ---------------------------------------------------------------------------
# composition terrain broadcast
# ---------------------------------------------------------------------------


def _terrain_collections():
    primary_latitude = np.asarray([51.5, 51.75, 52.0])
    primary_longitude = np.asarray([260.0, 260.25, 260.5, 260.75])
    terrain_latitude = np.asarray([52.5, 52.25, 52.0, 51.75, 51.5, 51.25])
    terrain_longitude = np.asarray([
        -100.5, -100.25, -100.0, -99.75, -99.5, -99.25, -99.0,
    ])

    def direct(name, valid_time, values):
        return ms._DirectValue(
            name=name, valid_time=valid_time, member=None,
            source_cycle=T0, axes=("y", "x"),
            values=np.asarray(values, dtype=np.float64),
            missing_count=0, references=(f"fixture:{name}",),
        )

    def collection(latitude, longitude, fields):
        from types import MappingProxyType
        cycles = {
            (valid_time, member): value.source_cycle
            for (valid_time, member, _name), value in fields.items()
        }
        return ms._DecodedCollection(
            latitude=np.asarray(latitude, dtype=np.float64),
            longitude=np.asarray(longitude, dtype=np.float64),
            vertical_values=np.asarray([1000.0, 850.0]),
            direct=MappingProxyType(fields),
            source_cycles=MappingProxyType(cycles),
            grid_fingerprint="fixture-grid",
        )

    primary = collection(primary_latitude, primary_longitude, {
        (time, None, "surface_pressure"): direct(
            "surface_pressure", time, np.full((3, 4), 100000.0 + index),
        )
        for index, time in enumerate((T0, T6))
    })
    base = np.arange(6 * 7, dtype=np.float64).reshape(6, 7)
    terrain = collection(terrain_latitude, terrain_longitude, {
        (T0, None, "terrain_height"): direct("terrain_height", T0, base),
    })
    return primary, terrain, base[[4, 3, 2], 2:6]


def test_terrain_broadcast_attaches_the_analysis_record_at_every_time():
    primary, terrain, expected = _terrain_collections()
    combined, receipt = _compose_terrain(
        primary, terrain, time_alignment="cycle_invariant_broadcast",
    )
    assert receipt["time_alignment"] == "cycle_invariant_broadcast"
    for key in ((T0, None), (T6, None)):
        actual = combined.direct[(*key, "terrain_height")]
        np.testing.assert_array_equal(actual.values, expected)


def test_terrain_exact_alignment_still_refuses_a_missing_time():
    primary, terrain, _expected = _terrain_collections()
    with pytest.raises(ValueError, match="lacks exact primary valid time"):
        _compose_terrain(primary, terrain)


def test_composition_accepts_the_broadcast_alignment_and_refuses_others(
    tmp_path,
):
    payload = json.loads(ERA5_COMPOSITION.read_text(encoding="utf-8"))
    payload["supplements"]["terrain_height"]["time_alignment"] = \
        "cycle_invariant_broadcast"
    contract = load_composition(
        _write(tmp_path / "broadcast.json", payload), ERA5_MAPPING,
    )
    assert contract["supplements"]["terrain_height"]["time_alignment"] == \
        "cycle_invariant_broadcast"

    payload["supplements"]["terrain_height"]["time_alignment"] = "whenever"
    with pytest.raises(ValueError, match="time alignment"):
        load_composition(
            _write(tmp_path / "bad.json", payload), ERA5_MAPPING,
        )


# ---------------------------------------------------------------------------
# declared bounded land soil repair
# ---------------------------------------------------------------------------


def _soil_frames(vsw_layer0):
    """Full canonical frames with a controllable soil-moisture layer 0."""

    mapping, records, land = _full_mapping_and_records()
    for field_name in ("soil_temperature", "volumetric_soil_moisture"):
        mapping["fields"][field_name]["missing"] = {"kind": "preserve_mask"}
    # All-land grid so the land gate governs every cell.
    for record in records:
        if record.discipline == 0 and record.category == 200 \
                and record.parameter == 12:  # land_fraction (13th field)
            record.values[...] = 1.0
        if record.parameter == 14 and record.category == 200:
            # volumetric_soil_moisture, both layers share one array here
            record.values[...] = np.asarray(vsw_layer0, dtype=np.float64)
    collection = ms._assemble_grib(mapping, records)
    return ms._materialize_frames(
        mapping, collection, mapping_sha256="0" * 64, input_sha256={},
    )


def test_declared_nearest_column_repair_fills_a_bounded_coastal_gap():
    frames = _soil_frames([[np.nan, 0.3], [0.25, 0.2]])
    snapshots = ms.mapped_frames_to_regular_snapshots(
        frames,
        soil_land_repair={
            "kind": "nearest_soil_column_within_cells", "maximum_cells": 1,
        },
    )
    from gpuwm.ingest.soil_contract import MAPPED_SOIL_MOISTURE
    for snapshot in snapshots:
        moisture = snapshot.fields[MAPPED_SOIL_MOISTURE]
        assert np.isfinite(moisture).all()
        # nearest donor of (0,0) at chebyshev radius 1 minimising squared
        # index distance: (0,1) and (1,0) tie at 1; (di,dj) lexicographic
        # order takes (0,1).
        assert moisture[0, 0, 0] == 0.3


def test_repair_beyond_the_declared_radius_still_refuses_with_counts():
    frames = _soil_frames([[np.nan, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="4 land"):
        ms.mapped_frames_to_regular_snapshots(
            frames,
            soil_land_repair={
                "kind": "nearest_soil_column_within_cells",
                "maximum_cells": 1,
            },
        )


def test_undeclared_repair_keeps_the_historical_refusal():
    frames = _soil_frames([[np.nan, 0.3], [0.25, 0.2]])
    with pytest.raises(ValueError, match="missing source-land"):
        ms.mapped_frames_to_regular_snapshots(frames)


def test_soil_contract_accepts_the_bounded_repair_and_refuses_wider(tmp_path):
    from gpuwm.mapped_composition import load_composition
    payload = json.loads(
        (ROOT / "gpuwm" / "authorities" /
         "rw-wps-aifs-single-grib2.composition.json").read_text(
            encoding="utf-8")
    )
    payload["soil_layers"]["missing"]["land"] = {
        "kind": "nearest_soil_column_within_cells", "maximum_cells": 3,
    }
    contract = load_composition(
        _write(tmp_path / "repair.json", payload),
        ROOT / "gpuwm" / "authorities" /
        "rw-wps-aifs-single-grib2.mapping.json",
    )
    assert contract["soil_layers"]["missing"]["land"]["maximum_cells"] == 3

    payload["soil_layers"]["missing"]["land"] = {
        "kind": "nearest_soil_column_within_cells", "maximum_cells": 40,
    }
    with pytest.raises(ValueError, match="maximum_cells"):
        load_composition(
            _write(tmp_path / "wide.json", payload),
            ROOT / "gpuwm" / "authorities" /
            "rw-wps-aifs-single-grib2.mapping.json",
        )
