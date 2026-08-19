"""Cross-source composition: a field may be sourced from ANOTHER source's decode.

The grammar under test is Drew's requested feature: a composition declares
per-field source bindings -- contributing source id, that source's own sealed
mapping authority pinned by SHA-256, and a cycle/time-alignment rule -- and a
same-grid contribution lands while a cross-grid contribution refuses by naming
the regrid capability this composition does not have.  Everything here is
tables plus one generic engine capability; no model name appears in any
identifier of the shipped code.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.mapped_composition import (
    _compose_bound_fields,
    load_composition,
)
from gpuwm.mapped_source import (
    _DecodedCollection,
    _DirectValue,
    _materialize_frames,
    load_mapping,
)


ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "data" / "cross_source"
MAPPING = FIXTURES / "atmosphere-13-level.mapping.json"
COMPOSITION = FIXTURES / "borrowed-analysis-surface.composition.json"
DONOR_MAPPING = ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"

BOUND_FIELDS = (
    "terrain_height", "land_fraction",
    "soil_temperature", "volumetric_soil_moisture",
)


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mapping_raw() -> dict:
    return json.loads(MAPPING.read_text(encoding="utf-8"))


def _composition_raw() -> dict:
    return json.loads(COMPOSITION.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mapping grammar: fields a composition must complete are DECLARED, not absent.
# ---------------------------------------------------------------------------

def test_fixture_mapping_declares_its_gaps_as_composition_bound():
    mapping = load_mapping(MAPPING)
    for name in BOUND_FIELDS:
        field = mapping["fields"][name]
        assert field["provider"] == "composition_bound"
        assert not field.get("selectors")
        assert field.get("derivation") is None
    # The atmosphere itself stays a plain direct decode.
    assert mapping["fields"]["air_temperature"]["selectors"]
    assert "provider" not in mapping["fields"]["air_temperature"]


def test_composition_bound_excludes_selectors_and_derivations(tmp_path):
    raw = _mapping_raw()
    raw["fields"]["terrain_height"]["selectors"] = [{
        "format": "grib2", "discipline": 0, "category": 3,
        "parameter": 5, "level_type": 1,
    }]
    with pytest.raises(ValueError, match="composition_bound"):
        load_mapping(_write(tmp_path, "with-selector.json", raw))

    raw = _mapping_raw()
    raw["fields"]["terrain_height"]["provider"] = "someone_else"
    with pytest.raises(ValueError, match="provider"):
        load_mapping(_write(tmp_path, "bad-provider.json", raw))


def test_direct_materialization_refuses_a_composition_bound_mapping():
    """A mapping with declared gaps must never decode as if it were whole."""

    mapping = load_mapping(MAPPING)
    valid_time = datetime(2026, 8, 17)
    collection = _DecodedCollection(
        latitude=np.asarray([10.0, 10.25]),
        longitude=np.asarray([40.0, 40.25]),
        vertical_values=np.asarray([100000.0]),
        direct=MappingProxyType({}),
        source_cycles=MappingProxyType({(valid_time, None): valid_time}),
        grid_fingerprint="fixture",
    )
    with pytest.raises(ValueError, match="contributing source"):
        _materialize_frames(
            mapping, collection, mapping_sha256="0" * 64, input_sha256={},
        )


# ---------------------------------------------------------------------------
# Composition grammar: per-field source bindings.
# ---------------------------------------------------------------------------

def test_fixture_composition_binds_every_gap_to_one_contributing_source():
    contract = load_composition(COMPOSITION, MAPPING)
    assert contract["supplements"] == {}
    bindings = contract["field_sources"]
    assert list(bindings) == ["physical_analysis_surface"]
    binding = bindings["physical_analysis_surface"]
    assert binding["source_id"] == "gfs-pressure-level-grib2-native"
    assert len(binding["mapping_sha256"]) == 64
    assert sorted(binding["fields"]) == sorted(BOUND_FIELDS)
    assert binding["grid_alignment"] == "exact_coordinate_subset"
    assert binding["time_alignment"] == "source_cycle_analysis_broadcast"


def test_an_unbound_gap_refuses(tmp_path):
    raw = _composition_raw()
    fields = raw["field_sources"]["physical_analysis_surface"]["fields"]
    fields.remove("land_fraction")
    path = _write(tmp_path, "unbound.json", raw)
    with pytest.raises(ValueError, match="no contributing source"):
        load_composition(path, MAPPING)


def test_a_field_bound_twice_refuses(tmp_path):
    raw = _composition_raw()
    binding = dict(raw["field_sources"]["physical_analysis_surface"])
    binding = json.loads(json.dumps(binding))
    binding["fields"] = ["land_fraction"]
    binding["mapping_role"] = "second_mapping"
    binding["data_role"] = "second_data"
    binding["provenance_role"] = "second_provenance"
    raw["field_sources"]["second_surface"] = binding
    path = _write(tmp_path, "double-bound.json", raw)
    with pytest.raises(ValueError, match="more than one contributing source"):
        load_composition(path, MAPPING)


def test_binding_a_directly_mapped_field_refuses(tmp_path):
    raw = _composition_raw()
    raw["field_sources"]["physical_analysis_surface"]["fields"].append(
        "skin_temperature"
    )
    path = _write(tmp_path, "two-providers.json", raw)
    with pytest.raises(ValueError, match="two providers"):
        load_composition(path, MAPPING)


def test_terrain_has_exactly_one_provider(tmp_path):
    # Provider via binding AND via supplement: refuse.
    raw = _composition_raw()
    raw["supplements"] = {
        "terrain_height": {
            "data_role": "in_band",
            "provenance_role": "in_band_provenance",
            "format": "grib2",
            "field": "terrain_height",
            "selector_authority": "mapping_field_exact",
            "grid_alignment": "exact_coordinate_subset",
            "time_alignment": "valid_time_exact",
            "require_invariant_across_time": True,
        },
    }
    path = _write(tmp_path, "terrain-twice.json", raw)
    with pytest.raises(ValueError, match="two providers"):
        load_composition(path, MAPPING)

    # No provider at all: refuse.
    raw = _composition_raw()
    raw["field_sources"]["physical_analysis_surface"]["fields"].remove(
        "terrain_height"
    )
    path = _write(tmp_path, "terrain-never.json", raw)
    with pytest.raises(ValueError, match="no contributing source"):
        load_composition(path, MAPPING)


def test_binding_grammar_is_closed(tmp_path):
    for key, value, match in (
        ("time_alignment", "whenever", "time.alignment|time_alignment"),
        ("grid_alignment", "bilinear", "grid.alignment|grid_alignment"),
        ("mapping_sha256", "abc", "SHA-256"),
        ("source_id", "", "source_id"),
    ):
        raw = _composition_raw()
        raw["field_sources"]["physical_analysis_surface"][key] = value
        path = _write(tmp_path, f"bad-{key}.json", raw)
        with pytest.raises(ValueError, match=match):
            load_composition(path, MAPPING)

    raw = _composition_raw()
    raw["field_sources"]["physical_analysis_surface"]["surprise"] = True
    path = _write(tmp_path, "unknown-key.json", raw)
    with pytest.raises(ValueError, match="unknown key"):
        load_composition(path, MAPPING)


def test_the_soil_pair_rides_one_contributing_source(tmp_path):
    """One donor owns the soil geometry; a split pair has no single
    mapping for the soil contract to validate against."""

    raw = _composition_raw()
    first = raw["field_sources"]["physical_analysis_surface"]
    first["fields"].remove("volumetric_soil_moisture")
    second = json.loads(json.dumps(first))
    second["fields"] = ["volumetric_soil_moisture"]
    second["mapping_role"] = "second_mapping"
    second["data_role"] = "second_data"
    second["provenance_role"] = "second_provenance"
    raw["field_sources"]["moisture_donor"] = second
    path = _write(tmp_path, "split-soil.json", raw)
    with pytest.raises(ValueError, match="one contributing source"):
        load_composition(path, MAPPING)


def test_binding_roles_must_not_collide(tmp_path):
    raw = _composition_raw()
    binding = raw["field_sources"]["physical_analysis_surface"]
    binding["provenance_role"] = binding["data_role"]
    path = _write(tmp_path, "role-collision.json", raw)
    with pytest.raises(ValueError, match="role"):
        load_composition(path, MAPPING)


# ---------------------------------------------------------------------------
# Compose-time semantics on synthetic collections.
# ---------------------------------------------------------------------------

def test_the_public_mapped_route_accepts_contributing_mapping_bindings():
    """Ship only what users can reach: the front door takes the donor table."""

    from gpuwm.mapped_direct import _parser, _role_bindings

    args = _parser().parse_args([
        "--source-format", "grib2",
        "--composition", "/case/composition.json",
        "--mapping", "/case/mapping.json",
        "--input", "/source/f000.grib2",
        "--contributing-mapping", "physical_analysis_surface_mapping=/case/donor.mapping.json",
        "--input-manifest", "/case/inputs.json",
        "--input-manifest-sha256", "0" * 64,
        "--wps-namelist", "/case/namelist.wps",
        "--geog-root", "/static/WPS_GEOG",
        "--experiment-config", "/case/experiment.toml",
        "--output-root", "/output/mapped",
    ])
    assert _role_bindings(args.contributing_mapping, multiple=False) == {
        "physical_analysis_surface_mapping": Path("/case/donor.mapping.json"),
    }


def _direct(
    name: str,
    valid_time: datetime,
    values: np.ndarray,
    cycle: datetime | None = None,
    member: str | None = None,
    axes: tuple[str, ...] = ("y", "x"),
) -> _DirectValue:
    return _DirectValue(
        name=name,
        valid_time=valid_time,
        member=member,
        source_cycle=cycle or valid_time,
        axes=axes,
        values=np.asarray(values, dtype=np.float64),
        missing_count=0,
        references=(f"fixture:{name}:{valid_time.isoformat()}",),
    )


def _collection(latitude, longitude, fields, vertical=(1000.0, 850.0)):
    cycles = {
        (valid_time, member): value.source_cycle
        for (valid_time, member, _name), value in fields.items()
    }
    return _DecodedCollection(
        latitude=np.asarray(latitude, dtype=np.float64),
        longitude=np.asarray(longitude, dtype=np.float64),
        vertical_values=np.asarray(vertical, dtype=np.float64),
        direct=MappingProxyType(fields),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint="fixture-grid",
    )


_BINDING = {
    "source_id": "donor-fixture",
    "mapping_role": "donor_mapping",
    "mapping_sha256": "0" * 64,
    "data_role": "donor_data",
    "provenance_role": "donor_provenance",
    "fields": ["terrain_height"],
    "grid_alignment": "exact_coordinate_subset",
    "time_alignment": "source_cycle_analysis_broadcast",
}


def _primary(times, latitude=(51.5, 51.75), longitude=(260.0, 260.25)):
    cycle = min(times)
    fields = {
        (time, None, "surface_pressure"): _direct(
            "surface_pressure", time,
            np.full((len(latitude), len(longitude)), 100000.0),
            cycle=cycle,
        )
        for time in times
    }
    return _collection(latitude, longitude, fields)


def _donor(times, latitude=(52.0, 51.75, 51.5), longitude=(-100.0, -99.75, -99.5),
           name="terrain_height", base=None):
    shape = (len(latitude), len(longitude))
    if base is None:
        base = np.arange(shape[0] * shape[1], dtype=np.float64).reshape(shape)
    fields = {
        (time, None, name): _direct(name, time, base)
        for time in times
    }
    return _collection(latitude, longitude, fields)


def test_analysis_broadcast_binds_the_donor_analysis_to_every_primary_time():
    start = datetime(2026, 8, 17)
    primary = _primary((start, start + timedelta(hours=6)))
    donor = _donor((start,))
    combined, receipt = _compose_bound_fields(
        primary, donor, binding_name="donor", binding=_BINDING,
    )
    assert receipt["status"] == "PASS"
    assert receipt["source_id"] == "donor-fixture"
    assert receipt["time_alignment"] == "source_cycle_analysis_broadcast"
    assert receipt["donor_valid_times"] == [start.isoformat()]
    assert receipt["broadcast_primary_valid_times"] == [
        (start + timedelta(hours=6)).isoformat()
    ]
    # Donor rows run north-first (52.0, 51.75, 51.5) and its longitudes
    # (-100.0, -99.75, -99.5) equal the primary's (260.0, 260.25) modulo
    # 360 at columns 0 and 1; the primary's latitudes (51.5, 51.75) are
    # donor rows 2 and 1.
    expected = np.asarray([[6.0, 7.0], [3.0, 4.0]])
    for time in (start, start + timedelta(hours=6)):
        injected = combined.direct[(time, None, "terrain_height")]
        np.testing.assert_array_equal(injected.values, expected)
        # Borrowed state keeps the DONOR's cycle in its provenance.
        assert injected.source_cycle == start


def test_analysis_broadcast_refuses_a_cycle_mismatched_donor():
    start = datetime(2026, 8, 17)
    primary = _primary((start, start + timedelta(hours=6)))
    donor = _donor((start + timedelta(hours=6),))
    with pytest.raises(ValueError, match="source cycle"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=_BINDING,
        )


def test_analysis_broadcast_refuses_a_multi_time_donor():
    start = datetime(2026, 8, 17)
    primary = _primary((start, start + timedelta(hours=6)))
    donor = _donor((start, start + timedelta(hours=6)))
    with pytest.raises(ValueError, match="exactly one"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=_BINDING,
        )


def test_valid_time_exact_requires_the_donor_at_every_primary_time():
    start = datetime(2026, 8, 17)
    binding = dict(_BINDING, time_alignment="valid_time_exact")
    primary = _primary((start, start + timedelta(hours=6)))
    donor = _donor((start, start + timedelta(hours=6)))
    combined, receipt = _compose_bound_fields(
        primary, donor, binding_name="donor", binding=binding,
    )
    assert receipt["broadcast_primary_valid_times"] == []
    assert len(receipt["matched_primary_valid_times"]) == 2
    assert (start + timedelta(hours=6), None, "terrain_height") in combined.direct

    donor = _donor((start,))
    with pytest.raises(ValueError, match="lacks.*valid time"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=binding,
        )


def test_cross_grid_contribution_refuses_naming_the_regrid_capability():
    start = datetime(2026, 8, 17)
    primary = _primary((start,), latitude=(51.5, 51.75), longitude=(260.0, 260.25))
    donor = _donor((start,), latitude=(52.0, 51.0), longitude=(-101.0, -99.0))
    with pytest.raises(ValueError, match="regrid capability"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=_BINDING,
        )


def test_a_member_bearing_donor_refuses():
    start = datetime(2026, 8, 17)
    primary = _primary((start,))
    fields = {}
    for member in ("0", "1"):
        fields[(start, member, "terrain_height")] = _direct(
            "terrain_height", start, np.zeros((3, 3)), member=member,
        )
    donor = _collection((52.0, 51.75, 51.5), (-100.0, -99.75, -99.5), fields)
    with pytest.raises(ValueError, match="member"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=_BINDING,
        )


def test_an_already_provided_field_refuses_two_providers():
    start = datetime(2026, 8, 17)
    primary = _primary((start,))
    fields = dict(primary.direct)
    fields[(start, None, "terrain_height")] = _direct(
        "terrain_height", start, np.zeros((2, 2)),
    )
    primary = _collection(primary.latitude, primary.longitude, fields)
    donor = _donor((start,))
    with pytest.raises(ValueError, match="two providers"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=_BINDING,
        )


def test_a_vertical_bearing_borrow_requires_the_same_ladder():
    start = datetime(2026, 8, 17)
    primary = _primary((start,))
    latitude = (52.0, 51.75, 51.5)
    longitude = (-100.0, -99.75, -99.5)
    fields = {
        (start, None, "air_temperature"): _direct(
            "air_temperature", start,
            np.zeros((3, len(latitude), len(longitude))),
            axes=("vertical", "y", "x"),
        ),
    }
    donor = _collection(latitude, longitude, fields, vertical=(500.0, 400.0, 300.0))
    binding = dict(_BINDING, fields=["air_temperature"])
    with pytest.raises(ValueError, match="vertical"):
        _compose_bound_fields(
            primary, donor, binding_name="donor", binding=binding,
        )
