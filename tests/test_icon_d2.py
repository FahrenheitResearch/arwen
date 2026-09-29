"""ICON-D2 is rows on the DWD route: the registry, the fetch, the normalization.

Every assertion reads a shipped document -- the registry row, the fetch route
table, the normalization authority or the packaged profile -- so a fact that
drifts between two of them fails here.  The real-byte checks read GRIB2
Sections 0 through 5 of 24 objects downloaded from opendata.dwd.de for the
2026-09-27 12 UTC cycle (tests/fixtures/icon-d2-2026092712-headers.json).

Source: Deutscher Wetterdienst (CC BY 4.0).
"""
from __future__ import annotations

import base64
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import fetch_routes, source_adapters, source_authorities
from gpuwm import source_normalization as norm
from gpuwm.source_cycles import cycle_grid_for

FIXTURE = Path(__file__).with_name("fixtures") / "icon-d2-2026092712-headers.json"
NORMALIZER = "icon-d2-gdt101-model-level-v1"
CYCLE = datetime(2026, 9, 27, 12)
MODEL_LEVELS = tuple(range(1, 66))
#: The native R19B07 mesh and the cells DWD publishes on it: every record
#: but the two coordinate records masks the same 16,968-cell boundary strip.
MESH_CELLS = 542040
PUBLISHED_CELLS = 525072


@pytest.fixture(scope="module")
def spec():
    return norm.load_normalization(NORMALIZER)


@pytest.fixture(scope="module")
def mapping(spec):
    path = source_authorities.packaged_authorities(spec.profile)["mapping"]
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The registry row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias", ["icon-d2", "icon-2km", "dwd-icon-d2", "ICON-D2"])
def test_every_spelling_names_the_one_row(alias, spec):
    row = source_adapters.get_source_adapter(alias)
    assert row.source_id == "icon-d2"
    assert row.runnable and row.runner == "mapped_composition_v1"
    assert row.status == source_adapters.AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert row.packaged_profile == spec.profile == "icon-d2-grib2-v1"
    assert row.forcing_interval_seconds == 3600
    assert row.max_forecast_hour == 48
    assert "Deutscher Wetterdienst" in row.notes and "CC BY 4.0" in row.notes


def test_the_model_top_a_config_is_given_is_the_top_dwd_publishes(mapping):
    """An emitted p_top above the source would refuse at preparation after
    the download was paid for. A measured column plus a seasonal margin
    bounds the default even though model-level numbers carry no pressure.
    """
    from gpuwm import domain_wizard

    row = source_adapters.get_source_adapter("icon-d2")
    assert mapping["coordinates"]["vertical"]["kind"] == "model_level"
    measurement = json.loads(FIXTURE.with_name("model-level-source-tops.json")
                             .read_text(encoding="utf-8"))["profiles"][row.packaged_profile]
    from math import exp, log

    reference = measurement["warm_season_reference"]
    low_height, high_height = reference["height_m"]
    low_pressure, high_pressure = reference["pressure_pa"]
    height = (measurement["upper_interface_height_m"]
              + measurement["lower_interface_height_m"]) / 2
    assert low_height < height < high_height
    assert low_pressure > high_pressure > 0
    fraction = (height - low_height) / (high_height - low_height)
    pressure = exp(log(low_pressure) + fraction * (log(high_pressure) - log(low_pressure)))
    assert pressure == pytest.approx(reference["pressure_at_mass_height_pa"])
    assert reference["coverage_rule"] == "lower altitude bracket pressure"
    assert reference["source_url"].startswith("https://")
    assert row.certified_source_top_pa >= max(measurement["measured_maximum_pa"], low_pressure)
    assert domain_wizard.emitted_model_top_pa("icon-d2") == row.certified_source_top_pa
    for name in ("cloud_water_mixing_ratio", "cloud_ice_mixing_ratio",
                 "rain_water_mixing_ratio", "snow_mixing_ratio",
                 "graupel_or_hail_mixing_ratio"):
        assert mapping["fields"][name]["derivation"]
        assert name not in mapping["target"]["initialization_policies"]



# ---------------------------------------------------------------------------
# One ladder, one cadence, one mesh across the documents
# ---------------------------------------------------------------------------

def test_route_document_and_mapping_declare_one_ladder_and_cadence(spec, mapping):
    route = fetch_routes.route_for("icon-d2")
    levels = {name: tuple(group.get("levels") or ())
              for group in route.axes["field"] for name in group["names"]}
    for field in ("p", "t", "u", "v", "qv", "qc", "qi", "qr", "qs", "qg"):
        assert levels[field] == MODEL_LEVELS
        assert spec.levels_for(field) == MODEL_LEVELS
    assert (mapping["coordinates"]["vertical"]["levels"]
            == list(MODEL_LEVELS))
    assert (mapping["target"]["boundary_interval_seconds"]
            == spec.cadence["forcing_interval_seconds"] == 3600)
    assert (list(route.cycle_hours) == list(spec.cadence["cycle_hours"])
            == list(range(0, 24, 3)))
    for hour in route.cycle_hours:
        assert spec.horizon_hours(hour) == 48
        assert fetch_routes.ladder_for(route, CYCLE.replace(hour=hour))[-1] == 48
    assert spec.native_grid == {"cells": MESH_CELLS, "grid_template": 101,
                                "originating_centre": 78}
    assert spec.target["step_degrees"] == 0.02


def test_the_profile_pins_its_normalization_and_binds_sea_ice_per_cycle(spec, mapping):
    profile = source_authorities.packaged_profile(spec.profile)
    assert profile["input_normalizer"] == NORMALIZER
    assert (source_authorities.packaged_normalization(NORMALIZER).name
            == "rw-wps-icon-d2-grib2.normalization.json")
    # DWD publishes ICON-D2's sea-ice fraction once per cycle, beside the
    # land fraction, where the global product publishes it at every lead.
    assert spec.fields["fr_ice"]["kind"] == "time-invariant"
    assert mapping["fields"]["sea_ice_fraction"]["time_binding"] == "cycle_invariant"
    assert spec.roles == {"terrain": "icon_d2_invariant_surface",
                          "provenance": "icon_d2_invariant_surface_provenance"}
    assert (profile["data_role"], profile["provenance_role"]) == (
        spec.roles["terrain"], spec.roles["provenance"])


# ---------------------------------------------------------------------------
# What the fetch downloads is what the normalization accepts
# ---------------------------------------------------------------------------

def test_a_fetched_window_is_exactly_the_inventory_the_normalization_accepts(spec):
    plan = fetch_routes.resolve_request("icon-d2", cycle=CYCLE, hours=3)
    paths = [Path("/not-downloaded") / name for name in plan.primary_files]
    objects = norm.validate_inventory(spec, paths)
    assert len(objects) == len(paths) == 675 * 4 + 71
    assert ({o.field for o in objects if o.kind == "time-invariant"}
            == {"clat", "clon", "fr_ice", "fr_land", "hsurf", "hhl"})
    assert {o.level for o in objects if o.field == "t"} == set(MODEL_LEVELS)
    assert all(o.level is None for o in objects if o.kind == "single-level")
    assert {o.lead for o in objects if o.lead is not None} == {0, 1, 2, 3}


@pytest.mark.parametrize("name", [
    # a level on a single-level field
    "icon-d2_germany_icosahedral_single-level_2026092712_000_5_t_2m.grib2.bz2",
    # a lead on a time-invariant object
    "icon-d2_germany_icosahedral_time-invariant_2026092712_003_0_hsurf.grib2.bz2",
    # a pressure level DWD does not publish for ICON-D2
    "icon-d2_germany_icosahedral_pressure-level_2026092712_000_150_t.grib2.bz2",
    # past the f048 horizon
    "icon-d2_germany_icosahedral_pressure-level_2026092712_049_500_t.grib2.bz2",
    # an hour ICON-D2 does not run
    "icon-d2_germany_icosahedral_pressure-level_2026092701_000_500_t.grib2.bz2",
    # DWD's regular-lat-lon product, which this route does not read
    "icon-d2_germany_regular-lat-lon_pressure-level_2026092712_000_500_t.grib2.bz2",
    # the global model's objects
    "icon_global_icosahedral_pressure-level_2026092712_000_500_T.grib2.bz2",
])
def test_the_object_grammar_refuses_what_this_route_does_not_read(name, spec):
    with pytest.raises(ValueError):
        norm.parse_object(spec, Path("/not-downloaded") / name)


# ---------------------------------------------------------------------------
# Coverage, schedule and price
# ---------------------------------------------------------------------------

def test_a_domain_inside_the_window_asks_only_for_published_cells(spec):
    """Measured 2026-09-27 12Z: the published cells fill lat 43.66..57.70,
    lon -0.42..17.64, and no larger lat/lon box."""

    window = source_adapters.get_source_adapter("icon-d2").coverage_window
    south, west, north, east = window.envelope()
    target = norm.target_from_points(spec, [south, north], [west, east])
    assert target.south >= 43.66 and target.west >= -0.42
    assert target.south + (target.ny - 1) * target.dy <= 57.70
    assert target.west + (target.nx - 1) * target.dx <= 17.64
    assert not window.outside(50.1, 8.7)   # Frankfurt
    assert window.outside(41.9, 12.5)      # Rome, south of the mesh


def test_latest_and_the_horizon_come_from_the_route_rows():
    grid = cycle_grid_for("icon-d2")
    assert grid.hours == tuple(range(0, 24, 3))
    assert grid.delay_hours == 2.0
    assert grid.search_hours == 24
    assert all(grid.horizon(CYCLE.replace(hour=hour)) == 48 for hour in grid.hours)


def test_a_download_is_priced_from_its_measured_row():
    from gpuwm import download_budget

    estimate = download_budget.download_estimate(
        {"source": "icon-d2", "cycle": "2026-09-27T12", "hours": 3})
    assert estimate["objects"] == 675 * 4 + 71
    assert estimate["bytes"] == pytest.approx(
        675 * 4 * 628500 + 69 * 208600 + 2 * 415200, rel=1e-6)


# ---------------------------------------------------------------------------
# Real DWD octets against the declared selectors
# ---------------------------------------------------------------------------

def _sections(header: bytes) -> dict[int, bytes]:
    assert header[:4] == b"GRIB" and header[7] == 2
    found: dict[int, bytes] = {}
    offset = 16
    while offset < len(header):
        length = int.from_bytes(header[offset:offset + 4], "big")
        assert 5 <= length <= len(header) - offset
        found[header[offset + 4]] = header[offset:offset + length]
        offset += length
    return found


def _surface(scale_octet: int, raw: int) -> float:
    if scale_octet == 0xFF or raw == 0xFFFF_FFFF:
        return 0.0
    scale = -(scale_octet & 0x7F) if scale_octet & 0x80 else scale_octet
    value = -(raw & 0x7FFF_FFFF) if raw & 0x8000_0000 else raw
    return value / 10.0 ** scale


def _observed(row) -> dict[str, object]:
    header = base64.b64decode(row["header_base64"])
    part = _sections(header)
    s1, s3, s4, s5 = part[1], part[3], part[4], part[5]
    word = lambda s, a: int.from_bytes(s[a:a + 4], "big")   # noqa: E731
    half = lambda s, a: int.from_bytes(s[a:a + 2], "big")   # noqa: E731
    return {
        "discipline": header[6], "centre": half(s1, 5),
        "reference_time": (half(s1, 12), s1[14], s1[15], s1[16], s1[17], s1[18]),
        "cells": word(s3, 6), "grid_template": half(s3, 12),
        "product_template": half(s4, 7),
        "category": s4[9], "parameter": s4[10],
        "time_unit": s4[17], "forecast_time": word(s4, 18),
        "level_type": s4[22], "level_value": _surface(s4[23], word(s4, 24)),
        "second_level_type": s4[28],
        "second_level_value": _surface(s4[29], word(s4, 30)),
        "published": word(s5, 5),
    }


@pytest.fixture(scope="module")
def fixture():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["schema"] == "gpuwm-source-real-byte-headers-v1"
    assert document["source_id"] == "icon-d2"
    assert document["licence"] == "CC BY 4.0"
    return document


def test_the_fixture_covers_every_field_the_document_declares(fixture, spec):
    named = {norm.parse_object(spec, Path("/f") / row["object"]).field
             for row in fixture["objects"] if "pressure-level" not in row["object"]}
    model = json.loads(FIXTURE.with_name("icon-d2-model-level-headers.json")
                       .read_text(encoding="utf-8"))
    named.update(row["field"] for row in model)
    assert named == set(spec.fields), sorted(set(spec.fields) - named)


@pytest.mark.parametrize("index", range(24))
def test_every_real_object_carries_its_declared_selector_and_mesh(fixture, spec, index):
    row = fixture["objects"][index]
    if "pressure-level" in row["object"]:
        with pytest.raises(ValueError):
            norm.parse_object(spec, Path("/not-downloaded") / row["object"])
        return
    obj = norm.parse_object(spec, Path("/not-downloaded") / row["object"])
    seen = _observed(row)
    assert seen["grid_template"] == 101 and seen["cells"] == MESH_CELLS
    assert seen["centre"] == 78 and seen["product_template"] == 0
    assert seen["reference_time"] == (2026, 9, 27, 12, 0, 0)
    # DWD stamps the lead in minutes (Code Table 4.4 value 0).
    assert seen["time_unit"] == 0
    assert seen["forecast_time"] * 60 == (obj.lead or 0) * 3600
    declared = obj.selector
    assert (seen["discipline"], seen["category"], seen["parameter"]) == declared[:3]
    assert seen["level_type"] == declared[3]
    assert seen["level_value"] == pytest.approx(declared[4], abs=1e-9)
    assert seen["second_level_type"] == declared[5]
    assert seen["second_level_value"] == pytest.approx(declared[6], abs=1e-9)


def test_dwd_masks_one_boundary_strip_in_every_record_but_the_coordinates(fixture):
    """Why the remap plan leaves cells out: the land fraction, like every
    field, publishes 525,072 of the mesh's 542,040 cells, while the two
    coordinate records publish all of them."""

    for row in fixture["objects"]:
        field = row["object"].rsplit("_0_", 1)[-1] if "time-invariant" in row["object"] else ""
        published = _observed(row)["published"]
        if field.startswith(("clat", "clon")):
            assert published == MESH_CELLS, row["object"]
        else:
            assert published == PUBLISHED_CELLS, row["object"]


@pytest.mark.parametrize("index", range(11))
def test_model_level_headers_match_normalization_and_mapping(spec, mapping, index):
    path = FIXTURE.with_name("icon-d2-model-level-headers.json")
    row = json.loads(path.read_text(encoding="utf-8"))[index]
    obj = norm.parse_object(spec, Path(row["url"].rsplit("/", 1)[1]))
    seen = _observed(row)
    assert (seen["discipline"], seen["category"], seen["parameter"],
            seen["level_type"], seen["level_value"], seen["second_level_type"],
            seen["second_level_value"]) == obj.selector
    assert seen["cells"] == MESH_CELLS
    assert seen["published"] == PUBLISHED_CELLS
    from gpuwm.mapped_source import _selector_matches_record
    parts = _sections(base64.b64decode(row["header_base64"]))
    record = SimpleNamespace(**seen, center=seen["centre"],
        subcenter=int.from_bytes(parts[1][7:9], "big"),
        master_table_version=parts[1][9], local_table_version=parts[1][10],
        member=None, time_semantics=(seen["product_template"],))
    matches = [name for name, field in mapping["fields"].items()
               for selector in field.get("selectors", [])
               if _selector_matches_record(selector, record, "grib2")]
    assert len(matches) == 1, (row["field"], matches)



def test_every_invariant_height_is_required(spec):
    plan = fetch_routes.resolve_request("icon-d2", cycle=CYCLE, hours=1)
    paths = [Path(name) for name in plan.primary_files]
    missing = [p for p in paths if "_66_hhl" not in p.name]
    with pytest.raises(ValueError, match="missing coordinate/invariant"):
        norm.validate_inventory(spec, missing)
