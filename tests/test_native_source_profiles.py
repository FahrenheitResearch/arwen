"""Native-level input profiles retain the analyzed column and aerosol meaning.

These gates prevent pressure-level truncation at 50 hPa, dropped aerosol
numbers, a partial native soil column, and lateral forcing outside its grid.
"""

from datetime import datetime
import json
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwm.fetch_routes import resolve_request
from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import load_mapping
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_authorities import packaged_authorities, packaged_contributing_mappings


@pytest.mark.parametrize("source,top", [("rap-native", 1000.0), ("hrrr-native", 1500.0)])
def test_native_columns_use_explicit_pressure_and_all_fifty_levels(source, top):
    adapter = get_source_adapter(source)
    assert adapter.runner == "mapped_composition_v1"
    authority = packaged_authorities(adapter.packaged_profile)
    mapping = load_mapping(authority["mapping"])
    vertical = mapping["coordinates"]["vertical"]
    assert vertical["kind"] == "model_level"
    assert vertical["levels"] == list(range(1, 51))
    assert vertical["model_top_pressure_pa"] == top
    assert mapping["fields"]["air_pressure"]["selectors"] == [
        {"format": "grib2", "discipline": 0, "category": 3,
         "parameter": 0, "level_type": 105}]
    assert "derivation" not in mapping["fields"]["air_pressure"]


@pytest.mark.parametrize("source", ["rap-native", "hrrr-native"])
def test_native_aerosol_identity_is_number_per_mass_from_operational_vtable(source):
    """PMTF/PMTC index labels must not cause a particulate-mass conversion."""
    mapping = load_mapping(packaged_authorities(source + "-grib2-v1")["mapping"])
    for name, parameter in (("water_friendly_aerosol_number", 193),
                            ("ice_friendly_aerosol_number", 192)):
        field = mapping["fields"][name]
        assert field["units"] == {"source": "kg-1", "target": "kg-1"}
        # Keep masks until the operational metgrid neighbor fallback finishes.
        assert field["missing"] == {"kind": "preserve_mask"}
        assert field["selectors"] == [{
            "format": "grib2", "discipline": 0, "category": 13,
            "parameter": parameter, "center": 7, "subcenter": 0,
            "master_table_version": 2, "local_table_version": 1,
            "level_type": 105}]
        assert any(row["name"] == name for row in mapping["target"]["required_fields"])


def test_profiles_are_name_independent_table_data(tmp_path):
    """A new source identity must not require another model-specific adapter."""
    authority = packaged_authorities("rap-native-grib2-v1")
    document = json.loads(authority["mapping"].read_text(encoding="utf-8"))
    document["name"] = "arbitrary-native-analysis"
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert load_mapping(path)["name"] == "arbitrary-native-analysis"


def test_number_masks_survive_decode_but_temperature_masks_still_refuse(tmp_path):
    """Only fields with a downstream finite-donor repair may preserve holes."""
    from gpuwm.ingest.analyzed_numbers import CANONICAL_NUMBER_FIELDS
    source = Path(__file__).resolve().parents[1] / "tools/rw_wps/crates/rw-wps/src/mapping.rs"
    rust = source.read_text(encoding="utf-8")
    table = re.search(r"pub const MASKED_NUMBER_FIELDS:.*?= \[(.*?)\];", rust, re.S).group(1)
    assert set(re.findall(r'"([a-z_]+)"', table)) == set(CANONICAL_NUMBER_FIELDS)
    authority = packaged_authorities("rap-native-grib2-v1")
    document = json.loads(authority["mapping"].read_text(encoding="utf-8"))
    document["fields"]["air_temperature"]["missing"] = {"kind": "preserve_mask"}
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="air_temperature preserve_mask"):
        load_mapping(path)


@pytest.mark.parametrize("source", ["rap-native", "hrrr-native"])
def test_native_hydrometeor_number_moments_are_not_cold_started(source):
    mapping = load_mapping(packaged_authorities(source + "-grib2-v1")["mapping"])
    for name, category, parameter in (("cloud_droplet_number", 6, 28),
                                     ("cloud_ice_number", 6, 29),
                                     ("rain_number", 1, 100)):
        field = mapping["fields"][name]
        assert field["selectors"][0]["category"] == category
        assert field["selectors"][0]["parameter"] == parameter
        assert field["selectors"][0]["level_type"] == 105
        assert field["units"] == {"source": "kg-1", "target": "kg-1"}
        assert field["missing"] == {"kind": "preserve_mask"}


def test_hybrid_boundary_fetch_honors_three_hour_cadence():
    plan = resolve_request("rap-native", cycle=datetime(2026, 10, 2, 18),
                           hours=3, cadence=3)
    assert plan.leads == (0, 3)
    assert [p.role for p in plan.objects] == ["awp130bgrb", "awp130bgrb"]
    assert plan.objects[0].url.endswith("rap.20261002/rap.t18z.awp130bgrbf00.grib2")
    assert plan.objects[1].url.endswith("rap.20261002/rap.t18z.awp130bgrbf03.grib2")


def test_native_analysis_binds_complete_soil_from_same_cycle_pressure_product():
    authority = packaged_authorities("hrrr-native-grib2-v1")
    mapping = load_mapping(authority["mapping"])
    assert mapping["target"]["require_lateral_boundaries"] is False
    contract = load_composition(authority["composition"], authority["mapping"])
    binding = contract["field_sources"]["soil_surface"]
    assert binding["time_alignment"] == "source_cycle_analysis_broadcast"
    assert set(binding["fields"]) == {
        "terrain_height", "soil_temperature", "volumetric_soil_moisture"}
    for name in binding["fields"]:
        assert mapping["fields"][name]["provider"] == "composition_bound"
    donor = load_mapping(packaged_contributing_mappings("hrrr-native-grib2-v1")[
        "soil_surface_mapping"])
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        assert len(donor["fields"][name]["selectors"]) == 9
    plan = resolve_request("hrrr-native", cycle=datetime(2026, 10, 2, 18),
                           hours=0)
    assert plan.donors[0].source == "hrrr-prs"
    assert plan.donors[0].leads == (0,)
    assert plan.donors[0].cycle == plan.cycle


def test_rap_hybrid_grid_covers_the_entire_hrrr_boundary_ring():
    """The larger-grid forcing must cover every edge, including the east edge."""
    source = get_source_adapter("hrrr-native").coverage_window
    forcing = get_source_adapter("rap-native").coverage_window
    i = np.arange(1, source.nx + 1, dtype=float)
    j = np.arange(1, source.ny + 1, dtype=float)
    x = np.concatenate((i, i, np.ones(j.size), np.full(j.size, source.nx)))
    y = np.concatenate((np.ones(i.size), np.full(i.size, source.ny), j, j))
    latitude, longitude = source.grid().ij_to_latlon(x, y)
    fx, fy = forcing.grid().latlon_to_ij(latitude, longitude)
    assert fx.min() >= 1 and fx.max() <= forcing.nx
    assert fy.min() >= 1 and fy.max() <= forcing.ny
