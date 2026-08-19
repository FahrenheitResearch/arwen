"""GEFS through the gauntlet: an ensemble member is TABLE DATA end to end.

Round 1 proved a deterministic model enters as three packaged JSON
documents plus registry rows.  This file pins the ensemble twin of that
shape: the SAME three documents and rows give one *member* of NCEP's
global ensemble a runnable IC route, on top of the member-addressing
grammar the ensemble capability already ships
(``rw-wps-gefs-ensemble-grib2.members.json``).  No GEFS decode code
exists anywhere; the two facts that make the ensemble hard are stated as
table data:

* ``pgrb2a`` and ``pgrb2b`` isobaric level sets are exactly disjoint
  (zero overlapping levels for any variable, MEASURED), so the mapping's
  31-level ladder is only satisfiable by passing BOTH files of the SAME
  member -- the frame assembler's one-GRIB-member rule then refuses a
  cross-member mix by construction.
* Every selector pins ``pdt`` 1 (instantaneous member forecast), so the
  ensemble mean/spread files that share the member directories (PDT 2)
  and the PDT-11 accumulation twins that appear from f003 can never
  satisfy a state selector.

Selectors were authored from real 2026-08-17 00Z bytes
(gec00/gep01/gep02 pgrb2a+pgrb2b, ``noaa-gefs-pds``) through the
converged grib-core inventory; the octets asserted here are the observed
ones.
"""

from __future__ import annotations

import json

from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import load_mapping
from gpuwm.physics_registry import physics_registry
from gpuwm.source_adapters import (AdapterStatus, SourceKind,
                                   get_source_adapter,
                                   packaged_profile_sources)
from gpuwm.source_authorities import packaged_authorities
from gpuwm import prepared_single_domain_forecast as runner


PROFILE_ID = "gefs-ensemble-grib2-v1"

#: The 31-level ladder MEASURED as the exact union of the ``pgrb2a`` and
#: ``pgrb2b`` isobaric sets (Pa).  1 hPa to 1000 hPa; 125 hPa and
#: 475 hPa are deliberately absent (o3mr-only / tcc-only levels).
EXPECTED_LEVELS = [
    100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 2000.0, 3000.0, 5000.0,
    7000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 35000.0,
    40000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0,
    75000.0, 80000.0, 85000.0, 90000.0, 92500.0, 95000.0, 97500.0,
    100000.0,
]

#: MEASURED per-file isobaric level counts for the state variables.  The
#: intersection of every pair is empty; the union is the 31-level ladder.
MEASURED_A_LEVELS = {
    "air_temperature": 10, "specific_humidity": 0,
    "eastward_wind": 12, "northward_wind": 12, "geopotential_height": 11,
}
MEASURED_B_LEVELS = {
    "air_temperature": 21, "specific_humidity": 31,
    "eastward_wind": 19, "northward_wind": 19, "geopotential_height": 20,
}


def test_the_gefs_row_is_a_packaged_profile_that_keeps_its_member_set():
    adapter = get_source_adapter("gefs")
    assert adapter.packaged_profile == PROFILE_ID
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.runnable is True
    assert adapter.file_family == "GRIB2"
    # The member axis survives the upgrade: the row still declares the
    # packaged member grammar, and the registry still says what this
    # source IS -- individual trajectories, never their statistics.
    assert adapter.member_set == "gefs-ensemble-grib2-members-v1"
    assert adapter.source_kind is SourceKind.ENSEMBLE_MEMBERS
    assert adapter.required_products == ("pgrb2a", "pgrb2b")
    assert packaged_profile_sources()["gefs"] == PROFILE_ID
    generic = get_source_adapter("mapped")
    assert adapter.runner == generic.runner


def test_the_gefs_mapping_is_the_measured_31_level_union_ladder():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    vertical = mapping["coordinates"]["vertical"]
    assert vertical["units"] == "Pa"
    assert [float(level) for level in vertical["levels"]] == EXPECTED_LEVELS
    # The ladder is longer than either file's contribution, so a lone
    # pgrb2a can never satisfy it, and a lone pgrb2b fails on the
    # pgrb2a-only surface fields: the split-product fact is enforced by
    # the level table, not by code.  (specific_humidity is the one state
    # variable published complete in pgrb2b -- and with ZERO messages in
    # pgrb2a, which is the other half of the same measured split.)
    for field, a_count in MEASURED_A_LEVELS.items():
        b_count = MEASURED_B_LEVELS[field]
        assert a_count + b_count == len(EXPECTED_LEVELS), field
        if field != "specific_humidity":
            assert max(a_count, b_count) < len(EXPECTED_LEVELS), field
    assert MEASURED_A_LEVELS["specific_humidity"] == 0


def test_every_gefs_state_selector_pins_the_instantaneous_member_template():
    """PDT 1 on every selector: statistics and accumulation twins refuse.

    geavg/gespr share the member directories and decode cleanly as
    ordinary fields (PDT 2/12); from f003 the same files carry PDT-11
    time-interval twins.  A selector without the template pin would
    accept either.  The pin is what makes the refusal byte-level.
    """

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    selector_count = 0
    soil_fields = {"soil_temperature", "volumetric_soil_moisture"}
    for name, field in mapping["fields"].items():
        for selector in field.get("selectors", ()):
            if name in soil_fields:
                # The composition soil-selector grammar excludes
                # template pins, and mapping/composition soil selectors
                # must carry identical semantics; the measured soil
                # records are PDT 1 at every staged step, and a
                # statistic file still cannot complete a frame because
                # every non-soil selector below pins the template.
                assert "pdt" not in selector, name
                continue
            assert selector.get("pdt") == 1, (
                f"{name} selector without the PDT-1 pin would accept an "
                "ensemble statistic or an accumulation twin")
            selector_count += 1
    assert selector_count >= 14


def test_gefs_selector_octets_match_the_recorded_real_inventory():
    """The soil quadruple and the 2 m/10 m pins are the observed octets."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    # Noah layers: 0-0.1 m rides pgrb2a, the other three ride pgrb2b;
    # the level keys are the scaled fixed-surface metre bounds (the
    # integer `level` octet collapses layers 1 and 2 -- the measured
    # level-key trap -- so the bounds are the only correct key).
    for name, parameter in (
        ("soil_temperature", 2), ("volumetric_soil_moisture", 192),
    ):
        selectors = mapping["fields"][name]["selectors"]
        bounds = [
            (selector["level_value"], selector["second_level_value"])
            for selector in selectors
        ]
        assert bounds == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
        for selector in selectors:
            assert selector["discipline"] == 2
            assert selector["category"] == 0
            assert selector["parameter"] == parameter
            assert selector["level_type"] == 106
            assert selector["second_level_type"] == 106
            assert selector["center"] == 7
            assert selector["subcenter"] == 2
        assert mapping["fields"][name]["missing"]["kind"] == "preserve_mask"
    # pgrb2b also publishes 80 m and 100 m records under the same
    # discipline/category/parameter at level_type 103, so the near-
    # surface selectors must pin their level.
    for name, level in (
        ("air_temperature_2m", 2), ("specific_humidity_2m", 2),
        ("eastward_wind_10m", 10), ("northward_wind_10m", 10),
    ):
        (selector,) = mapping["fields"][name]["selectors"]
        assert selector["level_type"] == 103
        assert selector["level_value"] == level
    # Fields that exist ONLY in pgrb2b: the mask, the ice analysis, the
    # skin temperature, and the whole humidity column.
    for name in ("land_fraction", "sea_ice_fraction", "skin_temperature"):
        (selector,) = mapping["fields"][name]["selectors"]
        assert selector["level_type"] == 1
    assert mapping["fields"]["specific_humidity"]["selectors"][0][
        "discipline"] == 0
    assert mapping["fields"]["specific_humidity"].get("derivation") is None


def test_the_gefs_composition_binds_noah_layers_and_in_band_terrain():
    authorities = packaged_authorities(PROFILE_ID)
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    soil = contract["soil_layers"]
    assert [
        (layer["top"], layer["bottom"]) for layer in soil["source_layers"]
    ] == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
    assert soil["target_layers"] == [
        {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
        {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0},
    ]
    assert soil["remap"]["kind"] == "conservative_layer_means"
    assert soil["missing"]["ocean"]["temperature"] == "skin_temperature"
    supplement = contract["supplements"]["terrain_height"]
    assert supplement["data_role"] == "gefs_member_in_band_surface"
    assert supplement["time_alignment"] == "valid_time_exact"
    assert supplement["require_invariant_across_time"] is True


def test_the_physics_registry_routes_gefs_through_the_prepared_runner():
    payload = physics_registry()
    route = payload["runner_routes"]["tools.prepared_single_domain_forecast"]
    assert "gefs" in route["source_ids"]
    assert route["source_template_ids"]["gefs"] \
        == route["source_template_ids"]["gem-gdps"]


def test_the_prepared_runner_tables_carry_the_gefs_rows():
    assert "gefs" in runner.SUPPORTED_SOURCES
    capabilities = runner.runner_capabilities()
    assert "gefs" in capabilities["supported_sources"]
    source = capabilities["source_profiles"]["gefs"]
    assert source["single_d01_gpu_execution"] is True
    cadence = capabilities["window"]["source_forcing_cadence_hours"]
    assert cadence["gefs"] == "uniform-positive-whole-hour"
