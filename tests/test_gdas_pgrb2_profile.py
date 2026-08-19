"""GDAS through the gauntlet: a model is TABLE DATA, and these pins hold it.

GDAS `pgrb2.0p25` is record-for-record byte-identical to GFS `pgrb2.0p25`
(696/696 at f000, 743/743 at f003, MEASURED against the same 06Z cycle),
so this profile is the GFS-shape field catalogue arriving as three
packaged JSON documents, one ``_PACKAGED_PROFILES`` row, one upgraded
``_adapter`` row, the physics-registry route, and the prepared-runner
table rows.  No GDAS decode code exists anywhere: the numeric-octet
selector grammar carries the NCEP local-table rows (soil moisture at
2.0.192 under localTablesVersion 1), and the type-106 depth-pair
selector binding -- the scaled fixed-surface quadruple that survives the
integer `level`-key collision between the 0-0.1 m and 0.1-0.4 m layers
-- already existed as the generic soil contract.

Selectors were authored from real 2026-08-17 06Z bytes
(gdas.t06z.pgrb2.0p25.f000/f003, noaa-gfs-bdp-pds) through the converged
grib-core inventory.

The honesty pins live here too: the certified route reads the hourly
f000..f009 files, whose bytes stamp even hour 0 a FORECAST
(typeOfProcessedData=fc, generating process 81 at hour 0, 96 after);
the one product stamped an ANALYSIS (`pgrb2.1p00.anl`) is a strict
610-record subset with no soil, no land mask and no 2 m/10 m state, and
is not an initialization route.
"""

from __future__ import annotations

import json

from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import load_mapping
from gpuwm.physics_registry import physics_registry
from gpuwm.source_adapters import (AdapterStatus, get_source_adapter,
                                   packaged_profile_sources)
from gpuwm.source_authorities import (packaged_authorities,
                                      packaged_profile)
from gpuwm import prepared_single_domain_forecast as runner


PROFILE_ID = "gdas-pgrb2-0p25-grib2-v1"

#: The 33-level isobaricInhPa ladder observed in the real pgrb2.0p25
#: bytes -- 1..1000 hPa, in Pa.  The eight sub-hPa `isobaricInPa`
#: surfaces (1..70 Pa) sit outside the declared ladder and are
#: admitted-and-ignored, exactly as AIFS's extra 10 hPa surfaces are.
EXPECTED_LEVELS = [
    100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0,
    3000.0, 4000.0, 5000.0, 7000.0, 10000.0, 15000.0, 20000.0,
    25000.0, 30000.0, 35000.0, 40000.0, 45000.0, 50000.0, 55000.0,
    60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0,
    92500.0, 95000.0, 97500.0, 100000.0,
]

#: The four Noah layer bounds encoded in the bytes as scaled type-106
#: fixed-surface pairs (scale factor 2: 0/10, 10/40, 40/100, 100/200).
EXPECTED_SOIL_LAYERS = [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]


def test_the_gdas_row_is_a_packaged_profile_and_nothing_more():
    adapter = get_source_adapter("gdas")
    assert adapter.packaged_profile == PROFILE_ID
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.runnable is True
    assert adapter.file_family == "GRIB2"
    assert adapter.default_product == "pgrb2.0p25"
    # The run really stops at f009: f010 is a MEASURED 404.
    assert adapter.max_forecast_hour == 9
    assert packaged_profile_sources()["gdas"] == PROFILE_ID
    generic = get_source_adapter("mapped")
    assert adapter.runner == generic.runner


def test_the_gdas_mapping_is_the_gfs_catalogue_on_the_global_grid():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    # Global regular lat/lon rides the embedded grid; no projected-grid
    # block and no wind rotation exist for this source.
    assert mapping.get("grid") is None
    assert mapping["coordinates"]["vertical"]["levels"] == EXPECTED_LEVELS
    # GDAS publishes specific humidity directly, on all declared levels
    # and at 2 m -- no RH derivation on either.
    for name in ("specific_humidity", "specific_humidity_2m"):
        field = mapping["fields"][name]
        assert field.get("derivation") is None
        selector = field["selectors"][0]
        assert (selector["discipline"], selector["category"],
                selector["parameter"]) == (0, 1, 0)
    # Geopotential height arrives in gpm; the same-unit scale is
    # declared explicitly so a real conversion can never hide behind a
    # defaulted 1.0.
    for name in ("geopotential_height", "terrain_height"):
        units = mapping["fields"][name]["units"]
        assert units["source"] == "gpm"
        assert units["target"] == "m"
        assert units["scale"] == 1.0


def test_the_gdas_soil_rows_are_the_scaled_fixed_surface_quadruple():
    """The integer `level` key collides (layers 1 and 2 both truncate to
    0-0); the selector rows bind the real metre depth pairs instead."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    for name, parameter, local in (
        ("soil_temperature", 2, False),
        ("volumetric_soil_moisture", 192, True),
    ):
        selectors = mapping["fields"][name]["selectors"]
        assert len(selectors) == 4
        for selector, (top, bottom) in zip(selectors, EXPECTED_SOIL_LAYERS):
            assert selector["discipline"] == 2
            assert selector["category"] == 0
            assert selector["parameter"] == parameter
            assert selector["level_type"] == 106
            assert selector["second_level_type"] == 106
            assert selector["level_value"] == top
            assert selector["second_level_value"] == bottom
            if local:
                # An NCEP local-table row is octets, not a name: the
                # profile never asks any table to spell "SOILW".
                assert selector["center"] == 7
                assert selector["subcenter"] == 0
                assert selector["master_table_version"] == 2
                assert selector["local_table_version"] == 1


def test_the_gdas_composition_binds_noah_layers_one_to_one():
    authorities = packaged_authorities(PROFILE_ID)
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    soil = contract["soil_layers"]
    assert [
        (layer["top"], layer["bottom"]) for layer in soil["source_layers"]
    ] == EXPECTED_SOIL_LAYERS
    assert [
        (layer["top"], layer["bottom"]) for layer in soil["target_layers"]
    ] == EXPECTED_SOIL_LAYERS
    assert soil["remap"]["kind"] == "conservative_layer_means"
    profile = packaged_profile(PROFILE_ID)
    supplement = contract["supplements"]["terrain_height"]
    assert supplement["data_role"] == profile["data_role"] \
        == "gdas_pgrb2_in_band_surface"
    assert supplement["provenance_role"] == profile["provenance_role"] \
        == "gdas_pgrb2_in_band_surface_provenance"
    # Terrain rides IN BAND in every forecast-hour file; nothing here is
    # a once-per-cycle invariant needing the broadcast alignment.
    assert supplement["time_alignment"] == "valid_time_exact"


def test_the_gdas_provenance_names_the_analysis_route_honestly():
    authorities = packaged_authorities(PROFILE_ID)
    provenance = json.loads(
        authorities["provenance"].read_text(encoding="utf-8"))
    policy = provenance["analysis_cycle_policy"]
    # f000 is stamped a forecast in the bytes; the analysis-stamped
    # 1-degree file has no land surface at all.  The route this profile
    # certifies is the field-complete f000..f009 set, said out loud.
    assert "forecast" in policy.lower()
    assert "anl" in policy.lower()


def test_the_physics_registry_routes_gdas_through_the_prepared_runner():
    payload = physics_registry()
    route = payload["runner_routes"]["tools.prepared_single_domain_forecast"]
    assert "gdas" in route["source_ids"]
    assert route["source_template_ids"]["gdas"] \
        == route["source_template_ids"]["hrrr-prs"]


def test_the_prepared_runner_tables_carry_the_gdas_rows():
    assert "gdas" in runner.SUPPORTED_SOURCES
    capabilities = runner.runner_capabilities()
    assert "gdas" in capabilities["supported_sources"]
    source = capabilities["source_profiles"]["gdas"]
    assert source["single_d01_gpu_execution"] is True
    cadence = capabilities["window"]["source_forcing_cadence_hours"]
    assert cadence["gdas"] == "uniform-positive-whole-hour"
