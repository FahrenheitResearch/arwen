"""RAP through the gauntlet: a model is TABLE DATA, and these pins hold it.

The arbitrary acceptance test names RAP explicitly.  This file pins the
shape its arrival must take: three packaged JSON documents, one
``_PACKAGED_PROFILES`` row, one upgraded ``_adapter`` row, the physics
registry route, and the prepared-runner table rows.  No RAP decode code
exists anywhere -- the generic engine capabilities HRRR's wrfprs profile
added (declared Lambert grid, decode-time wind rotation, node soil) and
the GFS profile's RH-to-specific-humidity derivation carry all of it.

Selectors were authored from real 2026-08-16 00Z bytes
(rap.t00z.awip32f00.grib2, NOAA noaa-rap-pds) through the converged
grib-core inventory; the grid octets asserted here are the observed ones.
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


PROFILE_ID = "rap-awip32-grib2-v1"

#: The 39-level pressure ladder observed in the real awip32 bytes --
#: 50..1000 hPa at 25 hPa -- which is byte-for-byte the HRRR wrfprs ladder.
EXPECTED_LEVELS = [float(p) for p in range(5000, 100001, 2500)]


def test_the_rap_row_is_a_packaged_profile_and_nothing_more():
    adapter = get_source_adapter("rap")
    assert adapter.packaged_profile == PROFILE_ID
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.runnable is True
    assert adapter.file_family == "GRIB2"
    assert adapter.default_product == "awip32"
    assert packaged_profile_sources()["rap"] == PROFILE_ID
    generic = get_source_adapter("mapped")
    assert adapter.runner == generic.runner


def test_the_rap_mapping_declares_the_observed_lambert_grid():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    grid = mapping["grid"]
    assert grid["family"] == "lambert_conformal"
    assert grid["wind_basis"] == "grid_relative_with_rotation"
    parameters = grid["parameters"]
    # Observed GRIB2 octets of every record in the real awip32 sample.
    assert parameters["nx"] == 349
    assert parameters["ny"] == 277
    assert parameters["lat1"] == 1.0
    assert parameters["lon1"] == 214.5
    assert parameters["latin1"] == 50.0
    assert parameters["latin2"] == 50.0
    assert parameters["lov"] == 253.0
    assert parameters["dx_m"] == 32463.0
    assert parameters["dy_m"] == 32463.0
    assert parameters["shape_of_earth"] == 6
    assert parameters["earth_radius_m"] == 6371229.0


def test_the_rap_mapping_is_the_hrrr_ladder_with_the_gfs_humidity_rule():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    assert mapping["coordinates"]["vertical"]["levels"] == EXPECTED_LEVELS
    # awip32 publishes RH, not SPFH, on pressure levels: the existing
    # generic derivation carries it, exactly as the GFS profile does.
    humidity = mapping["fields"]["specific_humidity"]
    assert humidity["derivation"] == "humidity-from-rh"
    assert humidity["selectors"] == []
    rh = mapping["fields"]["relative_humidity"]["selectors"][0]
    assert (rh["discipline"], rh["category"], rh["parameter"]) == (0, 1, 1)
    assert rh["level_type"] == 100
    # 2 m humidity is direct SPFH in the file; no derivation there.
    q2 = mapping["fields"]["specific_humidity_2m"]
    assert q2.get("derivation") is None
    # The nine-node RUC soil column, identical octets to HRRR's.
    for field in ("soil_temperature", "volumetric_soil_moisture"):
        assert len(mapping["fields"][field]["selectors"]) == 9


def test_the_rap_composition_binds_ruc_nodes_to_noah_layers():
    authorities = packaged_authorities(PROFILE_ID)
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    soil = contract["soil_layers"]
    depths = [node["depth"] for node in soil["source_nodes"]]
    assert depths == [0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0]
    assert [
        (layer["top"], layer["bottom"]) for layer in soil["target_layers"]
    ] == [(0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0)]
    profile = packaged_profile(PROFILE_ID)
    supplement = contract["supplements"]["terrain_height"]
    assert supplement["data_role"] == profile["data_role"] \
        == "rap_awip32_in_band_surface"
    assert supplement["provenance_role"] == profile["provenance_role"] \
        == "rap_awip32_in_band_surface_provenance"


def test_the_physics_registry_routes_rap_through_the_prepared_runner():
    payload = physics_registry()
    route = payload["runner_routes"]["tools.prepared_single_domain_forecast"]
    assert "rap" in route["source_ids"]
    assert route["source_template_ids"]["rap"] \
        == route["source_template_ids"]["hrrr-prs"]


def test_the_prepared_runner_tables_carry_the_rap_rows():
    assert "rap" in runner.SUPPORTED_SOURCES
    capabilities = runner.runner_capabilities()
    assert "rap" in capabilities["supported_sources"]
    source = capabilities["source_profiles"]["rap"]
    assert source["single_d01_gpu_execution"] is True
    cadence = capabilities["window"]["source_forcing_cadence_hours"]
    assert cadence["rap"] == "uniform-positive-whole-hour"


def test_rap_selector_octets_match_the_recorded_real_inventory():
    """The mapping's soil selectors are the observed awip32 octets."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    for name, parameter in (
        ("soil_temperature", 2), ("volumetric_soil_moisture", 192),
    ):
        for selector, depth in zip(
            mapping["fields"][name]["selectors"],
            (0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0),
        ):
            assert selector["discipline"] == 2
            assert selector["category"] == 0
            assert selector["parameter"] == parameter
            assert selector["center"] == 7
            assert selector["master_table_version"] == 2
            assert selector["local_table_version"] == 1
            assert selector["level_type"] == 106
            assert selector["level_value"] == depth
            assert selector["second_level_value"] == depth
