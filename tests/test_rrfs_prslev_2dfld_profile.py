"""RRFS through the gauntlet: HRRR's successor is TABLE DATA, and these pins hold it.

RRFS is the model the arbitrary acceptance test exists for: HRRR's
operational successor, flowing today on ``noaa-rrfs-ops-pds`` and NOMADS
``rrfs/v1.0``.  This file pins the shape its arrival must take: three
packaged JSON documents, one ``_PACKAGED_PROFILES`` row, one ``_adapter``
row, the physics registry route, and the prepared-runner table rows.  No
RRFS decode code exists anywhere -- the CONUS grid is bit-for-bit HRRR's
Lambert (measured from real bytes, every geolocating octet identical), so
the entire HRRR wrfprs machinery carries it; the two structural facts that
are RRFS's own -- the 45-level pressure ladder with 70 hPa where HRRR has
75 hPa, and the split of the state across a prslev/2dfld file PAIR -- are
declared data, not code.

Selectors were authored from real bytes (rrfs.t00z.prslev.3km.f000 and
rrfs.t00z.2dfld.3km.f000/f001, CONUS, 2026-08-12 00Z prototype cycle,
cross-checked against the live 2026-08-17 00Z operational cycle) through
the converged grib-core inventory.  Every one of the 19 mapped fields
matched exactly one record per valid time across the pair with HRRR's own
selector octets; the pair is disjoint (675 + 262/315 messages, union with
no overlap).
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


PROFILE_ID = "rrfs-prslev-2dfld-grib2-v1"

#: The 45-level pressure ladder observed in the real prslev bytes:
#: 2, 5, 7, 10, 20, 30, 50, 70, 100 hPa, then 125..1000 by 25.  NOT the
#: HRRR ladder: RRFS has 70 hPa where HRRR has 75, extends to 2 hPa where
#: HRRR stops at 50, and drops HRRR's 1013.2 hPa entry.
EXPECTED_LEVELS = [
    float(p) for p in (200, 500, 700, 1000, 2000, 3000, 5000, 7000, 10000)
] + [float(p) for p in range(12500, 100001, 2500)]


def test_the_rrfs_row_is_a_packaged_profile_and_nothing_more():
    adapter = get_source_adapter("rrfs")
    assert adapter.packaged_profile == PROFILE_ID
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.runnable is True
    assert adapter.file_family == "GRIB2"
    assert adapter.default_product == "prslev"
    # The HRRR analogue is prslev PLUS 2dfld: prslev alone carries zero
    # surface fields and zero soil, so the row requires the pair.
    assert adapter.required_products == ("prslev", "2dfld")
    assert packaged_profile_sources()["rrfs"] == PROFILE_ID
    generic = get_source_adapter("mapped")
    assert adapter.runner == generic.runner


def test_the_rrfs_grid_is_bit_for_bit_the_hrrr_grid():
    """Measured from real bytes: every geolocating octet matches HRRR."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    grid = mapping["grid"]
    assert grid["family"] == "lambert_conformal"
    # resolution_flags 0x38 on every record, interpolated CONUS product:
    # winds are grid-relative and must be rotated, exactly as HRRR's.
    assert grid["wind_basis"] == "grid_relative_with_rotation"
    hrrr = load_mapping(packaged_authorities("hrrr-prs-grib2-v1")["mapping"])
    assert grid["parameters"] == hrrr["grid"]["parameters"]
    # And the observed octets themselves, so this test cannot drift with
    # the HRRR document.
    parameters = grid["parameters"]
    assert parameters["nx"] == 1799
    assert parameters["ny"] == 1059
    assert parameters["lat1"] == 21.138123
    assert parameters["lon1"] == 237.280472
    assert parameters["latin1"] == 38.5
    assert parameters["latin2"] == 38.5
    assert parameters["lov"] == 262.5
    assert parameters["dx_m"] == 3000.0
    assert parameters["dy_m"] == 3000.0
    assert parameters["shape_of_earth"] == 6
    assert parameters["earth_radius_m"] == 6371229.0


def test_the_rrfs_ladder_is_declared_not_borrowed_from_hrrr():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    levels = mapping["coordinates"]["vertical"]["levels"]
    assert levels == EXPECTED_LEVELS
    # The two silent-mismatch traps, pinned by name: RRFS publishes
    # 70 hPa (7000 Pa) where HRRR publishes 75 hPa (7500 Pa), and no
    # 1013.2 hPa entry.  A shared level table would mismatch silently.
    assert 7000.0 in levels
    assert 7500.0 not in levels
    assert 101320.0 not in levels
    hrrr = load_mapping(packaged_authorities("hrrr-prs-grib2-v1")["mapping"])
    assert levels != hrrr["coordinates"]["vertical"]["levels"]


def test_the_rrfs_fields_are_the_hrrr_octets_with_direct_humidity():
    """prslev publishes SPFH directly on all 45 levels: no RH derivation."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    humidity = mapping["fields"]["specific_humidity"]
    assert humidity.get("derivation") is None
    selector = humidity["selectors"][0]
    assert (selector["discipline"], selector["category"],
            selector["parameter"]) == (0, 1, 0)
    assert selector["level_type"] == 100
    # The nine-node RUC soil column, octet-identical to HRRR's.
    for field in ("soil_temperature", "volumetric_soil_moisture"):
        assert len(mapping["fields"][field]["selectors"]) == 9


def test_the_rrfs_soil_selectors_carry_the_level_key_quadruple():
    """Soil identity is the metre-decoded fixed-surface QUADRUPLE.

    All nine RRFS soil records are zero-thickness nodes whose integer
    ecCodes ``level`` collides 5-into-1 (0.00/0.01/0.04/0.10/0.30 m all
    report level 0).  The only correct key is the decoded quadruple
    (level_type, level_value, second_level_type, second_level_value) in
    metres, which is what these selectors declare -- the same cross-source
    rule HRRR and RAP already pin.
    """

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    depths = (0.0, 0.01, 0.04, 0.1, 0.3, 0.6, 1.0, 1.6, 3.0)
    for name, parameter in (
        ("soil_temperature", 2), ("volumetric_soil_moisture", 192),
    ):
        selectors = mapping["fields"][name]["selectors"]
        assert [s["level_value"] for s in selectors] == list(depths)
        for selector, depth in zip(selectors, depths):
            assert selector["discipline"] == 2
            assert selector["category"] == 0
            assert selector["parameter"] == parameter
            assert selector["center"] == 7
            assert selector["master_table_version"] == 2
            assert selector["local_table_version"] == 1
            assert selector["level_type"] == 106
            assert selector["second_level_type"] == 106
            assert selector["second_level_value"] == depth


def test_the_rrfs_snow_fields_declare_the_water_zero_fill():
    """RRFS bitmap-masks WEASD/SNOD over water; HRRR publishes full grids.

    The masked cells are water cells with no snowpack state, so the
    mapping declares the grammar's value policy with 0.0 -- the WRF
    initialization value for snow over water -- instead of rejecting.
    Soil keeps preserve_mask exactly as HRRR's profile does.
    """

    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    for name in ("snow_water_equivalent", "snow_depth"):
        assert mapping["fields"][name]["missing"] == {
            "kind": "value", "value": 0.0}
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        assert mapping["fields"][name]["missing"] == {"kind": "preserve_mask"}


def test_the_rrfs_composition_binds_ruc_nodes_to_noah_layers():
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
        == "rrfs_prslev_2dfld_in_band_surface"
    assert supplement["provenance_role"] == profile["provenance_role"] \
        == "rrfs_prslev_2dfld_in_band_surface_provenance"
    # Terrain rides the 2dfld file at every valid time: exact alignment,
    # not a broadcast.
    assert supplement["time_alignment"] == "valid_time_exact"
    assert supplement["require_invariant_across_time"] is True


def test_the_physics_registry_routes_rrfs_through_the_prepared_runner():
    payload = physics_registry()
    route = payload["runner_routes"]["tools.prepared_single_domain_forecast"]
    assert "rrfs" in route["source_ids"]
    assert route["source_template_ids"]["rrfs"] \
        == route["source_template_ids"]["hrrr-prs"]


def test_the_prepared_runner_tables_carry_the_rrfs_rows():
    assert "rrfs" in runner.SUPPORTED_SOURCES
    capabilities = runner.runner_capabilities()
    assert "rrfs" in capabilities["supported_sources"]
    source = capabilities["source_profiles"]["rrfs"]
    assert source["single_d01_gpu_execution"] is True
    cadence = capabilities["window"]["source_forcing_cadence_hours"]
    assert cadence["rrfs"] == "uniform-positive-whole-hour"
