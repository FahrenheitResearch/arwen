"""AIGFS through the gauntlet: an atmosphere-only model is TABLE DATA.

NCEP's GraphCast-based 0.25-degree AI forecast is the barest product in
the catalog: six 3-D fields on 13 pressure levels plus 2 m temperature,
10 m wind and MSLP, and NO land surface of any kind.  This file pins the
shape its arrival must take: three packaged JSON documents (the
composition role being an explicit PENDING declaration), one
``_PACKAGED_PROFILES`` row, one upgraded ``_adapter`` row -- and no AIGFS
decode code anywhere.

Two facts dominate the profile and are held here by name:

* THE FRONT DOORS DISAGREE.  The operational product is NOMADS-only and
  stamps subCentre 0; the ``noaa-nws-graphcastgfs-pds`` S3 bucket serves
  a DIFFERENT experimental run under IDENTICAL filenames stamped
  subCentre 2 (500 hPa heights differ by up to 8.66 gpm, MEASURED
  2026-08-17 00Z).  Every selector pins ``subcenter: 0`` so the imposter
  refuses by named identity octet.
* A SOLO INIT IS IMPOSSIBLE.  No soil, mask, orography, skin or surface
  pressure exists in any file, so the target declares the seven missing
  canonicals pending the cross-source composition and the source stays
  non-runnable with a named reason.

Selectors were authored from real 2026-08-17 00Z bytes (NOMADS
aigfs.t00z.pres/sfc, both front doors inventoried) through the converged
grib-core inventory.
"""

from __future__ import annotations

import json

import pytest

from gpuwm.mapped_composition import (
    PENDING_COMPOSITION_SCHEMA,
    load_composition,
)
from gpuwm.mapped_source import load_mapping
from gpuwm.source_adapters import (AdapterStatus, get_source_adapter,
                                   packaged_profile_sources)
from gpuwm.source_authorities import (packaged_authorities,
                                      packaged_profile)


PROFILE_ID = "aigfs-nomads-grib2-v1"

#: The 13-level ladder observed in the real bytes -- exactly the AIFS
#: ladder minus its 10 hPa level; model top 50 hPa.
EXPECTED_LEVELS = [
    5000, 10000, 15000, 20000, 25000, 30000, 40000,
    50000, 60000, 70000, 85000, 92500, 100000,
]

#: The canonical state the operational product does not publish.
EXPECTED_PENDING = [
    "land_fraction",
    "skin_temperature",
    "soil_temperature",
    "specific_humidity_2m",
    "surface_pressure",
    "terrain_height",
    "volumetric_soil_moisture",
]


def test_the_aigfs_row_runs_hybrid_and_the_solo_refusal_record_ships():
    """The row moved on to the cross-source hybrid profile (see
    ``test_aigfs_hybrid_profile``); what THIS file's claim becomes is
    that the atmosphere-only profile stays shipped, stays pending, and
    still refuses a solo composition by name -- the record that a solo
    init was and remains impossible."""

    adapter = get_source_adapter("aigfs")
    assert adapter.packaged_profile == "aigfs-gdas-hybrid-grib2-v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    assert adapter.runnable is True
    assert adapter.file_family == "GRIB2"
    assert adapter.default_product == "pres"
    assert adapter.required_products == ("pres", "sfc")
    assert adapter.max_forecast_hour == 384
    assert packaged_profile_sources()["aigfs"] \
        == "aigfs-gdas-hybrid-grib2-v1"
    # The composition note still names what must be borrowed.
    for named in ("soil", "orography", "skin temperature",
                  "surface pressure", "GDAS"):
        assert named in adapter.composition_requirement
    # The atmosphere-only record is still shipped, still pending.
    profile = packaged_profile(PROFILE_ID)
    assert profile["composition_state"] == "pending_cross_source"
    assert profile["data_role"] is None
    assert profile["provenance_role"] is None


def test_the_mapping_is_the_observed_thirteen_level_atmosphere():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    assert mapping["coordinates"]["vertical"]["levels"] == EXPECTED_LEVELS
    fields = mapping["fields"]
    # The six 3-D fields the product publishes, and nothing invented.
    for name, (category, parameter) in {
        "air_temperature": (0, 0),
        "specific_humidity": (1, 0),
        "eastward_wind": (2, 2),
        "northward_wind": (2, 3),
        "pressure_vertical_velocity": (2, 8),
        "geopotential_height": (3, 5),
    }.items():
        selector = fields[name]["selectors"][0]
        assert (selector["discipline"], selector["category"],
                selector["parameter"]) == (0, category, parameter)
        assert selector["level_type"] == 100
    # Omega stays omega: the product has no w in m s-1, and pretending
    # otherwise would be an invention.
    omega = fields["pressure_vertical_velocity"]
    assert omega["units"] == {"source": "Pa s-1", "target": "Pa s-1"}
    policies = mapping["target"]["initialization_policies"]
    assert "omega" in policies["vertical_velocity"]
    # MSLP is mapped under NCEP's own parameter (0.3.1), not the ECMWF
    # msl (0.3.0) -- the same quantity carries two different parameter
    # numbers across the catalog and a shared row would miss one.
    mslp = fields["air_pressure_at_mean_sea_level"]["selectors"][0]
    assert (mslp["discipline"], mslp["category"], mslp["parameter"]) \
        == (0, 3, 1)
    assert mslp["level_type"] == 101


def test_the_missing_land_surface_is_declared_not_faked():
    authorities = packaged_authorities(PROFILE_ID)
    mapping = load_mapping(authorities["mapping"])
    target = mapping["target"]
    assert sorted(target["pending_composition_requirements"]) \
        == sorted(EXPECTED_PENDING)
    assert target["soil_layer_count"] == 0
    for name in EXPECTED_PENDING:
        assert name not in mapping["fields"]
    required = {item["name"] for item in target["required_fields"]}
    assert not required & set(EXPECTED_PENDING)


def test_every_selector_pins_the_operational_subcentre():
    """The S3 imposter (subCentre 2) must match ZERO selectors."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    selectors = [
        selector
        for field in mapping["fields"].values()
        for selector in field.get("selectors", [])
    ]
    assert selectors, "the mapping must select something"
    for selector in selectors:
        assert selector["center"] == 7
        assert selector["subcenter"] == 0
        assert selector["master_table_version"] == 2
        assert selector["local_table_version"] == 1


def test_the_composition_role_refuses_by_naming_the_missing_state():
    authorities = packaged_authorities(PROFILE_ID)
    declaration = json.loads(
        authorities["composition"].read_text(encoding="utf-8"))
    assert declaration["schema"] == PENDING_COMPOSITION_SCHEMA
    with pytest.raises(ValueError) as refusal:
        load_composition(authorities["composition"], authorities["mapping"])
    message = str(refusal.value)
    for name in EXPECTED_PENDING:
        assert name in message
    assert "GFS/GDAS" in message


def test_the_provenance_pins_the_front_door_and_names_the_imposter():
    authorities = packaged_authorities(PROFILE_ID)
    provenance = json.loads(
        authorities["provenance"].read_text(encoding="utf-8"))
    assert "nomads.ncep.noaa.gov" in provenance["front_door"]["operational"]
    identity = provenance["identity"]
    assert identity["subcentre"] == 0
    assert identity["generating_process_identifier"] == 137
    imposter = provenance["imposter"]
    assert "noaa-nws-graphcastgfs-pds" in imposter["name"]
    assert "subCentre" in imposter["in_band_discriminator"]
    # The precipitation two-window trap is a deliberate, named exclusion.
    assert "tp" in json.dumps(provenance["deliberate_exclusions"]) or \
        "precipitation" in json.dumps(provenance["deliberate_exclusions"])
