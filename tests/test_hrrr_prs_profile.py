"""HRRR's pressure-level profile carries the condensate the file publishes.

The packaged ``hrrr-prs-grib2-v1`` mapping is table data: what it declares
is what the mapped route decodes, and what it leaves out is zero-filled by
policy before the model ever sees the file.  The five hydrometeor mass
fields the wrfprs product publishes on every pressure level (cloud water,
cloud ice, rain, snow and graupel: GRIB2 discipline 0, category 1,
parameters 22, 82, 24, 25 and 32 at level type 100, kg kg-1) were absent
from the mapping and zeroed by its ``initialization_policies``, so every
run through ``--source hrrr-prs`` started with no condensate and had to
grow its storms from vapour, while ``--source hrrr`` mapped all five.

These pins hold the mapping to the file: the five rows exist with the
selectors the real 2026-08-16 00Z bytes carry, they are required, and no
explicit-zero policy remains for a field the mapping now maps.  Vertical
velocity is deliberately still zero-by-policy: the regular-source join has
no consumer for it and WRF real starts W at zero.
"""

from __future__ import annotations

import hashlib

from gpuwm.mapped_source import HYDROMETEOR_LEGACY_NAMES, load_mapping
from gpuwm.source_adapters import get_source_adapter, packaged_profile_sources
from gpuwm.source_authorities import packaged_authorities, packaged_profile


PROFILE_ID = "hrrr-prs-grib2-v1"

#: The wrfprs hydrometeor records, by canonical name: GRIB2 parameter
#: number in discipline 0, category 1 (CLMR 22, ICMR 82, RWMR 24, SNMR 25,
#: GRLE 32), every one on the 39 pressure levels (level type 100).
HYDROMETEOR_PARAMETERS = {
    "cloud_water_mixing_ratio": 22,
    "cloud_ice_mixing_ratio": 82,
    "rain_water_mixing_ratio": 24,
    "snow_mixing_ratio": 25,
    "graupel_or_hail_mixing_ratio": 32,
}


def _mapping():
    return load_mapping(packaged_authorities(PROFILE_ID)["mapping"])


def test_the_hrrr_prs_row_is_a_packaged_profile():
    adapter = get_source_adapter("hrrr-prs")
    assert adapter.packaged_profile == PROFILE_ID
    assert adapter.runner == "mapped_composition_v1"
    assert packaged_profile_sources()["hrrr-prs"] == PROFILE_ID
    assert set(adapter.aliases) == {"hrrr-pressure", "hrrr-wrfprs"}


def test_the_mapping_declares_every_hydrometeor_the_file_publishes():
    mapping = _mapping()
    fields = mapping["fields"]
    assert set(HYDROMETEOR_PARAMETERS) == set(HYDROMETEOR_LEGACY_NAMES), (
        "the pin table and the regular-source join disagree about which "
        "five fields are the hydrometeors")
    for name, parameter in HYDROMETEOR_PARAMETERS.items():
        assert name in fields, f"{name} is not mapped; the run starts dry"
        field = fields[name]
        selectors = field["selectors"]
        assert len(selectors) == 1, (name, selectors)
        selector = selectors[0]
        assert selector["format"] == "grib2"
        assert (selector["discipline"], selector["category"],
                selector["parameter"]) == (0, 1, parameter), (name, selector)
        assert selector["level_type"] == 100, (name, selector)
        assert "level_value" not in selector, (
            f"{name} must select every pressure level, not one")
        assert field["units"] == {"source": "kg kg-1", "target": "kg kg-1"}
        assert field["source_axes"] == ["vertical", "y", "x"]
        assert field["target_axes"] == ["vertical", "y", "x"]
        assert field["location"] == "mass"
        assert field["staggering"] == "none"
        assert field["missing"] == {"kind": "reject"}


def test_the_hydrometeors_are_required_so_a_dry_file_is_refused_at_prep():
    target = _mapping()["target"]
    required = {row["name"]: row for row in target["required_fields"]}
    for name in HYDROMETEOR_PARAMETERS:
        assert name in required, name
        assert required[name]["axes"] == ["vertical", "y", "x"]
        assert required[name]["location"] == "mass"
        assert required[name]["target_units"] == "kg kg-1"


def test_no_explicit_zero_policy_survives_for_a_field_the_mapping_maps():
    """The zero policy was the defect; a policy beside a mapped row is a
    document saying two things about one field."""
    target = _mapping()["target"]
    policies = target["initialization_policies"]
    for name in HYDROMETEOR_PARAMETERS:
        assert name not in policies, (
            f"{name} is mapped and still carries policy {policies[name]!r}")
    # W stays zero by policy and the document still says so: the regular
    # join drops a carried vertical velocity, and WRF real starts W at 0.
    assert policies["vertical_velocity"] == (
        "explicit_zero_with_adapter_validation")
    controlled = set(target["policy_controlled_fields"])
    assert set(HYDROMETEOR_PARAMETERS) <= controlled
    assert "vertical_velocity" in controlled


def test_the_packaged_pin_is_the_mapping_on_disk():
    """A re-pinned digest is what lets the wheel refuse a drifted table."""
    authorities = packaged_authorities(PROFILE_ID)
    pinned = packaged_profile(PROFILE_ID)["sha256"]["mapping"]
    observed = hashlib.sha256(authorities["mapping"].read_bytes()).hexdigest()
    assert pinned == observed
