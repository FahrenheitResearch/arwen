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

Cloud ice has had two GRIB2 identities.  HRRR before HRRRv3 (cycles
before July 2018) publishes it as CICE, discipline 0 category 6
parameter 0, in the same slot between CLMR and RWMR on the same 40
isobaric levels; HRRRv3 and v4 publish CIMIXR, 0/1/82.  With only the
newer identity mapped, every earlier cycle was refused at preparation
with ``mapped frame at <time> lacks required fields
['cloud_ice_mixing_ratio']`` (2017-01-19 00Z, the SNOWIE case date, is
the measured example).  The row lists both, newer first, so an earlier
file prepares with the cloud ice it publishes and a file that carries
neither is still refused.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm import mapped_source
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

#: Every (category, parameter) identity in discipline 0 under which a
#: public wrfprs file publishes each hydrometeor on its isobaric levels,
#: in the order the mapping prefers them.  Read from the archive's own
#: inventories (noaa-hrrr-bdp-pds wrfprsf00 .idx at 12Z) on 2026-09-28:
#: 2014-09-30, 2015-06-01, 2016-06-01, 2016-10-01, 2017-01-19 and
#: 2018-06-01 carry CICE (6, 0) on 40 levels and no CIMIXR; 2018-08-01,
#: 2019-06-01, 2020-06-01, 2020-12-01, 2020-12-03 and 2026-09-27 carry
#: CIMIXR (1, 82) on 40 levels and no CICE.  No sampled file carries both.
HYDROMETEOR_IDENTITIES = {
    "cloud_water_mixing_ratio": ((1, 22),),
    "cloud_ice_mixing_ratio": ((1, 82), (6, 0)),
    "rain_water_mixing_ratio": ((1, 24),),
    "snow_mixing_ratio": ((1, 25),),
    "graupel_or_hail_mixing_ratio": ((1, 32),),
}
CURRENT_ICE = (1, 82)
PRE_HRRRV3_ICE = (6, 0)


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
    assert set(HYDROMETEOR_IDENTITIES) == set(HYDROMETEOR_PARAMETERS)
    for name, parameter in HYDROMETEOR_PARAMETERS.items():
        assert HYDROMETEOR_IDENTITIES[name][0] == (1, parameter), (
            f"{name}: the identity current files carry must stay the "
            "preferred one")
        assert name in fields, f"{name} is not mapped; the run starts dry"
        field = fields[name]
        selectors = field["selectors"]
        assert [(selector["category"], selector["parameter"])
                for selector in selectors] \
            == list(HYDROMETEOR_IDENTITIES[name]), (name, selectors)
        for selector in selectors:
            assert selector["format"] == "grib2"
            assert selector["discipline"] == 0, (name, selector)
            assert selector["level_type"] == 100, (name, selector)
            assert "level_value" not in selector, (
                f"{name} must select every pressure level, not one")
            assert "scale" not in selector, (
                f"{name}: every identity publishes kg kg-1 as it is")
        assert field["units"] == {"source": "kg kg-1", "target": "kg kg-1"}
        assert field["source_axes"] == ["vertical", "y", "x"]
        assert field["target_axes"] == ["vertical", "y", "x"]
        assert field["location"] == "mass"
        assert field["staggering"] == "none"
        assert field["missing"] == {"kind": "reject"}


def test_no_other_field_answers_the_earlier_cloud_ice_identity():
    """A record of CICE on an isobaric level can only be cloud ice."""

    category, parameter = PRE_HRRRV3_ICE
    for name, field in _mapping()["fields"].items():
        if name == "cloud_ice_mixing_ratio":
            continue
        for selector in field.get("selectors", ()):
            assert (selector.get("discipline"), selector.get("category"),
                    selector.get("parameter")) != (0, category, parameter), (
                name, selector)


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


# --------------------------------------------------------------------
# The packaged hydrometeor rows decoded against each HRRR generation's
# record layout, through the Python engine's own assembly and frame
# materialization.  The rest of the state is a constant column under
# private parameters (category 200) so no record answers another row.
# --------------------------------------------------------------------

T0 = datetime(2017, 1, 19, 0)
LEVELS = (100000.0, 70000.0, 30000.0)
#: Cloud ice by level, kg kg-1, the size of what the 2017-01-19 00Z file
#: publishes at 850, 700 and 300 hPa (maxima 3.4e-5, 4.1e-5, 2.8e-4).
ICE = {100000.0: 3.4e-5, 70000.0: 4.1e-5, 30000.0: 2.8e-4}
OTHER_FIELDS = (
    ("air_temperature", ["vertical", "y", "x"], "mass", "K"),
    ("specific_humidity", ["vertical", "y", "x"], "mass", "kg kg-1"),
    ("eastward_wind", ["vertical", "y", "x"], "mass", "m s-1"),
    ("northward_wind", ["vertical", "y", "x"], "mass", "m s-1"),
    ("geopotential_height", ["vertical", "y", "x"], "mass", "m"),
    ("surface_pressure", ["y", "x"], "surface", "Pa"),
    ("terrain_height", ["y", "x"], "surface", "m"),
    ("skin_temperature", ["y", "x"], "surface", "K"),
    ("air_temperature_2m", ["y", "x"], "surface", "K"),
    ("specific_humidity_2m", ["y", "x"], "surface", "kg kg-1"),
    ("eastward_wind_10m", ["y", "x"], "surface", "m s-1"),
    ("northward_wind_10m", ["y", "x"], "surface", "m s-1"),
    ("land_fraction", ["y", "x"], "surface", "1"),
    ("soil_temperature", ["soil", "y", "x"], "soil", "K"),
    ("volumetric_soil_moisture", ["soil", "y", "x"], "soil", "m3 m-3"),
)


def _private_level_type(axes) -> int:
    return 100 if "vertical" in axes else 106 if "soil" in axes else 1


def _probe_mapping() -> dict:
    packaged = _mapping()["fields"]
    fields = {name: packaged[name] for name in HYDROMETEOR_IDENTITIES}
    for parameter, (name, axes, location, units) in enumerate(OTHER_FIELDS):
        fields[name] = {
            "selectors": [{"format": "grib2", "discipline": 0,
                           "category": 200, "parameter": parameter,
                           "level_type": _private_level_type(axes)}],
            "units": {"source": units, "target": units},
            "source_axes": axes, "target_axes": axes,
            "location": location, "staggering": "none",
            "missing": {"kind": "reject"},
        }
    fields["air_pressure"] = {
        "selectors": [], "derivation": "pressure-from-coordinate",
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["vertical", "y", "x"],
        "target_axes": ["vertical", "y", "x"],
        "location": "mass", "staggering": "none",
        "missing": {"kind": "reject"},
    }
    return {
        "schema": "rw-wps.mapping.v1", "name": "hrrr-prs-hydrometeor-probe",
        "format": "grib2",
        "coordinates": {
            "horizontal": {"kind": "embedded_grid"},
            "vertical": {"kind": "pressure", "units": "Pa",
                         "positive": "down", "levels": list(LEVELS)},
            "time": {"kind": "embedded_metadata"},
        },
        "fields": fields,
        "derivations": [{"name": "pressure-from-coordinate",
                         "operation": "pressure_from_vertical_coordinate"}],
        "target": {
            "required_fields": [
                {"name": name, "axes": field["target_axes"],
                 "location": field["location"],
                 "target_units": field["units"]["target"]}
                for name, field in fields.items() if name != "air_pressure"
            ],
            "soil_layer_count": 1,
            "initialization_policies": {
                "vertical_velocity": "explicit_zero_with_adapter_validation",
                "snow_water_equivalent": "explicit_zero_with_adapter_validation",
                "snow_depth": "explicit_zero_with_adapter_validation",
                "sea_ice_fraction": "explicit_zero_with_adapter_validation",
            },
        },
    }


class _Records:
    def __init__(self):
        self.items: list[mapped_source._GribRecord] = []

    def add(self, category, parameter, level_type, value, level=0.0):
        self.items.append(mapped_source._GribRecord(
            source=Path("hrrr.t00z.wrfprsf00.grib2"), index=len(self.items),
            reference_time=T0, valid_time=T0, member=None,
            parameter=parameter, level_type=level_type, level_value=level,
            table_version=None, center=7, subcenter=0,
            master_table_version=2, local_table_version=1,
            discipline=0, category=category,
            second_level_type=255, second_level_value=0.0,
            process_identity=(2, 83), time_semantics=(0,),
            values=np.full((2, 2), float(value)),
            latitude=np.array([44.0, 44.03]),
            longitude=np.array([-116.0, -115.96]),
            grid_fingerprint="one-grid",
        ))


def _file(*ice_identities) -> _Records:
    """One wrfprs valid time: the state, the four other species on their
    one identity, and cloud ice under each identity in ``ice_identities``
    (the value is scaled by the identity's position so a test can tell
    which one a level took)."""

    records = _Records()
    for parameter, (_name, axes, _location, _units) in enumerate(OTHER_FIELDS):
        level_type = _private_level_type(axes)
        for level in (LEVELS if level_type == 100 else (0.0,)):
            records.add(200, parameter, level_type, 1.0, level)
    for name, identities in HYDROMETEOR_IDENTITIES.items():
        if name == "cloud_ice_mixing_ratio":
            continue
        category, parameter = identities[0]
        for level in LEVELS:
            records.add(category, parameter, 100, 1.0e-4, level)
    for position, (category, parameter) in enumerate(ice_identities):
        for level in LEVELS:
            records.add(category, parameter, 100,
                        ICE[level] * (position + 1), level)
    return records


def _frame(records):
    mapping = _probe_mapping()
    collection = mapped_source._assemble_grib(mapping, records.items)
    frames = mapped_source._materialize_frames(
        mapping, collection, mapping_sha256="0" * 64, input_sha256={})
    assert len(frames) == 1
    return frames[0]


def _cloud_ice_by_level(frame) -> list[float]:
    field = frame.fields["cloud_ice_mixing_ratio"]
    assert field.axes == ("vertical", "y", "x")
    assert field.missing_count == 0
    return [float(plane.max()) for plane in field.values]


def test_a_pre_hrrrv3_file_prepares_with_the_cloud_ice_it_publishes():
    """The 2017-01-19 00Z layout: cloud ice as CICE, 0/6/0, and no CIMIXR.

    Before the row listed CICE, this frame was refused with ``lacks
    required fields ['cloud_ice_mixing_ratio']``: the refusal every
    pre-HRRRv3 cycle met at preparation.
    """

    frame = _frame(_file(PRE_HRRRV3_ICE))
    assert _cloud_ice_by_level(frame) == [ICE[level] for level in LEVELS]


def test_a_current_file_prepares_with_its_cimixr_cloud_ice():
    frame = _frame(_file(CURRENT_ICE))
    assert _cloud_ice_by_level(frame) == [ICE[level] for level in LEVELS]


def test_cimixr_is_taken_over_cice_where_a_file_carries_both():
    """No archived file carries both; if one did, the identity current
    files carry is the one read, level by level."""

    frame = _frame(_file(CURRENT_ICE, PRE_HRRRV3_ICE))
    assert _cloud_ice_by_level(frame) == [ICE[level] for level in LEVELS]


def test_a_file_with_no_cloud_ice_is_still_refused_at_prep():
    """Listing the earlier identity is not a zero fill: a wrfprs file
    carrying neither is refused, as a dry file always was."""

    with pytest.raises(ValueError,
                       match=r"lacks required fields \['cloud_ice_mixing_ratio'\]"):
        _frame(_file())
