"""The AIGFS hybrid: an AI atmosphere RUNS on a borrowed physical surface.

The atmosphere-only AIGFS profile proved a solo init impossible and refused
by name.  This file pins the shape finishing that story must take: the
source becomes runnable through a CROSS-SOURCE packaged profile -- a hybrid
mapping whose seven missing canonicals are ``composition_bound``, a
composition whose ``field_sources`` binding pins the same-cycle GDAS
analysis donor's own mapping by SHA-256, and a profile row that carries the
donor mapping as a fourth pinned authority so the front door can pass it as
``--contributing-mapping``.  Still zero AIGFS code: every fact above is a
row in a JSON document or the profile registry.

The donor mapping is the checked-in GFS pressure-level table with ONE
table-data change: 2 m specific humidity is DIRECTLY selected (0.1.0 at
heightAboveGround 2 m, MEASURED present in gdas.t00z.pgrb2.0p25.f000)
instead of derived from 2 m RH, because a borrowed field must be directly
selected in the donor -- the partitioned donor decode refuses a derived
donor field by name.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import load_mapping
from gpuwm.source_adapters import (AdapterStatus, get_source_adapter,
                                   packaged_profile_sources)
from gpuwm.source_authorities import (packaged_authorities,
                                      packaged_contributing_mappings,
                                      packaged_profile)
from gpuwm.source_cli import EXIT_USAGE, main


PROFILE_ID = "aigfs-gdas-hybrid-grib2-v1"
MAPPING_ROLE = "physical_analysis_surface_mapping"
DATA_ROLE = "physical_analysis_surface_data"
PROVENANCE_ROLE = "physical_analysis_surface_provenance"

#: The canonical state the operational AIGFS product does not publish;
#: every one of these is composition_bound in the hybrid mapping and
#: bound to the GDAS analysis donor in the composition.
BORROWED = [
    "land_fraction",
    "skin_temperature",
    "soil_temperature",
    "specific_humidity_2m",
    "surface_pressure",
    "terrain_height",
    "volumetric_soil_moisture",
]


def test_the_hybrid_profile_pins_a_fourth_authority_the_donor_mapping():
    profile = packaged_profile(PROFILE_ID)
    assert profile["composition_state"] == "composed"
    assert profile["data_role"] == DATA_ROLE
    assert profile["provenance_role"] == PROVENANCE_ROLE
    declared = profile["contributing_mappings"]
    assert set(declared) == {MAPPING_ROLE}
    resolved = packaged_contributing_mappings(PROFILE_ID)
    assert set(resolved) == {MAPPING_ROLE}
    path = resolved[MAPPING_ROLE]
    assert path.is_file()
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == declared[MAPPING_ROLE]["sha256"]
    # A profile with no bindings has no contributing mappings, and says so
    # as an empty mapping rather than an error.
    assert packaged_contributing_mappings("hrrr-prs-grib2-v1") == {}


def test_the_composition_binds_the_donor_by_the_packaged_pin():
    """One donor, one binding, and the hash chain has no loose end: the
    composition's pinned donor digest IS the digest of the donor mapping
    the profile ships."""

    authorities = packaged_authorities(PROFILE_ID)
    profile = packaged_profile(PROFILE_ID)
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    bindings = contract["field_sources"]
    assert set(bindings) == {"physical_analysis_surface"}
    binding = bindings["physical_analysis_surface"]
    assert sorted(binding["fields"]) == sorted(BORROWED)
    assert binding["mapping_role"] == MAPPING_ROLE
    assert binding["data_role"] == DATA_ROLE
    assert binding["provenance_role"] == PROVENANCE_ROLE
    assert binding["grid_alignment"] == "exact_coordinate_subset"
    assert binding["time_alignment"] == "source_cycle_analysis_broadcast"
    assert binding["mapping_sha256"] \
        == profile["contributing_mappings"][MAPPING_ROLE]["sha256"]
    donor_path = packaged_contributing_mappings(PROFILE_ID)[MAPPING_ROLE]
    donor = load_mapping(donor_path)
    assert donor["name"] == binding["source_id"]


def test_the_donor_mapping_selects_every_borrowed_field_directly():
    donor = load_mapping(
        packaged_contributing_mappings(PROFILE_ID)[MAPPING_ROLE])
    for name in BORROWED:
        field = donor["fields"][name]
        assert field.get("derivation") is None, name
        assert field["selectors"], name
    # The one deliberate divergence from the checked-in GFS table, held
    # by value: 2 m specific humidity is the direct 0.1.0 @ 2 m record.
    selector = donor["fields"]["specific_humidity_2m"]["selectors"][0]
    assert (selector["discipline"], selector["category"],
            selector["parameter"]) == (0, 1, 0)
    assert selector["level_type"] == 103
    assert selector["level_value"] == 2


def test_the_hybrid_mapping_keeps_the_operational_identity_pins():
    """Every SELECTED field still pins subcenter 0: the S3/EAGLE imposter
    must match zero selectors in the hybrid exactly as it does in the
    atmosphere-only profile."""

    authorities = packaged_authorities(PROFILE_ID)
    mapping = json.loads(authorities["mapping"].read_text(encoding="utf-8"))
    selected = bound = 0
    for name, field in mapping["fields"].items():
        if field.get("provider") == "composition_bound":
            bound += 1
            assert name in BORROWED
            assert not field.get("selectors")
            continue
        for selector in field.get("selectors", []):
            selected += 1
            assert selector["center"] == 7
            assert selector["subcenter"] == 0
            assert selector["master_table_version"] == 2
            assert selector["local_table_version"] == 1
    assert bound == len(BORROWED)
    assert selected, "the hybrid must still select the atmosphere"
    target = mapping["target"]
    assert "pending_composition_requirements" not in target
    assert target["soil_layer_count"] == 4
    required = {item["name"] for item in target["required_fields"]}
    assert set(BORROWED) <= required


def test_the_aigfs_row_is_runnable_through_the_hybrid_profile():
    adapter = get_source_adapter("aigfs")
    assert adapter.packaged_profile == PROFILE_ID
    assert packaged_profile_sources()["aigfs"] == PROFILE_ID
    assert adapter.runnable is True
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.status is AdapterStatus.RUNNABLE_NOT_CERTIFIED
    # The composition note still names what is borrowed and from where.
    for named in ("soil", "orography", "GDAS"):
        assert named in adapter.composition_requirement


def _prep_argv(tmp_path, *extra, supplement=True):
    inputs = []
    for name in ("pres.f000.grib2", "sfc.f000.grib2",
                 "pres.f006.grib2", "sfc.f006.grib2"):
        path = tmp_path / name
        path.write_bytes(b"not read by --dry-run")
        inputs.extend(["--input", str(path)])
    donor = tmp_path / "donor.f000.grib2"
    donor.write_bytes(b"not read by --dry-run")
    for name in ("grib2_inventory.exe", "grib2_dump.exe"):
        (tmp_path / name).write_bytes(b"not read by --dry-run")
    return [
        "--source", "aigfs", *inputs,
        "--grib2-inventory", str(tmp_path / "grib2_inventory.exe"),
        "--grib2-dump", str(tmp_path / "grib2_dump.exe"),
        *(["--supplement", str(donor)] if supplement else []),
        "--source-manifest", str(tmp_path / "inputs.json"),
        "--source-manifest-sha256", "0" * 64,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path),
        "--experiment-config", str(tmp_path / "experiment.toml"),
        "--output-root", str(tmp_path / "out"),
        *extra, "--dry-run",
    ]


def test_the_front_door_carries_the_donor_mapping_from_the_profile(
    tmp_path, capsys,
):
    assert main(_prep_argv(tmp_path)) == 0
    command = capsys.readouterr().out
    donor_path = packaged_contributing_mappings(PROFILE_ID)[MAPPING_ROLE]
    assert "-m gpuwm.mapped_direct" in command
    assert (
        f"--contributing-mapping {MAPPING_ROLE}="
        f"{str(donor_path)}".replace("\\", "/") in command.replace("\\", "/")
    )
    assert f"--supplement {DATA_ROLE}=" in command
    assert f"--provenance {PROVENANCE_ROLE}=" in command


def test_a_caller_may_not_substitute_their_own_donor_mapping(
    tmp_path, capsys,
):
    mine = tmp_path / "mine.mapping.json"
    mine.write_text("{}", encoding="utf-8")
    argv = _prep_argv(
        tmp_path, "--contributing-mapping", f"{MAPPING_ROLE}={mine}")
    assert main(argv) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--contributing-mapping is decided by the packaged aigfs " \
        "profile" in error


def test_a_donorless_call_still_refuses_with_the_supplement_named(
    tmp_path, capsys,
):
    """The solo-init refusal survives the upgrade: without the donor bytes
    there is no run, and the message names the missing piece instead of
    faking a surface."""

    assert main(_prep_argv(tmp_path, supplement=False)) == EXIT_USAGE
    assert "--supplement" in capsys.readouterr().err
