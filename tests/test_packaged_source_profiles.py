"""A shipped mapping is table data, and the front door proves it.

The arbitrary acceptance test says adding a model must be metadata work,
not a new code path.  A PACKAGED PROFILE is what that looks like when the
mapping can be written: three JSON documents in ``gpuwm/authorities``,
one row of :data:`gpuwm.source_authorities._PACKAGED_PROFILES` pinning
their digests, and one ``_adapter(...)`` row naming the profile.  No
runner, no module, no branch.

These tests hold that shape rather than describing it: the profile
documents must really load through the same validator every mapping goes
through, the digests must really match the shipped bytes, the front door
must really compose the generic mapped command from the profile, and a
caller must not be able to substitute their own mapping into a packaged
source's name.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from gpuwm.ingest.soil_contract import validate_soil_layer_contract
from gpuwm.mapped_composition import load_composition
from gpuwm.mapped_source import load_mapping
from gpuwm.source_adapters import (get_source_adapter, packaged_profile_sources,
                                   source_adapters)
from gpuwm.source_authorities import (PROFILE_ROLES, packaged_authorities,
                                      packaged_authority_sha256,
                                      packaged_contributing_mappings,
                                      packaged_contributing_sha256,
                                      packaged_profile, packaged_profile_ids)
from gpuwm.source_cli import EXIT_USAGE, main


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("profile_id", packaged_profile_ids())
def test_every_shipped_profile_is_three_documents_pinned_by_digest(profile_id):
    profile = packaged_profile(profile_id)
    authorities = packaged_authorities(profile_id)
    pins = packaged_authority_sha256(profile_id)
    assert set(authorities) == set(PROFILE_ROLES)
    # A profile whose source publishes bytes the mapped engine cannot read
    # names an input normalizer, and pins that fourth document the same way.
    # It is not one of PROFILE_ROLES: those are the roles the front door
    # spells on the command line, and this one is resolved by name instead.
    extra = {"normalization"} if profile.get("input_normalizer") else set()
    assert set(pins) == set(PROFILE_ROLES) | extra
    for role in PROFILE_ROLES:
        path = authorities[role]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[role]
        # Shipped INSIDE the package, not beside the checkout: a wheel
        # user has no `configs/`.
        assert path.parent == ROOT / "gpuwm" / "authorities"
    if extra:
        from gpuwm.source_authorities import packaged_normalization
        path = packaged_normalization(profile["input_normalizer"])
        assert path.is_file()
        assert path.parent == ROOT / "gpuwm" / "authorities"
        assert (hashlib.sha256(path.read_bytes()).hexdigest()
                == pins["normalization"])
    assert profile["source_format"] in {"grib1", "grib2", "netcdf"}


@pytest.mark.parametrize("profile_id", packaged_profile_ids())
def test_every_shipped_profile_validates_as_an_ordinary_mapping(profile_id):
    """No private validator: a packaged mapping is just a mapping.

    A ``pending_cross_source`` profile's mapping validates the same way;
    its composition role is the explicit PENDING declaration, whose whole
    contract is that loading it always refuses by naming the missing
    state -- held here so a pending profile can never quietly become
    runnable without shipping a real composition.
    """

    authorities = packaged_authorities(profile_id)
    mapping = load_mapping(authorities["mapping"])
    profile = packaged_profile(profile_id)
    assert mapping["format"] == profile["source_format"]
    if profile["composition_state"] == "pending_cross_source":
        assert profile["data_role"] is None
        assert profile["provenance_role"] is None
        pending = mapping["target"]["pending_composition_requirements"]
        assert pending, "a pending profile must name its missing state"
        with pytest.raises(ValueError) as refusal:
            load_composition(authorities["composition"], authorities["mapping"])
        for name in pending:
            assert name in str(refusal.value)
        return
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    profile = packaged_profile(profile_id)
    bindings = dict(contract.get("field_sources") or {})
    if bindings:
        # A cross-source profile: terrain and the soil pair ride a
        # contributing-source binding, so the soil-depth contract
        # validates against the pinned DONOR table -- exactly the check
        # decode performs once the donor's bytes are pinned -- and the
        # data/provenance roles live on the binding, not a supplement.
        assert "terrain_height" not in contract["supplements"]
        terrain_bindings = [
            binding for binding in bindings.values()
            if "terrain_height" in binding["fields"]
        ]
        assert len(terrain_bindings) == 1
        binding = terrain_bindings[0]
        assert binding["data_role"] == profile["data_role"]
        assert binding["provenance_role"] == profile["provenance_role"]
        contributing = packaged_contributing_mappings(profile_id)
        donor = load_mapping(contributing[str(binding["mapping_role"])])
        validate_soil_layer_contract(contract["soil_layers"], mapping=donor)
    else:
        # And its soil contract binds each declared depth to a selector.
        validate_soil_layer_contract(
            contract["soil_layers"], mapping=mapping)
        supplement = contract["supplements"]["terrain_height"]
        assert supplement["data_role"] == profile["data_role"]
        assert supplement["provenance_role"] == profile["provenance_role"]


@pytest.mark.parametrize("profile_id", packaged_profile_ids())
def test_contributing_mappings_are_pinned_table_data(profile_id):
    """A cross-source profile ships its donor's sealed mapping too.

    The composition pins the donor mapping by SHA-256; the profile row
    must ship the exact document those bytes hash to, under the exact
    role the binding declares, from inside the package -- otherwise the
    prepared runner could not pass ``contributing_mappings`` and the
    packaged name would describe a route a wheel user cannot run.
    """

    profile = packaged_profile(profile_id)
    if profile["composition_state"] == "pending_cross_source":
        # A pending profile has no runnable composition to bind donors
        # to, and _profile refuses the slot on one; nothing to resolve.
        assert dict(profile["contributing_mappings"]) == {}
        return
    authorities = packaged_authorities(profile_id)
    contract = load_composition(
        authorities["composition"], authorities["mapping"])
    bindings = dict(contract.get("field_sources") or {})
    declared_roles = {
        str(binding["mapping_role"]): binding
        for binding in bindings.values()
    }
    contributing = packaged_contributing_mappings(profile_id)
    pins = packaged_contributing_sha256(profile_id)
    assert set(contributing) == set(pins) == set(declared_roles)
    for role, path in contributing.items():
        assert path.is_file()
        assert path.parent == ROOT / "gpuwm" / "authorities"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == pins[role]
        assert digest == declared_roles[role]["mapping_sha256"]
        donor = load_mapping(path)
        assert donor["name"] == declared_roles[role]["source_id"]


def test_every_packaged_source_row_points_at_a_shipped_profile():
    sources = packaged_profile_sources()
    assert set(sources) <= {
        adapter.source_id for adapter in source_adapters()}
    for source, profile_id in sources.items():
        adapter = get_source_adapter(source)
        assert profile_id in packaged_profile_ids()
        state = packaged_profile(profile_id)["composition_state"]
        if state == "pending_cross_source":
            # An atmosphere-only profile decodes but must NOT be
            # runnable: its composition role is a named refusal, and the
            # adapter row must say what has to be composed.
            assert adapter.runnable is False
            assert adapter.composition_requirement
        else:
            assert adapter.runnable is True
        # The file family the registry advertises and the format the
        # profile decodes are one statement, not two.
        family = adapter.file_family.lower()
        assert packaged_profile(profile_id)["source_format"] in family \
            or "netcdf" in family


def test_the_20crv3_netcdf_row_costs_no_runner_of_its_own():
    """The arbitrary claim, asserted where it would break first."""

    adapter = get_source_adapter("20crv3-cf")
    generic = get_source_adapter("mapped")
    assert adapter.runner == generic.runner == "mapped_composition_v1"
    assert adapter.packaged_profile == "20crv3-netcdf-v1"
    assert generic.packaged_profile is None


def _prep_argv(tmp_path, *extra):
    inputs = []
    for name in ("air.nc", "sfc.nc", "invariant.nc"):
        path = tmp_path / name
        path.write_bytes(b"not read by --dry-run")
        inputs.extend(["--input", str(path)])
    return [
        "--source", "20crv3-cf", *inputs,
        "--supplement", str(tmp_path / "invariant.nc"),
        "--source-manifest", str(tmp_path / "inputs.json"),
        "--source-manifest-sha256", "0" * 64,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path),
        "--experiment-config", str(tmp_path / "experiment.toml"),
        "--output-root", str(tmp_path / "out"),
        *extra, "--dry-run",
    ]


def test_the_front_door_fills_the_mapped_command_in_from_the_profile(
    tmp_path, capsys,
):
    assert main(_prep_argv(tmp_path)) == 0
    command = capsys.readouterr().out
    authorities = packaged_authorities("20crv3-netcdf-v1")
    profile = packaged_profile("20crv3-netcdf-v1")

    # The generic mapped runner, with nothing 20CRv3-shaped in the command
    # except the three documents the distribution supplied.
    assert "-m gpuwm.mapped_direct" in command
    assert "--source-format netcdf" in command
    for role in ("mapping", "composition"):
        assert str(authorities[role]).replace("\\", "/") in command
    assert (
        f"--supplement {profile['data_role']}="
        in command)
    assert (
        f"--provenance {profile['provenance_role']}="
        in command)


def test_the_front_door_tells_the_mapped_runner_which_source_it_is(
    tmp_path, capsys,
):
    """So the mapped route can print its forecast command when it ends.

    The prepared-forecast command is bound to a source id, and the
    mapped runner is generic -- it has no idea which name the front door
    was called by.  Without the id forwarded, the route finished a
    preparation and printed nothing, while the GFS route printed a
    complete hash-bound line; that gap sent a pilot hunting digests
    inside proof.json and onto the one field that never matches.
    """

    assert main(_prep_argv(tmp_path)) == 0
    assert "--prepared-forecast-source 20crv3-cf" in capsys.readouterr().out


def test_a_caller_may_not_substitute_their_own_mapping_into_the_name(
    tmp_path, capsys,
):
    """The whole value of a packaged name is that it means one thing."""

    mapping = tmp_path / "mine.json"
    mapping.write_text("{}", encoding="utf-8")
    assert main(_prep_argv(tmp_path, "--mapping", str(mapping))) == EXIT_USAGE
    error = capsys.readouterr().err
    assert "--mapping is decided by the packaged 20crv3-cf profile" in error


def test_the_supplement_role_comes_from_the_profile_not_the_caller(
    tmp_path, capsys,
):
    argv = _prep_argv(tmp_path)
    index = argv.index("--supplement")
    argv[index + 1] = f"some_other_role={argv[index + 1]}"
    assert main(argv) == EXIT_USAGE
    assert "supplies the role" in capsys.readouterr().err


def _hybrid_prep_argv(tmp_path, *extra):
    inputs = []
    for name in ("pres.f000.grib2", "sfc.f000.grib2",
                 "pres.f006.grib2", "sfc.f006.grib2"):
        path = tmp_path / name
        path.write_bytes(b"not read by --dry-run")
        inputs.extend(["--input", str(path)])
    donor = tmp_path / "analysis.f000.grib2"
    donor.write_bytes(b"not read by --dry-run")
    for name in ("grib2_inventory.exe", "grib2_dump.exe"):
        (tmp_path / name).write_bytes(b"not read by --dry-run")
    return [
        "--source", "aigefs", *inputs,
        "--supplement", str(donor),
        "--grib2-inventory", str(tmp_path / "grib2_inventory.exe"),
        "--grib2-dump", str(tmp_path / "grib2_dump.exe"),
        "--source-manifest", str(tmp_path / "inputs.json"),
        "--source-manifest-sha256", "0" * 64,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path),
        "--experiment-config", str(tmp_path / "experiment.toml"),
        "--output-root", str(tmp_path / "out"),
        *extra, "--dry-run",
    ]


def test_the_front_door_fills_the_contributing_mapping_in_from_the_profile(
    tmp_path, capsys,
):
    """A cross-source packaged profile decides its donor mapping too.

    The composed decode refuses a missing contributing mapping, so a
    packaged cross-source name that did not fill the binding in would be
    a shipped route no caller can run -- and one the caller COULD point
    at a different donor table would let a run claim the packaged name
    while borrowing through a mapping nobody shipped.
    """

    assert main(_hybrid_prep_argv(tmp_path)) == 0
    command = capsys.readouterr().out
    contributing = packaged_contributing_mappings("aigefs-member-hybrid-grib2-v1")
    assert "-m gpuwm.mapped_direct" in command
    for role, path in contributing.items():
        assert (
            f"--contributing-mapping {role}={path}".replace("\\", "/")
            in command.replace("\\", "/")
        )


def test_a_caller_may_not_substitute_their_own_contributing_mapping(
    tmp_path, capsys,
):
    mine = tmp_path / "mine.json"
    mine.write_text("{}", encoding="utf-8")
    argv = _hybrid_prep_argv(
        tmp_path,
        "--contributing-mapping",
        f"physical_analysis_surface_mapping={mine}",
    )
    assert main(argv) == EXIT_USAGE
    error = capsys.readouterr().err
    assert (
        "--contributing-mapping is decided by the packaged aigefs profile"
        in error
    )


# ---------------------------------------------------------------------------
# A166: a preparation bound to a mapping that differs from the packaged one
# only in an admission-only declaration is still that profile's preparation.
# ---------------------------------------------------------------------------

def _toggled_declaration(profile_id: str) -> tuple[bytes, dict]:
    """The packaged mapping as a release on the other side of A159 bound it.

    The whole-multiples key removed where the profile declares it (what a
    2.8.0 preparation copied into its evidence) and added where it does
    not, re-serialized so the bytes, and therefore the digest, differ.
    """

    import json

    from gpuwm.source_authorities import BOUNDARY_MULTIPLES_KEY

    document = json.loads(
        packaged_authorities(profile_id)["mapping"].read_bytes())
    target = document["target"]
    if BOUNDARY_MULTIPLES_KEY in target:
        del target[BOUNDARY_MULTIPLES_KEY]
    else:
        target[BOUNDARY_MULTIPLES_KEY] = True
    data = json.dumps(document, indent=2).encode("utf-8")
    assert hashlib.sha256(data).hexdigest() \
        != packaged_authority_sha256(profile_id)["mapping"]
    return data, document


@pytest.mark.parametrize("profile_id", packaged_profile_ids())
def test_a_declaration_only_mapping_change_resolves_an_earlier_preparation(
        profile_id):
    """A159 moved ten mapping pins by one admission key; frames did not move.

    Every packaged profile, both directions of the toggle: the bound
    document resolves at the spacing the packaged target declares, and
    the packaged bytes themselves resolve as before.
    """

    from gpuwm.source_authorities import (bound_mapping_refusal,
                                          mapping_decode_identity,
                                          packaged_mapping_target)

    import json

    data, document = _toggled_declaration(profile_id)
    packaged = packaged_authorities(profile_id)["mapping"].read_bytes()
    assert mapping_decode_identity(document) \
        == mapping_decode_identity(json.loads(packaged))
    spacing = int(packaged_mapping_target(profile_id)[
        "boundary_interval_seconds"])
    assert bound_mapping_refusal(
        profile_id, data, spacings_seconds=(spacing,)) is None
    assert bound_mapping_refusal(profile_id, packaged) is None


@pytest.mark.parametrize("profile_id", packaged_profile_ids())
def test_a_mapping_that_decodes_differently_is_still_refused(profile_id):
    """Anything outside the admission keys is what the frames are made of."""

    import json

    from gpuwm.source_authorities import bound_mapping_refusal

    _, document = _toggled_declaration(profile_id)
    renamed = json.loads(json.dumps(document))
    first = sorted(renamed["fields"])[0]
    renamed["fields"][f"{first}_renamed"] = renamed["fields"].pop(first)
    levels = json.loads(json.dumps(document))
    levels["target"]["target_vertical_levels"] = int(
        levels["target"].get("target_vertical_levels") or 0) + 1
    for changed in (renamed, levels):
        reason = bound_mapping_refusal(
            profile_id, json.dumps(changed).encode("utf-8"))
        assert reason is not None and "decodes differently" in reason
    assert "is not a JSON document" in bound_mapping_refusal(
        profile_id, b"\xff not json")


def test_a_resolved_mapping_is_held_to_the_packaged_target_s_spacing():
    """A cadence the packaged target does not take is refused, by name.

    hrrr-prs declares 3600 s with whole multiples: a preparation bound to
    the 2.8.0 document resolves at 1 h and 3 h, and is refused at 90 min,
    which no hrrr-prs mapping ever took.
    """

    from gpuwm.source_authorities import bound_mapping_refusal

    data, _ = _toggled_declaration("hrrr-prs-grib2-v1")
    assert bound_mapping_refusal(
        "hrrr-prs-grib2-v1", data, spacings_seconds=(3600, 10800)) is None
    reason = bound_mapping_refusal(
        "hrrr-prs-grib2-v1", data, spacings_seconds=(5400,))
    assert reason == ("its boundary spacing 5400 seconds is not a whole "
                      "multiple of the 3600 seconds the target contract "
                      "declares")
    # A profile that keeps a single spacing (no multiples) refuses a
    # bound document's wider claim at any spacing but its own.  Chosen
    # from the packaged rows rather than named: A164 gave the profile
    # this used to name (aigefs) whole multiples, and the case went red.
    from gpuwm.source_authorities import (BOUNDARY_MULTIPLES_KEY,
                                          packaged_mapping_target)

    single_id = next(
        profile for profile in packaged_profile_ids()
        if BOUNDARY_MULTIPLES_KEY not in packaged_mapping_target(profile)
        and packaged_mapping_target(profile).get("boundary_interval_seconds"))
    spacing = int(packaged_mapping_target(single_id)["boundary_interval_seconds"])
    single, _ = _toggled_declaration(single_id)
    assert bound_mapping_refusal(
        single_id, single, spacings_seconds=(spacing,)) is None
    assert f"differs from target contract {spacing}" in bound_mapping_refusal(
        single_id, single, spacings_seconds=(2 * spacing,))


def test_a_three_hour_icon_global_preparation_made_before_a173_is_still_its():
    """A173 moved the icon-global mapping's spacing from 3 h to the 1 h DWD
    posts.  A preparation bound to the earlier document decodes the same
    frames, so it resolves at the 3 h it was made at and is held to the
    packaged target at any other spacing."""

    import json

    from gpuwm.source_authorities import bound_mapping_refusal

    profile = "icon-global-grib2-v1"
    document = json.loads(packaged_authorities(profile)["mapping"].read_bytes())
    assert document["target"]["boundary_interval_seconds"] == 3600
    document["target"]["boundary_interval_seconds"] = 10800
    earlier = json.dumps(document, indent=1).encode("utf-8")
    assert bound_mapping_refusal(
        profile, earlier, spacings_seconds=(10800,)) is None
    assert "not a whole multiple" in bound_mapping_refusal(
        profile, earlier, spacings_seconds=(5400,))
