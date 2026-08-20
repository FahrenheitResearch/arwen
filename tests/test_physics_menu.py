"""The per-source physics menu, and the refusals that read it.

Drew's design, 2026-08-20: "why cant every model have multiple unique
working default".  Every registered source gets its OWN default and its
own set of suites that actually run on it, and both are COMPUTED from
the admissibility rules that already refuse -- never a hand-kept table
one repo to the left.

The failure this closes was met while driving the GUI: a suite that runs
shortwave with longwave off met a night window, the refusal's remedy
named a Kain-Fritsch suite, and the HRRR route then refused that suite
for ``cu_physics = 1`` because a 3 km grid resolves its own convection.
A remedy that leads to the next refusal names nothing.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from gpuwm.runplan import run_plan_main


def _menu_document(capsys):
    """Run the REAL front door and return its one printed document."""

    from gpuwm.cli import build_parser

    assert run_plan_main(
        build_parser().parse_args(["run-plan", "--physics-profiles"])) == 0
    captured = capsys.readouterr().out
    # stdout is the machine channel: the registries print on import, and
    # the redirect is what keeps their chatter off the channel a
    # consumer calls json.loads on.
    return json.loads(captured), captured


def _cells(document, source_id):
    row = next(row for row in document["sources"]
               if row["source_id"] == source_id)
    return row, {cell["profile_id"]: cell for cell in row["profiles"]}


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_menu_query_answers_with_its_own_versioned_schema(capsys):
    from gpuwm.runplan import PHYSICS_PROFILES_SCHEMA

    document, raw = _menu_document(capsys)
    assert document["schema"] == PHYSICS_PROFILES_SCHEMA
    assert document["schema"] == "gpuwm.run-plan.physics-profiles.v1"
    assert raw.lstrip().startswith("{")
    # One document on the channel, not a document with a print in it.
    assert raw.count("\n{\n") == 0
    assert document["gpuwm_version"]
    assert document["registry_schema"] == "gpuwm-native-source-adapters-v1"
    assert document["physics_registry_sha256"]
    assert document["admissibility_rules"]


def test_the_probe_document_advertises_the_menu_schema():
    from gpuwm.runplan import PHYSICS_PROFILES_SCHEMA, probe_environment

    document = probe_environment(readiness=False)
    assert document["schemas"]["physics_profiles"] == PHYSICS_PROFILES_SCHEMA


# ---------------------------------------------------------------------------
# Both axes are the doors' own lists (the arbitrary gate)
# ---------------------------------------------------------------------------


def test_the_source_axis_is_the_registry_in_registry_order(capsys):
    """Fails the moment the reply is built from anything but the registry."""

    from gpuwm.source_adapters import source_adapters

    document, _raw = _menu_document(capsys)
    registered = [adapter.source_id for adapter in source_adapters()]
    assert [row["source_id"] for row in document["sources"]] == registered
    assert document["source_count"] == len(registered)


def test_the_profile_axis_is_the_wizards_own_list_in_its_own_order(capsys):
    """The order is the door's, because the order carries meaning.

    ``WIZARD_PHYSICS_PROFILES`` puts the nocturnally valid suites first;
    a menu that re-sorted them would break the derivation that picks a
    source's default and a refusal's remedy off the head of the list.
    """

    from gpuwm.domain_wizard import WIZARD_PHYSICS_PROFILES

    document, _raw = _menu_document(capsys)
    listed = [entry["profile_id"] for entry in document["profiles"]]
    assert listed == list(WIZARD_PHYSICS_PROFILES)
    assert document["profile_count"] == len(WIZARD_PHYSICS_PROFILES)
    for row in document["sources"]:
        assert [cell["profile_id"] for cell in row["profiles"]] == listed


# ---------------------------------------------------------------------------
# Every cell is the door's own verdict
# ---------------------------------------------------------------------------


def test_every_cell_is_the_pairing_predicate_the_wizard_refuses_by(capsys):
    """One predicate, two surfaces: the menu and the emission refusal.

    ``why_not`` is the blocker VERBATIM.  A menu that paraphrased would
    be a second vocabulary for the same fact, and the reader who follows
    the menu and then meets the refusal would read two different
    sentences about one switch.
    """

    from gpuwm.domain_wizard import profile_route_blocker

    document, _raw = _menu_document(capsys)
    for row in document["sources"]:
        for cell in row["profiles"]:
            blocker = profile_route_blocker(cell["profile_id"],
                                            row["source_id"])
            assert cell["admissible"] is (blocker is None), (
                row["source_id"], cell["profile_id"])
            assert cell["why_not"] == blocker


def test_the_hrrr_route_refuses_kain_fritsch_and_the_menu_says_which(capsys):
    """The contrasting pair Drew hit, both directions.

    ERA5 admits every shipped suite; the native HRRR route pins
    ``cu_physics = 0`` because its 3 km grid resolves convection, so
    every Kain-Fritsch suite is inadmissible there -- and the menu
    carries the route's own sentence saying so.
    """

    from gpuwm.domain_wizard import DEFAULT_PHYSICS_PROFILE

    document, _raw = _menu_document(capsys)
    _hrrr, hrrr = _cells(document, "hrrr")
    _era5, era5 = _cells(document, "era5")

    kf = [entry["profile_id"] for entry in document["profiles"]
          if entry["cumulus_scheme_id"] == 1]
    assert kf, "the shipped list must still carry a Kain-Fritsch suite"
    for profile in kf:
        assert era5[profile]["admissible"] is True, profile
        assert hrrr[profile]["admissible"] is False, profile
        assert "cu_physics=1 (the route requires 0)" in hrrr[profile][
            "why_not"]
    assert era5[DEFAULT_PHYSICS_PROFILE]["admissible"] is True
    assert hrrr[DEFAULT_PHYSICS_PROFILE]["admissible"] is False
    # And the menu is not empty for the route that refuses those: a
    # source with no runnable suite would be the real defect.
    assert any(cell["admissible"] for cell in hrrr.values())


def test_day_only_is_the_longwave_off_class_read_off_the_switches(capsys):
    """The nocturnal-validity class, derived from the pairing itself."""

    from gpuwm.physics_compat import single_domain_runtime_switches

    document, _raw = _menu_document(capsys)
    asymmetric = 0
    for entry in document["profiles"]:
        switches = single_domain_runtime_switches(entry["profile_id"])
        lw = int(switches.get("ra_lw_physics",
                              switches.get("ra_physics", 0)))
        sw = int(switches.get("ra_sw_physics",
                              switches.get("ra_physics", 0)))
        expected = sw > 0 and lw == 0
        assert entry["day_only"] is expected, entry["profile_id"]
        assert entry["longwave_scheme_id"] == lw
        assert entry["shortwave_scheme_id"] == sw
        if expected:
            asymmetric += 1
            assert "longwave" in entry["day_only_reason"]
        else:
            assert entry["day_only_reason"] is None
    assert asymmetric, "the shipped list still carries daytime suites"
    # The class reaches the per-source cell too -- a picker greys a cell,
    # not a footnote.
    for row in document["sources"]:
        for cell, entry in zip(row["profiles"], document["profiles"]):
            assert cell["day_only"] == entry["day_only"]


def test_maturity_is_the_physics_registry_template_maturity(capsys):
    """Copied from the registry, never a second vocabulary here."""

    from gpuwm.physics_registry import physics_registry

    templates = physics_registry()["templates"]
    document, _raw = _menu_document(capsys)
    for entry in document["profiles"]:
        declared = templates.get(entry["profile_id"], {}).get("maturity")
        assert entry["maturity"]["registry_maturity"] == declared, (
            entry["profile_id"])
        assert entry["maturity"]["verification_status"]
    for row in document["sources"]:
        for cell, entry in zip(row["profiles"], document["profiles"]):
            assert cell["maturity"] == entry["maturity"]


def test_vertical_bounds_are_the_preflights_own_bounds(capsys):
    """The gate that green-lit nz=130 under YSU answers here too.

    A menu that offered a suite without its level bound would hand a
    front end exactly the configuration `gpuwm check` used to pass and
    the first physics call then killed.
    """

    from gpuwm.physics_compat import (
        single_domain_runtime_switches,
        validate_resolved_physics_vertical_levels,
    )

    document, _raw = _menu_document(capsys)
    for entry in document["profiles"]:
        switches = dict(single_domain_runtime_switches(entry["profile_id"]))
        switches["nz"] = 40
        checks = validate_resolved_physics_vertical_levels(switches)["checks"]
        floors = [check["minimum"] for check in checks
                  if check.get("minimum") is not None]
        assert entry["vertical_levels"]["minimum"] == (
            max(floors) if floors else None), entry["profile_id"]
        assert entry["vertical_levels"]["basis"]
        assert entry["vertical_levels"]["bounds"]
    # YSU's ceiling is the one that killed a run; it must be present and
    # it must be the contract's number, not a number spelled here.
    from gpuwm.physics_vertical_contract import YSU_VERTICAL_LEVEL_BOUNDS
    from gpuwm.domain_wizard import DEFAULT_PHYSICS_PROFILE

    default = next(entry for entry in document["profiles"]
                   if entry["profile_id"] == DEFAULT_PHYSICS_PROFILE)
    assert default["vertical_levels"]["maximum"] == min(
        bound for bound in (YSU_VERTICAL_LEVEL_BOUNDS[1], 128, 256)
        if bound is not None)


# ---------------------------------------------------------------------------
# Every source gets its own WORKING default
# ---------------------------------------------------------------------------


def test_every_source_default_is_admissible_on_that_source(capsys):
    """Drew's ruling, as a gate.

    A default that its own route refuses is a source whose bare run
    cannot start, and "fixed" means a bare default run stops showing the
    defect.
    """

    document, _raw = _menu_document(capsys)
    for row in document["sources"]:
        default = row["default_profile_id"]
        assert default, row["source_id"]
        cells = {cell["profile_id"]: cell for cell in row["profiles"]}
        assert cells[default]["is_default"] is True
        assert cells[default]["admissible"] is True, row["source_id"]
        assert sum(cell["is_default"] for cell in row["profiles"]) == 1
        # A default that runs the surface into a nocturnal collapse is
        # not a default; the ordering exists to prevent exactly that.
        assert cells[default]["day_only"] is False, row["source_id"]


def test_the_default_is_what_the_wizard_actually_binds(capsys):
    """The menu and the emitting door cannot disagree.

    A menu is a promise about what a bare run does.  It is checked
    against the wizard's own resolution rather than recomputed here.
    """

    from gpuwm.domain_wizard import resolved_physics_profile

    document, _raw = _menu_document(capsys)
    for row in document["sources"]:
        assert row["default_profile_id"] == resolved_physics_profile(
            row["source_id"], None), row["source_id"]


def test_a_door_declaring_no_default_still_reaches_the_unnamed_suite():
    """``None`` is an answer, and the chain has a branch that needs it.

    ``DEFAULT_PHYSICS_PROFILE`` may be declared ``None``, meaning the
    unshipped product default suite rather than any registered profile.
    The first draft of the derivation lost that seam by always returning
    a shipped id, which silently closed ``gpuwm go``'s unnamed-suite
    branch (``tests/test_go_chain.py`` caught it).
    """

    from gpuwm import domain_wizard
    from gpuwm.physics_menu import default_profile_for

    bound = domain_wizard.DEFAULT_PHYSICS_PROFILE
    domain_wizard.DEFAULT_PHYSICS_PROFILE = None
    try:
        assert default_profile_for("gfs") is None
        assert domain_wizard.resolved_physics_profile("gfs", None) is None
    finally:
        domain_wizard.DEFAULT_PHYSICS_PROFILE = bound


def test_the_native_hrrr_routes_default_is_still_the_routes_own_constant(
        capsys):
    """The derivation must reproduce the constant it replaces.

    ``ROUTE_DEFAULT_PHYSICS_PROFILE`` is what the native HRRR preparer
    binds.  Deriving the wizard's default instead of tabling it is only
    correct if the derived answer IS that constant -- so this cell binds
    the two rather than restating either.
    """

    from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE

    document, _raw = _menu_document(capsys)
    row, _cell = _cells(document, "hrrr")
    assert row["default_profile_id"] == ROUTE_DEFAULT_PHYSICS_PROFILE


# ---------------------------------------------------------------------------
# THE ARBITRARY ACCEPTANCE TEST: a new source row AND a new profile
# ---------------------------------------------------------------------------


def _graft_profile(monkeypatch, profile_id, switches, maturity):
    """Register one more shipped suite, by table row alone."""

    from types import MappingProxyType

    from gpuwm import (domain_wizard, physics_compat, physics_menu,
                       physics_registry)

    monkeypatch.setattr(
        physics_compat, "_SINGLE_DOMAIN_RUNTIME_SWITCHES",
        MappingProxyType({
            **physics_compat._SINGLE_DOMAIN_RUNTIME_SWITCHES,
            profile_id: MappingProxyType(dict(switches))}))
    grafted = (*physics_menu.WIZARD_PHYSICS_PROFILES, profile_id)
    # The owner (gpuwm.physics_menu) and the wizard's re-export of it: a
    # suite registered tomorrow is registered in the table, and every
    # reader of that table sees it.
    monkeypatch.setattr(physics_menu, "WIZARD_PHYSICS_PROFILES", grafted)
    monkeypatch.setattr(domain_wizard, "WIZARD_PHYSICS_PROFILES", grafted)
    registry = physics_registry.physics_registry()
    registry["templates"][profile_id] = {
        "maturity": maturity,
        "components": dict(
            registry["templates"][domain_wizard.WSM6_PROFILE_ID][
                "components"]),
    }
    monkeypatch.setattr(physics_registry, "_REGISTRY", registry)


def test_a_grafted_source_row_and_a_grafted_profile_both_appear(
        capsys, monkeypatch):
    """THE arbitrary acceptance test, both axes at once.

    A model registered tomorrow and a suite registered tomorrow -- two
    table rows, no code edited anywhere -- come out of the real
    ``--physics-profiles`` door with the right admissibility, the right
    nocturnal class, the registry's maturity, and a default that works
    on them.
    """

    from gpuwm import source_adapters as registry
    from gpuwm.domain_wizard import WSM6_PROFILE_ID
    from gpuwm.physics_compat import single_domain_runtime_switches

    grafted_row = dataclasses.replace(
        registry.source_adapters()[0], source_id="probe-arbitrary-model",
        display_name="Probe Arbitrary Model 9000", aliases=(),
        credentials=())
    monkeypatch.setattr(
        registry, "_ADAPTERS", (*registry.source_adapters(), grafted_row))

    # A daytime suite that ALSO selects Kain-Fritsch: inadmissible on the
    # native HRRR route, admissible everywhere the route gate is silent.
    switches = dict(single_domain_runtime_switches(WSM6_PROFILE_ID))
    switches.update(cu_physics=1, cudt_minutes=5.0)
    _graft_profile(monkeypatch, "probe-arbitrary-suite-v1", switches,
                   "implemented-unverified")

    document, _raw = _menu_document(capsys)

    entry = document["profiles"][-1]
    assert entry["profile_id"] == "probe-arbitrary-suite-v1"
    assert entry["day_only"] is True
    assert entry["maturity"]["registry_maturity"] == "implemented-unverified"
    assert entry["cumulus_scheme_id"] == 1
    assert entry["vertical_levels"]["minimum"] is not None

    row = document["sources"][-1]
    assert row["source_id"] == "probe-arbitrary-model"
    assert row["display_name"] == "Probe Arbitrary Model 9000"
    cells = {cell["profile_id"]: cell for cell in row["profiles"]}
    grafted_cell = cells["probe-arbitrary-suite-v1"]
    assert grafted_cell["admissible"] is True
    assert grafted_cell["why_not"] is None
    assert grafted_cell["day_only"] is True
    assert grafted_cell["is_default"] is False
    # The new suite is offered on the new model AND refused by name on
    # the route whose grid resolves its own convection -- with no branch
    # anywhere that knows either id.
    _hrrr_row, hrrr = _cells(document, "hrrr")
    assert hrrr["probe-arbitrary-suite-v1"]["admissible"] is False
    assert "cu_physics=1 (the route requires 0)" in hrrr[
        "probe-arbitrary-suite-v1"]["why_not"]
    # And the new model still gets a working, nocturnally valid default.
    assert cells[row["default_profile_id"]]["admissible"] is True
    assert cells[row["default_profile_id"]]["day_only"] is False


def test_a_source_whose_route_refuses_the_listed_head_still_gets_a_default(
        capsys, monkeypatch):
    """The default is DERIVED, so a new gate moves it with no code edit.

    Grafting a route gate onto a source that never had one is the
    arbitrary test applied to defaults: a hand-tabled default would
    stay pointing at a suite the new gate refuses.
    """

    from gpuwm import domain_wizard, physics_menu
    from gpuwm.hrrr_route_inputs import route_physics_blocker

    grafted = {**physics_menu._route_emission_physics_gates(),
               "era5": route_physics_blocker}
    monkeypatch.setattr(physics_menu, "_route_emission_physics_gates",
                        lambda: grafted)

    document, _raw = _menu_document(capsys)
    row, cells = _cells(document, "era5")
    assert cells[domain_wizard.DEFAULT_PHYSICS_PROFILE]["admissible"] is False
    assert cells[row["default_profile_id"]]["admissible"] is True
    assert cells[row["default_profile_id"]]["day_only"] is False
    assert domain_wizard.resolved_physics_profile("era5", None) == row[
        "default_profile_id"]


# ---------------------------------------------------------------------------
# The refusal remedies read the same table (#233)
# ---------------------------------------------------------------------------


def test_the_menu_carries_each_sources_nocturnal_remedy(capsys):
    """The remedy is a menu cell, not a sentence assembled at each door."""

    document, _raw = _menu_document(capsys)
    for row in document["sources"]:
        remedy = row["nocturnal_remedy"]
        cells = {cell["profile_id"]: cell for cell in row["profiles"]}
        assert remedy["profile_id"] in cells, row["source_id"]
        assert cells[remedy["profile_id"]]["admissible"] is True
        assert cells[remedy["profile_id"]]["day_only"] is False
        assert remedy["instruction"]
        # The scheme is NAMED, never left as a bare selector: a pilot
        # report read `ra_physics: 0` as "radiation off" on a suite
        # running RTE+RRTMGP on both streams.
        assert "longwave" in remedy["instruction"]
        assert "longwave 0" not in remedy["instruction"]
        if remedy["profile_id"] == row["default_profile_id"]:
            assert "omit --physics-profile" in remedy["instruction"]
        else:
            assert f"--physics-profile {remedy['profile_id']}" in remedy[
                "instruction"]


def _wizard_refusal(source, profile):
    """The REAL wizard door's refusal text for one pairing."""

    from gpuwm.cli import build_parser, main

    argv = ["domain", "--source", source, "--cycle", "2024-05-06T18",
            "--point", "35.4,-97.5", "--hours", "12",
            "--physics-profile", profile, "--out", "unused.toml"]
    parser = build_parser()
    args = parser.parse_args(argv)
    with pytest.raises(ValueError) as refusal:
        args.func(args)
    del main
    return str(refusal.value)


def test_the_nocturnal_remedy_names_a_suite_the_active_source_admits():
    """THE defect Drew hit, closed at the door that printed it.

    Old behaviour: the remedy named the gfs/era5 default on every
    source, and ``--source hrrr`` then refused that suite for
    ``cu_physics = 1``.  Two refusals, no way forward.
    """

    from gpuwm.domain_wizard import (
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK, WSM6_PROFILE_ID,
        profile_route_blocker, resolved_physics_profile)

    text = _wizard_refusal("hrrr", WSM6_PROFILE_ID)
    assert "includes local night" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in text
    named = [profile for profile in
             __import__("gpuwm.domain_wizard", fromlist=["x"]
                        ).WIZARD_PHYSICS_PROFILES if profile in text]
    assert named, "the remedy must still name a suite"
    for profile in named:
        if profile == WSM6_PROFILE_ID:
            continue  # the refused suite, named as the cause
        assert profile_route_blocker(profile, "hrrr") is None, profile
    # And for this source the remedy IS the source's own default, so the
    # shortest way forward is to stop passing the flag.
    assert "omit --physics-profile" in text
    assert resolved_physics_profile("hrrr", None) in text


def test_following_the_nocturnal_remedy_emits_a_config(tmp_path):
    """The proof that the remedy is a way forward and not a next refusal."""

    from gpuwm.cli import build_parser
    from gpuwm.domain_wizard import WSM6_PROFILE_ID

    text = _wizard_refusal("hrrr", WSM6_PROFILE_ID)
    assert "omit --physics-profile" in text
    out = tmp_path / "emitted.toml"
    args = build_parser().parse_args(
        ["domain", "--source", "hrrr", "--cycle", "2024-05-06T18",
         "--point", "35.4,-97.5", "--hours", "12", "--out", str(out)])
    assert args.func(args) == 0
    assert out.exists()


def test_the_load_time_nocturnal_refusal_names_a_universally_admissible_suite():
    """The config-load guard has no source, so its example must fit all.

    ``build_experiment`` refuses a loaded config that has no forcing
    source on it, so the remedy cannot be route-aware -- it can only be
    route-SAFE, which means naming a suite no registered source's route
    refuses.
    """

    from gpuwm.domain_wizard import profile_route_blocker
    from gpuwm.physics_menu import universally_admissible_profile
    from gpuwm.source_adapters import source_adapters

    profile = universally_admissible_profile()
    assert profile is not None
    for adapter in source_adapters():
        assert profile_route_blocker(profile, adapter.source_id) is None, (
            adapter.source_id)
