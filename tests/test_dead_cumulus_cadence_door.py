"""One cumulus-off suite, two spellings of a switch it never reads.

``cudt_minutes`` is the interval between cumulus calls.  At
``cu_physics = 0`` there are none: ``gpuwm/core/clock.py`` builds a
cumulus calendar only for cu_physics in (1, 3, 16), and no cumulus step
is taken without one.  The value is dead.

Its two producers spell the dead value differently.  The WRF namelist
importer omits the key for a cumulus-off suite on purpose, so the
configuration inherits ``RunConfig``'s live 5.0; every shipped
cumulus-off profile states 0.0.  The capability door compared them raw
and refused the difference, which made every namelist-routed
preparation of a cumulus-off profile impossible -- including the
nowcast front door's own default profile on its HRRR route, the route
that reaches the door through a namelist.

``gpuwm/ingest/prepared_cache.py`` already pins the same key for the
same reason when it compares two prepared identities.  These cases hold
that rule at the capability door, and hold the door's answer for a
configuration that DOES run a cumulus scheme, where the interval is
read on every call it fires.

The prepared route asks the same question a second time, of the loaded
experiment, and refused the same two spellings there -- printing the
resolved radiation pair EQUAL in the sentence that refused the run.
The last case here holds that door to the same reconciliation, so the
two cannot answer differently.
"""

from __future__ import annotations

import pytest

from gpuwm.physics_compat import (single_domain_runtime_switches,
                                  validate_single_domain_physics_profile)

#: A cumulus-off suite: Thompson, YSU, Noah, legacy RRTMG on both
#: streams, and NO cumulus parameterization.  It is the spelling the
#: nowcast door defaulted to through 2.7.5; the door's default is now
#: the same composition on RTE+RRTMGP, and this fixture deliberately
#: stays on the legacy engine so the dead-cadence rule keeps being read
#: on both radiation arms rather than on whichever one is current.
CUMULUS_OFF_PROFILE = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"

#: A profile that runs Kain-Fritsch, where the cadence is live.
CUMULUS_ON_PROFILE = "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"


def test_the_profile_states_the_dead_value_and_the_importer_omits_the_key():
    """The precondition, read rather than assumed."""

    off = single_domain_runtime_switches(CUMULUS_OFF_PROFILE)
    assert off["cu_physics"] == 0
    assert off["cudt_minutes"] == 0.0

    from gpuwm.config import RunConfig
    assert RunConfig.__dataclass_fields__["cudt_minutes"].default == 5.0

    on = single_domain_runtime_switches(CUMULUS_ON_PROFILE)
    assert on["cu_physics"] == 1
    assert on["cudt_minutes"] == 5.0


def test_a_cumulus_off_profile_matches_the_cadence_the_importer_leaves_live():
    """The namelist route's configuration IS the profile's run.

    Before this, the door raised ``settings={'cudt_minutes':
    {'selected': 5.0, 'expected': 0.0}}`` and no HRRR preparation of
    this profile could complete.
    """

    switches = dict(single_domain_runtime_switches(CUMULUS_OFF_PROFILE))
    switches["cudt_minutes"] = 5.0
    validate_single_domain_physics_profile(
        CUMULUS_OFF_PROFILE, config=switches)


def test_any_cumulus_off_interval_is_the_same_dead_switch():
    """Not a 5.0 special case: no cumulus call reads any of them."""

    switches = dict(single_domain_runtime_switches(CUMULUS_OFF_PROFILE))
    for cadence in (5.0, 1.0, 30.0):
        switches["cudt_minutes"] = cadence
        validate_single_domain_physics_profile(
            CUMULUS_OFF_PROFILE, config=switches)


def test_a_running_cumulus_scheme_still_owns_its_cadence():
    """The pin is scoped to the dead case, and the refusal stands.

    With Kain-Fritsch selected the interval is read on every call it
    fires, so a configuration that departs from the profile's cadence
    has asked for different physics and is still refused, naming the
    key and both values.
    """

    switches = dict(single_domain_runtime_switches(CUMULUS_ON_PROFILE))
    switches["cudt_minutes"] = 30.0
    with pytest.raises(ValueError) as refusal:
        validate_single_domain_physics_profile(
            CUMULUS_ON_PROFILE, config=switches)
    sentence = str(refusal.value)
    assert "cudt_minutes" in sentence
    assert "30.0" in sentence and "5.0" in sentence


def test_a_cumulus_off_profile_still_refuses_a_real_difference():
    """The pin moved one dead switch, not the comparison."""

    switches = dict(single_domain_runtime_switches(CUMULUS_OFF_PROFILE))
    switches["cudt_minutes"] = 5.0
    switches["bl_pbl_physics"] = 5
    with pytest.raises(ValueError) as refusal:
        validate_single_domain_physics_profile(
            CUMULUS_OFF_PROFILE, config=switches)
    assert "cudt_minutes" not in str(refusal.value)


def test_the_prepared_route_reads_the_same_two_spellings_as_the_door(
        tmp_path, capsys):
    """The route's own comparison, on the namelist spelling.

    ``_validate_profile_switches`` compared the loaded experiment's
    switches against the profile key by key.  A configuration that
    reached this profile through a WRF namelist carries the aggregate
    radiation selector and the live cumulus interval, so the comparison
    refused it -- in a sentence that printed ``radiation=(4, 4)`` and
    ``resolved_radiation=(4, 4)`` beside each other, having already
    resolved the pair it then refused the spelling of.  That was every
    HRRR preparation of this profile through the nowcast front door.
    """

    import dataclasses
    from types import SimpleNamespace

    import tools.prepared_single_domain_forecast as runner
    from gpuwm.cli import main as cli_main
    from gpuwm.experiment import load_experiment

    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--physics-profile", CUMULUS_OFF_PROFILE,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    assert rc == 0, capsys.readouterr().out

    exp = load_experiment(out)
    # As it is written, from the profile's own spelling.
    runner._validate_profile_switches(
        exp, source="gfs", profile=CUMULUS_OFF_PROFILE)

    # And respelled the way the WRF namelist importer emits it.  The
    # validator reads exp.root, exp.domains and each domain's run and
    # grid_id, so a stand-in carrying the respelled run is the whole of
    # what it asks about.
    root = exp.root
    respelled = dataclasses.replace(
        root.run, ra_physics=4, ra_lw_physics=-1, ra_sw_physics=-1,
        cudt_minutes=5.0)
    imported = SimpleNamespace(run=respelled, grid_id=root.grid_id)
    exp_imported = SimpleNamespace(root=imported, domains=(imported,))
    validation = runner._validate_profile_switches(
        exp_imported, source="gfs", profile=CUMULUS_OFF_PROFILE)
    assert validation["profile"] == CUMULUS_OFF_PROFILE
    assert validation["validated_domains"][0]["radiation_scheme_ids"] == [4, 4]
def test_both_doors_name_a_departure_in_one_vocabulary(tmp_path, capsys):
    """The two refusals a namelist-routed run can meet, in one wording.

    They are one reconciliation now, and they were still two
    vocabularies for one idea: the capability door printed
    ``settings={name: {'selected': ..., 'expected': ...}}`` and the
    prepared route printed ``differs={name: {'observed': ...,
    'expected': ...}}`` on the same run.  A user who meets both on one
    command reads one word for one thing.
    """

    import dataclasses
    from types import SimpleNamespace

    import tools.prepared_single_domain_forecast as runner
    from gpuwm.cli import main as cli_main
    from gpuwm.experiment import load_experiment

    switches = dict(single_domain_runtime_switches(CUMULUS_ON_PROFILE))
    switches["cudt_minutes"] = 30.0
    with pytest.raises(ValueError) as door:
        validate_single_domain_physics_profile(
            CUMULUS_ON_PROFILE, config=switches)
    at_the_door = str(door.value)

    out = tmp_path / "area.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                   "--ladder", "12", "--source", "gfs",
                   "--physics-profile", CUMULUS_ON_PROFILE,
                   "--cycle", "2026-07-29T18", "--hours", "6",
                   "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    root = load_experiment(out).root
    departed = dataclasses.replace(root.run, cudt_minutes=30.0)
    domain = SimpleNamespace(run=departed, grid_id=root.grid_id)
    with pytest.raises(ValueError) as route:
        runner._validate_profile_switches(
            SimpleNamespace(root=domain, domains=(domain,)),
            source="gfs", profile=CUMULUS_ON_PROFILE)
    on_the_route = str(route.value)

    for sentence in (at_the_door, on_the_route):
        assert "settings={" in sentence
        assert "'selected'" in sentence and "'expected'" in sentence
        assert "differs=" not in sentence
        assert "'observed'" not in sentence
        assert "cudt_minutes" in sentence
