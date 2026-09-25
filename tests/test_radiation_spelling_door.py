"""One radiation choice, two spellings, one answer at every door.

``ra_physics = N`` means "engine N on both streams" and leaves
``ra_lw_physics``/``ra_sw_physics`` at -1; the split pair states the two
streams and leaves ``ra_physics`` at 0.  ``gpuwm.config
.radiation_scheme_ids`` has admitted both since it was written, and the
WRF namelist importer emits the aggregate for a coupled pair.

The capability door did not: it matched a component option on the RAW
selector keys, and one registry option was keyed on ``(-1, -1)`` -- the
sentinel for "the split pair is not stated here" -- so the aggregate
spelling of ANY radiation choice landed on that one option.  An imported
4/4 configuration therefore matched no option a shipped template
declares and could never equal its own profile, and an imported
radiation-off configuration was refused for not setting
``ra_physics = 4``.  Both run exactly what the split spelling runs.

The shortwave-only refusal is measured here too, beside them, because it
is the OTHER thing a radiation selector pair can be refused for and the
two were reported together: it stands, and this pins the sentence that
makes it stand.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpuwm.config import radiation_scheme_ids
from gpuwm.physics_compat import (PhysicsCapabilityError,
                                  constant_longwave_refusal,
                                  single_domain_runtime_switches,
                                  validate_physics_capabilities,
                                  validate_single_domain_physics_profile)
from gpuwm.physics_registry import physics_registry


def _aggregate(switches: dict, engine: int) -> dict:
    """``switches`` respelled the way a coupled WRF namelist imports."""

    spelled = dict(switches)
    spelled["ra_physics"] = engine
    spelled["ra_lw_physics"] = -1
    spelled["ra_sw_physics"] = -1
    return spelled


def _coupled_profiles() -> list[str]:
    """Every shipped profile whose radiation is the coupled 4/4 pair.

    Read off the registry rather than listed, so a template registered
    tomorrow is measured the day it lands; restricted to the templates
    that carry a single-domain runtime-switch row, which is what this
    door takes.
    """

    from gpuwm.physics_compat import _SINGLE_DOMAIN_RUNTIME_SWITCHES

    registry = physics_registry()
    return sorted(
        template_id
        for template_id, template in registry["templates"].items()
        if template.get("components", {}).get("radiation") == "rte-rrtmgp"
        and template_id in _SINGLE_DOMAIN_RUNTIME_SWITCHES)


def test_no_option_is_keyed_on_the_not_stated_here_sentinel():
    """The registry row that made the two spellings two selections.

    -1 is not a WRF scheme id.  A selector tuple of two of them is an
    ABSENCE, and an option keyed on an absence is selected by every
    configuration that spelled its choice some other way.
    """

    options = physics_registry()["components"]["radiation"]["options"]
    sentinel = {
        option_id: option["selectors"]
        for option_id, option in options.items()
        if any(int(value) < 0 for value in option["selectors"].values())
    }
    assert sentinel == {}


@pytest.mark.parametrize("profile", _coupled_profiles())
def test_a_coupled_profile_is_matched_by_either_spelling(profile):
    """The blocker: the aggregate spelling could not equal its profile."""

    switches = dict(single_domain_runtime_switches(profile))
    assert (switches["ra_lw_physics"], switches["ra_sw_physics"]) == (4, 4)
    assert switches["ra_physics"] == 0

    spelled = _aggregate(switches, 4)
    assert radiation_scheme_ids(SimpleNamespace(**spelled)) == (4, 4)
    assert (validate_physics_capabilities(spelled)
            == validate_physics_capabilities(switches))
    # The whole door, not only the component resolution: the settings
    # comparison beside it read the three keys literally too.
    validate_single_domain_physics_profile(profile, config=spelled)


def test_the_aggregate_spelling_of_radiation_off_resolves_to_off():
    """The second face of the same defect.

    A WRF namelist with radiation off imports as ``ra_physics = 0`` with
    the split pair unstated, which is the same sentinel tuple.  It was
    refused with a sentence about ``ra_physics = 4`` -- the requirement
    of an option it had not asked for.
    """

    switches = dict(single_domain_runtime_switches(
        "wsm6-ysu-mm5-noah-no-radiation-v1"))
    switches["ra_lw_physics"] = 0
    switches["ra_sw_physics"] = 0
    resolved = validate_physics_capabilities(_aggregate(switches, 0))
    assert resolved["radiation"] == "off"


def test_a_contradiction_between_the_two_spellings_is_still_refused():
    """What the door must keep refusing, and in whose words.

    The aggregate naming one engine while the split pair names another
    is not a spelling, it is two selections; the resolver refuses it and
    names both.
    """

    switches = dict(single_domain_runtime_switches(
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"))
    switches["ra_physics"] = 4
    switches["ra_lw_physics"] = 0
    switches["ra_sw_physics"] = 1
    with pytest.raises(ValueError, match="contradict"):
        validate_physics_capabilities(switches)


def test_naming_one_half_of_the_split_pair_keeps_its_own_refusal():
    """Collapsing the spellings may not swallow the together-keys refusal."""

    switches = dict(single_domain_runtime_switches(
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"))
    del switches["ra_sw_physics"]
    with pytest.raises(PhysicsCapabilityError, match="TOGETHER"):
        validate_physics_capabilities(switches)


def test_a_caller_that_names_no_radiation_selector_still_selects_none():
    """Absence is not a value, and the collapse may not invent one."""

    resolved, _options = _resolved_without_radiation()
    assert "radiation" not in resolved


def _resolved_without_radiation():
    from gpuwm.physics_compat import _resolve_physics_component_options

    switches = dict(single_domain_runtime_switches(
        "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"))
    for key in ("ra_physics", "ra_lw_physics", "ra_sw_physics"):
        switches.pop(key, None)
    return _resolve_physics_component_options(switches)


def test_an_imported_wrf_namelist_selecting_rrtmgp_matches_its_profile(
        tmp_path):
    """The reported route, end to end through the importer it came from."""

    from gpuwm.experiment import load_experiment
    from gpuwm.namelist_import import import_namelists
    try:
        from tests.test_namelist_import import INPUT_TEXT, WPS_TEXT
    except ModuleNotFoundError:
        from test_namelist_import import INPUT_TEXT, WPS_TEXT

    wps = tmp_path / "namelist.wps"
    inp = tmp_path / "namelist.input"
    wps.write_text(WPS_TEXT)
    inp.write_text(INPUT_TEXT)
    toml_text, _report = import_namelists(wps, inp, name="coupled_pair")
    # The aggregate spelling is what this importer emits for a coupled
    # pair, and the point of the test is that it no longer has to change.
    assert "ra_physics = 4" in toml_text
    written = tmp_path / "coupled_pair.toml"
    written.write_text(toml_text)
    run = load_experiment(written).root.run
    assert (run.ra_physics, run.ra_lw_physics, run.ra_sw_physics) == (4, -1, -1)
    assert radiation_scheme_ids(run) == (4, 4)
    assert validate_physics_capabilities(run)["radiation"] == "rte-rrtmgp"


def test_the_shortwave_only_refusal_names_its_consumer():
    """NOT a defect, and this is the sentence that keeps it from becoming one.

    A shortwave-only pair under a land-surface scheme has no producer
    for the downward longwave, so the surface energy budget integrates a
    fabricated constant for the whole forecast.  WRF v4.6.1 does not
    offer the pairing at all with shortwave on -- its ``lwrad_select``
    has no ``lw = 0`` case and calls ``wrf_error_fatal``.  The refusal
    therefore stands, and it has to keep NAMING the consumer and the
    number, because a refusal that names neither is the one that gets
    demoted to a warning.
    """

    switches = single_domain_runtime_switches(
        "wsm6-ysu-mm5-noah-no-radiation-v1")
    assert (switches["ra_lw_physics"], switches["ra_sw_physics"]) == (0, 1)
    message = constant_longwave_refusal(
        [SimpleNamespace(grid_id=1, **switches)])
    assert message is not None
    assert "Noah LSM" in message
    assert f"sf_surface_physics {switches['sf_surface_physics']}" in message
    assert "300 W m-2" in message
    # And it is lifted by the declaration it names, not by silence.
    from gpuwm.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in message
    assert constant_longwave_refusal(
        [SimpleNamespace(grid_id=1, **switches)],
        acknowledgements=(CONSTANT_DOWNWARD_LONGWAVE_ACK,)) is None
