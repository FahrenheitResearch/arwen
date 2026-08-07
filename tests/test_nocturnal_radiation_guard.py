"""The nocturnal-radiation guard and the wizard's nocturnally sane default.

Provenance: a wizard-emitted 48 h real case bound
``thompson-mp8-ysu-mm5-noah-validation-v1`` (ra_lw_physics 0,
ra_sw_physics 1).  Shortwave heated the surface by day; at night the
surface radiated with no downward longwave, skin temperature cratered
and 2 m dewpoints read in the 50s F inside a 70s airmass.  Two fixes,
both bound here:

* the wizard's real-case default is a certified full lw+sw profile
  (registry maturity read off the registry, never asserted);
* any real experiment whose window includes local night refuses to LOAD
  an undeclared shortwave-on/longwave-off pairing, at the one loader
  every front door shares, with the physics, the profile and both
  remedies named.

The instrument is tested in both directions (a night window trips, a
daylight window does not) because a guard that cannot pass is as wrong
as one that cannot fire.
"""
from __future__ import annotations

from datetime import datetime

import tomllib

import pytest

from gpuwm.domain_wizard import (DEFAULT_PHYSICS_PROFILE,
                                 experiment_from_text, render_config,
                                 resolved_physics_profile)
from gpuwm.experiment import build_experiment
from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                  MORRISON_PROFILE_ID, THOMPSON_PROFILE_ID,
                                  WSM6_PROFILE_ID, first_local_night_time,
                                  single_domain_runtime_switches,
                                  solar_elevation_deg)

#: The reporting user's geometry: an Alabama reference point whose April
#: evening includes local night about 12 h into a 12Z-start window.
_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 33.8, "ref_lon": -87.29,
    "truelat1": 23.8, "truelat2": 43.8, "stand_lon": -87.29,
}


def _emitted(profile, *, start=datetime(2011, 4, 26, 12), hours=48):
    return render_config(
        name="guardcase", start_time=start, hours=hours,
        projection=dict(_PROJECTION), dims=[(120, 100)], ratios=(),
        fetch_hints={"source": "era5"}, case_data=None, profile=profile)


def _raw(text):
    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


# ---------------------------------------------------------------------------
# The default: certified, full lw+sw, taken from the registry.
# ---------------------------------------------------------------------------

def test_real_case_default_is_the_certified_full_radiation_profile():
    assert DEFAULT_PHYSICS_PROFILE == MORRISON_PROFILE_ID
    assert resolved_physics_profile("era5", None) == MORRISON_PROFILE_ID
    assert resolved_physics_profile("gfs", None) == MORRISON_PROFILE_ID
    # HRRR keeps its route-constrained default; an explicit request
    # always wins on every source.
    assert resolved_physics_profile("hrrr", None) == WSM6_PROFILE_ID
    assert (resolved_physics_profile("era5", THOMPSON_PROFILE_ID)
            == THOMPSON_PROFILE_ID)


def test_default_profile_is_registry_certified_and_nocturnally_valid():
    """The choice is re-derived from the authorities, not from this test."""
    from gpuwm.physics_registry import physics_registry

    template = physics_registry()["templates"][DEFAULT_PHYSICS_PROFILE]
    assert template["maturity"] == "wrf-matched-run"
    switches = single_domain_runtime_switches(DEFAULT_PHYSICS_PROFILE)
    assert switches["ra_lw_physics"] == 4
    assert switches["ra_sw_physics"] == 4


def test_default_emission_is_nocturnally_valid_and_loads():
    text = _emitted(DEFAULT_PHYSICS_PROFILE)
    assert MORRISON_PROFILE_ID in text
    assert "# NOCTURNALLY VALID" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in text
    exp = experiment_from_text(text, source="<default-emission>")
    assert exp.root.run.ra_lw_physics == 4
    assert exp.root.run.ra_sw_physics == 4


def test_no_wizard_door_defaults_to_an_asymmetric_pairing():
    from gpuwm.domain_interactive import DEFAULT_PHYSICS_PROFILE_BY_SOURCE

    for source in ("gfs", "era5"):
        for profile in (resolved_physics_profile(source, None),
                        DEFAULT_PHYSICS_PROFILE_BY_SOURCE[source]):
            switches = single_domain_runtime_switches(profile)
            assert not (switches["ra_sw_physics"] > 0
                        and switches["ra_lw_physics"] == 0), (
                f"{source} default {profile} is asymmetric")


# ---------------------------------------------------------------------------
# The guard, both directions.
# ---------------------------------------------------------------------------

def test_legacy_asymmetric_night_config_refuses_at_load_naming_everything():
    """The reporting user's shape: no declaration anywhere in the file."""
    raw = _raw(_emitted(THOMPSON_PROFILE_ID))
    raw["experiment"].pop("acknowledgements", None)  # legacy file
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<legacy-night>")
    message = str(caught.value)
    assert "local night" in message
    assert "ra_lw_physics 0" in message
    assert THOMPSON_PROFILE_ID in message           # the profile that did it
    assert "skin temperature" in message            # the physics
    assert MORRISON_PROFILE_ID in message           # remedy 1: sane profile
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in message  # remedy 2: declare


def test_asymmetric_night_with_declared_acknowledgement_loads():
    raw = _raw(_emitted(THOMPSON_PROFILE_ID))
    raw["experiment"]["acknowledgements"] = [
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK]
    exp = build_experiment(raw, source="<declared-night>")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in exp.acknowledgements


def test_asymmetric_daylight_window_loads_without_declaration():
    """Negative control: the guard must be passable where the physics is."""
    raw = _raw(_emitted(THOMPSON_PROFILE_ID,
                        start=datetime(2011, 4, 26, 15), hours=3))
    raw["experiment"].pop("acknowledgements", None)
    exp = build_experiment(raw, source="<daylight>")
    assert exp.acknowledgements == ()


def test_full_radiation_night_window_is_not_guarded():
    raw = _raw(_emitted(MORRISON_PROFILE_ID))
    build_experiment(raw, source="<full-radiation-night>")


def test_idealized_experiment_without_projection_is_untouched():
    """No [projection] means no place and no clock: nothing to guard."""
    from gpuwm.experiment import ExperimentConfig

    assert ExperimentConfig.__dataclass_fields__[
        "projection"].default is None
    # The guard body is scoped on projection presence; a config with no
    # projection table is an idealized one and never reaches it.  Bound
    # structurally because idealized fixtures build their own vertical
    # coordinates and belong to their own suites.


# ---------------------------------------------------------------------------
# The wizard's own emissions of explicitly selected asymmetric profiles.
# ---------------------------------------------------------------------------

def test_wizard_declares_explicit_asymmetric_night_selection_in_ink():
    text = _emitted(THOMPSON_PROFILE_ID)
    assert "# NOT NOCTURNALLY VALID" in text
    assert f'acknowledgements = ["{ASYMMETRIC_RADIATION_NOCTURNAL_ACK}"]' \
        in text
    # And the file it emits still loads through every front door's guard.
    exp = experiment_from_text(text, source="<explicit-night>")
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in exp.acknowledgements


def test_wizard_daylight_asymmetric_emission_warns_without_declaring():
    text = _emitted(THOMPSON_PROFILE_ID,
                    start=datetime(2011, 4, 26, 15), hours=3)
    assert "NOT NOCTURNALLY VALID" in text
    assert "all-daylight" in text
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK not in text
    experiment_from_text(text, source="<explicit-day>")


# ---------------------------------------------------------------------------
# The instrument itself, against known answers in both directions.
# ---------------------------------------------------------------------------

def test_solar_elevation_known_answers():
    # Equinox, subsolar point: sun near the zenith at local noon...
    assert solar_elevation_deg(datetime(2026, 3, 20, 12, 0), 0.0, 0.0) > 80.0
    # ...and far below the horizon at local midnight.
    assert solar_elevation_deg(datetime(2026, 3, 20, 0, 0), 0.0, 0.0) < -80.0
    # June polar day: Svalbard's sun stays up at local midnight.
    assert solar_elevation_deg(datetime(2026, 6, 21, 23, 0),
                               78.0, 15.0) > 0.0


def test_first_local_night_time_both_directions():
    # The reporting user's window: night arrives the first evening.
    night = first_local_night_time(
        datetime(2011, 4, 26, 12), 48 * 3600.0,
        ref_lat=33.8, ref_lon=-87.29)
    assert night is not None
    assert datetime(2011, 4, 26, 22) < night < datetime(2011, 4, 27, 3)
    # A midday 3 h window at the same point has no night in it.
    assert first_local_night_time(
        datetime(2011, 4, 26, 15), 3 * 3600.0,
        ref_lat=33.8, ref_lon=-87.29) is None
