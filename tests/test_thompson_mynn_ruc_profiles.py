"""Thompson with MYNN surface layer and PBL over RUC, as named suites.

The MYNN + RUC pair shipped only with WSM6, so the operational class
that puts Thompson microphysics on that surface and boundary layer had
no name, and no menu, ``--physics-profile`` list or preset could offer
it.  Two rows close it, each its WSM6 row with the microphysics moved.
This file pins what they are for:

* each composes: the registry resolves it, its runtime product differs
  from its WSM6 row in the microphysics alone, and the run door accepts it;
* each is declared on exactly the routes and sources its WSM6 row is,
  immediately after it, because the RUC soil ingest is what limits both;
* the catalog, its preset and the wizard menu list it, and the engine's
  fit check sizes a plan that names it;
* the radiation-bearing row loads over a night window with no
  declaration, while the Dudhia row is refused on the SAME window.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.domain_wizard import experiment_from_text, render_config
from gpuwm.experiment import build_experiment
from gpuwm.physics_compat import (
    ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    first_local_night_time,
    identify_single_domain_profile,
    single_domain_runtime_switches,
)
from gpuwm.physics_registry import physics_registry

#: (Thompson row, the WSM6 row it moves the microphysics of).
PAIRS = (
    (THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, MYNN_RUC_RTE_RRTMGP_PROFILE_ID),
    (THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID, MYNN_RUC_PROFILE_ID),
)
NEW = tuple(new for new, _base in PAIRS)

#: The same reference geometry and window tests/test_mynn_radiation_profiles.py
#: measures the MYNN family on: local night falls inside the window.
_PROJECTION = {
    "map_proj": "lambert", "ref_lat": 33.8, "ref_lon": -87.29,
    "truelat1": 23.8, "truelat2": 43.8, "stand_lon": -87.29,
}
_NIGHT_START = datetime(2011, 4, 26, 12)
_NIGHT_HOURS = 48


#: A window that stays in daylight at the same point: 15Z to 18Z is
#: 10:00 to 13:00 local, where a Dudhia suite loads undeclared.
_DAY_START = datetime(2011, 4, 26, 15)
_DAY_HOURS = 3


def _emitted(profile, *, start=_NIGHT_START, hours=_NIGHT_HOURS):
    return render_config(
        name="thompsonmynnruc", start_time=start, hours=hours,
        projection=dict(_PROJECTION), dims=[(120, 100)], ratios=(),
        fetch_hints={"source": "era5"}, case_data=None, profile=profile)


def _raw(text):
    import tomllib

    raw = tomllib.loads(text)
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


# ---------------------------------------------------------------------------
# The suite composes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_suite_is_its_wsm6_row_with_thompson_in_it(profile, base):
    registry = physics_registry()
    template = registry["templates"][profile]
    base_template = registry["templates"][base]
    assert template["components"] == {
        **base_template["components"], "microphysics": "thompson-mp8"}
    assert template["components"]["pbl"] == "mynn"
    assert template["components"]["surface_layer"] == "mynn"
    assert template["components"]["land_surface"] == "ruc-lsm"
    # The maturity the siblings carry, derived rather than asserted.
    assert template["maturity"] == base_template["maturity"] \
        == "implemented-unverified"
    assert template["parameters"] == base_template["parameters"]
    assert template["warnings"]


@pytest.mark.parametrize("profile, base", PAIRS)
def test_no_warning_names_a_wsm6_row_but_the_one_pairing_sentence(
        profile, base):
    """A warning copied from the WSM6 base named that base's own partner as
    this row's only difference; the only sentence that may name a WSM6 row
    is the one saying which row this suite differs from."""

    warnings = physics_registry()["templates"][profile]["warnings"]
    naming = [warning for warning in warnings if "wsm6-" in warning]
    assert naming == [
        warning for warning in warnings
        if warning.startswith("This template differs from " + base)]


@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_runtime_product_moves_the_microphysics_and_nothing_else(
        profile, base):
    """A paired run against the WSM6 row isolates the microphysics.

    ``moist_cq`` moves with it because it is the Thompson option's own
    required setting, the value the Thompson validation suite carries too.
    """

    new = single_domain_runtime_switches(profile)
    old = single_domain_runtime_switches(base)
    moved = {name for name in set(new) | set(old)
             if new.get(name) != old.get(name)}
    assert moved <= {"mp_physics", "moist_cq"}
    assert (old["mp_physics"], new["mp_physics"]) == (6, 8)
    assert new["moist_cq"] == single_domain_runtime_switches(
        THOMPSON_PROFILE_ID)["moist_cq"]
    assert (int(new["bl_pbl_physics"]), int(new["sf_sfclay_physics"]),
            int(new["sf_surface_physics"]), int(new["num_soil_layers"])) == (
                5, 5, 3, 9)


@pytest.mark.parametrize("profile", NEW)
def test_the_run_door_accepts_the_suite(profile):
    switches = single_domain_runtime_switches(profile)
    cfg = RunConfig(nx=41, ny=41, nz=40, dx=1000.0, dy=1000.0,
                    ztop=20000.0, dt=5.0, run_seconds=60.0,
                    time_step_sound=4, **switches)
    validate_run_config(cfg)
    assert identify_single_domain_profile(cfg) == profile


# ---------------------------------------------------------------------------
# Declared where the WSM6 row is, and only there.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("profile, base", PAIRS)
def test_the_suite_follows_its_wsm6_row_on_every_route_and_source(
        profile, base):
    routes = physics_registry()["runner_routes"]
    declared_somewhere = False
    for route_id, route in routes.items():
        for group in ("source_template_ids", "expert_template_ids"):
            for source_id, declared in (route.get(group) or {}).items():
                assert (profile in declared) == (base in declared), (
                    route_id, group, source_id)
                if profile in declared:
                    declared_somewhere = True
                    assert declared.index(profile) == \
                        declared.index(base) + 1, (route_id, source_id)
    assert declared_somewhere
    assert profile in SINGLE_DOMAIN_PHYSICS_PROFILES


# ---------------------------------------------------------------------------
# The catalog, its preset and the wizard offer it.
# ---------------------------------------------------------------------------

def test_the_wizard_menu_offers_both_and_ranks_the_night_valid_one_first():
    from gpuwm.domain_wizard import WIZARD_PHYSICS_PROFILES

    menu = list(WIZARD_PHYSICS_PROFILES)
    assert set(NEW) <= set(menu)
    # The radiation-bearing row sits in the nocturnally valid block: ahead
    # of every Dudhia suite, beside its WSM6 sibling.
    assert menu.index(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID) == \
        menu.index(MYNN_RUC_RTE_RRTMGP_PROFILE_ID) + 1
    assert menu.index(THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID) == \
        menu.index(MYNN_RUC_PROFILE_ID) + 1
    assert menu.index(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID) < \
        menu.index(THOMPSON_PROFILE_ID)


@pytest.mark.parametrize("source", ("hrrr", "era5"))
def test_the_catalog_lists_the_suite_and_the_preset_runs(source):
    from gpuwm import physics_catalog as pc

    document = pc.catalog(source=source)
    offered = {row["id"] for row in document["suites"]
               if row["on_create_page"]}
    assert set(NEW) <= offered
    preset = pc.preset("coastal-fog-stratus")
    assert preset["suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID
    verdict = pc.check({"preset": "coastal-fog-stratus", "source": source})
    assert verdict["valid"], verdict.get("refusal")
    assert verdict["named_suite"] == THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID


@pytest.mark.parametrize("profile", NEW)
def test_the_fit_check_sizes_a_plan_that_names_the_suite(tmp_path, profile):
    """The engine's fit check is ``run-plan --resolve`` on the plan intent.

    The same resolve the Create page's Fit runs: the wizard emits the
    config for the named suite, the loader accepts it, and the resolved
    domain carries the suite's switches.  The window is 12Z to 18Z at a
    central-US point, all daylight, so the Dudhia row needs no declaration.
    """

    import json

    from gpuwm.runplan import PLAN_SCHEMA, load_plan, resolve_plan

    intent = {"point": "35.2,-97.4", "source": "hrrr", "root_dx_km": 3,
              "cycle": "2024-05-03T12", "hours": 6, "vram_gib": 24,
              "physics_profile": profile}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "thompson-mynn-ruc-plan",
        "route": "prepared", "config": {"intent": intent},
        "output_root": str(tmp_path / "run")}), encoding="utf-8")
    _resolution, exp, _data = resolve_plan(load_plan(path),
                                           require_inputs=False)
    run = exp.domains[0].run
    assert (run.mp_physics, run.bl_pbl_physics, run.sf_sfclay_physics,
            run.sf_surface_physics, run.num_soil_layers) == (8, 5, 5, 3, 9)


# ---------------------------------------------------------------------------
# Night: the radiation-bearing row loads, the Dudhia row is refused.
# ---------------------------------------------------------------------------

def test_the_window_contains_local_night():
    night = first_local_night_time(
        _NIGHT_START, _NIGHT_HOURS * 3600.0,
        ref_lat=_PROJECTION["ref_lat"], ref_lon=_PROJECTION["ref_lon"])
    assert night is not None


def test_the_radiation_bearing_row_loads_over_night_undeclared():
    raw = _raw(_emitted(THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID))
    raw["experiment"].pop("acknowledgements", None)
    experiment = build_experiment(raw, source="<thompson-mynn-ruc-night>")
    assert experiment.acknowledgements == ()
    run = experiment.root.run
    assert run.mp_physics == 8
    assert (run.ra_lw_physics, run.ra_sw_physics) == (4, 4)


def test_the_dudhia_row_is_refused_over_the_same_night_undeclared():
    text = _emitted(THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID)
    raw = _raw(text)
    raw["experiment"].pop("acknowledgements", None)
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<thompson-mynn-ruc-dudhia-night>")
    message = str(caught.value)
    assert "local night" in message
    assert "ra_lw_physics 0" in message
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in message


@pytest.mark.parametrize("profile", NEW)
def test_the_emitted_config_loads_through_the_shared_front_door(profile):
    """Over daylight, where both rows are valid; night is pinned above."""

    night = first_local_night_time(
        _DAY_START, _DAY_HOURS * 3600.0,
        ref_lat=_PROJECTION["ref_lat"], ref_lon=_PROJECTION["ref_lon"])
    assert night is None
    experiment = experiment_from_text(
        _emitted(profile, start=_DAY_START, hours=_DAY_HOURS),
        source="<thompson-mynn-ruc>")
    run = experiment.root.run
    expected = single_domain_runtime_switches(profile)
    assert {name: getattr(run, name) for name in expected} == expected
    assert identify_single_domain_profile(run) == profile
