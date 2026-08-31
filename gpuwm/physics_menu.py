"""Which physics suites each source can actually run, COMPUTED.

Owner design, 2026-08-20, verbatim: "why cant every model have multiple
unique working default".  Every registered source gets its own default
and its own set of shipped suites that actually run on it -- and both
are DERIVED from the admissibility rules that already refuse, never
transcribed into a table here.

THE FAILURE THIS CLOSES.  Driving the GUI, a suite that runs shortwave
with longwave OFF met a window with local night in it.  The wizard
refused, correctly, and its remedy named the gfs/era5 default -- a
Kain-Fritsch suite.  ``--source hrrr`` then refused THAT for
``cu_physics = 1``, because the native HRRR route's 3 km grid resolves
its own convection.  Two refusals, no way forward.  A remedy that leads
to the next refusal names nothing, so a remedy has to be picked from the
set the ACTIVE source's route admits, and that set has to exist as one
computed object rather than as a sentence written at each door.

The rules evaluated here are the ones that refuse elsewhere, called
through their own functions:

* :func:`gpuwm.physics_compat.profile_route_blocker` -- the emission
  route's physics gate (``gpuwm.hrrr_route_inputs.REQUIRED_PHYSICS``,
  ``ADMITTED_PBL_PHYSICS``, ``ADMITTED_RADIATION_PAIRS``,
  ``SUPPORTED_MICROPHYSICS``) plus the registry's land-surface offer
  declaration (:func:`gpuwm.physics_compat.land_surface_route_blocker`);
* the nocturnal-validity class, which is the same predicate
  :func:`gpuwm.domain_wizard.declared_nocturnal_night` and
  :func:`gpuwm.physics_compat.nocturnal_radiation_refusal` test:
  shortwave on with longwave off;
* the component-owned vertical level bounds
  :func:`gpuwm.physics_compat.validate_resolved_physics_vertical_levels`
  aggregates -- the gate that used to green-light nz=130 under YSU;
* the physics registry's own template maturity.

NOTHING HERE BRANCHES ON A SOURCE ID OR A PROFILE ID.  Both axes are
read from the doors that own them -- ``source_adapters()`` and
``WIZARD_PHYSICS_PROFILES`` -- so a model registered tomorrow and a
suite registered tomorrow both appear, correctly classified, with no
edit here.  ``tests/test_physics_menu.py`` grafts one of each to prove
it.

Every import below is deliberately inside a function: the wizard
imports this module's callers, and a module-level import would close
that loop.

TWO of them still name :mod:`gpuwm.domain_wizard`, and both are for its
PROSE -- ``physics_summary`` in :func:`profile_facts` and
``_radiation_words`` in :func:`nocturnal_remedy`.  The tables and the
pairing predicate moved to :mod:`gpuwm.physics_compat` on 2026-08-20,
because :func:`gpuwm.physics_compat.nocturnal_radiation_refusal` reaches
:func:`universally_admissible_profile` at the one experiment load every
front door shares, and a preparation-only install has no wizard: while
the predicate lived behind the door, that refusal raised ImportError in
the standalone RW-WPS wheel instead of refusing.  The two prose reaches
are safe there because their only callers are the wizard itself
(``domain_wizard.py``) and ``gpuwm run-plan --physics-profiles``
(``runplan.py``), neither of which that wheel stages.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from gpuwm.physics_compat import (MORRISON_PROFILE_ID, MYNN_PROFILE_ID,
                                  MYNN_RTE_RRTMGP_PROFILE_ID,
                                  MYNN_RUC_PROFILE_ID,
                                  MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  NSSL2_LEGACY_RRTMG_PROFILE_ID,
                                  NSSL2_PROFILE_ID,
                                  P3_LEGACY_RRTMG_PROFILE_ID,
                                  RUC_PROFILE_ID,
                                  THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                  THOMPSON_PROFILE_ID,
                                  THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
                                  WSM6_PROFILE_ID)

#: The level count each component's vertical bound is ASKED at.
#:
#: The bounds themselves are level-count-independent declarations, but
#: the aggregating preflight is a validator -- it takes a grid and says
#: yes or no -- so the menu asks it about a grid every shipped component
#: accepts and reads the bounds off the checks it reports.  40 is the
#: certified Grell-Freitas fixture's level count and sits inside every
#: shipped component's declared range, so no violation is raised and
#: every check is reported.  A component whose floor rose above this
#: would drop out of the reported checks rather than lie, which is why
#: :func:`vertical_levels` reports what it read rather than asserting a
#: count.
_BOUND_PROBE_LEVELS = 40


#: The profiles the prepared single-domain forecast runner accepts, in
#: the order the doors list them.
#:
#: THE ORDER CARRIES MEANING: the nocturnally valid suites come first, so
#: :func:`default_profile_for` taking the head of an admissible set gets
#: both radiation streams without knowing what "nocturnal" means.  The
#: legacy-RRTMG Thompson twins and the P3 composition are the only
#: full-radiation suites the nested HRRR route's physics gate admits --
#: every other 4/4 profile carries ``cu_physics = 1``, which that route
#: refuses because its 3 km grid is convection permitting -- so a list
#: without them could only default that route to a
#: shortwave-on/longwave-off suite.  P3 sits AFTER the Thompson twins
#: deliberately: the head of the HRRR-admissible set is that route's
#: default, and the default stays the Thompson legacy suite (what the
#: operational HRRR itself runs); P3 is chosen, never inherited.
#:
#: IT LIVES HERE, not in the wizard door that used to own it, and the
#: reason is a wheel: :func:`gpuwm.physics_compat.nocturnal_radiation_refusal`
#: reaches :func:`universally_admissible_profile` below at the one
#: experiment load every front door shares, INCLUDING the preparation-only
#: ones.  While the list and the pairing predicate sat in
#: :mod:`gpuwm.domain_wizard`, that made the refusal reach the wizard,
#: which the standalone RW-WPS preprocessing wheel does not stage (it pulls
#: the memory preflight and ``gpuwm.cli``) -- so that wheel raised
#: ImportError where a refusal belonged, and
#: ``tools/build_rw_wps_release.py`` refused to stage rather than ship it.
#: The wizard imports both names from here and re-exports them under the
#: spellings every reader in the tree already uses.
WIZARD_PHYSICS_PROFILES = (
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    P3_LEGACY_RRTMG_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
    MYNN_PROFILE_ID,
    RUC_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
)


def _route_emission_physics_gates() -> dict[str, Any]:
    """Emission-route physics gates, by source id.

    These sources' emission routes run through a runner module carrying
    its OWN physics gate, so the pairing predicate below consults that
    module's spelling of the gate -- never a restatement, which is how
    advice came to filter one hard-coded source while every other
    source's advice went out unchecked.  A TABLE, not a code path: a
    source that gains a routed emission gains its gate at every surface
    reading the predicate -- advice ranking, the pre-emission refusal,
    ``--help``'s caveat -- by adding a row here.

    Built on call, like every other import in this module, so nothing
    here runs at import time.
    """

    from gpuwm.hrrr_route_inputs import route_physics_blocker

    return {"hrrr": route_physics_blocker}


def profile_route_blocker(profile, source) -> str | None:
    """Why ``source`` cannot prepare ``profile``, or ``None``.

    The registry answer plus the emission route's own physics gate,
    resolved through the same calls the front doors make, so a caller
    asking "can this pairing run", a runner asking "may this pairing
    run" and a refusal RANKING alternative suites cannot give different
    answers.
    """

    from gpuwm.physics_compat import (
        land_surface_component_for_selector, land_surface_route_blocker,
        single_domain_runtime_switches,
    )

    if profile is None or source is None:
        return None
    try:
        switches = single_domain_runtime_switches(profile)
    except (KeyError, ValueError):
        return None
    named = land_surface_component_for_selector(
        switches.get("sf_surface_physics"))
    if named is not None:
        blocker = land_surface_route_blocker(named, source=str(source))
        if blocker is not None:
            return blocker
    emission_gate = _route_emission_physics_gates().get(str(source))
    if emission_gate is None:
        return None
    return emission_gate(switches)


def shipped_profiles() -> tuple[str, ...]:
    """The suites the wizard door offers, in the door's own order.

    The order carries meaning: nocturnally valid suites come first, so a
    derivation that takes the head of the admissible set gets a default
    with both radiation streams on without knowing what "nocturnal"
    means.  Read, never re-sorted.
    """

    return tuple(WIZARD_PHYSICS_PROFILES)


def registered_sources() -> tuple[str, ...]:
    """Every registry row's id, in registry order."""

    from gpuwm.source_adapters import source_adapters

    return tuple(adapter.source_id for adapter in source_adapters())


def _switches(profile: str) -> dict[str, Any]:
    from gpuwm.physics_compat import single_domain_runtime_switches

    return dict(single_domain_runtime_switches(profile))


def radiation_scheme_ids(switches: Mapping[str, Any]) -> tuple[int, int]:
    """``(longwave, shortwave)`` selectors, split or legacy spelling.

    The registry writes the split pair and v1.0.0 wrote the combined
    ``ra_physics``; both mean the same thing and a reader that knew only
    one of them would classify half the shipped suites wrong.
    """

    combined = int(switches.get("ra_physics", 0) or 0)
    longwave = switches.get("ra_lw_physics")
    shortwave = switches.get("ra_sw_physics")
    return (combined if longwave is None else int(longwave),
            combined if shortwave is None else int(shortwave))


def day_only(profile: str) -> bool:
    """Whether this suite runs shortwave with longwave OFF.

    THE nocturnal-validity class, spelled as the guards spell it
    (``ra_sw_physics > 0 and ra_lw_physics == 0``).  A suite in this
    class heats the surface by day with no longwave scheme to balance it
    after sunset, so a window containing local night is refused unless
    the operator declares the validation experiment themselves.
    """

    longwave, shortwave = radiation_scheme_ids(_switches(profile))
    return shortwave > 0 and longwave == 0


def day_only_reason(profile: str) -> str | None:
    """Why this suite is daytime-only, in the guards' own terms."""

    longwave, shortwave = radiation_scheme_ids(_switches(profile))
    if not (shortwave > 0 and longwave == 0):
        return None
    return (f"ra_sw_physics {shortwave} with ra_lw_physics {longwave}: "
            "shortwave heats the surface by day while no longwave scheme "
            "runs, so a window that includes local night is refused "
            "unless the validation experiment is declared")


def vertical_levels(profile: str) -> dict[str, Any]:
    """The level-count window every component of this suite accepts.

    Asked of the preflight that refuses at preparation time, so a picker
    cannot offer a suite at a level count the first physics call would
    kill -- the defect that let ``gpuwm check`` pass a 130-level YSU
    configuration.

    ``maximum`` is computed WITHOUT a model-top pressure, because a menu
    has no experiment yet.  The radiation adapters add cap layers above
    the model top, so a real ``p_top`` can only TIGHTEN this ceiling,
    never widen it; the basis sentence says so rather than leaving a
    reader to assume the number is final.
    """

    from gpuwm.physics_compat import (
        validate_resolved_physics_vertical_levels)

    settings = _switches(profile)
    settings["nz"] = _BOUND_PROBE_LEVELS
    checks = validate_resolved_physics_vertical_levels(settings)["checks"]
    bounds = []
    floors: list[int] = []
    ceilings: list[int] = []
    for check in checks:
        # Two shapes come back: a component bound (minimum/maximum on
        # MODEL levels) and a radiation cap (maximum on TOTAL layers,
        # model levels plus the above-model layers it adds).  The second
        # is converted to a model-level ceiling rather than mixed in raw,
        # which would have compared two different quantities.
        above = int(check.get("above_model_layers", 0) or 0)
        minimum = check.get("minimum")
        maximum = check.get("maximum")
        if maximum is not None:
            maximum = int(maximum) - above
            ceilings.append(maximum)
        if minimum is not None:
            minimum = int(minimum)
            floors.append(minimum)
        bounds.append({
            "component": check["component"],
            "minimum": minimum,
            "maximum": maximum,
            "above_model_layers": above,
        })
    return {
        "minimum": max(floors) if floors else None,
        "maximum": min(ceilings) if ceilings else None,
        "bounds": bounds,
        "basis": (
            "the component-owned bounds "
            "gpuwm.physics_compat.validate_resolved_physics_vertical_"
            "levels aggregates, read at nz="
            f"{_BOUND_PROBE_LEVELS} with no model-top pressure; a "
            "declared p_top adds radiation cap layers above the model "
            "top and can only lower this ceiling"),
    }


def maturity(profile: str) -> dict[str, Any]:
    """The physics registry's maturity for this suite, copied.

    A front end colours these words; it does not translate them into a
    second vocabulary.  A suite the registry declares no template for
    reads back ``None`` with the reason stated, rather than being
    silently promoted to the default rung.
    """

    from gpuwm.physics_compat import (
        VERIFICATION_EXPERIMENTAL, VERIFICATION_SUPPORTED,
        VERIFICATION_WRF_VERIFIED, _EXPERIMENTAL_MATURITY,
        _WRF_VERIFIED_MATURITY)
    from gpuwm.physics_registry import physics_registry

    template = physics_registry()["templates"].get(profile)
    declared = (template.get("maturity")
                if isinstance(template, Mapping) else None)
    status = VERIFICATION_SUPPORTED
    if declared == _WRF_VERIFIED_MATURITY:
        status = VERIFICATION_WRF_VERIFIED
    elif declared == _EXPERIMENTAL_MATURITY:
        status = VERIFICATION_EXPERIMENTAL
    return {
        "registry_maturity": declared,
        "verification_status": status,
        "registered_template": template is not None,
    }


def profile_facts(profile: str) -> dict[str, Any]:
    """Everything about one suite that does not depend on a source."""

    from gpuwm.domain_wizard import physics_summary

    switches = _switches(profile)
    longwave, shortwave = radiation_scheme_ids(switches)
    return {
        "profile_id": profile,
        "summary": physics_summary(profile),
        "microphysics_scheme_id": int(switches.get("mp_physics", 0)),
        "cumulus_scheme_id": int(switches.get("cu_physics", 0)),
        "pbl_scheme_id": int(switches.get("bl_pbl_physics", 0)),
        "land_surface_scheme_id": int(switches.get("sf_surface_physics", 0)),
        "longwave_scheme_id": longwave,
        "shortwave_scheme_id": shortwave,
        "day_only": day_only(profile),
        "day_only_reason": day_only_reason(profile),
        "maturity": maturity(profile),
        "vertical_levels": vertical_levels(profile),
        "switches": switches,
    }


def admissible_profiles(source: str) -> tuple[str, ...]:
    """The shipped suites ``source``'s emission route can prepare.

    The predicate is the wizard's own -- one call, so the menu and the
    refusal cannot disagree about a pairing.
    """

    return tuple(profile for profile in shipped_profiles()
                 if profile_route_blocker(profile, source) is None)


def default_profile_for(source: str) -> str | None:
    """The suite a bare run on ``source`` binds, DERIVED.

    The door's declared default (:data:`gpuwm.domain_wizard
    .DEFAULT_PHYSICS_PROFILE`) wins whenever THIS source's route admits
    it and it runs both radiation streams -- an owner directive from
    2026-08-06, after a shipped 48 h case, is that no door defaults to a
    longwave-off suite.  When the route refuses it, the next suite in
    the door's listed order that satisfies both conditions takes over.
    That is the whole derivation, and it reproduces every default this
    product shipped before the derivation existed
    (``tests/test_physics_menu.py`` binds the native HRRR route's answer
    to ``hrrr_route_inputs.ROUTE_DEFAULT_PHYSICS_PROFILE``) while giving
    a source registered tomorrow a working one with no edit here.

    ``None`` is a real answer, not a failure: the door's default may be
    declared ``None``, which means the unshipped product default suite
    (``DEFAULT_SUITE_PHYSICS``) rather than any registered profile.
    That seam is what reaches the chain's unnamed-suite branch, and
    ``tests/test_go_chain.py`` holds it.

    Falls back to the first admissible suite when every admissible suite
    is daytime-only, and to the door's declared default when nothing is
    admissible at all -- the second case is a source whose route refuses
    the entire shipped list, which the emission refusal then names
    switch by switch rather than this function inventing a way around.
    """

    from gpuwm.domain_wizard import DEFAULT_PHYSICS_PROFILE

    declared = DEFAULT_PHYSICS_PROFILE
    if declared is None:
        return None
    admissible = admissible_profiles(source)
    if declared in admissible and not day_only(declared):
        return declared
    for profile in admissible:
        if not day_only(profile):
            return profile
    if admissible:
        return admissible[0]
    return declared


def default_basis(source: str) -> str:
    """Why that suite is this source's default, in one sentence."""

    admissible = admissible_profiles(source)
    nocturnal = [profile for profile in admissible if not day_only(profile)]
    if nocturnal:
        return (f"the first suite in the door's listed order that "
                f"--source {source} can prepare and that runs both "
                f"radiation streams ({len(nocturnal)} of "
                f"{len(shipped_profiles())} qualify)")
    if admissible:
        return (f"the first suite --source {source} can prepare; no "
                "admissible suite runs both radiation streams, so this "
                "default is daytime-only and a window with local night "
                "must be declared")
    return (f"--source {source}'s route admits no shipped suite, so this "
            "is the door's listed default and the emission refusal names "
            "the offending switches")


def nocturnal_remedy(source: str) -> dict[str, Any]:
    """The way forward out of a nocturnal refusal on THIS source.

    Never a fixed profile id.  The suite named here is admissible on the
    active source by construction, which is the whole point: the remedy
    that shipped before this named one profile on every source, and on
    the native HRRR route that profile was refused one screen later for
    ``cu_physics = 1``.

    When the remedy IS the source's own default the instruction is to
    stop passing the flag, because that is the shortest true statement
    and it also survives a later change of default.
    """

    profile = None
    for candidate in admissible_profiles(source):
        if not day_only(candidate):
            profile = candidate
            break
    if profile is None:
        return {
            "profile_id": None,
            "is_default": False,
            "instruction": (
                f"--source {source} admits no shipped suite that runs "
                "both radiation streams, so this window can only run by "
                "declaring the daytime validation experiment"),
        }
    # The wizard's own radiation wording, not the raw selectors.  A
    # pilot report already read "ra_physics: 0" as "radiation off" on a
    # suite running RTE+RRTMGP on both streams, which is exactly the
    # misreading _radiation_words exists to prevent -- so the remedy
    # borrows that function rather than printing numbers beside a
    # sentence about radiation being on.
    from gpuwm.domain_wizard import _radiation_words

    default = default_profile_for(source)
    words = _radiation_words(_switches(profile))
    if profile == default:
        instruction = (
            f"omit --physics-profile, which binds --source {source}'s own "
            f"default {profile} ({words})")
    else:
        instruction = f"--physics-profile {profile} ({words})"
    return {
        "profile_id": profile,
        "is_default": profile == default,
        "instruction": instruction,
    }


def universally_admissible_profile() -> str | None:
    """The first suite NO registered source's route refuses, or ``None``.

    For the refusals that have no source in hand.  ``build_experiment``
    guards a loaded config, and a config carries no forcing source, so
    that refusal cannot be route-AWARE -- it can only be route-SAFE, and
    the only honest example it can give is a suite that runs everywhere.

    Both radiation streams are required for the same reason
    :func:`default_profile_for` requires them: the refusals that reach
    for this example are the nocturnal ones.
    """

    sources = registered_sources()
    for profile in shipped_profiles():
        if day_only(profile):
            continue
        if all(profile_route_blocker(profile, source) is None
               for source in sources):
            return profile
    return None


def admissibility_rules() -> list[dict[str, Any]]:
    """The rules the cells were computed against, and who owns each one.

    Emitted so a front end can SAY why a cell is closed rather than
    reproducing the reasoning -- a second implementation of these rules
    in a GUI is the hand-typed profile list one layer up.  Every list
    here is read from the module that enforces it.
    """

    from gpuwm.hrrr_route_inputs import (
        ADMITTED_PBL_PHYSICS, ADMITTED_RADIATION_PAIRS, REQUIRED_PHYSICS,
        SUPPORTED_MICROPHYSICS)
    from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      _ROUTE_FOR_SOURCE)

    return [
        {
            "rule": "route-emission-physics-gate",
            "owner": "gpuwm.hrrr_route_inputs.route_physics_blocker",
            "applies_to": sorted(_route_emission_physics_gates()),
            "declares": {
                "required_physics": dict(REQUIRED_PHYSICS),
                "admitted_pbl_physics": sorted(ADMITTED_PBL_PHYSICS),
                "admitted_radiation_pairs": sorted(
                    list(pair) for pair in ADMITTED_RADIATION_PAIRS),
                "supported_microphysics": sorted(SUPPORTED_MICROPHYSICS),
            },
        },
        {
            "rule": "land-surface-offer",
            "owner": "gpuwm.physics_compat.land_surface_route_blocker",
            "applies_to": sorted(_ROUTE_FOR_SOURCE),
            "declares": {
                "source_of_truth": "gpuwm/physics_registry_v2.json"
                                   "#/runner_routes",
            },
        },
        {
            "rule": "nocturnal-validity",
            "owner": "gpuwm.physics_compat.nocturnal_radiation_refusal",
            "applies_to": "every source",
            "declares": {
                "day_only_predicate": "ra_sw_physics > 0 and "
                                      "ra_lw_physics == 0",
                "acknowledgement":
                    ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
            },
        },
        {
            "rule": "vertical-level-bounds",
            "owner": "gpuwm.physics_compat."
                     "validate_resolved_physics_vertical_levels",
            "applies_to": "every source",
            "declares": {
                "reported_as": "profiles[].vertical_levels",
            },
        },
    ]


def source_menu(source: str, *, display_name: str | None = None,
                blocker: Callable[[str, str], str | None] | None = None,
                ) -> dict[str, Any]:
    """One source's row: its default, its remedy, and every cell.

    ``blocker`` exists so the whole-document build can pass the one
    predicate it already imported rather than re-importing it per row;
    it defaults to the wizard's own.
    """

    if blocker is None:
        blocker = profile_route_blocker

    default = default_profile_for(source)
    cells = []
    admissible_count = 0
    for profile in shipped_profiles():
        why_not = blocker(profile, source)
        if why_not is None:
            admissible_count += 1
        cells.append({
            "profile_id": profile,
            "admissible": why_not is None,
            # The route's own sentence, verbatim.  A paraphrase here
            # would be a second vocabulary for one switch, and the
            # reader who followed the menu and then met the refusal
            # would read two different accounts of the same fact.
            "why_not": why_not,
            "is_default": profile == default,
            "day_only": day_only(profile),
            "maturity": maturity(profile),
            "select_with": (f"--physics-profile {profile}"
                            if profile != default
                            else "omit --physics-profile"),
        })
    return {
        "source_id": source,
        "display_name": display_name if display_name is not None else source,
        "default_profile_id": default,
        "default_basis": default_basis(source),
        "admissible_count": admissible_count,
        "nocturnal_remedy": nocturnal_remedy(source),
        "profiles": cells,
    }


__all__ = [
    "WIZARD_PHYSICS_PROFILES", "profile_route_blocker",
    "admissibility_rules", "admissible_profiles", "day_only",
    "day_only_reason", "default_basis", "default_profile_for", "maturity",
    "nocturnal_remedy", "profile_facts", "radiation_scheme_ids",
    "registered_sources", "shipped_profiles", "source_menu",
    "universally_admissible_profile", "vertical_levels",
]
