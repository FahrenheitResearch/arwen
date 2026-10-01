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
  route's actual input-species requirements; source/template evidence membership
  does not restrict otherwise valid land-surface choices;
* :func:`source_soil_blocker` -- the soil the source publishes against the
  land surface's own soil ingest (RUC's nine-level remap), the question
  the preparation asks after the download;
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

from functools import lru_cache
from typing import Any, Callable, Mapping

from gpuwm.physics_compat import (SINGLE_DOMAIN_PHYSICS_PROFILES,
                                  MORRISON_PROFILE_ID, MYNN_PROFILE_ID,
                                  MYNN_RTE_RRTMGP_PROFILE_ID,
                                  MYNN_RUC_PROFILE_ID,
                                  MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  NSSL2_LEGACY_RRTMG_PROFILE_ID,
                                  NSSL2_PROFILE_ID,
                                  P3_LEGACY_RRTMG_PROFILE_ID,
                                  RUC_PROFILE_ID,
                                  THOMPSON_LEGACY_RRTMG_PROFILE_ID,
                                  THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
                                  THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
                                  THOMPSON_PROFILE_ID,
                                  THOMPSON_RTE_RRTMGP_PROFILE_ID,
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
#: cumulus-off Thompson suites and the P3 composition are the only
#: full-radiation suites the nested HRRR route's physics gate admits --
#: every other 4/4 profile carries ``cu_physics = 1``, which that route
#: refuses because its 3 km grid is convection permitting -- so a list
#: without them could only default that route to a
#: shortwave-on/longwave-off suite.  P3 sits AFTER them deliberately:
#: the head of the HRRR-admissible set is what a source with no adapter
#: recommendation inherits, so it has to be the arm the product
#: defaults to; P3 is chosen, never inherited.
#:
#: The RTE+RRTMGP Thompson suite therefore leads its legacy twin, which
#: it did not until the modern radiation arm became the default on every
#: route (owner ruling, 2026-09-19).  While the legacy twin led, a
#: source whose adapter named no recommendation would have inherited the
#: legacy engines from this order alone, which is the opposite of that
#: ruling.  ``--source hrrr`` never read it either way: that adapter's
#: own recommendation binds its answer.
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
#: The RANKING, not the membership (audit R-067).  Membership is derived
#: from the registry below, so a registered template can never be left
#: without a wizard row; this tuple says which suites lead, and the
#: paragraph above says why that order is the thing being compared.
_WIZARD_PROFILE_RANKING = (
    MORRISON_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    P3_LEGACY_RRTMG_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    # The Thompson member of the MYNN + RUC pair sits beside its WSM6
    # sibling in each block: both radiation streams here, Dudhia below.
    # Offered at every spacing, and the default below 1 km by the
    # spacing table that follows.
    THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    WSM6_PROFILE_ID,
    MYNN_PROFILE_ID,
    RUC_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    THOMPSON_MYNN_RUC_DUDHIA_PROFILE_ID,
)


#: The default suite by GRID SPACING, as table rows.  A row binds when the
#: finest grid a run asks for is finer than ``finest_dx_below_m`` and the
#: source's route admits its suite (the same pairing predicate every
#: refusal and every menu cell reads); the first row that binds wins, and
#: a source that admits no row's suite keeps its spacing-free default
#: (:func:`default_profile_for`).  Nothing here names a source.
#:
#: The one row: sub-km product domains take Thompson + MYNN surface layer
#: and PBL + RUC with radiation on both streams.  Scored against stations
#: and ceilometers on marine fog days at 750 m, that composition kept the
#: stratus that the source defaults' YSU and Noah each lost part of
#: (stratus CSI 0.71 against 0.52 with Noah and 0.55 with YSU + Noah),
#: and a custom sub-km domain with no --physics-profile got the source
#: default because the default keyed on the source alone.
SPACING_DEFAULTS: tuple[dict[str, Any], ...] = (
    {
        "finest_dx_below_m": 1000.0,
        "profile_id": THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
        "basis": (
            "the default below 1 km: Thompson, MYNN and RUC kept coastal "
            "fog and low stratus that YSU or Noah each lost part of, scored "
            "against stations and ceilometers on marine fog days"),
    },
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


def _land_surface_soil_admissions() -> dict[int, tuple[str, Callable]]:
    """Land surfaces whose soil ingest is a table lookup on the SOURCE'S soil.

    Keyed by ``sf_surface_physics``: the scheme's name and the admission
    the preparation's own soil ingest calls on the soil geometry a
    source's mapped composition declares.  RUC is the one row: its nine
    levels are remapped from whatever ladder the source publishes through
    :func:`gpuwm.ingest.soil_contract.ruc_soil_remap_policy`, which
    refuses a ladder WRF's ``init_soil_3_real`` cannot build a column
    from.  Noah's four layers are every mapped contract's own target
    (:func:`gpuwm.ingest.soil_contract.validate_soil_layer_contract`), so
    it has no row.  A TABLE, not a code path: a land surface that gains a
    source-geometry requirement is one row here, and every surface that
    reads the pairing predicate -- the wizard's refusal, the menu's cells,
    the default derivation -- answers with it.
    """

    from gpuwm.ingest.soil_contract import ruc_soil_remap_policy

    return {3: ("the RUC land surface (sf_surface_physics = 3)",
                ruc_soil_remap_policy)}


@lru_cache(maxsize=None)
def _source_soil_contract(source: str):
    """The soil geometry ``source``'s mapped composition declares, or None.

    None for a source with no packaged composition: its preparation
    dispatches on the native field names (ERA5, GFS, HRRR), whose soil
    arms the RUC ingest reproduces at 0 ULP against WRF
    (tests/test_ruc_soil_wiring.py), and for a composition that is still
    pending or declares no soil, which its own preparation refuses first.
    """

    from gpuwm.mapped_composition import load_composition
    from gpuwm.source_adapters import source_adapters
    from gpuwm.source_authorities import packaged_authorities

    profile_id = next((adapter.packaged_profile for adapter in source_adapters()
                       if adapter.source_id == source), None)
    if profile_id is None:
        return None
    try:
        authorities = packaged_authorities(profile_id)
        composition = load_composition(
            authorities["composition"], authorities["mapping"])
    except ValueError:
        return None
    return composition.get("soil_layers")


def source_soil_blocker(switches, source) -> str | None:
    """Why ``source``'s published soil cannot start this land surface.

    A runtime REQUIREMENT, never template membership (the rule
    gpuwm/physics_compat.py states where the membership gate was
    retired): the question asked is the one the preparation's soil
    ingest asks after the download, on the geometry the source's own
    composition declares, so the answer here and the refusal there are
    one answer.  The breakage it names is a preparation that fetched the
    whole cycle and then could not build the land surface's soil column;
    measured on 2.8.0, a RUC suite on ``--source gem-gdps`` (one 0-10 cm
    slab) passed ``gpuwm domain``, ``gpuwm check`` and ``gpuwm go
    --dry-run`` and could only be refused at preparation.
    """

    if switches is None or source is None:
        return None
    try:
        surface = int(dict(switches).get("sf_surface_physics"))
    except (TypeError, ValueError):
        return None
    admission = _land_surface_soil_admissions().get(surface)
    if admission is None:
        return None
    contract = _source_soil_contract(str(source))
    if contract is None:
        return None
    scheme, admit = admission
    try:
        admit(contract)
    except ValueError as error:
        return (f"{scheme} cannot be initialised from the soil --source "
                f"{source} publishes, so its preparation would download "
                f"the cycle and then refuse: {error}")
    return None


def switch_route_blocker(switches, source, *, domains: int = 1
                         ) -> str | None:
    """Why ``source``'s emission route refuses these resolved switches.

    The switch-level spelling of the same gate
    :func:`profile_route_blocker` asks about a shipped suite, for the
    callers that hold a resolved domain rather than a profile id -- the
    companion's option availability is one.  It reads the SAME tables, so
    a greyed cell in a front end and the refusal a run meets later
    cannot disagree, and a source registered tomorrow is answered here
    with no edit: the emission route's own physics gate, then the soil
    the source publishes against the land surface's soil ingest
    (:func:`source_soil_blocker`).

    ``domains`` is how many grids the run has, so each door asks for the
    run it holds.  Every route answers a tree with its single-domain gate:
    the nested HRRR route's hierarchy stage pins the soil column its land
    surface runs (four layers for Noah, nine for RUC), so it has no
    objection that binds only a tree.
    """

    if switches is None or source is None:
        return None
    emission_gate = _route_emission_physics_gates().get(str(source))
    if emission_gate is not None:
        blocked = emission_gate(dict(switches))
        if blocked is not None:
            return blocked
    return source_soil_blocker(switches, source)


def profile_route_blocker(profile, source, *, domains: int = 1
                          ) -> str | None:
    """Why ``source`` cannot prepare ``profile``, or ``None``.

    The registry answer plus the emission route's own physics gate,
    resolved through the same calls the front doors make, so a caller
    asking "can this pairing run", a runner asking "may this pairing
    run" and a refusal RANKING alternative suites cannot give different
    answers.
    """

    from gpuwm.physics_compat import (
        single_domain_runtime_switches,
    )

    if profile is None or source is None:
        return None
    try:
        switches = single_domain_runtime_switches(profile)
    except (KeyError, ValueError):
        return None
    return switch_route_blocker(switches, source, domains=domains)


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
    """``(longwave, shortwave)`` selectors, in every spelling.

    One line of delegation, on purpose.  The registry writes the split
    pair, v1.0.0 wrote the combined ``ra_physics``, and the aggregate
    radiation option writes -1 in both split keys with the resolved pair
    in the combined one.  This reader knew the first two and read the
    third literally, so the suite on that option reported scheme -1 on
    both streams -- which a picker prints as an id and a premise check
    reads as "runs no radiation at all".  It went unseen while no menu
    offered that suite and surfaced the moment one did.

    :func:`gpuwm.config.radiation_scheme_ids` is the engine's rule and
    :func:`gpuwm.config.radiation_scheme_ids_from_settings` is that rule
    asked of a switch map, so there is now one place to get it wrong.
    """

    from gpuwm.config import radiation_scheme_ids_from_settings

    return radiation_scheme_ids_from_settings(switches)


def day_only(profile: str) -> bool:
    """Whether this suite runs shortwave with longwave OFF.

    THE nocturnal-validity class, spelled as the guards spell it
    (``ra_sw_physics > 0 and ra_lw_physics == 0``).  A suite in this
    class heats the surface by day with no longwave scheme to balance it
    after sunset, so a window containing local night is refused unless
    the operator declares the validation experiment themselves.
    """

    return switches_day_only_reason(_switches(profile)) is not None


def day_only_reason(profile: str) -> str | None:
    """Why this suite is daytime-only, in the guards' own terms."""

    return switches_day_only_reason(_switches(profile))


def switches_day_only_reason(switches: Mapping[str, Any]) -> str | None:
    """:func:`day_only_reason` asked of a switch map rather than a suite id.

    The physics composer checks combinations no suite names, and asking
    the suite-keyed form would leave those without the answer or with a
    second copy of the predicate.
    """

    longwave, shortwave = radiation_scheme_ids(switches)
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
        "urban_scheme_id": int(switches.get("sf_urban_physics", 0)),
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


def _recommended_profile(source):
    from gpuwm.source_adapters import source_adapters
    return next((adapter.default_physics_profile for adapter in source_adapters()
                 if adapter.source_id == source), None)


def spacing_default_rows(source: str, domains: int = 1
                         ) -> list[dict[str, Any]]:
    """:data:`SPACING_DEFAULTS` as ``source`` answers it, row by row.

    Each row carries whether this source's route admits its suite for a
    run of ``domains`` domains and, when it does not, the pairing
    predicate's own sentence, so a front end that shows the default for a
    spacing reads the same answer the wizard binds rather than
    re-deriving it.
    """

    rows = []
    offered = set(shipped_profiles())
    for row in SPACING_DEFAULTS:
        profile = row["profile_id"]
        why_not = (profile_route_blocker(profile, source, domains=domains)
                   if profile in offered else
                   f"{profile} is not a suite this door offers")
        if why_not is None and day_only(profile):
            why_not = day_only_reason(profile)
        rows.append({**row, "admitted": why_not is None, "why_not": why_not})
    return rows


def spacing_default_menu_rows(source: str) -> list[dict[str, Any]]:
    """:func:`spacing_default_rows` for a menu, which knows no domain count.

    ``admitted`` and ``why_not`` answer for a single domain, and
    ``admitted_nested`` and ``why_not_nested`` for a domain with nests,
    each from the pairing predicate asked for that many grids, so a
    front end reads the answer the door gives the run it holds.
    """

    return [{**single, "admitted_nested": nested["admitted"],
             "why_not_nested": nested["why_not"]}
            for single, nested in zip(spacing_default_rows(source, 1),
                                      spacing_default_rows(source, 2))]


def spacing_default(source: str, finest_dx_m: float | None,
                    domains: int = 1) -> dict[str, Any] | None:
    """The :data:`SPACING_DEFAULTS` row that binds this grid, or ``None``."""

    if finest_dx_m is None:
        return None
    for row in spacing_default_rows(source, domains):
        if float(finest_dx_m) < float(row["finest_dx_below_m"]) \
                and row["admitted"]:
            return row
    return None


def default_profile_for(source: str, finest_dx_m: float | None = None,
                        domains: int = 1) -> str | None:
    """Prefer the spacing row, then source metadata, then the global default.

    ``finest_dx_m`` is the finest grid spacing the run asks for.  Given,
    the first :data:`SPACING_DEFAULTS` row that covers it and that this
    source's route admits for ``domains`` domains is the default; a
    source whose route admits no row's suite falls through to its
    spacing-free default below.

    The recommendation is independent of capability: allowing another suite
    cannot silently change the default. A global None retains the explicit
    unnamed-suite path; sources without a recommendation use the shared order.
    """

    from gpuwm.domain_wizard import DEFAULT_PHYSICS_PROFILE

    declared = DEFAULT_PHYSICS_PROFILE
    if declared is None:
        return None
    by_spacing = spacing_default(source, finest_dx_m, domains)
    if by_spacing is not None:
        return by_spacing["profile_id"]
    preferred = _recommended_profile(source) or declared
    admissible = admissible_profiles(source)
    if preferred in admissible and not day_only(preferred):
        return preferred
    if declared in admissible and not day_only(declared):
        return declared
    for profile in admissible:
        if not day_only(profile):
            return profile
    if admissible:
        return admissible[0]
    return declared


def default_basis(source: str, finest_dx_m: float | None = None,
                  domains: int = 1) -> str:
    """Why that suite is this source's default, in one sentence."""

    by_spacing = spacing_default(source, finest_dx_m, domains)
    if by_spacing is not None:
        return by_spacing["basis"]
    recommendation = _recommended_profile(source)
    if recommendation is not None and default_profile_for(source) == recommendation:
        return "the source's declared recommendation; other implemented suites remain selectable"
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

    default = default_profile_for(source)
    profile = (default if default is not None
               and default in admissible_profiles(source) and not day_only(default) else None)
    if profile is None:
        profile = next((candidate for candidate in admissible_profiles(source)
                        if not day_only(candidate)), None)
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
    the only accurate example it can give is a suite that runs everywhere.

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

    from gpuwm.physics_compat import ASYMMETRIC_RADIATION_NOCTURNAL_ACK

    return [
        {
            "rule": "route-emission-physics-gate",
            "owner": "gpuwm.hrrr_route_inputs.route_physics_blocker",
            "applies_to": sorted(_route_emission_physics_gates()),
            "declares": {
                "required_input": "active scheme prognostic boundary species",
                "configuration_authority": "gpuwm.config.validate_run_config",
                "profile_membership_required": False,
            },
        },
        {
            # The soil the source publishes against the land surface's
            # own soil ingest: the question the preparation asks after
            # the download, asked before it.
            "rule": "source-soil-geometry",
            "owner": "gpuwm.physics_menu.source_soil_blocker",
            "applies_to": "every source with a packaged composition",
            "declares": {
                "land_surfaces": {
                    str(surface): scheme for surface, (scheme, _admit)
                    in _land_surface_soil_admissions().items()},
                "admission": "gpuwm.ingest.soil_contract."
                             "ruc_soil_remap_policy",
                "profile_membership_required": False,
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
            # The pairing laws, which are TABLE DATA and not this
            # module's own: an option's constraints say which sibling
            # component option it requires and which combination is
            # refused outright.  Named here so a front end that greys a
            # cell for a coupling can say who owns the refusal it is
            # anticipating, instead of a GUI implying it invented the
            # pairing.  Where the cells come from is stated too, because
            # they arrive on a different payload from these profiles.
            "rule": "component-coupling",
            "owner": "gpuwm.physics_registry.validate_physics_plan",
            "applies_to": "every component option carrying constraints",
            "declares": {
                "requires_components":
                    "{component_id: [option_id, ...]} -- the sibling "
                    "options this option must be paired with",
                "refused_when":
                    "conjunction rules over sibling options and settings, "
                    "each carrying its reason and, where one exists, its "
                    "remedy_label and remedy_settings",
                "reported_as":
                    "physics_components[].options[].requires_components "
                    "and .refused_when",
                "configuration_authority":
                    "gpuwm.config.validate_run_config",
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
        # The default by grid spacing, which binds ahead of the one above
        # when the run's finest grid is finer than a row's bound, answered
        # for one domain and for a tree.
        "spacing_defaults": spacing_default_menu_rows(source),
        "admissible_count": admissible_count,
        "nocturnal_remedy": nocturnal_remedy(source),
        "profiles": cells,
    }


__all__ = [
    "SPACING_DEFAULTS", "WIZARD_PHYSICS_PROFILES", "profile_route_blocker",
    "spacing_default", "spacing_default_menu_rows", "spacing_default_rows",
    "admissibility_rules", "admissible_profiles", "day_only",
    "day_only_reason", "default_basis", "default_profile_for", "maturity",
    "nocturnal_remedy", "profile_facts", "radiation_scheme_ids",
    "registered_sources", "shipped_profiles", "source_menu",
    "source_soil_blocker", "switch_route_blocker",
    "switches_day_only_reason",
    "universally_admissible_profile", "vertical_levels",
]


# ---------------------------------------------------------------------------
# AGREEMENT WITH THE REGISTRY, AT IMPORT.  The wizard menu is a hand-kept
# list of template ids; an implemented composition it omits has no front
# door through the wizard.  Every omission is cited so the sweep is a grep:
# the Kessler probe is an HRRR-only ratification product, and audit R-068
# covers templates with no single-domain runtime product. The three
# Noah-MP expert templates remain selectable with advisory metadata.
#
# RETIRED HERE, with the defect it produced: the Kessler ratification
# probe's citation read "deliberately not offered as a wizard suite",
# which names no breakage -- and the nowcast front door offers every
# suite the single-domain runner resolves, so it offered a profile the
# configuration door it calls first would not accept.  Both of its rows
# died in argument parsing before any physics ran.  A template the
# single-domain door resolves a runtime product for is offered by every
# door that reaches that runner; a template it does not is cited below,
# by measurement.
_TEMPLATES_OUTSIDE_THE_WIZARD_MENU: dict[str, str] = {}
# The remaining omissions are named through the registry's own records --
# a composition, or the set difference against the door's own menu --
# rather than as id literals, the same way gpuwm/physics_compat.py cites
# them: one id carries the forcing source it was registered on, and a
# citation keyed on a spelling stops being a citation when the template
# is renamed.


def _templates_without_a_runtime_switch_row() -> dict[str, str]:
    """The omissions that remain, each one MEASURED rather than asserted.

    RETIRED here, against the measurement that retired it: the audit
    R-068 citation for the one WSM6 + Kain-Fritsch template on the
    aggregate RTE+RRTMGP radiation option, which read "no
    _SINGLE_DOMAIN_RUNTIME_SWITCHES row, so no runner accepts it".  Since
    the single-domain menu became every fixed-template route's own
    declaration, one of those routes declares that template and
    :func:`gpuwm.physics_compat.single_domain_runtime_switches` resolves a
    complete twenty-three-value product for it, so the reason had stopped
    being true while the wizard still excluded the template on it.
    gpuwm/physics_compat.py retired its sibling citation; this one was not
    swept with it.

    The replacement cannot go stale the same way, because R-068 is no
    longer a list of ids with a sentence attached.  It is the QUESTION
    "does the single-domain door resolve a runtime product for this
    template", asked of the door itself: a template that gains one leaves
    this map by arithmetic, and :func:`_wizard_menu` then offers it in the
    same import.  A template that loses one is cited automatically instead
    of failing the agreement check with no reason to give.

    RETIRED here too, with the fix it waited on: the R-067 citation for
    the aerosol-aware Thompson suite, whose blocker (R-044) was a missing
    mp_physics=28 arm in gpuwm/ingest/microphysics_cold_start.py.  The arm
    exists and the prepared single-domain route declares the suite, so the
    single-domain door resolves a runtime product for it and the wizard
    offers it by the arithmetic above.
    """

    import os

    from gpuwm.physics_registry import REGISTRY_REBUILD_ENV, physics_registry

    if os.environ.get(REGISTRY_REBUILD_ENV) == "1":
        # Skipped while the registry is being regenerated, exactly as the
        # agreement check these citations feed is: the templates they name
        # may not exist on disk until the rebuild finishes.
        return {}
    cited: dict[str, str] = {}
    reason = (
        "audit R-068: no fixed-template runner route declares it, so the "
        "single-domain door resolves no runtime product for it and no "
        "runner accepts it; declaring it on a fixed-template route is what "
        "retires this citation")
    offered = set(SINGLE_DOMAIN_PHYSICS_PROFILES)
    for template_id in sorted(physics_registry()["templates"]):
        if (template_id in offered or template_id in cited
                or template_id in _TEMPLATES_OUTSIDE_THE_WIZARD_MENU):
            continue
        cited[template_id] = reason
    return cited


_TEMPLATES_OUTSIDE_THE_WIZARD_MENU.update(
    _templates_without_a_runtime_switch_row())


def _wizard_menu() -> tuple[str, ...]:
    """The wizard menu: ranking above, membership from the registry.

    AUDIT R-067.  A hand-kept tuple of template ids is a scheme table, and
    an implemented composition with no menu row has no front door -- which
    is how eleven implemented schemes came to be registered, selectable in
    principle, and offered by nothing.  Every template the single-domain
    door validates now earns a wizard row unless its absence is CITED
    above, and the citations are what the retirement sweep greps for.

    The ranking stays declared because its order carries meaning that no
    registry row states: the nocturnally valid suites lead, so
    :func:`default_profile_for` taking the head of an admissible set gets
    both radiation streams without knowing what nocturnal means.  A suite
    the ranking does not name follows it, in the order the route declares.
    """

    ordered = list(_WIZARD_PROFILE_RANKING)
    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        if profile in _TEMPLATES_OUTSIDE_THE_WIZARD_MENU or profile in ordered:
            continue
        ordered.append(profile)
    return tuple(ordered)


WIZARD_PHYSICS_PROFILES = _wizard_menu()


def _require_agreement_with_the_registry() -> None:
    from gpuwm.physics_registry import require_template_menu_agreement

    require_template_menu_agreement(
        "gpuwm.physics_menu.WIZARD_PHYSICS_PROFILES", WIZARD_PHYSICS_PROFILES,
        cited_absences=_TEMPLATES_OUTSIDE_THE_WIZARD_MENU)


_require_agreement_with_the_registry()
