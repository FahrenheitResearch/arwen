"""Author every input file the nested HRRR route needs, from one experiment.

The HRRR domain-tree route takes five documents, and until now the
product emitted exactly one of them.  ``gpuwm domain --source hrrr``
wrote an experiment TOML and a ``namelist.wps`` that was missing
``&share/interval_seconds`` -- the one key that route's own raw-WPS
contract gate demands -- and nothing at all for
``--root-domain-spec``, ``--namelist-input`` or
``--stock-wrf-namelist-input``.  So wizard output could not drive the
route it was sized for, and the gate that proved the route ran at all
had to borrow a lane's proof harness to author the missing files.

This module is that authoring, in the product, derived from ONE source
of truth: the :class:`~gpuwm.experiment.ExperimentConfig` the wizard
already emitted.  Every geometry number, every physics switch, every
clock key here is read off that experiment rather than restated, and
:func:`write_hrrr_route_inputs` re-imports what it wrote through the
REAL importer and refuses if the round trip does not reproduce the
experiment it started from.  A set of files that disagrees with the
TOML beside them is exactly the failure this replaces; it cannot be
written and then discovered three stages downstream.

What the route requires of these files is not restated here as prose --
it is enforced by the gates in :mod:`gpuwm.hrrr_hierarchy_direct`,
which the tests run over these exact bytes.
"""

from __future__ import annotations

from datetime import timedelta
from fractions import Fraction
import json
from pathlib import Path

from gpuwm.ingest.hrrr_target import TARGET_DOMAIN_SCHEMA

#: The route's fixed forcing cadence.  HRRR publishes hourly and the
#: raw-WPS/raw-runtime contract gates pin the integer 3600 in both
#: namelists; 3600.0 is refused, so the spelling matters.
FORCING_INTERVAL_SECONDS = 3600

#: HRRR's native vertical and soil inventories, as the route's runtime
#: contract pins them.
NUM_METGRID_LEVELS = 51
NUM_METGRID_SOIL_LEVELS = 9

#: The physics slice the public HRRR hierarchy gate admits.  A config
#: outside it is refused here, at emission, naming the switch -- not
#: after a root preparation has been paid for.  Two entries are not
#: single pinned values: radiation is admitted as a PAIR and the PBL as
#: an enumerated SET, both below.
REQUIRED_PHYSICS = {
    "sf_sfclay_physics": 91,
    "sf_surface_physics": 2,
    "cu_physics": 0,
}

#: The ``bl_pbl_physics`` values the route admits.  This was a single
#: pinned 1 in :data:`REQUIRED_PHYSICS` until the gray-zone template was
#: registered; it is an enumerated set now, on exactly the shape the
#: radiation pair took, and it is NOT a statement that any PBL selector
#: is fine.  Each member is a value a REGISTERED HRRR preparation profile
#: pins:
#:
#: * ``1`` -- YSU, pinned by every route-compatible profile the route has
#:   always carried (``wsm6-ysu-mm5-noah-no-radiation-v1``,
#:   ``kessler-mp1-ysu-mm5-noah-dudhia-v1``,
#:   ``thompson-mp8-ysu-mm5-noah-validation-v1`` and its legacy-RRTMG
#:   sibling).
#: * ``11`` -- Shin-Hong 2015, pinned by
#:   ``thompson-mp8-shinhong-mm5-noah-rrtmg-legacy-v1``, the registered
#:   composition a physics-fidelity arm selecting divergence-ledger entry
#:   L3 resolves to.
#:
#: A value no registered profile pins still refuses here by name, and it
#: would refuse one gate later anyway: the root preparation is a shipped
#: profile or it does not happen.  ``tests/test_battery_route.py``
#: recomputes this set from the shipped profile table, so the enumeration
#: cannot drift from the registrations that justify it.
ADMITTED_PBL_PHYSICS = frozenset({1, 11})

#: The (ra_lw_physics, ra_sw_physics) pairs the route admits (the B4
#: route-qualification motion, items 1-3): the certified (0, 1)
#: validation suite -- longwave off natively, Dudhia shortwave -- and
#: the resolved RRTMG (4, 4) pair the shipped registry's own
#: prepared-tree row already claims for HRRR.  Explicit pairs only:
#: :func:`gpuwm.config.validate_run_config` separately requires
#: ``ra_physics = 0`` alongside explicit values, and that gate still
#: binds on every domain of the experiment, so this set widens nothing
#: beyond the two named compositions.
ADMITTED_RADIATION_PAIRS = frozenset({(0, 1), (4, 4)})

#: Microphysics the route's nest-transition resolver knows.
SUPPORTED_MICROPHYSICS = frozenset({1, 6, 8, 10, 18})

#: THE profile every door binds for this source when none is named.
#:
#: One constant, imported by all of them, because 1.7.1 proved the
#: alternative: the wizard door, the interactive door and the native
#: preparer each carried their own default, they disagreed, and a fix
#: applied to one left the others emitting the old suite.  A door that
#: wants this source's default reads it here or it is not a door.
#:
#: The value is the strongest suite that satisfies this module's OWN
#: gates above with both radiation streams on.  It is Thompson mp8
#: microphysics with RRTMG longwave and shortwave at
#: ``cu_physics = 0`` -- which is also, and not by coincidence, what the
#: operational High-Resolution Rapid Refresh runs (NOAA/GSL; the CCPP
#: ``HRRR_suite`` pairs Thompson aerosol-aware microphysics with RRTMG
#: radiation on a convection-permitting 3 km grid).  gpuwm diverges from
#: operations on two components it has no route-admissible
#: implementation for: YSU rather than MYNN-EDMF, and Noah rather than
#: the RUC LSM.  Both of those shipped profiles run longwave OFF and are
#: refused above on ``sf_sfclay_physics``/``sf_surface_physics`` anyway.
#:
#: It is deliberately NOT the gfs/era5 default
#: (``morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1``): that suite selects
#: Kain-Fritsch, and ``REQUIRED_PHYSICS`` pins ``cu_physics = 0`` because
#: this source's native grid already resolves convection.  The refusal is
#: physics, not a limitation.
#:
#: ``tests/test_nocturnal_radiation_guard.py`` re-derives this from the
#: shipped switch table and the gates above rather than restating it, so
#: a profile that becomes admissible -- or one that stops being -- moves
#: the assertion, not just this comment.
ROUTE_DEFAULT_PHYSICS_PROFILE = "thompson-mp8-ysu-mm5-noah-rrtmg-legacy-v1"

#: The four -- and only four -- differences between the native namelist
#: gpuwm integrates and the stock-WRF namelist beside it.  The route
#: compares the two parsed files key for key and refuses any other
#: difference, so they are generated from one renderer with one switch.
#: Under the (4, 4) RRTMG pair the longwave delta collapses -- both
#: arms run 4 -- and the two stock-only keys stay stock-only for the
#: same reason: each names a setting the native arm answers in CODE, so
#: the native namelist would be claiming to control something gpuwm does
#: not read from it.  ``ghg_input = 0`` mirrors what
#: gpuwm/core/rrtmg_legacy.py pins; ``do_radar_ref = 1`` mirrors the
#: REFL_10CM gpuwm evaluates unconditionally at output time.  The
#: certified raw-runtime contract
#: (:func:`gpuwm.hrrr_hierarchy_direct._require_raw_stock_delta`)
#: requires both ABSENT from the native namelist and pins both in the
#: stock one; it is the enforcing half of this sentence.
_STOCK_DELTAS = ("ra_lw_physics 0->1 (native longwave off only), "
                 "use_theta_m 0->1, stock-only ghg_input=0, stock-only "
                 "do_radar_ref=1")


class HrrrRouteInputError(ValueError):
    """This experiment cannot drive the nested HRRR route."""


def _f(value) -> str:
    """A namelist float that never renders as an integer."""
    text = repr(float(value))
    return text if ("." in text or "e" in text or "E" in text) else text + "."


def _column(values) -> str:
    """One WRF per-domain column: comma separated, trailing comma."""
    return ", ".join(str(value) for value in values) + ","


def _logical(value: bool) -> str:
    return ".true." if value else ".false."


def _repeated(value, count: int) -> str:
    return _column([value] * count)


#: Every switch the route's physics gate reads -- resolved per domain
#: at emission, per shipped profile at the wizard's pairing gate.
ROUTE_GATED_SWITCHES = (*REQUIRED_PHYSICS, "bl_pbl_physics",
                        "ra_lw_physics", "ra_sw_physics", "mp_physics")


def route_physics_problems(switches, *, label: str = "") -> list[str]:
    """The route's objections to one resolved switch set, or ``[]``.

    THE one spelling of the physics slice this route admits.
    :func:`validate_route_physics` applies it to every domain of a real
    experiment at emission, and the wizard's pairing predicate
    (``gpuwm.domain_wizard.profile_route_blocker``) applies it to a
    shipped profile's switch table both when refusing a pairing and
    when RANKING lighter-suite advice.  One predicate, shared, so a
    refusal can never advise a suite this gate turns away and the two
    readings can never drift.
    """

    problems = []
    for switch, required in REQUIRED_PHYSICS.items():
        observed = switches[switch]
        if int(observed) != required:
            problems.append(
                f"{label}{switch}={observed} "
                f"(the route requires {required})")
    pbl = int(switches["bl_pbl_physics"])
    if pbl not in ADMITTED_PBL_PHYSICS:
        problems.append(
            f"{label}bl_pbl_physics={pbl} (the route "
            f"admits {sorted(ADMITTED_PBL_PHYSICS)}, each pinned by a "
            "registered HRRR preparation profile)")
    pair = (int(switches["ra_lw_physics"]), int(switches["ra_sw_physics"]))
    if pair not in ADMITTED_RADIATION_PAIRS:
        problems.append(
            f"{label}(ra_lw_physics, ra_sw_physics)="
            f"{pair} (the route admits "
            f"{sorted(ADMITTED_RADIATION_PAIRS)})")
    if int(switches["mp_physics"]) not in SUPPORTED_MICROPHYSICS:
        problems.append(
            f"{label}mp_physics={switches['mp_physics']} "
            f"(the route supports "
            f"{sorted(SUPPORTED_MICROPHYSICS)})")
    return problems


def route_physics_blocker(switches) -> str | None:
    """Why this switch set cannot drive the nested HRRR route, or None.

    The profile-level projection of :func:`validate_route_physics`, for
    callers holding a suite's switch table rather than a whole
    experiment -- the wizard's pairing predicate.  The offending
    switches ride in the FIRST clause, ahead of the layered remedy,
    because a refusal that hides them behind ``--explain`` names no
    breakage.
    """

    problems = route_physics_problems(switches)
    if not problems:
        return None
    return ("this suite cannot drive the nested HRRR route -- "
            + "; ".join(problems)
            + ": pass --physics-profile with a route-compatible suite "
              "(the wizard's --source hrrr default is one)")


def validate_route_physics(exp) -> None:
    """Refuse, at emission, a suite the route's own gate will refuse.

    The alternative is what the field met: a wizard that reports PASS,
    a fetch, a root preparation, and only then a refusal naming a
    switch the wizard had already chosen.
    """
    problems = []
    for domain in exp.domains:
        run = domain.run
        problems.extend(route_physics_problems(
            {switch: getattr(run, switch)
             for switch in ROUTE_GATED_SWITCHES},
            label=f"d{domain.grid_id:02d} "))
    if exp.feedback != 0 or exp.smooth_option != 0:
        problems.append(
            f"feedback={exp.feedback}, smooth_option={exp.smooth_option} "
            "(the route is one-way only; both must be 0)")
    if problems:
        raise HrrrRouteInputError(
            "this suite cannot drive the nested HRRR route: "
            + "; ".join(problems)
            + ".  Pass --physics-profile with a route-compatible suite "
              "(the wizard's --source hrrr default is one).")


def verify_axis_authored_keys(exp, text: str, *, stock: bool = False) -> None:
    """Every key the physics-fidelity axis governs, in the emitted bytes.

    The axis (:mod:`gpuwm.physics_mode`) is the AUTHOR of its resolved
    keys: the loader writes the resolved vector onto every domain's
    RunConfig and refuses any other author.  This renderer therefore has
    exactly one correct source for such a key -- the resolved config --
    and a literal spelled here instead is invisible until a root
    preparation refuses the drift.  That is what happened to the first
    Shin-Hong arm: the config resolved ``bl_pbl_physics`` to 11, the
    renderer said 1, and the profile validator was right to stop.

    So the renderer reads back its own bytes and holds every governed key
    to the RESOLVED value.  This is deliberately generic -- it names no
    ledger entry and no scheme number, and it grows with the ledger --
    because the defect class is "a governed key is spelled as a literal
    somewhere in this function", not "L3 was missed".

    An APPLIED patch whose key never reaches the namelist is refused
    outright: a WRF arm that cannot express the patch is not a mirror of
    the gpuwm arm, and finding that out at emission costs nothing while
    finding it out after a root preparation costs a node.  A patch
    resolved to its faithful side is not held to that, because the
    faithful arm is what the emitter has always written.
    """
    from gpuwm.namelist_import import parse_namelist_text

    resolution = getattr(exp, "physics_mode", None)
    if resolution is None or not getattr(resolution, "governed", False):
        return

    sections = parse_namelist_text(text)
    flavor = "stock-WRF" if stock else "native gpuwm"
    problems = []
    for patch in resolution.resolved:
        # parse_namelist_text lower-cases every key it reads, and Fortran
        # namelist names are case-insensitive, so the ledger's spelling is
        # folded rather than trusted to already match.
        wanted = patch.key.lower()
        found = [(name, values) for name, entries in sections.items()
                 for key, values in entries.items()
                 if key == wanted]
        if not found:
            if patch.patched:
                problems.append(
                    f"{patch.entry_id} resolves {patch.key} = "
                    f"{patch.value!r} (patched) but the {flavor} namelist "
                    "carries no such key, so the WRF arm would run this "
                    "entry unpatched")
            continue
        if len(found) > 1:
            problems.append(
                f"{patch.key} appears in {sorted(n for n, _ in found)}; a "
                "governed key must be authored in exactly one section")
            continue
        section, values = found[0]
        # Every column element, not just d01: one experiment is one arm,
        # so a nest carrying a different resolved value would be a tree
        # that cannot be compared across its own boundary.
        if any(value != patch.value for value in values):
            problems.append(
                f"{patch.entry_id} resolves {patch.key} = {patch.value!r} "
                f"but &{section}/{patch.key} was emitted as {values!r}; the "
                "renderer must author this key from the resolved config, "
                "never as a literal")
    if problems:
        raise HrrrRouteInputError(
            "the emitted namelist contradicts the physics-fidelity axis "
            f"it was rendered from (physics_mode = {resolution.mode!r}, "
            f"patchset = {resolution.patchset!r}): "
            + "; ".join(problems)
            + ".  gpuwm.physics_mode.governed_keys() is the list every "
              "emitter owes the resolved value.")


def render_target_domain(exp) -> str:
    """The ``--root-domain-spec`` document for this experiment's d01.

    Its key set is closed: the loader accepts the dataclass fields and
    nothing else, so this is generated from the experiment rather than
    hand-kept.
    """
    root = exp.domains[0]
    run = root.run
    projection = exp.projection
    if projection.map_proj != "lambert":
        raise HrrrRouteInputError(
            f"the nested HRRR route is Lambert-only; this domain is "
            f"{projection.map_proj}.  Re-run the wizard with "
            "--projection lambert, or pick a point where 'auto' selects "
            "it (|lat| between 25 and 60 degrees).")
    # The root clock, decomposed exactly the way WRF's registry and the
    # experiment TOML spell it: whole seconds plus a proper rational
    # remainder.  A 1.5 km root runs 7.5 s = 7 + 1/2; the integer-only
    # spelling this replaced refused that ladder outright while the
    # same clock ran all night on the GFS route.
    dt = Fraction(exp.dt_exact(root.grid_id))
    dt_whole = dt.numerator // dt.denominator
    dt_rem = dt - dt_whole
    document = {
        "schema": TARGET_DOMAIN_SCHEMA,
        "name": exp.name,
        "map_proj": "lambert",
        "nx": int(run.nx),
        "ny": int(run.ny),
        "nz": int(run.nz),
        "dx_m": float(exp.dx_exact(root.grid_id)),
        "dy_m": float(exp.dx_exact(root.grid_id)),
        "ref_lat": float(projection.ref_lat),
        "ref_lon": float(projection.ref_lon),
        "truelat1": float(projection.truelat1),
        "truelat2": float(projection.truelat2),
        "stand_lon": float(projection.stand_lon),
        "time_step_seconds": int(dt_whole),
        "spec_bdy_width": int(run.spec_bdy_width),
        "spec_zone": int(run.spec_zone),
        "relax_zone": int(run.relax_zone),
    }
    if dt_rem:
        document["time_step_fract_num"] = dt_rem.numerator
        document["time_step_fract_den"] = dt_rem.denominator
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def target_domain(exp):
    """This experiment's d01 as the route's own target-domain object."""
    from gpuwm.ingest.hrrr_target import HrrrTargetDomain

    document = json.loads(render_target_domain(exp))
    document.pop("schema")
    return HrrrTargetDomain(**document)


def target_coverage_refusal(target) -> str | None:
    """Why HRRR cannot initialize this d01, or ``None`` when it can.

    Two demands, not one.  The WINDOW demand is the root preparer's own
    (``required_hrrr_source_window``): every interpolation and donor
    cell inside the native 1799 x 1059 grid.  The MARGIN demand is the
    wizard's promise: the required window must stay a further
    ``surface_fallback_radius_cells`` clear of every native edge,
    because a domain pinned exactly against the edge is one whose soil
    donor searches have no remediation left -- the only knob the
    mapping refusal can name (raising the radius) enlarges the required
    window past the edge and is refused by this very guard.

    Field 2026-08 (39,-98, 3 km root): the auto-fitted 1234x986 passed
    the window demand exactly on the top edge (j=36..1058), soil
    mapping then found two land cells with no donor within 8 cells, and
    the recommended raise to 16 was refused (j=28..1066 against j max
    1058).  Trimmed to 1186x938 -- 24 cells a side, exactly this margin
    -- it prepared, ran, and rendered.
    """
    from gpuwm.ingest.hrrr_target import (HRRR_SOURCE_NX, HRRR_SOURCE_NY,
                                          required_hrrr_source_window)

    try:
        window = required_hrrr_source_window(target)
    except ValueError as error:
        return str(error)
    radius = window.surface_fallback_radius_cells
    shortfalls = [
        (side, gap) for side, gap in (
            ("west", radius - window.i_start),
            ("east", window.i_end + radius - (HRRR_SOURCE_NX - 1)),
            ("south", radius - window.j_start),
            ("north", window.j_end + radius - (HRRR_SOURCE_NY - 1)),
        ) if gap > 0]
    if not shortfalls:
        return None
    worst = max(gap for _side, gap in shortfalls)
    named = ", ".join(f"the {side} side is short {gap} cell(s)"
                      for side, gap in shortfalls)
    return (
        "this domain leaves no usable surface-donor margin: its required "
        f"source window i={window.i_start}..{window.i_end}, "
        f"j={window.j_start}..{window.j_end} must stay a further "
        f"{radius} cells (its own surface_fallback_radius_cells) inside "
        f"the native HRRR limits i=0..{HRRR_SOURCE_NX - 1}, "
        f"j=0..{HRRR_SOURCE_NY - 1}, and {named}.  A land cell whose "
        "donor search fails near that edge would have no acceptable "
        "remedy -- every larger radius leaves HRRR coverage -- so trim "
        f"the domain by at least {worst} HRRR cell(s) (~{3 * worst} km) "
        "on the named side(s), or move it away from the edge")


def coverage_refusal(exp) -> str | None:
    """Why HRRR cannot force this d01, or ``None`` when it can.

    HRRR's native grid is 1799 x 1059 cells, and the interpolation
    stencil plus the surface-fallback halo need real source cells on
    every side of the target.  A domain sized purely against VRAM can
    therefore be a perfectly legal experiment that no HRRR fetch can
    ever force -- and that is what a card-filling ladder near the edge
    of HRRR's coverage produces.  Asked at sizing time, this makes the
    fit loop pick a domain HRRR can carry; asked at emission, it names
    the overflow instead of leaving it to a root preparation minutes
    later.  The donor-search margin rides along
    (:func:`target_coverage_refusal`), so what the fit loop accepts is
    a domain whose soil mapping keeps a usable remediation.

    This answers the COVERAGE question and nothing else.  A spec that
    cannot even be constructed (a non-Lambert projection, a malformed
    clock) raises :class:`HrrrRouteInputError` with its own cause and
    remedy; it is NOT returned as if the polygon were off the grid.
    Absorbing those errors here once wrapped "the root domain spec
    carries an integer time step" inside "falls outside HRRR coverage
    ...  Move --polygon inside the HRRR grid", and the user it refused
    was sent chasing a phantom coverage problem (field, 2026-08-06,
    dx 1.5 km).
    """
    return target_coverage_refusal(target_domain(exp))


def coverage_advisory(exp) -> list[str]:
    """Say when HRRR's grid, not the card, is what stopped the sizing.

    The sizing line reports headroom in GiB, so a domain the fit loop
    stopped growing for a reason that is not memory reads as an
    unexplained shortfall.  When the required source window plus the
    reserved donor-search margin reaches an edge of HRRR's native grid,
    that IS the reason.
    """
    from gpuwm.ingest.hrrr_target import (HRRR_SOURCE_NX, HRRR_SOURCE_NY,
                                          required_hrrr_source_window)

    if coverage_refusal(exp) is not None:
        return []
    window = required_hrrr_source_window(target_domain(exp))
    radius = window.surface_fallback_radius_cells
    touched = [name for name, at_edge in (
        ("west", window.i_start - radius <= 0),
        ("east", window.i_end + radius >= HRRR_SOURCE_NX - 1),
        ("south", window.j_start - radius <= 0),
        ("north", window.j_end + radius >= HRRR_SOURCE_NY - 1),
    ) if at_edge]
    if not touched:
        return []
    return [
        "this domain is bounded by HRRR's own grid, not by your card: "
        f"its interpolation window plus the reserved {radius}-cell "
        f"surface-donor margin reaches the {'/'.join(touched)} edge of "
        f"the {HRRR_SOURCE_NX}x{HRRR_SOURCE_NY} HRRR grid, so headroom "
        "left on the card cannot be spent here"]


def render_namelist_input(exp, *, stock: bool = False) -> str:
    """One WRF ``namelist.input`` for this experiment.

    ``stock`` renders the unchanged-WRF twin.  It differs from the
    native file in exactly three ways -- the route parses both and
    refuses a fourth -- so both come out of this one function.
    """
    validate_route_physics(exp)
    domains = list(exp.domains)
    count = len(domains)
    root = domains[0]
    runs = [domain.run for domain in domains]

    start = exp.start_time
    end = start + timedelta(seconds=float(exp.run_seconds))
    run_seconds = int(exp.run_seconds)
    hours, remainder = divmod(run_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    def dt_columns(moment, prefix):
        parts = {
            "year": f"{moment.year:04d}", "month": f"{moment.month:02d}",
            "day": f"{moment.day:02d}", "hour": f"{moment.hour:02d}",
            "minute": f"{moment.minute:02d}",
            "second": f"{moment.second:02d}",
        }
        return "".join(
            f" {prefix}_{key:<28} = {_repeated(value, count)}\n"
            for key, value in parts.items())

    eta = exp.vertical.eta_levels
    eta_rows = []
    for offset in range(0, len(eta), 5):
        chunk = ", ".join(f"{value:.5f}" for value in eta[offset:offset + 5])
        eta_rows.append(("              " if offset else "") + chunk + ",")
    clock = {"time_step": root.time_step}
    if root.time_step_fract_num:
        clock["time_step_fract_num"] = root.time_step_fract_num
        clock["time_step_fract_den"] = root.time_step_fract_den

    # Radiation is rendered from the config, per domain, not restated as
    # a literal.  The stock twin substitutes RRTM longwave (1) exactly
    # where the native arm runs with longwave off (0) -- WRF cannot run
    # the native preparation's disabled longwave -- and carries every
    # other radiation selection unchanged, so under the (4, 4) RRTMG
    # pair the longwave delta collapses and the two arms are identical
    # here.
    native_lw = [int(r.ra_lw_physics) for r in runs]
    shortwave = [int(r.ra_sw_physics) for r in runs]
    longwave = ([1 if value == 0 else value for value in native_lw]
                if stock else native_lw)
    theta_m = 1 if stock else 0
    ghg = " ghg_input                           = 0,\n" if stock else ""
    # STOCK-ONLY, and mandatory there: the mirrored arm has to PRODUCE
    # what the registration scores.  gpuwm evaluates REFL_10CM at output
    # time unconditionally -- it is an output product, not a
    # namelist-gated one, which is why the native namelist has no
    # counterpart to render from and the certified raw-runtime contract
    # requires the key ABSENT there.  WRF gates the same quantity three
    # deep on this one switch, Registry default 0:
    #
    #   * module_check_a_mundo.F:3477 -- do_radar_ref == 1 sets the
    #     derived compute_radar_ref;
    #   * Registry.EM_COMMON:3059 -- `package radar_refl
    #     compute_radar_ref==1 - state:refl_10cm,refd_max` allocates the
    #     array only then;
    #   * module_mp_thompson.F:1450 -- `if (diagflag .and.
    #     do_radar_ref == 1)` computes it, and solve_em.F:365 raises
    #     diagflag on every history step with ke_diag at full column
    #     depth, which is the column maximum the registration scores.
    #
    # At the default, under a scheme that is not Milbrandt/NSSL, the
    # array is never allocated: every history frame comes out missing
    # REFL_10CM with all other variables intact -- what the first
    # WRF-arm scoring attempt measured.  A SCALAR:
    # Registry.EM_COMMON:2447 declares it `namelist,physics` nentries 1,
    # run/README.namelist:1057 spells it without the `(max_dom)` its
    # per-domain neighbours there carry, and the Fortran above tests it
    # without a domain index.
    radar_ref = " do_radar_ref                        = 1," if stock else ""

    lines = [
        "! Generated by `gpuwm domain --source hrrr`.  This is the "
        + ("stock-WRF" if stock else "native gpuwm")
        + " half of the pair;",
        "! the two differ only by " + _STOCK_DELTAS + ", which the "
        "HRRR hierarchy",
        "! route verifies key for key before it prepares anything.",
        "&time_control",
        " run_days                            = 0,",
        f" run_hours                           = {hours},",
        f" run_minutes                         = {minutes},",
        f" run_seconds                         = {seconds},",
    ]
    lines.append(dt_columns(start, "start").rstrip("\n"))
    lines.append(dt_columns(end, "end").rstrip("\n"))
    lines.extend([
        f" interval_seconds                    = "
        f"{FORCING_INTERVAL_SECONDS},",
        f" input_from_file                     = "
        f"{_repeated('.true.', count)}",
        f" history_interval_s                  = "
        f"{_column(int(d.history_interval_s) for d in domains)}",
        f" frames_per_outfile                  = {_repeated(1, count)}",
        " restart                             = .false.,",
        " io_form_history                     = 2,",
        " io_form_restart                     = 2,",
        " io_form_input                       = 2,",
        " io_form_boundary                    = 2,",
        "/",
        "",
        "&domains",
    ])
    for key, value in clock.items():
        lines.append(f" {key:<35} = {value},")
    lines.extend([
        f" max_dom                             = {count},",
        f" e_we                                = "
        f"{_column(r.nx + 1 for r in runs)}",
        f" e_sn                                = "
        f"{_column(r.ny + 1 for r in runs)}",
        f" e_vert                              = "
        f"{_column(r.nz + 1 for r in runs)}",
        " eta_levels = " + ("\n".join(eta_rows)),
        f" p_top_requested                     = "
        f"{_f(exp.vertical.p_top)},",
        # &domains, and a SCALAR.  WRF declares this key
        # `namelist,domains` with nentries 1 (Registry.EM_COMMON:2283;
        # run/README.namelist documents it inside the &domains block as
        # a single value), so BOTH halves of the old spelling were
        # fatal: emitted into &dynamics, the Fortran namelist read of
        # that group fails and wrf.exe stops before the first timestep
        # -- measured on the campaign nodes -- and a per-domain column
        # would overrun a scalar namelist object here for the other
        # reason.  The root's value is the tree's: gpuwm/experiment.py
        # refuses hypsometric_opt as a [[domain]] override, so it
        # reaches every domain from [shared] and this loses nothing.
        f" hypsometric_opt                     = "
        f"{root.run.hypsometric_opt},",
        f" dx                                  = "
        f"{_column(_f(exp.dx_exact(d.grid_id)) for d in domains)}",
        f" dy                                  = "
        f"{_column(_f(exp.dx_exact(d.grid_id)) for d in domains)}",
        f" grid_id                             = "
        f"{_column(d.grid_id for d in domains)}",
        f" parent_id                           = "
        f"{_column(d.parent_id for d in domains)}",
        f" i_parent_start                      = "
        f"{_column(d.i_parent_start for d in domains)}",
        f" j_parent_start                      = "
        f"{_column(d.j_parent_start for d in domains)}",
        f" parent_grid_ratio                   = "
        f"{_column(d.parent_grid_ratio for d in domains)}",
        f" parent_time_step_ratio              = "
        f"{_column(d.parent_time_step_ratio for d in domains)}",
        " feedback                            = 0,",
        " smooth_option                       = 0,",
        f" num_metgrid_levels                  = {NUM_METGRID_LEVELS},",
        f" num_metgrid_soil_levels             = "
        f"{NUM_METGRID_SOIL_LEVELS},",
        " sfcp_to_sfcp                        = .true.,",
        "/",
        "",
        "&physics",
        f" mp_physics                          = "
        f"{_column(r.mp_physics for r in runs)}",
        f" ra_lw_physics                       = "
        f"{_column(longwave)}",
        f" ra_sw_physics                       = {_column(shortwave)}",
        f" radt                                = "
        f"{_column(_f(r.radt) for r in runs)}",
        " icloud                              = 1,",
        f" swrad_scat                          = "
        f"{_f(root.run.swrad_scat)},",
        f" sf_sfclay_physics                   = {_repeated(91, count)}",
        f" sf_surface_physics                  = {_repeated(2, count)}",
        # Per domain, from the RESOLVED config -- never a literal.  The
        # physics-fidelity axis writes its resolved vector onto every
        # domain's RunConfig (gpuwm/experiment.py), so an arm selecting
        # divergence-ledger entry L3 arrives here already carrying its
        # PBL selector.  A literal 1 shadowed it: the arm resolved to
        # Shin-Hong, the mirrored namelist said YSU, and the root
        # preparation refused the drift it was right to refuse.
        f" bl_pbl_physics                      = "
        f"{_column(r.bl_pbl_physics for r in runs)}",
        f" bldt                                = "
        f"{_column(_f(r.bldt) for r in runs)}",
        f" cu_physics                          = {_repeated(0, count)}",
        f" cudt                                = "
        f"{_column(_f(r.cudt_minutes) for r in runs)}",
        " isfflx                              = 1,",
        " ifsnow                              = 1,",
        " surface_input_source                = 1,",
        " num_soil_layers                     = 4,",
        " num_land_cat                        = 21,",
        f" sf_urban_physics                    = {_repeated(0, count)}",
        " sst_update                          = 0,",
    ])
    # The two stock-only &physics keys, together: each is a setting the
    # native arm answers in code and the mirrored arm can only be told.
    if ghg:
        lines.append(ghg.rstrip("\n"))
    if radar_ref:
        lines.append(radar_ref)
    lines.extend([
        "/",
        "",
        "&fdda",
        "/",
        "",
        "&dynamics",
        f" use_theta_m                         = {theta_m},",
        f" hybrid_opt                          = "
        f"{exp.vertical.hybrid_opt},",
        f" etac                                = "
        f"{_f(exp.vertical.etac)},",
        f" top_lid                             = "
        f"{_repeated(_logical(root.run.top_lid), count)}",
        f" w_damping                           = {root.run.w_damping},",
        # Per domain, explicitly.  A scalar `epssm = 0.5` assigns d01
        # only and leaves every nest on WRF's Registry default of 0.1 --
        # which is how a nested run over steep terrain came to lose its
        # vertical-acoustic off-centering exactly where it needed it.
        f" epssm                               = "
        f"{_column(_f(r.epssm) for r in runs)}",
        f" diff_opt                            = {_repeated(2, count)}",
        f" km_opt                              = "
        f"{_column(r.km_opt for r in runs)}",
        f" mix_full_fields                     = "
        f"{_repeated('.true.', count)}",
        f" diff_6th_opt                        = "
        f"{_column(r.diff_6th_opt for r in runs)}",
        f" diff_6th_factor                     = "
        f"{_column(_f(r.diff_6th_factor) for r in runs)}",
        f" diff_6th_slopeopt                   = "
        f"{_column(r.diff_6th_slopeopt for r in runs)}",
        # WRF's own moist-filter switch (Registry.EM_COMMON:2889,
        # max_domains, default .false.; divergence-ledger entry L4).
        # Rendered explicitly, per domain, so the mirrored WRF arm runs
        # the same moist-filter policy the config declares -- an omitted
        # key would silently hand the WRF arm the Registry default while
        # the ArWen arm read the TOML.
        f" moist_mix6_off                      = "
        f"{_column(_logical(r.moist_mix6_off) for r in runs)}",
        f" c_s                                 = "
        f"{_column(_f(r.c_s) for r in runs)}",
        f" diff_6th_thresh                     = "
        f"{_column(_f(r.diff_6th_thresh) for r in runs)}",
        f" base_temp                           = "
        f"{_f(root.run.base_temp)},",
        f" damp_opt                            = {root.run.damp_opt},",
        f" zdamp                               = "
        f"{_column(_f(r.zdamp) for r in runs)}",
        f" dampcoef                            = "
        f"{_column(_f(r.dampcoef) for r in runs)}",
        f" khdif                               = "
        f"{_column(_f(r.khdif) for r in runs)}",
        f" kvdif                               = "
        f"{_column(_f(r.kvdif) for r in runs)}",
        f" non_hydrostatic                     = "
        f"{_repeated('.true.', count)}",
        f" moist_adv_opt                       = {_repeated(1, count)}",
        f" scalar_adv_opt                      = {_repeated(1, count)}",
        f" time_step_sound                     = "
        f"{_column(r.time_step_sound for r in runs)}",
        f" smdiv                               = "
        f"{_column(_f(r.smdiv) for r in runs)}",
        f" emdiv                               = "
        f"{_column(_f(r.emdiv) for r in runs)}",
        f" h_sca_adv_order                     = "
        f"{_column(r.h_sca_adv_order for r in runs)}",
        "/",
        "",
        "&bdy_control",
        f" spec_bdy_width                      = "
        f"{root.run.spec_bdy_width},",
        f" spec_zone                           = {root.run.spec_zone},",
        f" relax_zone                          = {root.run.relax_zone},",
        f" spec_exp                            = "
        f"{_f(root.run.spec_exp)},",
        f" specified                           = "
        f"{_column(_logical(r.specified) for r in runs)}",
        f" nested                              = "
        f"{_column(_logical(r.nested) for r in runs)}",
        "/",
        "",
        "&grib2",
        "/",
        "",
        "&namelist_quilt",
        " nio_tasks_per_group                 = 0,",
        " nio_groups                          = 1,",
        "/",
        "",
    ])
    text = "\n".join(lines)
    verify_axis_authored_keys(exp, text, stock=stock)
    return text


def route_input_paths(config_path: Path) -> dict[str, Path]:
    """Where :func:`write_hrrr_route_inputs` puts each file.

    Named once, here, so the writer, the printed next steps and the
    tests cannot drift apart.
    """
    stem = config_path.stem
    parent = config_path.parent
    return {
        "wps_namelist": parent / f"{stem}.namelist.wps",
        "target_domain": parent / f"{stem}.d01-target.json",
        "namelist_input": parent / f"{stem}.namelist.input",
        "stock_namelist_input": parent / f"{stem}.stock.namelist.input",
    }


def verify_round_trip(exp, wps_namelist: Path, namelist_input: Path) -> None:
    """Re-import what was just written and demand the same experiment.

    The route reads these namelists, not the TOML, so a set of files
    that describes a different tree than the TOML beside it is a defect
    that only surfaces after a fetch and a root preparation.  Running
    the REAL importer over the REAL bytes at emission is the only check
    that cannot drift from what the route will do.
    """
    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.prepared_cache import (
        effective_prepared_domain_config, prepared_domain_config_identity)
    from gpuwm.namelist_import import import_namelists
    import tomllib

    # The TOML's governance declarations ([experiment].acknowledgements)
    # have no namelist spelling, so the round trip inherits them from
    # the authoritative experiment rather than failing the very guard
    # the TOML beside these files already satisfies.
    #
    # WHICH RRTMG is the same class of fact, and it was missed.  A WRF
    # namelist spells ``ra_lw_physics = ra_sw_physics = 4`` for both the
    # legacy-RRTMG transcription and the RTE+RRTMGP substitution -- the
    # selection between them is gpuwm's, carried in ``ra_rrtmg_variant``,
    # and WRF has no key for it.  Re-importing without saying so resolved
    # every 4/4 emission to RTE+RRTMGP and then reported the difference
    # as though the emitted files were wrong, which made the route's own
    # legacy-RRTMG profiles unemittable.  The route already inherits this
    # out of band everywhere else it matters
    # (:func:`gpuwm.hrrr_hierarchy_direct._native_experiment` takes it
    # from the prepared-cache identity header); this is the same
    # inheritance at emission.
    variant = getattr(exp.root.run, "ra_rrtmg_variant", None)
    text, _report = import_namelists(
        wps_namelist, namelist_input, name=exp.name,
        acknowledgements=tuple(exp.acknowledgements),
        **({} if variant is None else {"rrtmg_variant": variant}))
    imported = build_experiment(
        tomllib.loads(text),
        source=f"round trip of {namelist_input.name}")

    def identity(candidate):
        return [
            effective_prepared_domain_config(
                prepared_domain_config_identity(domain))
            for domain in candidate.domains]

    differences = []
    if len(imported.domains) != len(exp.domains):
        differences.append(
            f"domain count {len(imported.domains)} != {len(exp.domains)}")
    else:
        for grid_id, (left, right) in enumerate(
                zip(identity(exp), identity(imported)), start=1):
            left_run = left.get("run", {})
            right_run = right.get("run", {})
            for key in sorted(set(left_run) | set(right_run)):
                if left_run.get(key) != right_run.get(key):
                    differences.append(
                        f"d{grid_id:02d} {key}: config "
                        f"{left_run.get(key)!r} vs namelist "
                        f"{right_run.get(key)!r}")
    if imported.start_time != exp.start_time:
        differences.append(
            f"start_time {imported.start_time} != {exp.start_time}")
    if float(imported.run_seconds) != float(exp.run_seconds):
        differences.append(
            f"run_seconds {imported.run_seconds} != {exp.run_seconds}")
    if differences:
        raise HrrrRouteInputError(
            "the emitted HRRR namelists do not reproduce the emitted "
            "config -- refusing to write a set the route would read "
            "differently from the TOML beside it: "
            + "; ".join(differences[:8])
            + ("" if len(differences) <= 8
               else f" (+{len(differences) - 8} more)"))


def write_hrrr_route_inputs(config_path: Path, exp, *, wps_text: str,
                            writer) -> list[Path]:
    """Write every file the nested HRRR route needs beside the config.

    ``writer`` is the caller's atomic write function so the whole set
    lands the same way the config does.  Returns the paths written, in
    the order a reader meets them.
    """
    paths = route_input_paths(Path(config_path))
    writer(paths["wps_namelist"], wps_text)
    writer(paths["target_domain"], render_target_domain(exp))
    writer(paths["namelist_input"], render_namelist_input(exp))
    writer(paths["stock_namelist_input"],
           render_namelist_input(exp, stock=True))
    verify_round_trip(exp, paths["wps_namelist"], paths["namelist_input"])
    return [paths["wps_namelist"], paths["target_domain"],
            paths["namelist_input"], paths["stock_namelist_input"]]


__all__ = [
    "FORCING_INTERVAL_SECONDS",
    "HrrrRouteInputError",
    "REQUIRED_PHYSICS",
    "ROUTE_DEFAULT_PHYSICS_PROFILE",
    "SUPPORTED_MICROPHYSICS",
    "render_namelist_input",
    "render_target_domain",
    "route_input_paths",
    "route_physics_blocker",
    "route_physics_problems",
    "validate_route_physics",
    "verify_axis_authored_keys",
    "verify_round_trip",
    "write_hrrr_route_inputs",
]
