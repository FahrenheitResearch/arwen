"""Author a GFS 12/3 km moving-nest setup from an explicitly selected center.

The map request reads only published f000 MSLP and wind fields. Configuration
creation reuses the ordinary domain author, physics suite and vortex tracker;
neither entry point starts a forecast or computes a new tracking algorithm.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

#: v2, and the bump is the point.  v1 documents were a map or a
#: configuration; a v2 result can also be a PROPOSAL -- `kind` gains
#: "proposal", the result gains `fitting` and `created`, and `--out` on
#: a proposal exits 0 having written nothing until `--accept-fit` names
#: the reviewed fit.  A caller that reads exit 0 plus `--out` as "the
#: file is there" is correct under v1 and wrong under v2, and the
#: version string is the only part of the document such a caller is
#: guaranteed to look at.  Left at v1 the change would have been silent.
SCHEMA = "arwen.cyclone-setup.v2"
ROOT_DIMS = (200, 160)
CHILD_DIMS = (160, 160)
ROOT_DX_M = 12000.0
RATIO = 4
# Five-percent, aspect-preserving rungs. Both child axes stay divisible by
# 2*RATIO, so the shared author can center each nest on whole parent cells.
FIT_SCALE_STEPS = 20


def _cycle(raw: str, *, latest: bool = False) -> datetime:
    from gpuwm import domain_wizard as dw
    text = raw.strip()
    if text.lower() == "latest" and not latest:
        raise ValueError("Select a center on a resolved GFS map first; creation requires that map's exact cycle")
    if len(text) == 10 and text.isdecimal():
        text = datetime.strptime(text, "%Y%m%d%H").strftime("%Y-%m-%dT%H")
    result = dw._resolve_cycle(text, source="gfs", hours=0)
    if result.hour not in (0, 6, 12, 18) or result.minute or result.second:
        raise ValueError("GFS initialization must name a published 00/06/12/18 UTC cycle")
    return result


def latest_map(cycle: str = "latest") -> dict:
    moment = _cycle(cycle, latest=True)
    return {
        "schema": SCHEMA, "kind": "map", "cycle": moment.strftime("%Y%m%d%H"),
        "map_request": {"source": "gfs", "date": moment.strftime("%Y-%m-%d"),
                        "hour": moment.hour, "forecast_hour": 0, "member": 0,
                        "product": "mslp_10m_winds", "bounds": [-85., -180., 85., 180.]},
        "forecast_started": False,
        "selection": "Click the circulation center on this exact GFS f000 pressure-and-wind map.",
    }


def configuration_text(*, cycle: str, point: tuple[float, float], hours: int = 6,
                       name: str = "GFS cyclone 12 km to 3 km", tiles: str = "auto",
                       source: str = "cyclone-setup.toml",
                       dimensions=None) -> tuple[str, object]:
    from gpuwm import domain_wizard as dw
    from gpuwm.companion_domains import VORTEX_PRESET, VORTEX_PRESET_SOURCE
    from gpuwm.starter_template import render_tables

    moment = _cycle(cycle)
    if type(hours) is not int or not 1 <= hours <= 384:
        raise ValueError("Cyclone duration must be an integer from 1 to 384 hours")
    if tiles not in ("off", "auto", "on"):
        raise ValueError("Tile mode must be off, auto or on")
    lat, lon = point
    if not all(math.isfinite(v) for v in point) or not -85 <= lat <= 85 or not -180 <= lon <= 180:
        raise ValueError("Select a finite center on the displayed GFS map")
    dims = [ROOT_DIMS, CHILD_DIMS] if dimensions is None else list(dimensions)
    if (len(dims) != 2 or any(len(pair) != 2 for pair in dims)
            or any(type(n) is not int or n <= 0 for pair in dims for n in pair)):
        raise ValueError("Cyclone dimensions must be two positive integer axis pairs")
    projection = dw._projection_entries(lat, lon, "auto")
    area = dw.fetch_area_hint(projection, *dims[0], source="gfs", root_dx_m=ROOT_DX_M)
    profile = dw.resolved_physics_profile("gfs", None)
    text = dw.render_config(
        name=name, start_time=moment, hours=hours, projection=projection,
        dims=dims, ratios=(RATIO,), root_dx_m=ROOT_DX_M,
        profile=profile, cumulus_requested=False, tiles=tiles,
        fetch_hints={"source": "gfs", "cycle": moment.strftime("%Y-%m-%dT%H"),
                     "hours": max(3, math.ceil(hours / 3) * 3), "cadence": 3,
                     "area": area, "out": f"data/gfs-cyclone-{moment:%Y%m%d%H}"},
        case_data=None, history_interval_s=3600., nest_history_interval_s=900.)
    raw = tomllib.loads(text)
    child = next(row for row in raw["domain"] if row["grid_id"] == 2)
    child["follow"] = dict(VORTEX_PRESET)
    # This is one immediate following nest. Spawn/retire decisions are not part
    # of this quick-start; the chosen center is its initial registration.
    text = ("# GFS cyclone quick-start: 12 km parent and 3 km following nest.\n"
            "# Center and cycle were selected explicitly on a GFS f000 map.\n"
            f"# Existing vortex-lock preset: {VORTEX_PRESET_SOURCE}\n"
            "# Following uses the 850 hPa circulation; the selection map uses MSLP.\n"
            + render_tables(raw))
    experiment = dw.experiment_from_text(text, source=source)
    return text, experiment


def _fit_dimensions(scale):
    from gpuwm import domain_wizard as dw
    return [tuple(dw._even(n * scale) for n in ROOT_DIMS),
            tuple(RATIO * dw._even(n * scale / RATIO) for n in CHILD_DIMS)]


def _fit_scales(experiment):
    from gpuwm.companion_domains import VORTEX_PRESET
    # Keep the whole tracker search window clear of the boundary/blend zone
    # even after one maximum requested move. Runtime still enforces overlap,
    # movement bounds and containment on EVERY actual relocation.
    margin = (experiment.spec_bdy_width + experiment.blend_width
              + VORTEX_PRESET["search_margin_cells"]
              + max(VORTEX_PRESET["max_shift_cells"],
                    VORTEX_PRESET["max_move_parent_cells"]))
    scales = tuple(step / FIT_SCALE_STEPS for step in range(FIT_SCALE_STEPS - 1, 0, -1)
                   if all((parent - child // RATIO) // 2 >= margin
                          for parent, child in zip(*_fit_dimensions(step / FIT_SCALE_STEPS))))
    if not scales:
        # NAMED, because the alternative is fit_ladder's internal contract
        # message ("candidate_scales must be a tuple of 1..64 decreasing
        # positive finite scales") reaching a reader who never chose a
        # scale ladder.  What breaks: every rung of this ladder puts the
        # tracker's search window inside the parent's boundary/blend zone,
        # so a following nest could relocate into cells the parent does
        # not integrate.  The way out is the requested domain itself.
        raise ValueError(
            "No smaller cyclone layout keeps the following nest's tracker "
            f"search window {margin} cells clear of the 12 km parent's "
            "boundary and blend zone, so there is nothing to propose; run "
            "the requested domain with --tiles off, or use a larger card")
    return scales


#: The two budgets these floors are compared against are NOT the same
#: number, and the direction is what makes that safe.  The streaming
#: floor is the term `decide_tree` compares against its own, TIGHTER
#: tree budget (`_tree_budget_bytes`); `budget` here is the wizard's
#: `sizing_budget_bytes`, which is larger.  So a floor that exhausts THIS
#: budget certainly exhausts the tree's: the refusal cannot fire falsely
#: on the loose comparison, only stay silent where the tighter one would
#: have spoken -- and staying silent costs a bounded search, not a wrong
#: answer.  Reversing the direction (or swapping in the tree budget
#: without saying so) would turn it into a false "resizing cannot help".
def _fixed_floors(phases, budget, tiles):
    # Read the estimator/planner's own lower bounds, not a second byte model.
    forecast = getattr(phases, "forecast", None)
    resident = getattr(forecast, "fixed_envelope_bytes", None)
    streamed = getattr(getattr(phases, "tree_road", None),
                       "streaming_fixed_floor_bytes", None)
    required = ([resident] if tiles == "off" else [streamed] if tiles == "on"
                else [resident, streamed])
    # A STREAMING floor alone cannot rule out a smaller all-resident AUTO tree.
    impossible = all(floor is not None and floor >= budget for floor in required)
    return impossible, {"resident_fixed_floor_bytes": resident,
                        "streaming_fixed_floor_bytes": streamed}


#: What the REQUESTED, unreduced cyclone tree costs when it is asked for
#: as one resident allocation (``--tiles off``), and whether the declared
#: card admits it.
#:
#: The reduction this door performs is a SCIENCE reduction: a 2,400 x
#: 1,920 km 12 km parent is there to carry the steering environment, and
#: 1,560 x 1,248 km does not.  On the 6-8 GiB band the tile planner's
#: tree road refuses layouts the same card holds resident, so the door
#: was proposing degraded coverage -- or refusing outright -- while a
#: mode the user can select ran the domain they asked for, and never
#: said so.  It is REPORTED, never applied: streaming is off by choice
#: and a door that silently switched a memory mode would be lying about
#: what it emitted.
#:
#: Priced by the same estimator, on the same operands, against the same
#: budget, and through the SAME one hardware snapshot as every other
#: candidate here -- nothing is redetected and no number is inflated to
#: make the resident route look admissible.  The operands and the machine
#: reach this function only inside ``budget_of`` and ``price_off``, which
#: is the point: there is no second set of them to get wrong, and no
#: argument here a caller could vary to change what is priced.  A
#: resident experiment does not consult the tile planner at all, so the
#: snapshot is carried only so that this probe cannot become a second,
#: differently-measured machine.
#: ``None`` means the unreduced request is not admitted that way either,
#: and there is nothing to name.
def _unreduced_resident_admission(intent, budget_of, price_off):
    """Price the unreduced request with ``--tiles off``; None if refused."""
    if intent["tiles"] == "off":
        return None
    try:
        _text, exp = configuration_text(**{**intent, "tiles": "off"})
        phases = price_off(exp)
        budget = budget_of(exp)
    except (ValueError, OSError):
        # Not an admission answer -- the resident route could not even be
        # priced.  Claim nothing; the caller's own refusal stands as it is.
        return None
    if phases.peak_envelope_bytes > budget:
        return None
    return {"tiles": "off", "dimensions": [list(ROOT_DIMS), list(CHILD_DIMS)],
            "peak_envelope_bytes": phases.peak_envelope_bytes,
            "budget_bytes": budget}


#: What `--tiles off` actually authors when the resident route cannot
#: hold the requested domain either.  Asked ONLY on the refusal path,
#: where the door is about to hand back nothing at all: a refusal that
#: names no way through is the defect the refusal law is about, and
#: "try --tiles off" without a layout behind it is a suggestion, not an
#: answer.  So it is measured -- the same bounded ladder, the same
#: estimator, the same budget -- and reported with the dimensions it
#: found.  It is a REDUCTION too, and says so; it is not silently
#: applied, and the requested mode is what the caller asked for.
#: It answers TWO questions, so it returns two things.  ``found`` is the
#: layout, for the refusal that names a way through.  ``measured`` says
#: whether the resident ladder was actually WALKED -- because a refusal
#: with no layout is either "the resident route was priced and admits
#: nothing on this card" (a measurement, and the one thing that makes
#: "the computer cannot admit" a true sentence for a tiled request) or
#: "the resident route could not be priced at all" (no evidence, claim
#: nothing).  A memory-typed refusal out of the ladder IS the first:
#: every rung was priced and every rung was refused.  Any other failure
#: -- an unloadable candidate, a coverage or extent bound, an OSError --
#: is the second, and does not license a claim about the card.
def _resident_alternative(intent, sizing, dims_of, scales_of):
    """``(measured, found)``: the largest resident rung `--tiles off`
    admits, and whether the resident route was priced to find out."""
    from gpuwm import domain_wizard as dw

    if intent["tiles"] == "off":
        return False, None
    resident = dict(intent, tiles="off")
    try:
        _text, exp = configuration_text(**resident)
        dims, _fitted = dw.fit_ladder(
            ratios=(RATIO,), free_bytes=sizing.free_bytes,
            vram_gib=sizing.vram_gib, device_profile=sizing.device_profile,
            target_machine=None, hours=intent["hours"],
            start_time=_cycle(intent["cycle"]),
            projection=tomllib.loads(_text)["projection"], source="gfs",
            name=intent["name"], root_dx_m=ROOT_DX_M,
            profile=dw.resolved_physics_profile("gfs", None), tiles="off",
            forcing_interval_seconds=10800.,
            candidate_builder=lambda proposed: configuration_text(
                **resident, dimensions=proposed)[1],
            dimensions_builder=dims_of, candidate_scales=scales_of(exp),
            layout_label="cyclone 12/3 km resident")
    except dw.DomainFitError as error:
        return error.resource in {"vram", "host", "memory"}, None
    except (ValueError, OSError):
        return False, None
    return True, {"tiles": "off", "dimensions": [list(pair) for pair in dims]}


def _resident_alternative_sentence(found) -> str:
    dims = found["dimensions"]
    return (f"--tiles off is not refused here: it authors "
            f"{dims[0][0]}x{dims[0][1]} / {dims[1][0]}x{dims[1][1]} on this "
            "computer as one resident allocation.  That is smaller ground "
            "than was requested, so review it as a reduction")


#: The way out when there is no tile mode left to try.  Said only after
#: the resident ladder was walked and admitted nothing, so it is a
#: measurement and not a guess: the refusal that carries it has already
#: named the computer as the bound, and this is the sentence that says
#: what would move it.  Without it the hardware refusal on this band
#: ended at "no smaller candidate passed" -- true, and no way out.
def _no_resident_sentence(budget) -> str:
    return ("--tiles off is not a way through here either: the same "
            "bounded ladder, priced resident on this same computer, "
            f"admitted no layout under its {budget} byte budget, so what "
            "this needs is more free VRAM -- a larger card, or this one "
            "to itself -- not a different tile mode")


#: The way out of the refusal that has NO layout to point at.
#:
#: ``impossible`` means the estimator's own fixed floors -- the process
#: and radiation costs a run pays before one grid cell is stored --
#: already reach the budget, so no rung of any ladder is walked and there
#: is nothing measured to offer.  The refusal law still wants a way out,
#: and this was the one refusal on this door that named none: the HARDER
#: refused case got less guidance than the softer one beside it, which
#: names "more free VRAM -- a larger card, or this one to itself".
#:
#: What is said here is what the floors already on the payload show, and
#: nothing further.  The floor is compared against the BUDGET, so more
#: free VRAM moves the comparison; the floor IS the selected physics'
#: fixed cost, so a lighter selection moves the floor; the grid moves
#: neither, which is what the head sentence already says.  Where the
#: other memory mode's floor was measured and sits UNDER the budget it is
#: named as a floor that is not exhausted -- never as a layout, because
#: on this path no layout was priced.
def _fixed_floor_way_out(floors, budget, tiles) -> str:
    other, mode = (("resident_fixed_floor_bytes", "off") if tiles == "on" else
                   ("streaming_fixed_floor_bytes", "on") if tiles == "off" else
                   (None, None))
    value = floors.get(other) if other else None
    unexhausted = (
        f"; --tiles {mode}'s own fixed floor is {value} bytes, under that "
        "budget, so its floor is not what is exhausted here -- but no "
        "layout was priced on that road, and none is offered"
        if value is not None and value < budget else "")
    return ("What moves this is the budget or the floor itself, never the "
            f"grid: more free VRAM raises the {budget} byte budget -- a "
            "larger card, or this one to itself -- and a lighter physics "
            "selection lowers the fixed cost" + unexhausted)


#: Why the proposal stopped where it did, when something other than the
#: card stopped it.  The reason string is the fitter's own -- the same
#: one `stop_out` carries to the wizard's plan summary -- so this door
#: and that one cannot disagree about what bound a fit.  Stated as fact
#: on the document, never as a warning, and it names no flag this door
#: does not have.
def _stopped_by_sentence(stopped) -> str:
    return (f"The next larger layout was rejected on the {stopped['scope']}, "
            f"not the card: {stopped['reason']}")


def _keeps_coverage_sentence(admitted) -> str:
    """The one sentence that names ``--tiles off`` as the way to keep the
    requested ground, with the numbers that make it a claim."""

    dims = admitted["dimensions"]
    return (f"--tiles off admits the requested {dims[0][0]}x{dims[0][1]} / "
            f"{dims[1][0]}x{dims[1][1]} domain on this computer as one "
            f"resident allocation ({admitted['peak_envelope_bytes']} bytes "
            f"against a {admitted['budget_bytes']} byte budget); re-run with "
            "--tiles off to keep the requested coverage")


def plan_cyclone(*, cycle: str, point: tuple[float, float], sizing, target_machine=None,
                 hours: int = 6, name: str = "GFS cyclone 12 km to 3 km",
                 tiles: str = "auto", source: str = "cyclone-setup.toml",
                 cancelled=None) -> dict:
    from gpuwm import domain_wizard as dw
    from gpuwm.companion_domains import VORTEX_PRESET, VORTEX_PRESET_SOURCE
    from gpuwm.configuration_recovery import MemoryAdmissionError
    from gpuwm.starter_template import changes

    dw.check_fit_cancelled(cancelled)
    intent = dict(cycle=cycle, point=point, hours=hours, name=name, tiles=tiles, source=source)
    original_text, experiment = configuration_text(**intent)
    text = original_text
    # One hardware snapshot for the whole search. Never redetect/inflate VRAM
    # or force a different streaming mode to make a candidate appear to fit.
    if tiles != "off" and target_machine is None:
        from gpuwm.core.streaming import planner_machine
        target_machine = planner_machine(vram_bytes=sizing.free_bytes,
                                         name="gpuwm cyclone budget")
        if target_machine is None:
            # Same breakage and same way out the shared planner names ten
            # lines into dw._sizing_phases -- said once, in one wording,
            # and typed so it leaves plan_cyclone with a memory payload
            # instead of as an untyped ValueError with no resource.
            raise dw.DomainFitError(
                f"--tiles {tiles}: --tiles needs host RAM available to the "
                "shared planner; run the wizard on the forecast host or use "
                "--tiles off",
                resource="host")
        from dataclasses import replace
        # The planner machine carries the card that was MEASURED, not the
        # reference card, so the four sites in gpuwm/core/streaming.py that
        # fall back to machine.device_profile cannot price one card's
        # shader count as another's.
        #
        # MEASURED on an RTX 5070 Ti (70 SM, Linux, CUDA 13), on this
        # door's own 12/3 km cyclone tree, at two real card states --
        # 15.28 GiB free and 5.56 GiB free, the band where the tree road
        # binds, the second produced by holding 9.5 GiB on the card.  What was
        # measured: every number _sizing_phases produces (phase peak,
        # binding phase, the forecast non-pool/intercept/column-workspace/
        # fixed-envelope/subtotal/alloc terms, and the tree road's priced
        # flag, refusal, resource and streaming fixed floor), under three
        # planner machines identical but for this field: none, the measured
        # profile, and a deliberately absurd 999-SM profile.  All three are
        # byte-identical at both card states, including the refusal string
        # and the 6,978,986,310 B streaming fixed floor.  Calibration of
        # the instrument: the absurd arm is the positive control -- priced
        # through the estimator directly, 999 SMs move the same term from
        # 1,181,036,544 B to 7,214,964,736 B, so an arm that reached this
        # field could not have come back equal.
        #
        # So this line moves no admission threshold on this door:
        # _sizing_phases builds the resident estimate itself with
        # profile=sizing.device_profile and hands it to all four sites,
        # where machine.device_profile is only the fallback for a missing
        # estimate.  It is kept because that fallback, if a future caller
        # reaches it, must price this card -- 1,181,036,544 B here against
        # the reference card's 2,322,194,432 B, 1,088.3 MiB apart on the
        # one term shrinking the grid cannot move.
        target_machine = replace(target_machine, device_profile=sizing.device_profile)
    operands = dict(free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                    profile=sizing.device_profile, forcing_interval_seconds=10800.)
    budget = dw.sizing_budget_bytes(experiment, **operands)

    def price(exp):
        dw.check_fit_cancelled(cancelled)
        phases = dw._sizing_phases(exp, source="gfs", machine=target_machine, **operands)
        dw.check_fit_cancelled(cancelled)
        return phases

    # What stopped the SEARCH, carried out of the fitter rather than
    # re-derived from the emitted grid.  A cyclone fit is a POINT fit on
    # the shared bounded road, so the same bounds apply to it as to the
    # wizard's own point door: a high-latitude centre grows a 12 km parent
    # toward the projection pole and the fit shrinks off it, and a source
    # window can stop the search below what the card affords.  Without
    # this the door emitted a smaller grid than the budget allows and said
    # nothing about why -- the invisible saturation the plan summary
    # exists to prevent.  Reported on the document, which is this door's
    # only surface: its stdout is the JSON result and everything else is
    # redirected to stderr.
    fit_stop: dict = {}
    # The bounds the REQUEST carries, asked of the requested layout
    # itself (see below) as well as of every rung the fit tries.
    request_bound = dw.point_request_bound(
        tomllib.loads(original_text)["projection"], *ROOT_DIMS, ROOT_DX_M)
    refusal = None
    try:
        phases = price(experiment)
        if phases.peak_envelope_bytes > budget:
            refusal = dw.DomainFitError(phases.verdict(budget), resource="vram", phases=phases)
    except dw.DomainFitError as error:
        dw.check_fit_cancelled(cancelled)
        if error.resource not in {"vram", "host", "memory"}:
            raise
        refusal = error
        phases = error.phases

    if refusal is None and request_bound is not None:
        # The card is not the only thing that decides how big a domain
        # grown from a single point may be, and on this door it was the
        # only thing that did: the REQUESTED 200x160 was authored without
        # ever facing the bounds the fit below is bounded by.  Above
        # about 81 N that emitted a 12 km parent whose footprint contains
        # the projection pole -- where lat-lon source interpolation and
        # static-tile windowing are not pole-capable -- while the SAME
        # door, one gigabyte lower, refused a SMALLER pole-reaching rung
        # for exactly that reason.  One door, two answers, and the larger
        # one could not be prepared.  So the request faces the bound
        # first, and a request the bound stops is a reduction like any
        # other: the ladder shrinks off the pole and the proposal says
        # why.
        refusal = dw.DomainFitError(request_bound[1], resource="extent",
                                    phases=phases)

    reduced = refusal is not None
    memory_bound = reduced and refusal.resource in {"vram", "host", "memory"}
    keeps_coverage = None
    if reduced:
        # Asked BEFORE anything is proposed or refused, so both the
        # proposal's notice and the refusal's text can name it.  Only for
        # a MEMORY refusal AND only when the REQUESTED layout faces no
        # request bound of its own: `--tiles off` moves where the bytes
        # land, not where the domain sits, so offering it against a
        # request bound names a remedy that cannot help.
        #
        # Both tests, not either: a request can be bound BOTH ways at
        # once.  The memory refusal is raised first and leaves
        # `refusal.resource` reading `vram`, so the memory test alone
        # sent the UNREDUCED 200x160 through the estimator, saw it
        # admitted resident, and said so -- in the same notice that then
        # reported the projection had rejected a SMALLER root for
        # reaching the pole.  One notice, two contradictory claims, and
        # the named remedy did not do what it said: `--tiles off` at that
        # point and budget authors the reduced layout auto had already
        # proposed, because the pole bound stops the resident ladder in
        # exactly the same place.  Reproduced on the real door at
        # --point=82,-20 --vram-gib 8.45.  `request_bound` above is asked
        # of the requested dimensions, which is the layout this admission
        # would be naming, so it is the test that belongs here.
        keeps_coverage = _unreduced_resident_admission(
            intent,
            lambda exp: dw.sizing_budget_bytes(exp, **operands),
            lambda exp: dw._sizing_phases(exp, source="gfs",
                                          machine=target_machine, **operands)
        ) if memory_bound and request_bound is None else None
        # Measured once, on the refusal path only, and memoised: a refusal
        # has to name a way through, and the way through has to be a
        # layout somebody measured rather than a mode somebody suggested.
        probed: list = []

        def _probe(exhausted: bool) -> tuple[bool, dict | None]:
            # Nothing to measure when BOTH fixed floors exhaust the budget:
            # the resident ladder is then refused by its own floor before
            # any rung, so a search would cost prices to prove what the
            # floor already said.  `measured` stays False there because
            # this probe did not run -- but `exhausted` is itself the
            # measurement that the resident route admits nothing, and the
            # caller reads it directly.
            if not probed:
                probed.append(
                    (False, None) if keeps_coverage or exhausted else
                    _resident_alternative(intent, sizing,
                                          _fit_dimensions, _fit_scales))
            return probed[0]

        def _alternative(exhausted: bool):
            return _probe(exhausted)[1]

        def _way_through(exhausted: bool, floors: dict) -> str:
            if keeps_coverage:
                return f"  {_keeps_coverage_sentence(keeps_coverage)}."
            measured, found = _probe(exhausted)
            if found:
                return f"  {_resident_alternative_sentence(found)}."
            if measured and tiles != "off":
                return f"  {_no_resident_sentence(budget)}."
            if exhausted:
                # The refusal law's harder half: nothing was priced, so
                # nothing is offered, but what would move this IS known.
                return f"  {_fixed_floor_way_out(floors, budget, tiles)}."
            return ""

        impossible, floors = _fixed_floors(phases, budget, tiles)
        if impossible:
            raise MemoryAdmissionError(
                "The selected computer's fixed process/radiation costs exhaust its memory "
                "budget before grid storage; resizing cannot help with the selected physics "
                "and tile mode."
                + _way_through(True, floors),
                reason="fixed-floor", bound_by="computer", budget_bytes=budget,
                keeps_coverage=keeps_coverage,
                resident_alternative=_alternative(True), **floors)
        candidates = {}

        def build(dims):
            proposed_text, exp = configuration_text(**intent, dimensions=dims)
            candidates[tuple(dims)] = proposed_text
            return exp

        try:
            dims, experiment = dw.fit_ladder(
                ratios=(RATIO,), free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                device_profile=sizing.device_profile, target_machine=target_machine,
                hours=hours, start_time=_cycle(cycle),
                projection=tomllib.loads(original_text)["projection"], source="gfs", name=name,
                root_dx_m=ROOT_DX_M, profile=dw.resolved_physics_profile("gfs", None),
                tiles=tiles, forcing_interval_seconds=10800.,
                candidate_builder=build, dimensions_builder=_fit_dimensions,
                candidate_scales=_fit_scales(experiment),
                layout_label="cyclone 12/3 km", cancelled=cancelled,
                stop_out=fit_stop)
            text = candidates[tuple(dims)]
            # Re-admit the final, fully authored moving tree, including the
            # wizard's headroom. No stale candidate or admission bypass.
            phases = price(experiment)
            budget = dw.sizing_budget_bytes(experiment, **operands)
            if phases.peak_envelope_bytes > budget - dw.fit_headroom_bytes(budget):
                raise dw.DomainFitError("The final cyclone proposal no longer fits with headroom",
                                        resource="vram", phases=phases)
        except dw.DomainFitError as error:
            if error.resource not in {"vram", "host", "memory"}:
                raise
            impossible, floors = _fixed_floors(error.phases or phases, budget, tiles)
            # WHOSE refusal this is.  "The selected computer cannot admit"
            # is a claim about HARDWARE, and it was false on exactly the
            # band this door serves: at 6 GiB the resident fixed floor sat
            # a gigabyte and a half under the budget and `--tiles off`
            # authored a real layout, so what refused was the tile
            # planner's tree road for `--tiles auto`, not the card.  A
            # refusal that misnames what refused sends the reader after
            # the wrong remedy -- a bigger card -- and hides the one that
            # works.  The hardware claim is made only when the fixed floor
            # the estimator itself reports actually exhausts the budget.
            resident_floor = floors.get("resident_fixed_floor_bytes")
            # And WHOSE it is not.  The floor comparison alone is an
            # INFERENCE -- "a resident tree could start here" -- and it
            # misattributed in the opposite direction on a band this door
            # reaches in ordinary use: budgets between the resident fixed
            # floor and the smallest rung's actual cost (roughly 3-5 GiB
            # free on this tree, which "measured-available" sizing hands a
            # busy 16 GiB card).  There the floor sits under the budget,
            # so the door blamed the tree road, while `--tiles off` on the
            # same card was refused too.  The measurement that settles it
            # is already computed: the resident ladder is WALKED on this
            # path, and when it admits nothing the computer is the bound.
            # So the tile planner is blamed only when a resident layout
            # was actually found -- either the unreduced request priced
            # resident (`keeps_coverage`) or a rung of the resident ladder
            # (`resident_alternative`).
            #
            # `--tiles off` has no tree road to blame, so a refusal there is
            # the resident route against the card and the original sentence
            # is the true one.  Only a TILED request can be refused by the
            # tile planner while the card still holds a resident tree.
            hardware_bound = impossible or tiles == "off" or not (
                resident_floor is not None and resident_floor < budget) or (
                keeps_coverage is None and _alternative(impossible) is None)
            if hardware_bound:
                head = "The selected computer cannot admit a cyclone proposal: "
                tail = ("; fixed process/radiation costs exhaust the budget, so resizing "
                        "cannot help." if impossible else
                        "; no smaller candidate in this bounded policy passed. Physics, "
                        "duration, halos and movement margins were not reduced.")
            else:
                head = f"No --tiles {tiles} cyclone proposal was admitted: "
                tail = ("; no smaller candidate in this bounded policy passed. Physics, "
                        "duration, halos and movement margins were not reduced. This is "
                        "the tile planner's tree road refusing, not the computer: its "
                        f"resident fixed floor is {resident_floor} bytes against a "
                        f"{budget} byte budget.")
            raise MemoryAdmissionError(
                head + str(error) + tail + _way_through(impossible, floors),
                reason="fixed-floor" if impossible else "bounded-search",
                bound_by="computer" if hardware_bound else f"tiles-{tiles}-tree-road",
                resource=error.resource, budget_bytes=budget,
                keeps_coverage=keeps_coverage,
                resident_alternative=_alternative(impossible), **floors) from error

    dw.check_fit_cancelled(cancelled)
    stopped_by = dict(fit_stop) if fit_stop else None
    changed_fields = [{"field": field, "before": before, "after": after}
                      for field, before, after in changes(tomllib.loads(original_text),
                                                        tomllib.loads(text))]
    return {
        "schema": SCHEMA, "kind": "proposal" if reduced else "configuration",
        "cycle": _cycle(cycle).strftime("%Y%m%d%H"),
        "source": "gfs", "hours": hours, "point": list(point), "tiles": tiles,
        "domains": [{"grid_id": d.grid_id, "parent_id": d.parent_id,
                     "nx": d.run.nx, "ny": d.run.ny, "nz": d.run.nz,
                     "dx_m": d.run.dx, "dy_m": d.run.dy,
                     "following": d.grid_id == 2} for d in experiment.domains],
        "follow": dict(VORTEX_PRESET), "follow_preset_source": VORTEX_PRESET_SOURCE,
        "profile": dw.resolved_physics_profile("gfs", None),
        "memory": {"peak_envelope_bytes": phases.peak_envelope_bytes, "budget_bytes": budget,
                   "binding_phase": phases.binding_phase, "free_bytes": sizing.free_bytes,
                   "fit_headroom_bytes": dw.fit_headroom_bytes(budget) if reduced else 0,
                   "sizing_basis": "measured-available" if sizing.measured else "declared-capacity"},
        "fitting": {"changed": reduced, "review_required": reduced,
                    "fit_id": hashlib.sha256(text.encode()).hexdigest(),
                    "original_dimensions": [list(ROOT_DIMS), list(CHILD_DIMS)],
                    "proposed_dimensions": [[d.run.nx, d.run.ny] for d in experiment.domains],
                    "changes": changed_fields,
                    "reason": str(refusal) if reduced else None,
                    "policy": "largest admitted 5%-scale rung; preserve tracker search and movement clearance",
                    "keeps_coverage": keeps_coverage,
                    "stopped_by": stopped_by,
                    "notice": (("Smaller geographic coverage proposed. Review dimensions and all field "
                                "changes; use --accept-fit FIT_ID to save this exact proposal."
                                + (f"  {_keeps_coverage_sentence(keeps_coverage)}."
                                   if keeps_coverage else "")
                                + (f"  {_stopped_by_sentence(stopped_by)}."
                                   if stopped_by else ""))
                               if reduced else
                               "Original dimensions fit; configuration unchanged.")},
        "config_text": text, "created": False, "forecast_started": False,
    }


def _cancelled(error) -> bool:
    """Is this the author cancelling a fit, rather than a fit failing?

    Resolved late because this module imports the domain wizard lazily,
    and asked of an exception that has already been raised, so the import
    only happens on a path that already loaded it.
    """

    from gpuwm import domain_wizard as dw
    return isinstance(error, dw.DomainFitCancelled)


def main(args) -> int:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.latest_map:
                result = latest_map(args.cycle)
            else:
                from gpuwm import domain_wizard as dw
                from gpuwm.companion_query import inspect_configuration
                from gpuwm.hrrr_prepared_bundle import render_wps_namelist
                from gpuwm.starter_template import _publish_new_files
                if args.point is None:
                    raise ValueError("Select the cyclone center on the resolved GFS map first")
                _cycle(args.cycle)
                sizing, machine, _ = dw._domain_target_hardware(args)
                out = args.out.expanduser().resolve() if args.out else None
                result = plan_cyclone(cycle=args.cycle, point=dw._parse_point(args.point),
                    hours=args.hours, name=args.name, tiles=args.tiles, sizing=sizing,
                    target_machine=machine, source=str(out or "cyclone-setup.toml"))
                acceptance = getattr(args, "accept_fit", None)
                if acceptance is not None and acceptance != result["fitting"]["fit_id"]:
                    raise ValueError("The reviewed fit does not match this proposal; review the new proposal before saving")
                if acceptance is not None:
                    result["fitting"]["review_required"] = False
                    result["kind"] = "configuration"
                if out is not None and not result["fitting"]["review_required"]:
                    if out.suffix.lower() != ".toml":
                        raise ValueError("Save the cyclone configuration as a new .toml file")
                    text = result["config_text"]
                    experiment = dw.experiment_from_text(text, source=str(out))
                    wps = out.with_suffix(".namelist.wps")
                    receipt = out.with_suffix(".cyclone.json")
                    if any(path.exists() for path in (out, wps, receipt)):
                        raise ValueError("Choose a new output path; cyclone setup never overwrites an existing configuration")
                    wps_text = render_wps_namelist(experiment).replace(
                        " interval_seconds = 3600,", " interval_seconds = 10800,")
                    proof = {key: value for key, value in result.items() if key != "config_text"}
                    proof.update(created=True, output=str(out), output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                 wps_sha256=hashlib.sha256(wps_text.encode()).hexdigest())
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _publish_new_files(((wps, wps_text),
                        (receipt, json.dumps(proof, indent=2, allow_nan=False) + "\n"), (out, text)))
                    result["configuration"] = inspect_configuration(out)
                    result.update(created=True, config_path=str(out), receipt_path=str(receipt))
        print(json.dumps(result, allow_nan=False, default=str))
        return 0
    except KeyboardInterrupt:
        print(json.dumps({"schema": SCHEMA, "error": "Cyclone fitting cancelled",
                          "cancelled": True, "created": False, "forecast_started": False}))
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        # ONE cancellation answer for both seams.  The predicate path
        # raises DomainFitCancelled, which IS a RuntimeError, so this
        # handler used to turn a cancelled fit into exit 1 with no
        # `cancelled` key -- indistinguishable from a fit that failed by
        # the only two things a caller reads, the exit code and that key.
        if _cancelled(error):
            print(json.dumps({"schema": SCHEMA, "error": "Cyclone fitting cancelled",
                              "cancelled": True, "created": False,
                              "forecast_started": False}))
            return 130
        result = {"schema": SCHEMA, "error": str(error), "created": False,
                  "forecast_started": False}
        if hasattr(error, "memory"):
            result["memory"] = error.memory
        print(json.dumps(result, allow_nan=False))
        return 1


def register_cli(subparsers):
    from gpuwm.domain_wizard import CARD_VRAM_GIB
    parser = subparsers.add_parser("cyclone-setup", help="select a GFS f000 cyclone and author a 12/3 km following nest")
    parser.add_argument("--latest-map", action="store_true")
    parser.add_argument("--cycle", default="latest")
    parser.add_argument("--point")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--name", default="GFS cyclone 12 km to 3 km")
    parser.add_argument("--tiles", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--hardware-json", type=Path)
    parser.add_argument("--target-host-memory-json", type=Path)
    parser.add_argument("--vram-gib", type=float)
    parser.add_argument("--card", help="a tier (12gb/16gb/24gb/32gb), a size "
                        "('10gb') or a model with a recorded size ('RTX 3080')")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--accept-fit", metavar="FIT_ID",
                        help="save only the exact reviewed proposal identified by fitting.fit_id")
    parser.add_argument("--json", action="store_true", help="emit the JSON result (the default)")
    parser.set_defaults(func=main)
