"""Derive the hybrid coordinate the run's own ground can order.

THE DEFECT THIS EXISTS FOR.  WRF's cubic hybrid coordinate
(``hybrid_opt = 2``) can only order a column whose surface pressure stays
above a floor set by ``etac`` and ``p_top``
(:func:`gpuwm.core.grid.hybrid_surface_pressure_floor`), and WRF itself
calls a column below that floor fatal (v4.6.1
``dyn_em/nest_init_utils.F:1158-1182``, whose message says the cause
"tends to be caused by very high topography" and whose only remedy is
"reduce etac").  This package reproduced that refusal with every number
WRF's own message leaves out -- INCLUDING the largest ``etac`` that would
order the column -- and then handed the arithmetic to the user.  A
shipped preset placed over a bay whose 12 km parent reaches the world's
highest terrain stopped at the prepare stage on a coordinate the engine
had already solved.

So the run derives the coordinate it can use.  ``etac`` becomes the
largest supported value for the LOWEST base surface pressure over every
terrain field the run can touch: each domain's static terrain at its own
resolution (a 3 km nest carries higher peaks than its 12 km parent),
every declared domain including the ones that spawn later, and for a
following nest the whole statics corridor it may traverse -- a
relocation must never be the first place the coordinate fails.

What this is NOT.  It is not a change to the base state or to the
coefficient formulas: both stay transcribed from WRF.  It does not touch
``p_top``: ``etac`` is WRF's own named remedy and it keeps the model top
where the user put it.  And it does not retire the refusal, which still
stands for the one case it always named: terrain so high that no positive
``etac`` orders it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from gpuwm.core.grid import (analytic_base_pressure_field,
                             analytic_base_terrain_height,
                             hybrid_surface_pressure_floor,
                             largest_supported_etac)

#: Receipt schema for the derivation this module performs.
ADAPTATION_SCHEMA = "gpuwm-vertical-coordinate-adaptation-v1"

#: WRF's analytic base-state surface pressure over a terrain field:
#: ``module_initialize_real.F:3787-3803``,
#: ``p_s = p00*exp(-t00/a + sqrt((t00/a)^2 - 2*g*z/(a*Rd)))``, the exact
#: expression :func:`gpuwm.ingest.real._make_real_base_serial` builds
#: ``mub`` from.
#:
#: This is :func:`gpuwm.core.grid.analytic_base_pressure_field` under the
#: name this module's readers look for, NOT a second transcription: the
#: formula and its 50 K lapse live once, beside the base state and beside
#: :func:`~gpuwm.core.grid.analytic_base_terrain_height`, which inverts
#: them and which this module also imports.  Cells whose square-root
#: argument goes negative -- ground above the profile's own ceiling,
#: about 24.6 km for the 290 K default -- come back as NaN rather than as
#: a fabricated pressure, because the base state refuses them by name and
#: a survey must not quietly disagree with it.
analytic_base_surface_pressure = analytic_base_pressure_field


@dataclass(frozen=True)
class TerrainField:
    """One terrain array a run can touch, and what it belongs to.

    ``label`` is what the printed line and the receipt name, so it says
    the DOMAIN and WHICH of its terrains: "d01 static terrain",
    "d02 statics corridor".  ``base_temp`` is that domain's own, because
    the base-state surface pressure depends on it and a tree may carry
    different values per domain.
    """

    label: str
    terrain: np.ndarray
    base_temp: float = 290.0


@dataclass(frozen=True)
class VerticalAdaptation:
    """What the survey found, and the coordinate the run will use.

    ``etac`` is ``None`` for the one case the refusal still owns: no
    positive ``etac`` orders the governing column, and only a lower model
    top could.  ``adapted`` is False when the configured coordinate
    already ordered everything, and the record is kept anyway -- a
    receipt that only appears when something changed cannot be used to
    show that nothing did.
    """

    configured_etac: float
    etac: float | None
    hybrid_opt: int
    p_top: float
    configured_floor_pa: float
    surface_pressure_pa: float
    terrain_height_m: float
    column: tuple[int, ...]
    label: str
    surveyed: tuple[tuple[str, int, float], ...] = ()

    @property
    def adapted(self) -> bool:
        return self.etac is not None and self.etac != self.configured_etac

    @property
    def representable(self) -> bool:
        return self.etac is not None

    def sentence(self) -> str:
        """The one plain line a run prints when the coordinate changed."""

        ceiling = analytic_base_terrain_height(self.configured_floor_pa)
        return (
            f"vertical coordinate: etac {self.configured_etac:g} orders only "
            f"columns above {self.configured_floor_pa:.0f} Pa (about "
            f"{ceiling:.0f} m of terrain), and {self.label} mass point "
            f"{self.column} is at {self.terrain_height_m:.0f} m "
            f"({self.surface_pressure_pa:.0f} Pa), so this run uses etac "
            f"{self.etac:.3f} -- the largest the WRF cubic supports for that "
            f"column at the model top the configuration asked for "
            f"({self.p_top:g} Pa)")

    def receipt(self) -> dict:
        """The derivation, for the prepared receipt and the run document."""

        return {
            "schema": ADAPTATION_SCHEMA,
            "status": "ADAPTED" if self.adapted else (
                "AS_CONFIGURED" if self.representable else "UNREPRESENTABLE"),
            "hybrid_opt": int(self.hybrid_opt),
            "p_top_pa": float(self.p_top),
            "configured_etac": float(self.configured_etac),
            "etac": None if self.etac is None else float(self.etac),
            "configured_floor_pa": float(self.configured_floor_pa),
            "configured_floor_terrain_m": float(
                analytic_base_terrain_height(self.configured_floor_pa)),
            "governing_column": {
                "field": self.label,
                "index": [int(value) for value in self.column],
                "terrain_height_m": float(self.terrain_height_m),
                "surface_pressure_pa": float(self.surface_pressure_pa),
            },
            "surveyed_terrain": [
                {"field": label, "cells": int(cells),
                 "max_terrain_m": float(peak)}
                for label, cells, peak in self.surveyed
            ],
            "reference": (
                "WRF v4.6.1 dyn_em/nest_init_utils.F:1158-1182; "
                "doc/README.hybrid_vert_coord:65-74"),
        }


def survey_vertical_coordinate(
        znw, hybrid_opt: int, etac: float, p_top: float,
        fields: Sequence[TerrainField]) -> VerticalAdaptation | None:
    """The derivation itself: pure arrays in, one coordinate decision out.

    ``None`` means there is nothing to decide -- ``B = eta`` exactly
    (``hybrid_opt`` 0/1) has no terrain ceiling at all, and neither does a
    coordinate whose floor is already at zero pressure.  Otherwise the
    record always comes back, adapted or not, naming the column that
    governs the answer.
    """

    hybrid_opt = int(hybrid_opt)
    etac = float(etac)
    p_top = float(p_top)
    znw = np.asarray(znw, dtype=np.float64)
    fields = tuple(fields)
    if not fields:
        raise ValueError(
            "a vertical-coordinate survey needs at least one terrain field: "
            "deriving the coordinate from no ground at all would accept "
            "every column by vacuum")
    floor = hybrid_surface_pressure_floor(znw, hybrid_opt, etac, p_top)
    if floor <= 0.0:
        return None

    worst_label = ""
    worst_pressure = np.inf
    worst_height = 0.0
    worst_column: tuple[int, ...] = ()
    surveyed = []
    for field in fields:
        terrain = np.asarray(field.terrain, dtype=np.float64)
        if terrain.size == 0:
            raise ValueError(
                f"terrain field {field.label!r} is empty; a survey that "
                "counts an empty field as surveyed would report ground it "
                "never looked at")
        pressure = analytic_base_surface_pressure(terrain, field.base_temp)
        surveyed.append(
            (field.label, terrain.size, float(np.nanmax(terrain))))
        # NaN is ground the analytic base state itself refuses, and it has
        # to LOSE every comparison rather than be skipped: it is the
        # highest ground in the field, not the absent one.
        filled = np.where(np.isnan(pressure), -np.inf, pressure)
        index = np.unravel_index(int(np.argmin(filled)), filled.shape)
        candidate = float(filled[index])
        if candidate < worst_pressure:
            worst_pressure = candidate
            worst_height = float(terrain[index])
            worst_column = tuple(int(value) for value in index)
            worst_label = field.label

    if worst_pressure > floor:
        return VerticalAdaptation(
            configured_etac=etac, etac=etac, hybrid_opt=hybrid_opt,
            p_top=p_top, configured_floor_pa=floor,
            surface_pressure_pa=worst_pressure,
            terrain_height_m=worst_height, column=worst_column,
            label=worst_label, surveyed=tuple(surveyed))
    remedy = (None if not np.isfinite(worst_pressure)
              else largest_supported_etac(znw, p_top, worst_pressure))
    return VerticalAdaptation(
        configured_etac=etac, etac=remedy, hybrid_opt=hybrid_opt,
        p_top=p_top, configured_floor_pa=floor,
        surface_pressure_pa=worst_pressure, terrain_height_m=worst_height,
        column=worst_column, label=worst_label, surveyed=tuple(surveyed))


def adapt_experiment_vertical(exp, fields: Sequence[TerrainField], *,
                              announce=None):
    """Return ``(experiment, adaptation)`` on the coordinate this run can use.

    The experiment's one :class:`~gpuwm.experiment.VerticalConfig` is the
    single vertical authority (F1 amendment) and every ``RunConfig.etac``
    is a derived copy of it that
    :func:`gpuwm.experiment._assert_derived_copies` pins equal, so the
    adaptation replaces BOTH or the experiment would no longer load.
    Replacing them here is what makes every consumer -- the root
    coordinate, ``nest_init._shared_vertical_coord``, a tile buffer's
    rebuild, a nest spawned at runtime, an offline child -- take the
    derived coordinate without knowing this module exists.

    ``announce`` receives the one plain line when the coordinate changed;
    ``None`` prints nothing.  An UNREPRESENTABLE survey changes nothing
    and returns the experiment untouched: the refusal downstream owns
    that case and already names it.
    """

    vertical = exp.vertical
    if not vertical.eta_levels:
        return exp, None
    adaptation = survey_vertical_coordinate(
        np.asarray(vertical.eta_levels, dtype=np.float64),
        vertical.hybrid_opt, vertical.etac, vertical.p_top, fields)
    if adaptation is None or not adaptation.adapted:
        return exp, adaptation
    etac = float(adaptation.etac)
    domains = tuple(
        replace(dc, run=replace(dc.run, etac=etac)) for dc in exp.domains)
    adapted = replace(exp, vertical=replace(vertical, etac=etac),
                      domains=domains)
    if announce is not None:
        announce(adaptation.sentence())
    return adapted, adaptation


def _domain_start(domain, exp):
    """The date a domain's dated statics are built for.

    A domain that declares no start of its own begins with the
    experiment, which is what the loader writes onto every ordinary
    domain; reading it this way keeps the two spellings one answer.
    """

    return getattr(domain, "start_time", None) or exp.start_time


def run_terrain_fields(exp, grids, *, root_terrain, static_catalog,
                       static_highres=None, corridors: bool = True
                       ) -> list[TerrainField]:
    """Every terrain field this experiment's run can touch, in one list.

    The root's is the array the caller already built (post-overlay, so it
    is the terrain the root will integrate).  Each other declared domain
    -- including one that only spawns later, because a dormant nest is
    declared at prepare time and shares this one coordinate -- is built at
    its own resolution.  A following nest additionally contributes its
    statics corridor: child-resolution ground over the whole frame extent
    it may traverse, which is the only terrain that can answer "where
    could this nest be in six hours" before the run starts.

    Terrain alone is built, not the whole field set
    (:func:`gpuwm.static.build.build_terrain`), except where a
    ``[static.highres]`` overlay is active: that overlay replaces
    ``HGT_M`` out of the complete field set it merges, so those domains
    are surveyed on the complete build to stay equal to the terrain they
    will run on.
    """

    from gpuwm.static.build import (build_static_for_domain, build_terrain,
                                    geog_selection_from_catalog)
    from gpuwm.static.corridor import (corridor_frame_kwargs,
                                       corridor_geometry, corridor_grid,
                                       moving_grid_ids)

    highres_on = bool(static_highres is not None
                      and getattr(static_highres, "enabled", False))
    needs_catalog = (len(exp.domains) > 1
                     or any(int(dc.parent_id) != 0 for dc in exp.domains
                            if int(dc.grid_id) in moving_grid_ids(exp)))
    if static_catalog is None and needs_catalog:
        raise ValueError(
            "surveying a domain tree's terrain needs the WPS_GEOG static "
            "catalog: a child is built at its own resolution and carries "
            "higher peaks than its parent, so a coordinate derived from "
            "the root alone would be refused by the first nest that "
            "initializes")
    root = exp.domains[0]
    fields = [TerrainField(f"d{int(root.grid_id):02d} static terrain",
                           np.asarray(root_terrain, dtype=np.float64),
                           float(root.run.base_temp))]
    by_id = {int(dc.grid_id): dc for dc in exp.domains}
    grid_by_id = {int(dc.grid_id): grid
                  for dc, grid in zip(exp.domains, grids)}
    for dc in exp.domains[1:]:
        gid = int(dc.grid_id)
        selection = geog_selection_from_catalog(static_catalog, gid)
        if highres_on:
            from gpuwm.static.highres_production import apply_highres_statics
            built = build_static_for_domain(
                grid_by_id[gid], static_catalog, gid)
            built, _ = apply_highres_statics(
                built, grid_by_id[gid], config=static_highres, domain_id=gid,
                case_date=_domain_start(dc, exp).date(),
                landuse_attrs=selection.landuse_global_attrs())
            terrain = built["HGT_M"]
        else:
            terrain = build_terrain(grid_by_id[gid], selection.root,
                                    selection=selection)
        fields.append(TerrainField(f"d{gid:02d} static terrain",
                                   np.asarray(terrain, dtype=np.float64),
                                   float(dc.run.base_temp)))
    if not corridors:
        return fields
    for gid in sorted(moving_grid_ids(exp)):
        dc = by_id.get(gid)
        if dc is None or int(dc.parent_id) == 0:
            continue
        parent = by_id[int(dc.parent_id)]
        frame_kwargs = corridor_frame_kwargs(exp, dc)
        geometry = corridor_geometry(dc, parent.run, **frame_kwargs)
        frame_grid = grid_by_id[int(geometry["frame_grid_id"])]
        selection = geog_selection_from_catalog(static_catalog, gid)
        if highres_on:
            from gpuwm.static.corridor import build_child_statics_corridor
            built = build_child_statics_corridor(
                child_dc=dc, parent_run=parent.run,
                reference_grid=frame_grid, static_catalog=static_catalog,
                frame_kwargs=frame_kwargs, static_highres=static_highres)
            terrain = built.fields["HGT_M"]
        else:
            terrain = build_terrain(corridor_grid(frame_grid, geometry),
                                    selection.root, selection=selection)
        fields.append(TerrainField(
            f"d{gid:02d} statics corridor",
            np.asarray(terrain, dtype=np.float64), float(dc.run.base_temp)))
    return fields


def adapt_experiment_for_statics(exp, grids, *, root_terrain, static_catalog,
                                 static_highres=None, announce=None):
    """The prepare door's one call: survey this run's ground, then adapt.

    Every source door goes through here, so adding a source adds no
    coordinate logic; and the answer depends only on the experiment and
    its geography, never on which product the forcing came from.
    """

    fields = run_terrain_fields(
        exp, grids, root_terrain=root_terrain, static_catalog=static_catalog,
        static_highres=static_highres)
    return adapt_experiment_vertical(exp, fields, announce=announce)


def static_catalog_for_survey(catalog):
    """The WPS_GEOG catalog the survey reads: the one the nests initialise from.

    A nested run's input catalog comes in two shapes.  The source catalog
    itself (:class:`gpuwm.ingest.preflight.InputCatalog`) carries the GEOG
    roles among its own files and IS the static catalog; a wrapper that
    joins source snapshots to an independently verified static catalog
    (:class:`gpuwm.ingest.nest_init.NestedInputCatalog`) carries it as
    ``static_catalog``.  Child initialisation selects between the two
    with exactly this rule (``gpuwm.ingest.nest_init._static_catalog``),
    so the survey selects the same way and the coordinate is derived
    from the ground each child will actually be built on.

    It lives here and not beside its caller in :mod:`gpuwm.core.model`
    because that module is a clock module under an AST audit that bans
    reflection outright
    (tests/test_clock.py::test_no_float_elapsed_accumulation_audit).
    """

    return getattr(catalog, "static_catalog", catalog)


def not_applicable_why(vertical) -> str:
    """Why this run had no coordinate to derive, in the words of ITS route.

    Three returns in this module hand ``None`` back instead of a survey
    record, and they are three different facts about the run.  The
    receipt is the only place a reader of a prepared bundle, or of the
    run document written from it, learns which one happened, so each
    route says its own reason rather than borrowing another's.  Those
    two records are the whole readership of this string: the desktop's
    run view keys on ``status`` and never reads it.
    The order below is the order the code takes them in:

    * no eta ladder -- :func:`adapt_experiment_vertical` returns before
      it surveys any terrain, whatever ``hybrid_opt`` says;
    * ``hybrid_opt`` 0 or 1 -- :func:`survey_vertical_coordinate` gets a
      floor of zero from
      :func:`gpuwm.core.grid.hybrid_surface_pressure_floor` because
      ``B = eta`` makes every discrete ``dB/deta`` exactly 1;
    * ``hybrid_opt`` 2 with a floor already at zero pressure -- the same
      return, reached with a ladder configured and a cubic whose
      steepest segment cannot push the floor above zero.  The ladder
      and ``etac`` set that between them:
      :func:`gpuwm.core.grid.hybrid_surface_pressure_floor` returns zero
      exactly where the steepest discrete ``dB/deta`` is not above 1,
      a comparison ``p_top`` does not enter.

    Reading ``vertical`` alone is what the routes leave behind: a
    configuration with no ladder took the first, an identity option with
    a ladder took the second, and a cubic with a ladder took the third.
    """

    hybrid_opt = int(vertical.hybrid_opt)
    if not vertical.eta_levels:
        return (
            "this configuration carries no eta ladder, so preparation "
            "returned before it surveyed any terrain and there was no "
            "real-data vertical coordinate to derive")
    if hybrid_opt in (0, 1):
        return (
            f"hybrid_opt {hybrid_opt} sets B(eta) = eta exactly, so the "
            "reference column cannot invert at any surface pressure and "
            "this coordinate has no terrain ceiling to derive")
    return (
        f"hybrid_opt {hybrid_opt} with etac {float(vertical.etac):g} "
        "leaves this eta ladder no segment where dB/deta rises above 1, "
        "so this coordinate's surface-pressure floor is zero, no ground "
        "can reach it and there is nothing to derive")


def vertical_coordinate_receipt(exp, adaptation) -> dict:
    """The EFFECTIVE vertical configuration, plus how it was arrived at.

    A prepared bundle is run from its own artifacts, so the receipt has
    to answer "which coordinate is in these files" without the
    configuration beside it -- and, when that is not the configured one,
    why.  Present on every preparation, whether or not anything changed:
    an absent key would read as "prepared before this existed", which is
    a different claim and one no reader of the bundle could check.
    """

    vertical = exp.vertical
    if adaptation is None:
        derivation = {
            "schema": ADAPTATION_SCHEMA,
            "status": "NOT_APPLICABLE",
            "why": not_applicable_why(vertical),
        }
    else:
        derivation = adaptation.receipt()
    return {
        "hybrid_opt": int(vertical.hybrid_opt),
        "etac": float(vertical.etac),
        "p_top_pa": float(vertical.p_top),
        # Derived here rather than read off the config object: the
        # receipt describes the LADDER it is reporting, and a caller
        # holding any object with eta_levels can write one.
        "mass_levels": (len(vertical.eta_levels) - 1
                        if vertical.eta_levels else None),
        "derivation": derivation,
    }


def prepared_coordinate_refusal(*, label: str, configured_etac: float,
                                prepared_etac: float, znw, p_top: float,
                                surface_pressure=None) -> str | None:
    """Why a prepared coordinate may not be the one the forecast runs on.

    The prepared artifacts carry the coefficient arrays themselves, the
    way WRF's ``wrfinput`` carries ``C3H``/``C4H``, so the forecast takes
    its coordinate from them and never rebuilds it from the
    configuration.  What still has to hold is that the value it adopts is
    the DERIVATION and not a free-form number: it may only be at or below
    the configured ``etac``, and it must order every column the prepared
    artifact itself contains.  Both are checkable from the artifact with
    no geography at all, which is what lets a bundle be verified where it
    runs.  ``None`` when the prepared coordinate is admissible; omitting
    ``surface_pressure`` checks only the first half, which is the whole
    of what a preparation DOCUMENT can answer before any array is read.
    """

    prepared_etac = float(prepared_etac)
    configured_etac = float(configured_etac)
    if not (0.0 < prepared_etac <= configured_etac):
        return (
            f"{label} prepared etac {prepared_etac:g} is not a derivation of "
            f"the configured etac {configured_etac:g}: the coordinate this "
            "run adopts from its prepared inputs may only be at or below the "
            "configured value, because a HIGHER etac orders fewer columns "
            "than the configuration asked for and nothing in the bundle "
            "would show it")
    if surface_pressure is None:
        return None
    floor = hybrid_surface_pressure_floor(
        np.asarray(znw, dtype=np.float64), 2, prepared_etac, float(p_top))
    pressure = np.asarray(surface_pressure, dtype=np.float64)
    if floor > 0.0 and np.any(pressure <= floor):
        index = np.unravel_index(int(np.argmin(pressure)), pressure.shape)
        return (
            f"{label} prepared etac {prepared_etac:g} does not order the "
            f"prepared column at mass point "
            f"{tuple(int(value) for value in index)} "
            f"({float(pressure[index]):.0f} Pa, below this coordinate's "
            f"{floor:.0f} Pa floor); the prepared inputs and the "
            "coefficients the model would integrate describe different "
            "atmospheres")
    return None


def adopt_prepared_vertical(exp, preparation, *, announce=None):
    """The forecast runs on the coordinate its PREPARED INPUTS carry.

    THE SEAM.  A prepared bundle stores the coefficient arrays
    themselves -- ``ingest/prepared_cache.py`` restores
    ``VerticalCoord(**coord_scalars, **coord_arrays)``, ``c1f`` through
    ``c4h`` included -- exactly as WRF's ``wrfinput`` carries ``C3H`` and
    ``C4H`` and ``start_domain_em.F`` uses them rather than rebuilding
    from the namelist.  So the model already integrates the prepared
    coordinate.  What could still disagree is everything the RUNNER
    derives from the configuration beside it: the identity comparison
    against each cache, a tile buffer's rebuilt coordinate
    (``core/streaming.domain_vertical_coord``), a nest spawned mid-run
    and a relocated child's re-initialization, all of which read
    ``RunConfig.etac``.  Adopting the prepared value into the experiment
    HERE, before any of them exist, leaves one etac in the process.

    The adoption is checked, not trusted: the prepared value may only be
    at or below the configured one, ``hybrid_opt`` and ``p_top`` must
    match exactly, and every domain's own prepared column is then held
    against it as its cache is opened.  A bundle with no coordinate
    record is left alone -- it was written before this existed, by an
    engine that could only ever have used the configured value.
    """

    record = (preparation.get("vertical_coordinate")
              if isinstance(preparation, dict) else None)
    if not isinstance(record, dict):
        return exp, None
    try:
        prepared_etac = float(record["etac"])
        prepared_hybrid = int(record["hybrid_opt"])
        prepared_top = float(record["p_top_pa"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "the preparation document carries a vertical-coordinate record "
            "without a usable etac/hybrid_opt/p_top triple, so the "
            "coordinate the prepared inputs were built on cannot be "
            "established; re-prepare this case") from None
    vertical = exp.vertical
    if (prepared_hybrid != int(vertical.hybrid_opt)
            or prepared_top != float(vertical.p_top)):
        raise ValueError(
            f"the prepared inputs were built on hybrid_opt "
            f"{prepared_hybrid} with p_top {prepared_top:g} Pa and this "
            f"configuration asks for hybrid_opt {vertical.hybrid_opt} with "
            f"p_top {vertical.p_top:g} Pa; the prepared coefficient arrays "
            "describe a different vertical grid and the forecast would "
            "integrate them under the wrong model top")
    if prepared_etac == float(vertical.etac):
        return exp, record
    refusal = prepared_coordinate_refusal(
        label="the prepared bundle", configured_etac=float(vertical.etac),
        prepared_etac=prepared_etac, znw=vertical.eta_levels,
        p_top=float(vertical.p_top))
    if refusal is not None:
        raise ValueError(refusal)
    domains = tuple(
        replace(dc, run=replace(dc.run, etac=prepared_etac))
        for dc in exp.domains)
    adopted = replace(exp, vertical=replace(vertical, etac=prepared_etac),
                      domains=domains)
    if announce is not None:
        column = record.get("derivation", {}).get("governing_column", {})
        where = column.get("field", "its terrain")
        index = tuple(column.get("index", ()) or ())
        height = column.get("terrain_height_m")
        because = (
            f"{where} mass point {index} reaches {float(height):.0f} m"
            if index and height is not None else
            f"{where} reaches ground the configured coordinate cannot order")
        announce(
            f"vertical coordinate: these prepared inputs were built on etac "
            f"{prepared_etac:g}, not the configured "
            f"{float(vertical.etac):g}, because {because}; the forecast "
            f"integrates the prepared coefficients")
    return adopted, record


def prepared_domain_coordinate_refusal(*, label: str, vertical,
                                       coord_scalars, base_arrays) -> str | None:
    """Hold ONE prepared domain's stored coordinate to the adopted one.

    Called as each cache is opened, with that cache's own coordinate
    scalars and its own base-state ``mub``.  Every domain in a tree must
    carry the same coordinate -- they share one vertical grid by
    construction -- and that coordinate must order that domain's own
    columns.  ``None`` when the domain is admissible.
    """

    stored_etac = coord_scalars.get("etac")
    stored_hybrid = coord_scalars.get("hybrid_opt")
    if stored_etac is None or stored_hybrid is None:
        return (
            f"{label} prepared cache records no hybrid coordinate "
            "(etac/hybrid_opt); the coefficients it restores cannot be "
            "shown to be the ones this run adopted")
    if (int(stored_hybrid) != int(vertical.hybrid_opt)
            or float(stored_etac) != float(vertical.etac)):
        return (
            f"{label} prepared cache was built on hybrid_opt "
            f"{int(stored_hybrid)}, etac {float(stored_etac):g} and this "
            f"run integrates hybrid_opt {int(vertical.hybrid_opt)}, etac "
            f"{float(vertical.etac):g}; one domain of a tree cannot sit on "
            "a different vertical coordinate from its siblings")
    mub = base_arrays.get("mub")
    if mub is None:
        return None
    return prepared_coordinate_refusal(
        label=label, configured_etac=float(vertical.etac),
        prepared_etac=float(stored_etac), znw=vertical.eta_levels,
        p_top=float(vertical.p_top),
        surface_pressure=np.asarray(mub, dtype=np.float64)
        + float(vertical.p_top))


def refuse_tree_off_the_experiment_coordinate(exp, domains, *,
                                              route: str) -> None:
    """Hold a domain tree taken from an experiment to that experiment's coordinate.

    THE BREAKAGE THIS REFUSAL PREVENTS.  A builder takes its startup tree
    from the experiment BY VALUE (``pre_spawn_experiment`` returns fresh
    ``DomainConfig`` objects) and then stores those objects on the domain
    nodes, where :func:`gpuwm.core.streaming.domain_vertical_coord`
    rebuilds a streamed tile buffer's coordinate from ``cfg.etac`` and
    :class:`gpuwm.core.spawn_runner` prices a mid-run spawn from the
    experiment.  Derive the coordinate AFTER that copy is taken and the
    run holds two: the root and the experiment on the derived value, every
    nest's ``RunConfig`` on the configured one, and a streamed nest
    rebuilding its buffer on a coordinate its own domain is not on.  The
    order alone made that impossible; this asks, so the order cannot drift
    back silently.
    """

    vertical = exp.vertical
    wrong = [dc for dc in domains
             if float(dc.run.etac) != float(vertical.etac)
             or int(dc.run.hybrid_opt) != int(vertical.hybrid_opt)]
    if not wrong:
        return
    named = ", ".join(
        f"d{int(dc.grid_id):02d} on hybrid_opt {int(dc.run.hybrid_opt)}, "
        f"etac {float(dc.run.etac):g}" for dc in wrong)
    raise RuntimeError(
        f"the {route} tree left a domain off this run's vertical "
        f"coordinate: the experiment integrates hybrid_opt "
        f"{int(vertical.hybrid_opt)}, etac {float(vertical.etac):g} and "
        f"{named}. A domain whose RunConfig disagrees with the experiment "
        "rebuilds a streamed tile buffer, and prices a mid-run spawn, on "
        "a coordinate its own state is not on. The vertical derivation "
        "must run before the tree is taken from the experiment.")


__all__ = [
    "adopt_prepared_vertical",
    "refuse_tree_off_the_experiment_coordinate",
    "prepared_domain_coordinate_refusal",
    "ADAPTATION_SCHEMA",
    "not_applicable_why",
    "TerrainField",
    "VerticalAdaptation",
    "adapt_experiment_for_statics",
    "static_catalog_for_survey",
    "adapt_experiment_vertical",
    "analytic_base_surface_pressure",
    "prepared_coordinate_refusal",
    "run_terrain_fields",
    "survey_vertical_coordinate",
    "vertical_coordinate_receipt",
]
