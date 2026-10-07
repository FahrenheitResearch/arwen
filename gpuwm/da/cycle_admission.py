"""The prepared DA cycle's fit decision, taken once before its first upload.

A cycle runs its trajectories one after another on one card: each is a
whole forecast (the restored state, its physics, the lateral boundary
intervals the prepared cache holds and, on a nesting trajectory, a child
with its own state, physics and scratch), and the leg's model-grid
reflectivity observations stay on the card beside it.  A fresh ensemble's
first leg also perturbs each member on the card before it steps.  Nothing
priced that before :func:`gpuwm.ingest.prepared_cache.restore_prepared_cache`
allocated the first state, so a domain the card could not hold ran out of
memory inside the restore, the physics or the first member's perturbation,
after the preflight had reported the case sound.

The decision prices the LARGEST trajectory once and every trajectory of
every leg runs under it: the members are the same forecast, and the loop
releases each one before the next is wired.

The analysis runs after the leg's trajectories are released, beside the
leg's observations, so it is weighed against the trajectory rather than
added to it.  It is priced by the solver's own sizing
(:func:`gpuwm.da.letkf.analysis_device_price`, which calls the
:func:`gpuwm.da.letkf.chunk_points_for_budget` the solve chooses its chunk
with) on each route the solve can take, in the order it takes them: the
resident route at its configured chunk, the resident route at the
smallest chunk it shrinks to, and the host-staged fallback it retries on
when the resident route runs out of memory.  The first route whose
envelope fits the card is the one admitted and recorded; a card that
holds none of them is refused.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

#: Device bytes per model mass point of one leg's reflectivity
#: observations: the float32 ``z_obs`` and the Boolean ``z_mask`` the hot
#: start reads, uploaded once per leg and held through it.
OBSERVATION_BYTES_PER_POINT = 5

#: Device bytes per model mass point that radar latent heating
#: (:mod:`gpuwm.da.radar_tten`, ``--radar-tten``) holds through a forced
#: leg: the leg's reflectivity in NOAA's convention, the member's one
#: tendency slot and the pre-microphysics theta the forcing keeps, all
#: float32.
RADAR_TTEN_HELD_BYTES_PER_POINT = 12

#: Device bytes per model mass point the radar heating builder holds for a
#: moment before the leg's first step, beside the held arrays: the
#: background read off the state (theta, pressure and height, float32), the
#: cone-filled reflectivity (float32) and the two float64 volumes the
#: smoother alternates between.  Freed before the step, so it competes with
#: the step working set rather than adding to it.
RADAR_TTEN_BUILD_BYTES_PER_POINT = 32

#: Device bytes per mass point each slot beyond the first adds
#: (``--radar-tten-windows``: one slot per 15-minute window): the slot
#: itself, float32, held through the leg, and the window's reflectivity,
#: float32, held while the slots are built.
RADAR_TTEN_EXTRA_SLOT_HELD_BYTES_PER_POINT = 4
RADAR_TTEN_EXTRA_SLOT_BUILD_BYTES_PER_POINT = 4

#: The analysis routes, in the order the solve takes them.  ``resident``
#: holds every whole-domain array on the card and solves at the
#: configured chunk; ``reduced-chunk`` is the same route when the solve
#: finds less free memory than that chunk needs and shrinks it (the same
#: analysis to rounding, in more batches); ``host-staged`` is the fallback
#: that keeps the whole-domain arrays in host memory and moves one bounded
#: chunk at a time.
ANALYSIS_ROUTES = ("resident", "reduced-chunk", "host-staged")


class CycleMemoryRefused(MemoryError):
    """The cycle's largest trajectory or its analysis does not fit.

    Raised before the first upload.  ``admission`` carries every term by
    name, so a caller quoting the refusal states the same numbers.
    """

    def __init__(self, message: str, *, admission: "CycleAdmission"):
        super().__init__(message)
        self.admission = admission


@dataclass(frozen=True)
class CycleAdmission:
    """The priced device peak of one trajectory and one leg analysis."""

    #: The forecast's persistent arrays: state, physics, boundary tables
    #: and per-domain scratch, over every domain the trajectory carries.
    forecast_resident_bytes: int
    #: The forecast's step working set: the radiation chunk workspace and
    #: the largest domain's step transients.
    forecast_step_bytes: int
    observation_bytes: int
    #: The first leg's member perturbation (:func:`gpuwm.da.perturb.
    #: device_working_bytes`, cuFFT's plan work areas included), held
    #: before the first step, so it competes with the step working set
    #: rather than adding to it.
    perturbation_bytes: int
    #: The envelope the other forecast doors admit against
    #: (:func:`gpuwm.core.preflight.machine_peak_envelope_bytes`): the sum
    #: above with allocator headroom, plus the CUDA context and kernel
    #: local memory.  With an analysis, the envelope of its first route;
    #: after :func:`admit_cycle`, of the route admitted.
    required_bytes: int
    domains: int
    forcing_intervals: int
    basis: str
    #: ``((route, analysis_bytes, required_bytes), ...)`` for every route
    #: the analysis can take, in :data:`ANALYSIS_ROUTES` order; empty when
    #: the analysis solves on the host or no leg is observed.
    analysis_routes: tuple = ()
    #: The route priced (or admitted), and what it holds beside the
    #: observations.
    analysis_route: str | None = None
    analysis_bytes: int = 0
    #: The solver's own figures behind the routes, for the receipt.
    analysis_detail: dict | None = None
    free_bytes: int | None = None
    budget_bytes: int | None = None
    #: Radar latent heating: held through a forced leg, and the builder's
    #: transient before its first step (0 when the cycle forces nothing).
    radar_tten_held_bytes: int = 0
    radar_tten_build_bytes: int = 0

    @property
    def fits(self) -> bool:
        return self.budget_bytes is None or self.required_bytes <= self.budget_bytes

    def receipt(self) -> dict:
        return {
            "scope": ("the largest trajectory and the leg analysis, "
                      "reused for every member and every leg"),
            "required_bytes": int(self.required_bytes),
            "forecast_resident_bytes": int(self.forecast_resident_bytes),
            "forecast_step_bytes": int(self.forecast_step_bytes),
            "observation_bytes": int(self.observation_bytes),
            "perturbation_bytes": int(self.perturbation_bytes),
            "radar_tten_held_bytes": int(self.radar_tten_held_bytes),
            "radar_tten_build_bytes": int(self.radar_tten_build_bytes),
            "analysis_route": self.analysis_route,
            "analysis_bytes": int(self.analysis_bytes),
            "analysis_routes": [
                {"route": route, "analysis_bytes": int(nbytes),
                 "required_bytes": int(required)}
                for route, nbytes, required in self.analysis_routes],
            "analysis_solver": self.analysis_detail,
            "domains": int(self.domains),
            "forcing_intervals": int(self.forcing_intervals),
            "free_bytes": self.free_bytes,
            "budget_bytes": self.budget_bytes,
            "fits": bool(self.fits),
            "basis": self.basis,
        }


def _gib(value: int) -> str:
    return f"{int(value) / 2 ** 30:.2f} GiB ({int(value):,} bytes)"


def worst_analysis(prices):
    """One analysis price that covers every leg's, or None.

    Each observed leg reads its own observation file, so each leg's
    analysis is priced on its own; the cycle is admitted against the
    field-by-field worst of them (the largest arrays, scratch and row,
    the smallest chunk and budget), which no leg's route exceeds.
    """
    prices = [price for price in prices if price is not None]
    if not prices:
        return None
    worst = prices[0]
    for price in prices[1:]:
        worst = dataclasses.replace(
            worst,
            setup_bytes=max(worst.setup_bytes, price.setup_bytes),
            finish_bytes=max(worst.finish_bytes, price.finish_bytes),
            solve_bytes_per_point=max(worst.solve_bytes_per_point,
                                      price.solve_bytes_per_point),
            stencil_slots=max(worst.stencil_slots, price.stencil_slots),
            chunk_points=min(worst.chunk_points, price.chunk_points),
            scratch_bytes=max(worst.scratch_bytes, price.scratch_bytes),
            staged_row_bytes=max(worst.staged_row_bytes,
                                 price.staged_row_bytes),
            budget_bytes=min(worst.budget_bytes, price.budget_bytes),
            explicit_chunk=worst.explicit_chunk and price.explicit_chunk)
    return worst


def price_cycle(exp_leg, *, forcing_intervals: int, observation_points: int,
                perturbation_bytes: int, profile=None,
                analysis=None, radar_tten_points: int = 0,
                radar_tten_slots: int = 1
                ) -> CycleAdmission:
    """The device peak of the cycle's largest trajectory and its analysis.

    ``exp_leg`` is the experiment that trajectory runs: the root alone,
    or the root with its child when the cycle nests.  Every domain keeps
    its own scratch and dycore workspace on this route (the cycle's model
    carries no shared arena), so the shared-arena saving a multi-domain
    forecast estimate takes is put back.  ``analysis`` is the leg
    analysis's :class:`gpuwm.da.letkf.AnalysisDevicePrice` (see
    :func:`worst_analysis`), or None when no analysis runs on the card.
    ``radar_tten_points`` is the mass-point count of a cycle that forces
    its members with radar latent heating, 0 when it does not, and
    ``radar_tten_slots`` the slots each forced leg holds (one per window).
    """
    from gpuwm.core import preflight

    estimate = preflight.estimate_experiment(
        exp_leg, forcing_intervals=int(forcing_intervals), profile=profile)
    if estimate.uses_shared_scratch_arena \
            or estimate.uses_shared_dycore_state_workspace:
        estimate = dataclasses.replace(
            estimate, uses_shared_scratch_arena=False, scratch_arena_bytes=0,
            uses_shared_dycore_state_workspace=False,
            dycore_state_workspace_bytes=0)
    resident = int(estimate.resident_bytes)
    step = int(estimate.workspace_bytes + estimate.transient_peak_bytes)
    extra_slots = max(int(radar_tten_slots), 1) - 1
    radar_held = (RADAR_TTEN_HELD_BYTES_PER_POINT
                  + RADAR_TTEN_EXTRA_SLOT_HELD_BYTES_PER_POINT * extra_slots
                  ) * int(radar_tten_points)
    radar_build = (RADAR_TTEN_BUILD_BYTES_PER_POINT
                   + RADAR_TTEN_EXTRA_SLOT_BUILD_BYTES_PER_POINT * extra_slots
                   ) * int(radar_tten_points)
    observations = (OBSERVATION_BYTES_PER_POINT * int(observation_points)
                    + radar_held)
    perturbation = int(perturbation_bytes)
    # A member is perturbed on the root before its first step and before
    # a newborn child is built, so the perturbation competes with the
    # step working set, beside the root's arrays alone.
    root_id = int(exp_leg.root.grid_id)
    root_resident = int(sum(domain.resident_bytes
                            for domain in estimate.domains
                            if int(domain.grid_id) == root_id)
                        + estimate.k_tables_bytes)
    # The radar heating builder runs after the restore, the increment and
    # any perturbation, and its volumes are freed before the first step.
    trajectory = max(resident + step, root_resident + perturbation,
                     resident + radar_build)

    def envelope(peak: int) -> int:
        return int(preflight.machine_peak_envelope_bytes(
            alloc_estimate_bytes=math.ceil(
                estimate.headroom * (observations + peak)),
            non_pool_bytes=estimate.envelope_intercept_bytes,
            domains=len(estimate.domains), family=estimate.envelope_family,
            legacy_radiation=estimate.uses_legacy_radiation))

    routes = []
    detail = None
    if analysis is not None:
        tiers = (analysis.resident_bytes, analysis.reduced_bytes,
                 analysis.staged_bytes)
        routes = [(route, int(nbytes), envelope(max(trajectory, nbytes)))
                  for route, nbytes in zip(ANALYSIS_ROUTES, tiers)
                  if nbytes is not None]
        detail = {
            "setup_bytes": int(analysis.setup_bytes),
            "finish_bytes": int(analysis.finish_bytes),
            "scratch_bytes": int(analysis.scratch_bytes),
            "chunk_points": int(analysis.chunk_points),
            "solve_bytes_per_point": int(analysis.solve_bytes_per_point),
            "stencil_slots": int(analysis.stencil_slots),
            "staged_row_bytes": int(analysis.staged_row_bytes),
            "memory_budget_bytes": int(analysis.budget_bytes),
        }
    first = routes[0] if routes else (None, 0, envelope(trajectory))
    return CycleAdmission(
        forecast_resident_bytes=resident, forecast_step_bytes=step,
        observation_bytes=observations, perturbation_bytes=perturbation,
        required_bytes=int(first[2]), domains=len(estimate.domains),
        forcing_intervals=int(estimate.retained_forcing_intervals),
        basis=estimate.envelope_basis, analysis_routes=tuple(routes),
        analysis_route=first[0], analysis_bytes=int(first[1]),
        analysis_detail=detail, radar_tten_held_bytes=radar_held,
        radar_tten_build_bytes=radar_build)


def admit_cycle(price: CycleAdmission, *, free_bytes: int) -> CycleAdmission:
    """Judge ``price`` against the card's free memory, or refuse.

    The budget is the free memory less the external margin, the same
    budget :func:`gpuwm.core.streaming.decide` gives a resident forecast.
    The admitted analysis route is the first of ``price.analysis_routes``
    whose envelope fits.  The refusal is raised before a byte is
    uploaded: without it the cycle ran out of card memory inside the
    first restore, physics or member perturbation of a domain it could
    never have held, or ran a whole leg of forecasts and then ran out at
    an analysis none of whose routes the card could hold.
    """
    from gpuwm.core.preflight import EXTERNAL_MARGIN_BYTES

    budget = max(0, int(free_bytes) - int(EXTERNAL_MARGIN_BYTES))
    decided = dataclasses.replace(price, free_bytes=int(free_bytes),
                                  budget_bytes=budget)
    for route, nbytes, required in price.analysis_routes:
        if required <= budget:
            return dataclasses.replace(decided, analysis_route=route,
                                       analysis_bytes=int(nbytes),
                                       required_bytes=int(required))
    if price.analysis_routes:
        # Refused: quote the smallest envelope any route needs.
        route, nbytes, required = price.analysis_routes[-1]
        decided = dataclasses.replace(decided, analysis_route=route,
                                      analysis_bytes=int(nbytes),
                                      required_bytes=int(required))
    elif decided.fits:
        return decided
    parts = [f"the forecast's persistent arrays "
             f"{_gib(decided.forecast_resident_bytes)} over "
             f"{decided.domains} domain(s) with {decided.forcing_intervals} "
             f"boundary interval(s) held",
             f"its step working set {_gib(decided.forecast_step_bytes)}"]
    if decided.observation_bytes:
        parts.append(f"the leg's reflectivity observations "
                     f"{_gib(decided.observation_bytes)}")
    if decided.radar_tten_held_bytes:
        parts.append(f"of which the radar latent heating holds "
                     f"{_gib(decided.radar_tten_held_bytes)} through the leg "
                     f"and builds with {_gib(decided.radar_tten_build_bytes)} "
                     "before the first step")
    if decided.perturbation_bytes:
        parts.append(f"the first leg's member perturbation "
                     f"{_gib(decided.perturbation_bytes)} on the root, "
                     "held before the first step and before a newborn "
                     "child, so it is weighed against the step working "
                     "set rather than added to it")
    if decided.analysis_route is not None:
        parts.append(f"the leg analysis on its smallest route "
                     f"({decided.analysis_route}) "
                     f"{_gib(decided.analysis_bytes)}, run after the "
                     "trajectories are released, so it is weighed against "
                     "the trajectory rather than added to it")
    raise CycleMemoryRefused(
        f"this DA cycle needs {_gib(decided.required_bytes)} on the card for "
        f"its largest trajectory and its analysis ({'; '.join(parts)}; with "
        f"allocator headroom, the CUDA context and kernel local memory), and "
        f"the card has {_gib(decided.free_bytes)} free, {_gib(budget)} after "
        f"the {_gib(EXTERNAL_MARGIN_BYTES)} external margin.  It is refused "
        "here, before the first upload, because the first trajectory would "
        "run out of card memory partway through its restore, physics or "
        "perturbation, or the leg's analysis would, after the leg's whole "
        "ensemble had run.  Free the card of other work, or cycle a smaller "
        "domain, nest or ensemble", admission=decided)


def unsolvable_analysis_message(analysis) -> str | None:
    """Why no card can run this analysis, or None when some card can.

    When not one gridpoint fits the configured analysis budget, the
    resident route refuses, and the host-staged route, which caps its
    device scratch at the same budget, refuses too: the leg would reach
    its analysis after its whole ensemble ran and stop there.
    """
    if analysis is None or analysis.resident_bytes is not None \
            or analysis.staged_bytes is not None:
        return None
    return (f"the leg analysis cannot run under its memory budget of "
            f"{_gib(analysis.budget_bytes)}: one packed gridpoint of the "
            f"host-staged route needs {_gib(analysis.staged_row_bytes)} "
            f"({analysis.stencil_slots} stencil slots) and one gridpoint of "
            f"the resident route {_gib(analysis.solve_bytes_per_point)}.  It "
            "is refused before the first upload because the leg would "
            "otherwise run its whole ensemble and stop at the analysis.  "
            "Raise --memory-budget-mib, shrink the localisation radii, or "
            "cycle fewer members")
