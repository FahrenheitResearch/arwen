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
releases each one before the next is wired.  The analysis is not a term
here: it runs after the leg's trajectories are released, beside the
observations this decision already holds, and the batched solve sizes its
own chunks against the card it finds then
(:func:`gpuwm.da.letkf.chunk_points_for_budget`), so its admission stays
the solver's.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

#: Device bytes per model mass point of one leg's reflectivity
#: observations: the float32 ``z_obs`` and the Boolean ``z_mask`` the hot
#: start reads, uploaded once per leg and held through it.
OBSERVATION_BYTES_PER_POINT = 5


class CycleMemoryRefused(MemoryError):
    """The cycle's largest trajectory does not fit on this card.

    Raised before the first upload.  ``admission`` carries every term by
    name, so a caller quoting the refusal states the same numbers.
    """

    def __init__(self, message: str, *, admission: "CycleAdmission"):
        super().__init__(message)
        self.admission = admission


@dataclass(frozen=True)
class CycleAdmission:
    """One trajectory's priced device peak, and the card it was judged on."""

    #: The forecast's persistent arrays: state, physics, boundary tables
    #: and per-domain scratch, over every domain the trajectory carries.
    forecast_resident_bytes: int
    #: The forecast's step working set: the radiation chunk workspace and
    #: the largest domain's step transients.
    forecast_step_bytes: int
    observation_bytes: int
    #: The first leg's member perturbation (:func:`gpuwm.da.perturb.
    #: device_working_bytes`), held before the first step, so it competes
    #: with the step working set rather than adding to it.
    perturbation_bytes: int
    #: The envelope the other forecast doors admit against
    #: (:func:`gpuwm.core.preflight.machine_peak_envelope_bytes`): the sum
    #: above with allocator headroom, plus the CUDA context and kernel
    #: local memory.
    required_bytes: int
    domains: int
    forcing_intervals: int
    basis: str
    free_bytes: int | None = None
    budget_bytes: int | None = None

    @property
    def fits(self) -> bool:
        return self.budget_bytes is None or self.required_bytes <= self.budget_bytes

    def receipt(self) -> dict:
        return {
            "scope": ("the largest trajectory, reused for every member "
                      "and every leg"),
            "required_bytes": int(self.required_bytes),
            "forecast_resident_bytes": int(self.forecast_resident_bytes),
            "forecast_step_bytes": int(self.forecast_step_bytes),
            "observation_bytes": int(self.observation_bytes),
            "perturbation_bytes": int(self.perturbation_bytes),
            "domains": int(self.domains),
            "forcing_intervals": int(self.forcing_intervals),
            "free_bytes": self.free_bytes,
            "budget_bytes": self.budget_bytes,
            "fits": bool(self.fits),
            "basis": self.basis,
        }


def _gib(value: int) -> str:
    return f"{int(value) / 2 ** 30:.2f} GiB ({int(value):,} bytes)"


def price_cycle(exp_leg, *, forcing_intervals: int, observation_points: int,
                perturbation_bytes: int, profile=None) -> CycleAdmission:
    """The device peak of the cycle's largest trajectory.

    ``exp_leg`` is the experiment that trajectory runs: the root alone,
    or the root with its child when the cycle nests.  Every domain keeps
    its own scratch and dycore workspace on this route (the cycle's model
    carries no shared arena), so the shared-arena saving a multi-domain
    forecast estimate takes is put back.
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
    observations = OBSERVATION_BYTES_PER_POINT * int(observation_points)
    perturbation = int(perturbation_bytes)
    # A member is perturbed on the root before its first step and before
    # a newborn child is built, so the perturbation competes with the
    # step working set, beside the root's arrays alone.
    root_id = int(exp_leg.root.grid_id)
    root_resident = int(sum(domain.resident_bytes
                            for domain in estimate.domains
                            if int(domain.grid_id) == root_id)
                        + estimate.k_tables_bytes)
    subtotal = observations + max(resident + step,
                                  root_resident + perturbation)
    required = preflight.machine_peak_envelope_bytes(
        alloc_estimate_bytes=math.ceil(estimate.headroom * subtotal),
        non_pool_bytes=estimate.envelope_intercept_bytes,
        domains=len(estimate.domains), family=estimate.envelope_family,
        legacy_radiation=estimate.uses_legacy_radiation)
    return CycleAdmission(
        forecast_resident_bytes=resident, forecast_step_bytes=step,
        observation_bytes=observations, perturbation_bytes=perturbation,
        required_bytes=int(required), domains=len(estimate.domains),
        forcing_intervals=int(estimate.retained_forcing_intervals),
        basis=estimate.envelope_basis)


def admit_cycle(price: CycleAdmission, *, free_bytes: int) -> CycleAdmission:
    """Judge ``price`` against the card's free memory, or refuse.

    The budget is the free memory less the external margin, the same
    budget :func:`gpuwm.core.streaming.decide` gives a resident forecast.
    The refusal is raised before a byte is uploaded: without it the cycle
    ran out of card memory inside the first restore, physics or member
    perturbation of a domain it could never have held.
    """
    from gpuwm.core.preflight import EXTERNAL_MARGIN_BYTES

    budget = max(0, int(free_bytes) - int(EXTERNAL_MARGIN_BYTES))
    decided = dataclasses.replace(price, free_bytes=int(free_bytes),
                                  budget_bytes=budget)
    if decided.fits:
        return decided
    parts = [f"the forecast's persistent arrays "
             f"{_gib(decided.forecast_resident_bytes)} over "
             f"{decided.domains} domain(s) with {decided.forcing_intervals} "
             f"boundary interval(s) held",
             f"its step working set {_gib(decided.forecast_step_bytes)}"]
    if decided.observation_bytes:
        parts.append(f"the leg's reflectivity observations "
                     f"{_gib(decided.observation_bytes)}")
    if decided.perturbation_bytes:
        parts.append(f"the first leg's member perturbation "
                     f"{_gib(decided.perturbation_bytes)} on the root, "
                     "held before the first step and before a newborn "
                     "child, so it is weighed against the step working "
                     "set rather than added to it")
    raise CycleMemoryRefused(
        f"this DA cycle needs {_gib(decided.required_bytes)} on the card for "
        f"its largest trajectory ({'; '.join(parts)}; with allocator "
        f"headroom, the CUDA context and kernel local memory), and the card "
        f"has {_gib(decided.free_bytes)} free, {_gib(budget)} after the "
        f"{_gib(EXTERNAL_MARGIN_BYTES)} external margin.  It is refused here, "
        "before the first upload, because the first trajectory would run "
        "out of card memory partway through its restore, physics or "
        "perturbation.  Free the card of other work, or cycle a smaller "
        "domain or nest", admission=decided)
