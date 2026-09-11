"""One price and one ``[tiles]`` decision for a standalone offline child.

``gpuwm downscale`` reviews a child before it runs and the child runner
integrates it.  Both used to answer the memory question on their own:
the plan review priced the configured envelope with the itemized
estimator and said ``fits: true``; the runner, minutes later and after it
had already filled the card with the interpolated initial state, the
boundary tables and the physics driver, asked the tile planner with no
estimate and no machine.  The planner then measured the card the process
had just spent on, charged the whole rung's fixed cost against what was
left and refused a child the review had admitted.  Measured on a 10 GiB
card: 7.32 GiB free at review, ``fits true`` at 2.90 GiB of envelope, then
"no tile fits in 3.49 GiB of VRAM" after preprocessing.

This module is the one function both doors call.  Given the child
RunConfig, its ``[tiles]`` options and a planning machine captured before
any device allocation, it builds the experiment, prices it once, stamps
the resident-admission context onto the options the way
:class:`gpuwm.experiment.ExperimentConfig` does for its own tree, and asks
:func:`gpuwm.core.streaming.decide` with that machine and that estimate.
The review writes the answer into the plan; the runner hands the same
answer to ``make_stepper``.  A refusal, when the card is genuinely too
small, is raised here with the measured figure and the way out, before
anything is interpolated or allocated on the device.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone

from gpuwm.core import streaming

#: The memory model is start-time independent; the experiment wrapper needs
#: A datetime and the child's real clock comes from the parent frames.
PRICING_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

#: The basis a plan or a receipt names for the card the decision was taken
#: on: a card this process measured, or a capacity somebody declared.
MEASURED_BASIS = "measured-local"
DECLARED_BASIS = "declared"


@dataclass(frozen=True)
class ChildPricing:
    """The estimate, the decision and the budget it was judged on."""

    #: The decision ``make_stepper`` runs with; ``None`` only when the
    #: options needed a card and none could be planned against.
    decision: streaming.StreamingDecision | None
    #: The itemized estimate, ``None`` when the estimator refused this
    #: configuration (``pricing_error`` then says why).
    estimate: object | None
    #: The ``[tiles]`` options the decision was taken on, with the
    #: resident-admission context stamped so a later ``decide`` on them
    #: answers from the same envelope.
    options: streaming.StreamingOptions
    #: What the decision charged the envelope against, in bytes.
    budget_bytes: int | None
    #: The card figure the machine carried, and where it came from.
    machine_free_bytes: int | None
    machine_name: str | None
    basis: str
    pricing_error: str | None = None

    @property
    def peak_envelope_bytes(self) -> int | None:
        if self.estimate is None:
            return None
        return int(self.estimate.peak_envelope_bytes)

    @property
    def mode(self) -> str | None:
        if self.decision is None:
            return None
        return "streamed" if self.decision.stream else "resident"

    @property
    def admission_budget_bytes(self) -> int | None:
        """The whole-process budget the resident question was judged on.

        ``decide`` records it on the decision whenever the options went
        through the resident admission (mode ``auto`` with a card).  A
        streamed decision's own ``budget_bytes`` is the tile planner's,
        which reserves the radiation transient separately, so the plan's
        ``memory`` block asks for THIS one: ``fits`` is then answered on
        the same budget the mode was decided on, and the two blocks
        cannot disagree about one child.  ``None`` when no admission
        was taken (tiles off, a pinned tiling, no card).
        """
        if self.decision is None:
            return None
        detail = getattr(self.decision, "detail", None) or {}
        admission = detail.get("resident_admission")
        if admission is None:
            return None
        return int(admission["budget_bytes"])

    def plan_entry(self) -> dict:
        """The ``streaming`` block of the plan document and the receipt."""
        decision = self.decision
        if decision is None:
            why = ("[tiles] needs a card to plan against and this review "
                   "holds none; the run decides on the card it starts on"
                   if self.pricing_error is None else
                   f"not decided: {self.pricing_error}")
        else:
            why = decision.reason
        entry = {
            "mode": self.mode,
            "why": why,
            "basis": self.basis,
            "budget_bytes": self.budget_bytes,
            "peak_envelope_bytes": self.peak_envelope_bytes,
            "machine_free_bytes": self.machine_free_bytes,
            "machine": self.machine_name,
            "tile": None,
        }
        if decision is not None and decision.stream:
            entry["tile"] = {
                "nx": decision.tile_nx, "ny": decision.tile_ny,
                "nbuffers": decision.nbuffers, "halo": decision.halo,
                "store": decision.store, "write_mode": decision.write_mode,
            }
        if self.pricing_error is not None:
            entry["pricing_error"] = self.pricing_error
        return entry


def needs_machine(options: streaming.StreamingOptions | None) -> bool:
    """Whether ``decide`` will consult a card for these options.

    Off asks nothing; a pinned tiling IS the decision.  Only ``auto`` and
    an unpinned ``on`` need the planner, and the planner needs a card.
    """
    options = streaming.OFF if options is None else options
    return bool(options.enabled and options.tile_nx is None)


def cold_machine(options: streaming.StreamingOptions | None):
    """Measure the card NOW, before this process allocates anything on it.

    The runner calls this before ``interpolate_parent_initial_state``.
    ``Machine.detect`` reads free VRAM through CuPy, which stands the
    CUDA context up; that is the same cost the prepared route pays in
    ``streaming.cold_planning_machine`` and it is the last thing this
    process does on the card before the decision.  ``None`` when the
    options need no card.
    """
    options = streaming.OFF if options is None else options
    if not needs_machine(options):
        return None
    from tilestream.autoplan import Machine
    return Machine.detect(host_bytes=options.host_budget_bytes)


def declared_machine(*, free_bytes: int | None, name: str,
                     device_profile=None):
    """A planning machine from a figure ALREADY READ, for the review.

    ``free_bytes`` is what the review knows the card presents: the sizing
    probe's measurement under ``--auto-vram``, or the free figure a
    declared capacity is assumed to present.  Built through
    :func:`gpuwm.core.streaming.planner_machine` so the host budget comes
    from the same reader the run uses.  ``None`` when there is no card
    figure or the host RAM cannot be read.
    """
    if free_bytes is None:
        return None
    machine = streaming.planner_machine(vram_bytes=int(free_bytes), name=name)
    if machine is None:
        return None
    return replace(machine, device_profile=device_profile)


def price_child(cfg, options: streaming.StreamingOptions | None, *,
                machine, basis: str, vram_gib: float | None = None,
                profile=None, forcing_intervals: int | None = None,
                start_time: datetime | None = None) -> ChildPricing:
    """Price one standalone child and decide its ``[tiles]`` mode, once.

    ``machine`` is the planning machine both doors judge on: a cold
    ``Machine.detect`` in the runner, a :func:`declared_machine` in the
    review.  ``vram_gib`` and ``profile`` reach the estimator exactly as
    the fitted sizing route hands them: the measured card's own profile
    when the door measured one, the declared capacity otherwise, so a
    Noah-MP child is priced from the card's own reading when there is one.

    Raises :class:`tilestream.autoplan.CannotPlan` when the card is too
    small for the child even streamed, with the measured figure and the
    way out appended, so the sentence a reviewer reads at plan time is the
    sentence the runner would have raised before preprocessing.
    """
    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.experiment import experiment_from_run_config
    from tilestream import autoplan

    options = streaming.OFF if options is None else options
    if profile is None:
        profile = getattr(machine, "device_profile", None)
    exp = experiment_from_run_config(
        cfg, PRICING_EPOCH if start_time is None else start_time)
    estimate = None
    pricing_error = None
    try:
        estimate = estimate_experiment(
            exp, forcing_intervals=forcing_intervals, vram_gib=vram_gib,
            profile=profile)
    except Exception as error:  # noqa: BLE001 - a price is not a gate
        pricing_error = f"{type(error).__name__}: {error}"
    if estimate is not None and options.enabled and options.mode == "auto":
        # The same stamp ExperimentConfig puts on its own tree: the
        # configured, tiles-free experiment, so ``decide``'s admission path
        # answers resident when the configured envelope fits the budget
        # instead of falling through to the tile table.
        options = replace(
            options,
            resident_context=streaming.ResidentAdmissionContext(exp))
    free_bytes = (None if machine is None
                  else int(getattr(machine, "vram_bytes")))
    name = None if machine is None else str(getattr(machine, "name", ""))
    if needs_machine(options) and machine is None:
        return ChildPricing(
            decision=None, estimate=estimate, options=options,
            budget_bytes=None, machine_free_bytes=None, machine_name=None,
            basis=basis, pricing_error=pricing_error)
    try:
        decision = streaming.decide(
            cfg, options, machine=machine, resident_estimate=estimate)
    except autoplan.CannotPlan as error:
        if machine is None or error.resource != "vram":
            # A geometry refusal (the domain is too SMALL to tile) already
            # names its own way out; the card sentence belongs to a card
            # that is too small, and only there.
            raise
        gib = 1024 ** 3
        envelope = ("" if estimate is None else
                    f" The configured resident envelope is "
                    f"{estimate.peak_envelope_bytes / gib:.2f} GiB.")
        if basis == MEASURED_BASIS:
            card = (f"The card measured {free_bytes / gib:.2f} GiB free "
                    "when this decision was taken, before anything was "
                    "interpolated or allocated on the device")
            way_out = ("The way out is a smaller child (--child-size), "
                       "fewer levels (--child-levels), or freeing the card "
                       "before the run.")
        else:
            card = (f"The declared card is assumed to present "
                    f"{free_bytes / gib:.2f} GiB free")
            way_out = ("The way out is a smaller child (--child-size), "
                       "fewer levels (--child-levels), or pricing the real "
                       "card: --auto-vram measures the one in front of "
                       "you, --card or --vram-gib declares a larger one.")
        raise autoplan.CannotPlan(
            f"{error}  {card}.{envelope}  {way_out}",
            error.resource,
            dict(error.detail, machine_free_bytes=free_bytes,
                 basis=basis)) from error
    budget = (None if decision.budget_bytes is None
              else int(decision.budget_bytes))
    return ChildPricing(
        decision=decision, estimate=estimate, options=options,
        budget_bytes=budget, machine_free_bytes=free_bytes,
        machine_name=name, basis=basis, pricing_error=pricing_error)


__all__ = ["ChildPricing", "DECLARED_BASIS", "MEASURED_BASIS",
           "PRICING_EPOCH", "cold_machine", "declared_machine",
           "needs_machine", "price_child"]
