"""How often to assimilate, and what that cadence does to the tuning.

A WSR-88D in a precipitation VCP finishes a volume every 4-7 minutes.  A
cycle that fires every 15 minutes therefore ignores roughly two of every
three volumes the radar produced -- not because they arrived too late (the
live chunk feed delivers them mid-scan at a few seconds' lag) and not
because the cycle cannot afford them, but because the interval was a
constant.  This module makes the interval a *policy*, with the fixed one
preserved exactly.

Two modes:

``fixed``
    Analyses on a regular interval, choosing the volume nearest each mark.
    The behaviour the lane has today; reproduced here so that the A/B has
    a baseline arm and not a reconstruction of one.

``per-volume``
    One analysis per volume the radar actually produced, so the cycle
    length is whatever the antenna did -- 295 s and 418 s in the same hour
    when the VCP changes underneath it.

**The tuning is not cadence-invariant, and this module refuses to pretend
it is.**  Observation error, localization and inflation were all set
against 15-minute increments.  Two of the three have a defensible scaling
and one does not, and the difference is stated rather than smoothed over.

*Inflation (RTPS) scales, and here is the argument.*  RTPS relaxes the
posterior spread back toward the prior: ``sigma_a <- sigma_a [alpha
sigma_b/sigma_a + (1 - alpha)]``.  The quantity ``1 - alpha`` is the
fraction of the raw filter contraction that is allowed to stand at each
analysis -- a per-*cycle* deficit standing in for a continuous process.
Cycling k times more often applies that deficit k times more often, so
holding it per cycle draws spread down k times faster per unit time, which
is exactly the failure mode a shorter cadence is accused of.  Holding it
per unit *time* instead:

    (1 - alpha') / dt' = (1 - alpha) / dt      =>
    alpha' = 1 - (1 - alpha) * dt' / dt

is the discretization-consistent choice: the ensemble sees the same total
relaxation per quarter hour however many analyses that quarter hour is cut
into, which is the property the 15-minute tuning actually established.  At
alpha = 0.9 and dt = 900 s, a 340 s cadence wants alpha' = 0.962.

*Observation error scales, and here is the weaker argument.*  Volumes 5.7
minutes apart share beam geometry, clutter, ground targets and
representativeness error against a 3 km mass grid; their errors are
correlated on a timescale comparable to the volume period itself.
Assimilating them as independent multiplies the effective observation
count by ``dt / dt'``.  Inflating sigma_o by ``sqrt(dt / dt')`` restores
the information content per unit time to the tuned value, so a
shorter-cadence arm can only win through the *timeliness* of its
corrections and never through quietly counting the same radar twice.
The scaling is **one-sided**: ``max(1, sqrt(dt / dt'))``.  Run backwards
it would say a slower cadence makes each observation more accurate,
which is a claim about the radar rather than about the cadence -- and
the filter refuses an observation-error inflation below 1 for that exact
reason, which is how a first per-volume leg longer than the baseline
took an arm down.
This is deliberately conservative.  If the A/B shows the scaled arm losing
to the unscaled one, the correlation assumption is too strong and the
exponent belongs somewhere between 0 and 0.5 -- which is a measurement
this module cannot make for the caller.

*Localization does NOT scale, and inventing one would be dressing a guess
as a derivation.*  The radius is set by the ensemble's sampling-error
structure and by the physical correlation length of the increment, neither
of which is a function of how often the filter runs.  Shorter cycles do
mean less error growth between analyses, which argues weakly for a shorter
radius; that is second order, it was never measured at 15 minutes either,
and it is left exposed and flagged for retuning rather than scaled.

**Nothing here is applied silently.**  :func:`scaled_settings` returns the
scaled values *and* the reasoning *and* the untouched originals, every
caller records the whole block, and ``scaling="none"`` is a first-class
choice that keeps the old constants on purpose -- which is the control arm
an A/B needs, not an oversight.

No site names, no case names: volume times and intervals are arguments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

#: Cadence policies.  Membership is checked, so a typo is a refusal rather
#: than a silent fall-through to whichever one came first.
CADENCE_MODES = ("fixed", "per-volume")

#: What to do when volumes arrive faster than a cycle can be computed.
#: There is no fourth option that drops one quietly.
OVERRUN_POLICIES = ("refuse", "skip", "queue")

#: How the 15-minute tuning is carried to another cadence.
SCALING_MODES = ("none", "documented")

#: The interval every scaling below is expressed relative to: the cadence
#: the shipped observation-error, localization and inflation settings were
#: tuned at.  It is a *reference*, not a default cadence -- a caller states
#: its own baseline, and this is what the lane's was.
TUNED_BASELINE_INTERVAL_S = 900.0

#: Beyond this ratio of baseline to actual interval the scalings above are
#: extrapolations rather than interpolations, and say so in the receipt.
EXTRAPOLATION_WARN_RATIO = 4.0


class CadenceError(ValueError):
    """A cadence that cannot mean what the caller needs it to mean."""


def _utc(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(stamp: datetime) -> str:
    return _utc(stamp).strftime("%Y-%m-%dT%H:%M:%SZ")


def quantize_seconds(seconds: float, dt: float) -> float:
    """Snap a leg boundary onto the model's own timestep lattice.

    The cycling driver places a resumed clock with
    ``steps = round(start_seconds / dt)``, so a leg boundary that is not a
    whole number of steps is rounded *there*, silently, after the plan has
    already been written down.  Doing it here instead makes the shift a
    recorded quantity: a volume at 04:17:53 is 1073 s into a case that
    starts at 04:00 with dt = 15 s, which is 71.53 steps, and the cycle
    actually lands at 1065 s -- 8 s early, far inside the observation
    offset ceiling, and now written in the plan rather than discovered in
    a clock.
    """

    step = float(dt)
    if not math.isfinite(step) or step <= 0.0:
        raise CadenceError(
            f"dt is {dt!r}; a timestep lattice needs a finite positive step")
    return round(float(seconds) / step) * step


@dataclass(frozen=True)
class Cycle:
    """One analysis: when it lands, and how long the leg into it runs."""

    #: Seconds from the case anchor to this analysis, on the dt lattice.
    elapsed_seconds: float
    #: Duration of the forecast leg that ends at this analysis.
    leg_seconds: float
    #: The analysis time this cycle asks the feed for.
    valid_time: datetime
    #: The radar volume this cycle was planned around, when the plan came
    #: from real volume times; ``None`` for a fixed-interval mark, which
    #: is a request rather than an observation that exists.
    volume_time: datetime | None
    #: Seconds the dt lattice moved the analysis away from ``volume_time``.
    quantization_shift_seconds: float

    def to_payload(self) -> dict:
        return {
            "elapsed_seconds": float(self.elapsed_seconds),
            "leg_seconds": float(self.leg_seconds),
            "valid_time": _iso(self.valid_time),
            "volume_time": (None if self.volume_time is None
                            else _iso(self.volume_time)),
            "quantization_shift_seconds": round(
                float(self.quantization_shift_seconds), 3),
        }


@dataclass(frozen=True)
class CadencePlan:
    """A cycling schedule, and everything needed to defend it."""

    mode: str
    cycles: tuple[Cycle, ...]
    anchor: datetime
    dt_seconds: float
    #: Volumes inside the window that this plan does NOT assimilate, with
    #: the reason.  Never empty for ``fixed`` over a real feed -- that is
    #: the whole point of measuring it.
    unused_volumes: tuple[dict, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def intervals(self) -> tuple[float, ...]:
        return tuple(cycle.leg_seconds for cycle in self.cycles)

    @property
    def mean_interval_seconds(self) -> float:
        legs = self.intervals
        return sum(legs) / len(legs) if legs else 0.0

    def to_payload(self) -> dict:
        legs = self.intervals
        return {
            "schema": "gpuwm-da.cadence-plan.v1",
            "mode": self.mode,
            "anchor": _iso(self.anchor),
            "dt_seconds": float(self.dt_seconds),
            "cycle_count": len(self.cycles),
            "interval_seconds": {
                "min": min(legs) if legs else None,
                "max": max(legs) if legs else None,
                "mean": round(self.mean_interval_seconds, 3) if legs else None,
            },
            "cycles": [cycle.to_payload() for cycle in self.cycles],
            "unused_volumes": list(self.unused_volumes),
            "unused_volume_count": len(self.unused_volumes),
            "notes": list(self.notes),
        }


def _clean_volume_times(volume_times) -> list[datetime]:
    stamps = sorted({_utc(stamp) for stamp in volume_times})
    if not stamps:
        raise CadenceError(
            "no volume times were supplied; a cadence cannot be planned "
            "against a feed that listed nothing, and assuming a nominal "
            "VCP period here would invent data the radar did not send")
    return stamps


def plan_fixed(volume_times, *, anchor: datetime, interval_seconds: float,
               cycles: int, dt_seconds: float,
               max_offset_seconds: float = 480.0) -> CadencePlan:
    """The behaviour the lane has today, written down so it can be a control.

    Analyses land on ``anchor + n * interval_seconds``.  The volume nearest
    each mark is the one that serves it, and every other volume in the
    window is recorded in ``unused_volumes`` -- which is the measurement
    that makes the case for the other mode.
    """

    if interval_seconds <= 0.0:
        raise CadenceError(
            f"interval_seconds is {interval_seconds!r}; a fixed cadence "
            "needs a positive interval")
    if int(cycles) < 1:
        raise CadenceError(
            f"cycles is {cycles!r}; a plan with no analysis is not a plan")
    stamps = _clean_volume_times(volume_times)
    anchor = _utc(anchor)

    planned: list[Cycle] = []
    served: set[datetime] = set()
    for index in range(1, int(cycles) + 1):
        elapsed = quantize_seconds(index * float(interval_seconds),
                                   dt_seconds)
        mark = anchor + timedelta(seconds=elapsed)
        nearest = min(stamps, key=lambda s: abs((s - mark).total_seconds()))
        offset = abs((nearest - mark).total_seconds())
        if offset > float(max_offset_seconds):
            raise CadenceError(
                f"the fixed mark {_iso(mark)} has no volume within "
                f"{float(max_offset_seconds):.0f} s (nearest is "
                f"{_iso(nearest)}, {offset:.0f} s away); a cycle fed a "
                "stale volume should be a decision, not an accident")
        served.add(nearest)
        planned.append(Cycle(
            elapsed_seconds=elapsed,
            leg_seconds=(elapsed - planned[-1].elapsed_seconds
                         if planned else elapsed),
            valid_time=mark, volume_time=None,
            quantization_shift_seconds=0.0))

    first = planned[0].valid_time - timedelta(
        seconds=planned[0].leg_seconds)
    last = planned[-1].valid_time
    unused = tuple(
        {"volume_time": _iso(stamp),
         "reason": "no fixed mark selected it"}
        for stamp in stamps
        if stamp not in served and first <= stamp <= last)
    notes = (
        f"fixed {float(interval_seconds):.0f} s cadence; "
        f"{len(served)} of {len(served) + len(unused)} volumes inside the "
        "cycled window were assimilated",
    )
    return CadencePlan(mode="fixed", cycles=tuple(planned), anchor=anchor,
                       dt_seconds=float(dt_seconds), unused_volumes=unused,
                       notes=notes)


def plan_per_volume(volume_times, *, anchor: datetime, dt_seconds: float,
                    window_start: datetime | None = None,
                    window_end: datetime | None = None,
                    min_interval_seconds: float = 0.0) -> CadencePlan:
    """One analysis per volume the radar actually produced.

    ``min_interval_seconds`` collapses volumes closer together than the
    filter can usefully separate -- a site restarting a VCP can publish
    two volumes a minute apart, and two analyses against what is nearly
    one scan is double counting rather than extra information.  Every
    volume it drops is recorded with that reason.  It defaults to 0, which
    keeps every volume and makes the collapsing an explicit request.
    """

    stamps = _clean_volume_times(volume_times)
    anchor = _utc(anchor)
    start = _utc(window_start) if window_start is not None else None
    end = _utc(window_end) if window_end is not None else None

    unused: list[dict] = []
    inside: list[datetime] = []
    for stamp in stamps:
        if start is not None and stamp < start:
            continue
        if end is not None and stamp > end:
            continue
        if stamp <= anchor:
            unused.append({"volume_time": _iso(stamp),
                           "reason": "at or before the case anchor"})
            continue
        inside.append(stamp)
    if not inside:
        raise CadenceError(
            "no volume falls inside the requested window and after the "
            f"case anchor {_iso(anchor)}; there is nothing to cycle on")

    planned: list[Cycle] = []
    previous = 0.0
    for stamp in inside:
        raw = (stamp - anchor).total_seconds()
        elapsed = quantize_seconds(raw, dt_seconds)
        leg = elapsed - previous
        if leg <= 0.0:
            unused.append({
                "volume_time": _iso(stamp),
                "reason": (f"quantizing to the {float(dt_seconds):.0f} s "
                           "timestep put it at or before the previous "
                           "analysis")})
            continue
        if leg < float(min_interval_seconds):
            unused.append({
                "volume_time": _iso(stamp),
                "reason": (f"{leg:.0f} s after the previous analysis, "
                           f"below the {float(min_interval_seconds):.0f} s "
                           "floor")})
            continue
        planned.append(Cycle(
            elapsed_seconds=elapsed, leg_seconds=leg,
            valid_time=anchor + timedelta(seconds=elapsed),
            volume_time=stamp,
            quantization_shift_seconds=elapsed - raw))
        previous = elapsed

    if not planned:
        raise CadenceError(
            "every volume in the window was collapsed away; the cadence "
            "floor is above the radar's own volume period")
    legs = [cycle.leg_seconds for cycle in planned]
    notes = (
        f"per-volume cadence from {len(planned)} volumes; leg lengths "
        f"{min(legs):.0f}-{max(legs):.0f} s (mean {sum(legs)/len(legs):.0f} s)"
        " -- the antenna's own rhythm, not a constant",
        "analysis times are snapped to the model timestep lattice; the "
        "per-cycle shift is recorded and is bounded by half a timestep",
    )
    return CadencePlan(mode="per-volume", cycles=tuple(planned),
                       anchor=anchor, dt_seconds=float(dt_seconds),
                       unused_volumes=tuple(unused), notes=notes)


def scaled_settings(*, cycle_interval_s: float,
                    baseline_interval_s: float = TUNED_BASELINE_INTERVAL_S,
                    rtps_alpha: float, error_inflation: float,
                    horizontal_loc_m: float, vertical_loc_m: float,
                    scaling: str = "documented") -> dict:
    """Carry a 15-minute tuning to another cadence, showing the working.

    Returns a block with the values to use, the values it started from,
    and the reasoning for each -- including for the one setting that is
    deliberately NOT scaled.  A caller records the whole thing; nothing
    here is meant to be read as a number alone.
    """

    if scaling not in SCALING_MODES:
        raise CadenceError(
            f"unknown scaling {scaling!r}; expected one of "
            f"{', '.join(SCALING_MODES)}")
    dt_new = float(cycle_interval_s)
    dt_ref = float(baseline_interval_s)
    for name, value in (("cycle_interval_s", dt_new),
                        ("baseline_interval_s", dt_ref)):
        if not math.isfinite(value) or value <= 0.0:
            raise CadenceError(
                f"{name} is {value!r}; both intervals must be finite and "
                "positive for their ratio to mean anything")
    if not (0.0 <= float(rtps_alpha) <= 1.0):
        raise CadenceError(
            f"rtps_alpha is {rtps_alpha!r}; RTPS relaxes between the raw "
            "posterior and the prior spread and must lie in [0, 1]")
    if float(error_inflation) <= 0.0:
        raise CadenceError(
            f"error_inflation is {error_inflation!r}; it multiplies an "
            "observation-error standard deviation and cannot be "
            "non-positive")

    ratio = dt_new / dt_ref
    block = {
        "schema": "gpuwm-da.cadence-scaling.v1",
        "scaling": scaling,
        "baseline_interval_s": dt_ref,
        "cycle_interval_s": dt_new,
        "interval_ratio": round(ratio, 6),
        "baseline": {
            "rtps_alpha": float(rtps_alpha),
            "error_inflation": float(error_inflation),
            "horizontal_loc_m": float(horizontal_loc_m),
            "vertical_loc_m": float(vertical_loc_m),
        },
    }

    if scaling == "none":
        block["applied"] = dict(block["baseline"])
        block["reasoning"] = {
            "rtps_alpha": (
                "UNSCALED BY REQUEST. At a shorter cadence this draws "
                "ensemble spread down faster per unit time than the "
                "15-minute tuning did, because the same per-cycle "
                "relaxation deficit is applied more often. This is the "
                "control arm, and it is expected to under-disperse"),
            "error_inflation": (
                "UNSCALED BY REQUEST. Successive volumes are treated as "
                "independent, so a shorter cadence assimilates more "
                "nominal information per unit time than the tuning "
                "assumed"),
            "localization": (
                "unscaled, as in every mode; see the 'documented' mode's "
                "reasoning for why no scaling is offered"),
        }
        block["needs_retuning"] = [
            "rtps_alpha", "error_inflation", "horizontal_loc_m",
            "vertical_loc_m"]
        return block

    alpha_new = 1.0 - (1.0 - float(rtps_alpha)) * ratio
    # A cadence longer than the baseline drives alpha below 0; clamping is
    # not a repair, so it is refused rather than silently floored.
    if not (0.0 <= alpha_new <= 1.0):
        raise CadenceError(
            f"scaling rtps_alpha {float(rtps_alpha)} from "
            f"{dt_ref:.0f} s to {dt_new:.0f} s gives {alpha_new:.4f}, "
            "outside [0, 1]. The linear per-unit-time relaxation argument "
            "does not reach this far; state an alpha for this cadence "
            "directly rather than having a clamp invent one")
    # ONE-SIDED, and the asymmetry is the point.  The correlation
    # argument says that assimilating the same radar more often counts
    # one atmosphere several times, so sigma_o must go UP as the cadence
    # shortens.  Run it backwards and it claims that cycling more slowly
    # makes each observation more accurate -- which is a statement about
    # the radar, not about the cadence, and nobody measured it.  The
    # filter refuses a velocity_error_inflation below 1 for exactly that
    # reason, so a leg longer than the baseline holds the tuned value
    # instead of deflating below it.
    inflation_new = float(error_inflation) * max(
        1.0, math.sqrt(dt_ref / dt_new))

    block["applied"] = {
        "rtps_alpha": round(alpha_new, 6),
        "error_inflation": round(inflation_new, 6),
        "horizontal_loc_m": float(horizontal_loc_m),
        "vertical_loc_m": float(vertical_loc_m),
    }
    block["formulae"] = {
        "rtps_alpha": "alpha' = 1 - (1 - alpha) * dt' / dt",
        "error_inflation": "infl' = infl * max(1, sqrt(dt / dt'))",
        "localization": "unchanged",
    }
    block["reasoning"] = {
        "rtps_alpha": (
            "(1 - alpha) is the fraction of the raw filter contraction "
            "left standing at each analysis -- a per-cycle deficit "
            "discretizing a continuous relaxation. Holding it per cycle "
            "while cycling k times more often draws spread down k times "
            "faster per unit time. Holding (1 - alpha)/dt invariant makes "
            "the total relaxation per unit time cadence-independent, "
            "which is the property the 15-minute tuning established"),
        "error_inflation": (
            "volumes minutes apart share beam geometry, clutter, ground "
            "targets and representativeness error, so their errors are "
            "correlated on the volume period itself. Assimilating them as "
            "independent multiplies the effective observation count by "
            "dt/dt'; inflating sigma_o by sqrt(dt/dt') restores the "
            "information per unit time to the tuned value, so a faster "
            "arm can only win on the timeliness of its corrections. This "
            "is deliberately conservative -- if the scaled arm loses to "
            "the unscaled one the exponent belongs below 0.5, and only "
            "the A/B can say. The scaling is ONE-SIDED: run backwards it "
            "would claim a slower cadence makes each observation more "
            "accurate, which is a statement about the radar and not "
            "about the cadence, so a leg longer than the baseline holds "
            "the tuned value rather than deflating below it"),
        "localization": (
            "NOT SCALED, and no scaling is offered. The radius is set by "
            "ensemble sampling error and the physical correlation length "
            "of the increment, neither a function of cycling frequency. "
            "Shorter cycles mean less error growth between analyses, "
            "which argues weakly for a shorter radius; that is second "
            "order, was never measured at 15 minutes either, and a "
            "formula for it would be a guess wearing a derivation"),
    }
    block["needs_retuning"] = ["horizontal_loc_m", "vertical_loc_m"]
    if dt_ref / dt_new > EXTRAPOLATION_WARN_RATIO:
        block["extrapolation_warning"] = (
            f"the cadence is {dt_ref / dt_new:.1f}x faster than the tuned "
            f"baseline, past the {EXTRAPOLATION_WARN_RATIO:.0f}x point "
            "where these are extrapolations rather than interpolations; "
            "treat the applied values as a starting point for retuning "
            "and not as a result")
    return block


def check_overrun(plan: CadencePlan, *, cycle_cost_seconds: float,
                  policy: str = "refuse") -> tuple[CadencePlan, dict]:
    """Decide what happens when volumes outrun the cost of using them.

    ``cycle_cost_seconds`` is a MEASURED wall time for one assimilation
    cycle on the hardware that will run it, not an estimate: the whole
    point of this check is that a plan which cannot keep up is caught
    before it runs rather than discovered as a growing lag.

    Three policies, and there is deliberately no fourth that drops a
    volume quietly:

    ``refuse``
        Fail closed.  The default, because a nowcast silently falling
        behind real time is worse than one that never started.
    ``skip``
        Keep the cadence real-time by dropping volumes that would start a
        cycle before the previous one finished.  Every dropped volume is
        recorded with its reason, so a thinner analysis always says so.
    ``queue``
        Assimilate every volume and let the analysis lag.  Honest for a
        replay, where there is no real time to fall behind; for a live
        feed the lag grows without bound and the returned record says by
        how much per cycle.
    """

    if policy not in OVERRUN_POLICIES:
        raise CadenceError(
            f"unknown overrun policy {policy!r}; expected one of "
            f"{', '.join(OVERRUN_POLICIES)}")
    cost = float(cycle_cost_seconds)
    if not math.isfinite(cost) or cost <= 0.0:
        raise CadenceError(
            f"cycle_cost_seconds is {cycle_cost_seconds!r}; the check "
            "compares a measured wall time against the cadence and needs "
            "a positive one")

    legs = plan.intervals
    tight = [(index, leg) for index, leg in enumerate(legs) if leg < cost]
    record = {
        "schema": "gpuwm-da.cadence-overrun.v1",
        "policy": policy,
        "cycle_cost_seconds": cost,
        "shortest_interval_seconds": min(legs) if legs else None,
        "duty_cycle_at_mean_interval": (
            round(cost / plan.mean_interval_seconds, 4)
            if plan.mean_interval_seconds else None),
        "cycles_shorter_than_cost": len(tight),
    }
    if not tight:
        record["outcome"] = "clear"
        record["detail"] = (
            f"every cycle interval is at least {cost:.0f} s, so no volume "
            "arrives before its predecessor has been assimilated")
        return plan, record

    detail = "; ".join(
        f"cycle {index} runs {leg:.0f} s against a {cost:.0f} s cost"
        for index, leg in tight[:5])
    if policy == "refuse":
        record["outcome"] = "refused"
        record["detail"] = detail
        raise CadenceError(
            f"{len(tight)} of {len(legs)} cycles are shorter than the "
            f"measured {cost:.0f} s cost of one cycle ({detail}). The "
            "overrun policy is 'refuse', so nothing was planned. Choose "
            "'skip' to hold real time and drop the volumes that will not "
            "fit -- recorded, never silent -- or 'queue' to assimilate "
            "every volume and accept a growing lag")

    if policy == "queue":
        backlog = sum(cost - leg for _, leg in tight)
        record["outcome"] = "queued"
        record["projected_backlog_seconds"] = round(backlog, 1)
        record["detail"] = (
            f"{detail}. Every volume is assimilated and the analysis "
            f"falls {backlog:.0f} s behind real time over the plan. "
            "Sound for a replay; against a live feed this lag grows "
            "without bound")
        return plan, record

    kept: list[Cycle] = []
    dropped: list[dict] = list(plan.unused_volumes)
    carried = 0.0
    for cycle in plan.cycles:
        available = cycle.leg_seconds + carried
        if available < cost:
            carried = available
            dropped.append({
                "volume_time": (None if cycle.volume_time is None
                                else _iso(cycle.volume_time)),
                "reason": (f"skipped: only {available:.0f} s since the "
                           f"last analysis against a {cost:.0f} s cycle "
                           "cost")})
            continue
        kept.append(Cycle(
            elapsed_seconds=cycle.elapsed_seconds, leg_seconds=available,
            valid_time=cycle.valid_time, volume_time=cycle.volume_time,
            quantization_shift_seconds=cycle.quantization_shift_seconds))
        carried = 0.0
    if not kept:
        raise CadenceError(
            f"skipping to fit a {cost:.0f} s cycle cost left no analysis "
            "at all; the hardware cannot cycle this case at any cadence "
            "the radar offers")
    record["outcome"] = "skipped"
    record["cycles_dropped"] = len(plan.cycles) - len(kept)
    record["detail"] = (
        f"{detail}. {len(plan.cycles) - len(kept)} volume(s) dropped to "
        "hold real time; each one is listed in unused_volumes with its "
        "reason")
    thinned = CadencePlan(
        mode=plan.mode, cycles=tuple(kept), anchor=plan.anchor,
        dt_seconds=plan.dt_seconds, unused_volumes=tuple(dropped),
        notes=plan.notes + (record["detail"],))
    return thinned, record
