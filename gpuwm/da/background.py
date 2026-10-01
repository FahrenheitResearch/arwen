"""Source metadata for the background the regional cycling driver starts from.

The driver has always ridden ONE deterministic background and made its
ensemble by perturbing it (:mod:`gpuwm.da.perturb`, applied once at leg
0).  That is source-independent: nothing in the spread mechanism reads
the background's provenance.  What IS source-dependent is everything
around it -- how far back the newest usable cycle is, how many forecast
hours that cycle publishes, whether the source's native grid even
reaches the domain, and what the first guess contains.

This module is that source-dependent half, in one place, as data.  It
is deliberately a REGISTRY rather than an if/else on a source name:
HRRR is on NOAA's retirement path behind RRFS, so the next background
this driver learns must be an entry here, not another branch.

Three things live here and nothing else:

1. :data:`BACKGROUND_SOURCES` -- a live source-registry projection of cycle cadence, publication
   lag, forecast horizon and forcing interval, plus the coverage
   question, with the numbers each source's own contract module owns
   (:mod:`gpuwm.hrrr_forecast` for HRRR's 18 h/48 h horizon,
   :mod:`gpuwm.ingest.hrrr_target` for its grid envelope).
2. :func:`plan_background_cycle` -- the newest cycle at or before a
   requested model init whose files are plausibly published, and the
   forecast hour that init falls on.  Refuses past the horizon.
3. :func:`plan_member_backgrounds` -- HOW each trajectory's background
   was constructed, one record per trajectory, with the refusals that
   keep an ensemble from being fabricated.

Nothing here reaches the network, the GPU, or a prepared bundle.  It is
plan-time arithmetic and refusals, so a front door can prove a case is
buildable before it pays for a byte.

Compatibility timing defaults are not a capability roster.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

#: Schema of the provenance block :func:`background_receipt` returns.
BACKGROUND_RECEIPT_SCHEMA = "gpuwm-da.background-provenance.v1"

#: The only ensemble construction this driver implements, named so the
#: receipt says which one ran rather than leaving a reader to infer it
#: from the absence of an alternative.
PERTURBED_DETERMINISTIC = "perturbed-deterministic"

#: Constructions a caller may ASK for, mapped to why they are or are
#: not reachable.  A named-but-unreachable mode refuses with its own
#: sentence; an unknown name refuses with the list.  Both are better
#: than quietly building the only mode that exists.
ENSEMBLE_CONSTRUCTIONS = MappingProxyType({
    PERTURBED_DETERMINISTIC: None,
    "lagged-cycle": (
        "a lagged-cycle ensemble gives each member a DIFFERENT prepared "
        "case (its own cycle, its own boundary series, its own prepared "
        "cache digest).  This driver restores one prepared cache for "
        "every trajectory and tools/da_ensemble_state.py's "
        "EnsembleIdentity binds prepared_content_sha256 for the whole "
        "generation, so N lagged members cannot share one generation "
        "today.  Building it means N prepared cases and a cross-case "
        "carry-forward, which is unspecced engineering, not a flag"),
    "multi-source": (
        "a multi-source ensemble mixes backgrounds from different "
        "models in one generation, and has the same one-prepared-cache "
        "blocker as lagged-cycle.  It is the shape a public "
        "convection-allowing ensemble feed would want; revisit it when "
        "one exists"),
})


class BackgroundError(ValueError):
    """A background source, cycle, domain or ensemble that is refused."""


@dataclass(frozen=True)
class BackgroundSource:
    """Everything plan-time arithmetic needs to know about one source.

    ``publication_lag_seconds`` is a FUNCTION of the newest lead the
    window needs, not a constant, because the two sources publish
    differently: GFS's hourly files for a cycle land together and are
    treated as one four-hour wait, while HRRR streams its leads out over
    roughly an hour, so a window ending at f10 is usable well before the
    same cycle's f18 exists.  Modelling that as one number would either
    make HRRR wait for leads it does not want or claim leads that are
    not there yet.
    """

    name: str
    label: str
    cycle_step_hours: int | None
    forcing_interval_seconds: float | None
    publication_lag_seconds: Callable[[int], float]
    horizon_hours: Callable[[datetime], int]
    #: ``True`` when the source's native grid is smaller than the globe,
    #: so a domain has to be proved against it before a fetch.
    coverage_bounded: bool
    #: What the first guess carries that the DA cannot create from
    #: radial velocity alone.  Reported, never gated on.
    initial_hydrometeors: str
    #: Where a refused domain should go instead.  Source-neutral words:
    #: it names another registry entry, not a case or a site.
    coverage_fallback: str | None
    #: ``True`` when ``publication_lag_seconds`` is the source's posting
    #: row on its expected line (:func:`_row_lag_seconds`), so a planned
    #: cycle can still be posting its last leads and the run's fetch
    #: waits for them (:func:`late_background_wait_minutes`).  ``False``
    #: for a fixed shipped wait that is not a posting row.
    plans_on_posting_row: bool = True

    def horizon(self, cycle: datetime) -> int:
        return int(self.horizon_hours(cycle))

    def lag_seconds(self, last_lead_hours: int) -> float:
        return float(self.publication_lag_seconds(int(last_lead_hours)))


def _gfs_horizon(cycle: datetime) -> int:
    # Hourly GFS output stops at f120; the driver's window never goes
    # near it, but the ceiling is the source's, not the driver's.
    del cycle
    return 120


def _hrrr_horizon(cycle: datetime) -> int:
    from gpuwm.hrrr_forecast import hrrr_cycle_horizon

    return hrrr_cycle_horizon(cycle)


#: GFS: the driver's shipped wait, unchanged.  One flat number, matching
#: what the GFS front door has always applied, so a GFS plan derived
#: here is the plan the GFS route already produced.
GFS_AVAILABILITY_LAG_S = 4 * 3600.0

def _gfs_lag(last_lead_hours: int) -> float:
    del last_lead_hours
    return GFS_AVAILABILITY_LAG_S


def _row_lag_seconds(source_id: str, last_lead_hours: int) -> float:
    """When the source's ``last_lead_hours`` is due, from its table row, seconds.

    The source's ``publication_lag`` rules (the fetch route table's rows,
    or its ``legacy_posting`` row for a legacy transport) on the EXPECTED
    line, the time :func:`gpuwm.source_posting.expected_at` gives a lead,
    for the slowest of the cycle hours whose forecast reaches that lead,
    since no cycle is named here.

    The expected line, not the latest posting seen: a plan on the latest
    line waits out the slowest run any watch saw (HRRR's late 30 Sep 06Z
    and 07Z runs put every HRRR plan about an hour behind the cycle that
    was out).  A background that runs later than its expected time is
    waited for by the run's fetch until its row calls it late
    (:func:`late_background_wait_minutes`); it is not skipped for an
    older cycle and it does not fail.

    Only the hours whose horizon reaches the lead: HRRR's off-synoptic
    runs stop at f018, so their steeper per-lead rule must not date a
    synoptic run's f019 to f048.  A lead no hour reaches is refused by
    the source's ladder in words that name its horizon, so the maximum
    over every hour there only keeps this arithmetic total.
    """

    from gpuwm.source_cycles import cycle_grid_for

    grid = cycle_grid_for(source_id, posting=True)
    if grid is None:
        return 0.0
    lead = int(last_lead_hours)
    cycles = [datetime(2001, 1, 1, hour) for hour in grid.hours]
    reaching = [cycle for cycle in cycles
                if grid.horizon(cycle) is None or grid.horizon(cycle) >= lead]
    return max(grid.delay(cycle, lead) for cycle in (reaching or cycles)) * 3600.0


def _hrrr_lag(last_lead_hours: int) -> float:
    # HRRR posts its leads over the better part of two hours, so the wait
    # tracks the window's last lead.  The curve is the hrrr row of the
    # route table's legacy_posting, measured, not a constant here.
    return _row_lag_seconds("hrrr", last_lead_hours)


_LEGACY_BACKGROUND_DEFAULTS: Mapping[str, BackgroundSource] = MappingProxyType({
    "gfs": BackgroundSource(
        name="gfs",
        label="GFS 0.25 degree",
        cycle_step_hours=6,
        forcing_interval_seconds=3600.0,
        publication_lag_seconds=_gfs_lag,
        horizon_hours=_gfs_horizon,
        coverage_bounded=False,
        initial_hydrometeors="explicit zero (WRF Vtable.GFS parity)",
        coverage_fallback=None,
        plans_on_posting_row=False,
    ),
    "hrrr": BackgroundSource(
        name="hrrr",
        label="HRRR 3 km CONUS",
        cycle_step_hours=1,
        forcing_interval_seconds=3600.0,
        publication_lag_seconds=_hrrr_lag,
        horizon_hours=_hrrr_horizon,
        coverage_bounded=True,
        initial_hydrometeors="decoded natively (QC/QI/QR/QS/QG)",
        coverage_fallback="gfs",
    ),
})

#: The source a caller gets by saying nothing.  Changing this changes
#: what every existing invocation does, so it is named once, here.
DEFAULT_BACKGROUND_SOURCE = "gfs"


class _BackgroundSources(Mapping):
    """Live projection. Compatibility defaults do not decide membership."""

    def __iter__(self):
        from gpuwm.source_adapters import source_adapters
        return iter(row.source_id for row in source_adapters() if row.runnable)

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, name):
        from gpuwm.source_adapters import get_source_adapter
        from gpuwm.source_cycles import cycle_grid_for
        adapter = get_source_adapter(name)
        if not adapter.runnable:
            raise KeyError(name)
        if adapter.source_id in _LEGACY_BACKGROUND_DEFAULTS:
            return _LEGACY_BACKGROUND_DEFAULTS[adapter.source_id]
        grid = cycle_grid_for(adapter.source_id)
        gaps = (() if grid is None else tuple(
            (grid.hours[(i + 1) % len(grid.hours)] - hour) % 24 or 24
            for i, hour in enumerate(grid.hours)))
        step = gaps[0] if gaps and len(set(gaps)) == 1 else None
        def horizon(cycle):
            declared = None if grid is None else grid.horizon(cycle)
            return adapter.max_forecast_hour if declared is None else declared
        return BackgroundSource(
            name=adapter.source_id, label=adapter.display_title,
            cycle_step_hours=step, forcing_interval_seconds=adapter.forcing_interval_seconds,
            # No cycle is named here, so the slowest of the source's
            # cycle hours that reach that lead, on the expected line
            # (_row_lag_seconds says why).
            publication_lag_seconds=lambda lead: _row_lag_seconds(
                adapter.source_id, lead),
            horizon_hours=horizon, coverage_bounded=adapter.coverage_window is not None,
            initial_hydrometeors="See the bound native or mapped preparation inventory; not inferred from transport.",
            coverage_fallback=None)


BACKGROUND_SOURCES: Mapping[str, BackgroundSource] = _BackgroundSources()


def resolve_background_source(name: str) -> BackgroundSource:
    """Resolve aliases through the source owner, without a regional roster."""
    try:
        return BACKGROUND_SOURCES[name]
    except KeyError:
        raise BackgroundError(f"Source {name!r} declares no runnable preparation route") from None
    except ValueError as exc:
        raise BackgroundError(str(exc)) from exc


def late_background_wait_minutes(name: str, cycle: datetime, last_lead: int,
                                 *, now: datetime) -> int | None:
    """Minutes a run's fetch still waits for a planned background, or None.

    A plan dated on the posting row's expected line (:func:`_row_lag_seconds`)
    takes the newest cycle whose window is due, not one every watch has
    seen posted, so a cycle running late can still be posting its last
    leads when the fetch starts.  The fetch waits for them, saying which
    lead it waits on, until the window's last lead is late on its row
    (:func:`gpuwm.source_posting.late_at`: its expected time plus the
    row's ``late_after_minutes``, which is never shorter than the row's
    measured late spread), instead of refusing the cycle as unpublished.
    The answer is the minutes from ``now`` to that time, rounded up.

    None where nothing is left to wait for: the window's late time has
    passed (every lead is out, or the source fell behind its budget and
    the fetch's own publication check says so at once), the source's plan
    is a fixed shipped wait rather than its posting row, or no run waits
    on its row.
    """

    entry = resolve_background_source(name)
    if not entry.plans_on_posting_row:
        return None
    from gpuwm import fetch_routes, source_posting

    if entry.name not in fetch_routes.posting_sources():
        return None
    late = source_posting.late_at(entry.name, cycle, int(last_lead))
    if late is None:
        return None
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    remaining = (late - now).total_seconds()
    return math.ceil(remaining / 60.0) if remaining > 0 else None


@dataclass(frozen=True)
class BackgroundCycle:
    """One resolved background: which cycle, and where init sits in it."""

    source: str
    cycle: datetime
    forecast_start_hour: int
    forecast_end_hour: int
    init: datetime
    horizon_hours: int
    publication_lag_seconds: float
    selected_forecast_hours: tuple[int, ...] = ()

    @property
    def forecast_hours(self) -> tuple[int, ...]:
        if self.selected_forecast_hours:
            return self.selected_forecast_hours
        return tuple(range(self.forecast_start_hour,
                           self.forecast_end_hour + 1))

    def as_receipt(self) -> dict[str, object]:
        return {
            "source": self.source,
            "cycle": self.cycle.strftime("%Y-%m-%dT%H"),
            "init": self.init.isoformat(),
            "forecast_start_hour": self.forecast_start_hour,
            "forecast_end_hour": self.forecast_end_hour,
            "forecast_hours": list(self.forecast_hours),
            "cycle_horizon_hours": self.horizon_hours,
            "background_age_hours": self.forecast_start_hour,
            "publication_lag_seconds": self.publication_lag_seconds,
        }


def plan_background_cycle(
        source: str, *, init: datetime, now: datetime, run_seconds: float,
) -> BackgroundCycle:
    """Compatibility forecast-cycle view of the shared selection contract.

    The shared planner checks every required lead and can try an older extended
    cycle. Analysis sequences use background_contract.plan directly because
    their absolute valid times must not be relabeled as forecast leads.
    """
    from gpuwm.background_contract import plan
    try:
        selected = plan(source, init=init, now=now, run_seconds=run_seconds)
    except ValueError as exc:
        raise BackgroundError(str(exc)) from exc
    if selected.time_axis != "forecast_leads":
        raise BackgroundError("This is an analysis sequence; use background_contract.plan and its absolute valid times, not a forecast-cycle receipt")
    resolved = datetime.fromisoformat(selected.cycle)
    if init.tzinfo is None:
        resolved = resolved.replace(tzinfo=None)
    entry = resolve_background_source(selected.source)
    return BackgroundCycle(
        source=selected.source, cycle=resolved, init=init,
        forecast_start_hour=selected.forecast_leads[0],
        forecast_end_hour=selected.forecast_leads[-1],
        horizon_hours=entry.horizon(resolved),
        publication_lag_seconds=selected.publication_lag_seconds,
        selected_forecast_hours=selected.forecast_leads)


def refuse_uncovered_area(source: str, area) -> None:
    """Refuse, before any fetch, a box the source's grid does not carry.

    Delegates to :func:`gpuwm.fetch.validate_fetch_area`, which derives
    the envelope from the native grid definition rather than a
    hand-held box, so this gate and ``gpuwm fetch``'s own cannot drift
    apart.  A global source accepts every box and returns silently.
    """

    entry = resolve_background_source(source)
    if not entry.coverage_bounded:
        return
    from gpuwm.fetch import validate_fetch_area

    try:
        validate_fetch_area(entry.name, area)
    except ValueError as error:
        raise BackgroundError(str(error)) from None


def refuse_uncovered_domain(source: str, experiment) -> None:
    """Refuse a domain the source cannot force, halo included.

    Stricter than the lat/lon box: the interpolation stencil and the
    surface-donor halo both need real source cells on every side, and a
    domain that clears the envelope can still run its halo off the
    grid.  That failure has been paid for once already, after the
    download rather than before it.
    """

    entry = resolve_background_source(source)
    if not entry.coverage_bounded:
        return
    from gpuwm.source_adapters import get_source_adapter
    if get_source_adapter(entry.name).runner != "hrrr_f00_f12_v1":
        raise BackgroundError(
            f"{entry.label} needs native target geometry for its coverage proof; "
            "use domain_wizard.source_coverage_refusal at configuration time and "
            "retain the preparation owner's interpolation-halo check")
    from gpuwm.hrrr_route_inputs import HrrrRouteInputError, coverage_refusal

    try:
        refusal = coverage_refusal(experiment)
    except (HrrrRouteInputError, ValueError) as error:
        # The spec could not even be constructed.  That refusal carries
        # its own cause and remedy; wrapping it in coverage words (and
        # the --background-source fallback) sent a user chasing a
        # phantom coverage problem.
        raise BackgroundError(str(error)) from None
    if refusal is None:
        return
    fallback = (
        f"; prepare this case on --background-source "
        f"{entry.coverage_fallback} instead, whose grid is global"
        if entry.coverage_fallback else "")
    raise BackgroundError(
        f"{entry.label} cannot force this domain: {refusal}{fallback}")


@dataclass(frozen=True)
class MemberBackground:
    """How ONE trajectory's initial condition was built.

    The point of recording this per trajectory rather than once per run
    is that a reader can tell a control from a member, and two members
    from each other, without trusting a summary.  Two records that
    compare equal are two identical members, which is what
    :func:`plan_member_backgrounds` refuses.
    """

    trajectory: str
    construction: str
    #: ``None`` for the unperturbed control.
    perturbation_seed: int | None
    #: The fields/species this trajectory's perturbation touches, in the
    #: order the engine applies them.  Empty for the control.
    perturbed: tuple[str, ...]

    def as_receipt(self) -> dict[str, object]:
        return {
            "trajectory": self.trajectory,
            "construction": self.construction,
            "perturbation_seed": self.perturbation_seed,
            "perturbed": list(self.perturbed),
        }

    def identity(self) -> tuple:
        return (self.construction, self.perturbation_seed, self.perturbed)


def plan_member_backgrounds(
        *, control_name: str, members: int, seed: int,
        perturbed_fields: Sequence[Mapping[str, object]],
        perturbed_species: Sequence[Mapping[str, object]],
        construction: str = PERTURBED_DETERMINISTIC,
) -> tuple[MemberBackground, ...]:
    """One construction record per trajectory, or a refusal.

    The construction this driver implements gives every trajectory the
    SAME deterministic background and makes them distinct by perturbing
    each member with its own seed.  Three things make that accurate, and
    each is a refusal here rather than a comment:

    * a construction this driver does not implement is named and
      refused, with the reason it is not reachable, instead of silently
      falling through to the one that is;
    * an ensemble whose perturbation touches NOTHING -- every amplitude
      zero, no species -- would hand the filter N bit-identical copies
      of the control.  That is a fabricated ensemble with a real
      member count, and it is refused;
    * any two trajectories that would carry identical construction
      records are refused by comparison of the records themselves, so
      the guarantee survives a future construction nobody has written.

    None of this depends on which background was chosen.  Substituting
    a convection-allowing first guess for a global one changes the
    central estimate and leaves the spread mechanism untouched -- which
    is exactly why the spread mechanism is the thing that has to say so
    out loud.
    """

    if construction not in ENSEMBLE_CONSTRUCTIONS:
        raise BackgroundError(
            f"unknown ensemble construction {construction!r}; this "
            "driver knows " + ", ".join(sorted(ENSEMBLE_CONSTRUCTIONS)))
    blocked = ENSEMBLE_CONSTRUCTIONS[construction]
    if blocked is not None:
        raise BackgroundError(
            f"ensemble construction {construction!r} is not reachable: "
            + blocked)
    if not isinstance(members, int) or isinstance(members, bool) \
            or members < 1:
        raise BackgroundError("--members must be a positive integer")

    def _named(entry: Mapping[str, object], key: str) -> str:
        name = entry.get(key)
        if not isinstance(name, str) or not name:
            raise BackgroundError(
                f"perturbation entry is missing a {key}: {dict(entry)!r}")
        return name

    def _live(entry: Mapping[str, object]) -> bool:
        amplitude = entry.get("amplitude")
        return (isinstance(amplitude, (int, float))
                and not isinstance(amplitude, bool)
                and math.isfinite(float(amplitude))
                and float(amplitude) != 0.0)

    touched = tuple(
        [_named(entry, "name") for entry in perturbed_fields if _live(entry)]
        + [_named(entry, "mass_field") for entry in perturbed_species
           if _live(entry)])
    if not touched:
        raise BackgroundError(
            f"{members} member(s) were requested and the perturbation "
            "touches no field: every amplitude is zero and no species is "
            "listed, so all members would be bit-identical copies of the "
            "control.  An ensemble with no spread is a fabricated "
            "ensemble -- give the perturbation an amplitude, or run one "
            "deterministic trajectory instead")

    plan = [MemberBackground(
        trajectory=str(control_name), construction=construction,
        perturbation_seed=None, perturbed=())]
    for index in range(members):
        plan.append(MemberBackground(
            trajectory=str(index), construction=construction,
            perturbation_seed=int(seed) + index, perturbed=touched))

    seen: dict[tuple, str] = {}
    for entry in plan:
        identity = entry.identity()
        duplicate = seen.get(identity)
        if duplicate is not None:
            raise BackgroundError(
                f"trajectories {duplicate!r} and {entry.trajectory!r} "
                "would be constructed identically; duplicated members "
                "are a fabricated ensemble, not an ensemble of that "
                "size")
        seen[identity] = entry.trajectory
    return tuple(plan)


def background_receipt(
        *, source: str, cycle: BackgroundCycle | None,
        members: Sequence[MemberBackground],
        prepared_content_sha256: str | None = None,
        notes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The provenance block a report carries so a comparison can attribute.

    A skill difference between two runs is either the background or
    everything else.  This block is what lets a reader tell which,
    without re-deriving the plan: the source and its cycle, what the
    first guess carries, the prepared case's own digest, and one record
    per trajectory saying how that member was made.
    """

    entry = resolve_background_source(source)
    receipt: dict[str, object] = {
        "schema": BACKGROUND_RECEIPT_SCHEMA,
        "stability": "experimental",
        "source": entry.name,
        "label": entry.label,
        "cycle_step_hours": entry.cycle_step_hours,
        "forcing_interval_seconds": entry.forcing_interval_seconds,
        "initial_hydrometeors": entry.initial_hydrometeors,
        "coverage_bounded": entry.coverage_bounded,
        "cycle": None if cycle is None else cycle.as_receipt(),
        "prepared_content_sha256": prepared_content_sha256,
        "ensemble": {
            "construction": (
                members[0].construction if members else None),
            "member_count": max(0, len(members) - 1),
            "spread_origin": (
                "smooth Gaussian IC perturbations applied once at leg 0 to "
                "the single deterministic background; the mechanism is "
                "source-independent and unchanged by the source selection"),
            "members": [member.as_receipt() for member in members],
        },
    }
    if notes:
        receipt["notes"] = dict(notes)
    return receipt


__all__ = [
    "BACKGROUND_RECEIPT_SCHEMA",
    "BACKGROUND_SOURCES",
    "BackgroundCycle",
    "BackgroundError",
    "BackgroundSource",
    "DEFAULT_BACKGROUND_SOURCE",
    "ENSEMBLE_CONSTRUCTIONS",
    "MemberBackground",
    "PERTURBED_DETERMINISTIC",
    "background_receipt",
    "late_background_wait_minutes",
    "plan_background_cycle",
    "plan_member_backgrounds",
    "refuse_uncovered_area",
    "refuse_uncovered_domain",
    "resolve_background_source",
]
