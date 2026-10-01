"""When each lead of a source's cycle is due, and how long a run waits for it.

A run that starts on a source's first hours and waits for the rest needs
two facts per lead: when the lead should be on the server, and how long
past that the run keeps waiting before it says the source fell behind.
Both are table rows.  The first is the source's ``publication_lag`` rule
(the EARLIEST posting seen of each lead, per cycle hour and per lead,
:class:`gpuwm.source_cycles.PublicationRule`), read through the source's
cycle grid so a route and a legacy transport answer through one
function.  The second is the source's ``posting`` block
(:class:`gpuwm.fetch_routes.PostingRow`): its shape, how a lead is asked
about, the lateness budget and the poll ceiling.

Nothing here names a source.  A model added as a route row with a
``posting`` block has a schedule with no edit here, which is the
arbitrary acceptance test.  Nothing here reaches a network either: the
schedule is arithmetic on the rows, and asking a host whether a lead is
there belongs to the fetch loop that reads this schedule.

A tree adds start needs of its own (:func:`nest_start_needs`): a nest that
starts after the forecast (``start_time`` later than the root's) is
prepared from the source's analysis at its own start time
(``gpuwm.ingest.nest_init``), so its lead joins the run's start needs
(``nest_start:dNN``) beside the root's analysis and first boundary lead.
Children that start with the root add nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from fractions import Fraction
import math
from typing import Iterable, Mapping

from gpuwm import fetch_routes
from gpuwm.fetch_routes import PostingRow
from gpuwm.source_cycles import CycleGrid, PublicationRule, cycle_grid_for

#: The document ``<fetch out>/posting/schedule.json`` carries.
SCHEDULE_SCHEMA = "gpuwm.posting-schedule.v1"

#: The folder, schedule and failure file an as-posted fetch writes under
#: its output folder, named here because the writer (the fetch, which the
#: RW-WPS wheel stages) and every reader (go's event relay, the boundary
#: stream's lead wait) must agree on them, and the relay's module is not
#: in that wheel.  ``gpuwm.chain_events`` carries the same three names;
#: tests/test_fetch_as_posted.py pins that they are equal.
POSTING_DIRNAME = "posting"
POSTING_SCHEDULE_NAME = "schedule.json"
POSTING_FAILED_NAME = "failed.json"

#: Every lead of a new schedule starts here; the fetch loop moves it on
#: (``waiting``, ``posted``, ``ready``, ``late``).
SCHEDULED = "scheduled"


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def instant(moment: datetime | None) -> str | None:
    """An ISO UTC instant as every posting document spells one."""

    return None if moment is None else _utc(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def posting(source: str, provider: str | None = None) -> PostingRow:
    """``source``'s posting block (its route's, or its legacy row's)."""

    return fetch_routes.posting_for(source, provider=provider)


def grid(source: str) -> CycleGrid:
    """The cycle grid whose rules say when ``source``'s leads are due.

    Refused by naming the missing declaration, the same way ``--cycle
    latest`` is: a source with no grid has no time a lead is due.
    """

    found = cycle_grid_for(source, posting=True)
    if found is None:
        raise ValueError(
            f"{source!r} declares no cycle grid, so nothing says when its "
            "leads are due; name the source's cycle hours and publication "
            "lag in the fetch route table (or its registry row)")
    return found


def publication_rule(source: str, cycle: datetime,
                     lead: int) -> PublicationRule | None:
    """The table rule that dates ``lead`` of ``cycle``, or None.

    None where the source declares one delay for every cycle and lead
    (its grid's ``delay_hours``), which a reader then reads as that.
    """

    cycle = _utc(cycle)
    for rule in grid(source).delays:
        if rule.matches(cycle.hour, lead):
            return rule
    return None


def expected_at(source: str, cycle: datetime, lead: int) -> datetime:
    """When ``lead`` of ``cycle`` should be on the source's first host.

    The cycle plus A134's :meth:`~gpuwm.source_cycles.CycleGrid.delay`
    for that cycle hour and lead: the earliest posting seen.  A lead is
    not asked about much before this (DESIGN A136 2.2), so an earlier
    line only costs a probe that answers no, and a later one would miss
    a lead that is out.
    """

    cycle = _utc(cycle)
    hours = grid(source).delay(cycle, int(lead))
    return cycle + timedelta(hours=hours)


def late_after_minutes(source: str, override: float | None = None, *,
                       provider: str | None = None) -> float | None:
    """The lateness budget a run of ``source`` waits a lead for, minutes.

    The posting row's, or ``override`` (``[fetch] late_after_minutes``),
    checked as the row is.  None for a shape no run waits on lead by
    lead or for the whole cycle (an archive, a keyed job, private bytes).
    """

    row = posting(source, provider)
    if override is not None:
        if (isinstance(override, bool) or not isinstance(override, (int, float))
                or not math.isfinite(float(override)) or float(override) <= 0):
            raise ValueError(
                f"late_after_minutes = {override!r}: a budget must be a "
                "positive number of minutes, because zero fails every run "
                "whose lead is a second past its scheduled time")
        return float(override) if row.waited else None
    return row.late_after_minutes if row.waited else None


def late_at(source: str, cycle: datetime, lead: int, *,
            late_after_minutes_override: float | None = None,
            provider: str | None = None) -> datetime | None:
    """When a run stops waiting for ``lead`` and says the source fell behind.

    ``expected_at`` plus the budget; None where no run waits on the
    source (see :func:`late_after_minutes`).
    """

    budget = late_after_minutes(source, late_after_minutes_override,
                                provider=provider)
    if budget is None:
        return None
    return expected_at(source, cycle, lead) + timedelta(minutes=budget)


def lead_rows(source: str, cycle: datetime, leads: Iterable[int], *,
              late_after_minutes_override: float | None = None,
              provider: str | None = None) -> list[dict[str, object]]:
    """One schedule row per lead, in lead order, every lead ``scheduled``."""

    cycle = _utc(cycle)
    ordered = sorted({int(lead) for lead in leads})
    if not ordered or ordered[0] < 0:
        raise ValueError("a posting schedule needs one or more nonnegative leads")
    budget = late_after_minutes(source, late_after_minutes_override,
                                provider=provider)
    rows = []
    for lead in ordered:
        due = expected_at(source, cycle, lead)
        rows.append({
            "lead": lead,
            "valid_time": instant(cycle + timedelta(hours=lead)),
            "expected_at": instant(due),
            "late_at": (None if budget is None
                        else instant(due + timedelta(minutes=budget))),
            "first_seen_at": None,
            "fetched_at": None,
            "endpoint": None,
            "state": SCHEDULED,
        })
    return rows


def _rule_declaration(rule: PublicationRule) -> dict[str, object]:
    return {"cycle_hours": (None if rule.cycle_hours is None
                            else list(rule.cycle_hours)),
            "from_lead": int(rule.from_lead), "hours": float(rule.hours),
            "per_lead_hours": float(rule.per_lead_hours),
            "late_hours": float(rule.late_hours), "measured": rule.measured}


def schedule(source: str, cycle: datetime, leads: Iterable[int], *,
             member: str | None = None, as_posted: bool = True,
             late_after_minutes_override: float | None = None,
             provider: str | None = None) -> dict[str, object]:
    """The ``gpuwm.posting-schedule.v1`` document for one window.

    Every lead with its ``valid_time``, ``expected_at``, ``late_at`` and
    ``state: scheduled``, the posting row it was computed from and the
    table it came from.  The window's start needs (and so its
    ``expected_ready_at``) are the fetch loop's to add: they depend on
    the window's donors and delayed nests, not on this table alone.
    """

    cycle = _utc(cycle)
    row = posting(source, provider)
    rows = lead_rows(source, cycle, leads,
                     late_after_minutes_override=late_after_minutes_override,
                     provider=provider)
    rules = [_rule_declaration(rule) for rule in grid(source).delays]
    return {
        "schema": SCHEDULE_SCHEMA,
        "source": source,
        "member": member,
        "cycle": cycle.strftime("%Y-%m-%dT%H"),
        "as_posted": bool(as_posted),
        "shape": row.shape,
        "streams": row.streams,
        "why": row.why,
        "ready_check": row.ready_check,
        "late_after_minutes": late_after_minutes(
            source, late_after_minutes_override, provider=provider),
        "poll_seconds": row.poll_seconds,
        "publication_lag": rules,
        "expected_final_at": rows[-1]["expected_at"],
        "leads": rows,
        "table_sha256": fetch_routes.packaged_route_table_sha256(),
    }


def declaration(source: str) -> dict[str, object]:
    """``source``'s posting facts as the registry row carries them.

    What a site plans launches from without asking a host: the posting
    block, the ``publication_lag`` rules every ``expected_at`` is
    computed from (A134's rows, with what they were measured on), each
    provider transport's own block where the row declares one, and the
    table they came from.
    """

    source_id = fetch_routes.canonical_source(source)
    row = posting(source_id)
    found = cycle_grid_for(source_id, posting=True)
    value: dict[str, object] = dict(row.declaration())
    value["publication_lag"] = ([] if found is None else
                                [_rule_declaration(rule) for rule in found.delays])
    legacy = fetch_routes.legacy_posting(source_id)
    if legacy is not None and legacy.providers:
        value["providers"] = {name: provider.declaration()
                              for name, provider in legacy.providers.items()}
    value["table_sha256"] = fetch_routes.packaged_route_table_sha256()
    return value


def declarations() -> Mapping[str, dict[str, object]]:
    """:func:`declaration` for every source with a posting row."""

    return {source: declaration(source)
            for source in fetch_routes.posting_sources()}


def nest_start_needs(exp, *, source: str, cycle: datetime | None = None,
                     start_lead: int = 0,
                     cadence_hours: int = 1) -> list[dict]:
    """The source leads a tree's delayed nests start from.

    One row per delayed nest, in lead order: ``{role: "nest_start:dNN",
    source, lead, valid_time}``, where ``lead`` is the window's
    ``start_lead`` plus the nest's start offset in hours.  ``cycle``, when
    given, is held to that arithmetic.  A delayed nest whose start is not
    on the window's lead ``cadence_hours`` is refused by name, which is
    the breakage it prevents: a start need for a lead the window never
    fetches, for a nest the preparation would refuse anyway (it takes the
    one source snapshot at the nest's start time).
    """

    cadence = int(cadence_hours)
    if cadence <= 0:
        raise ValueError(f"cadence_hours must be positive, got {cadence_hours!r}")
    rows = []
    for domain in exp.domains:
        if int(domain.parent_id) == 0:
            continue
        grid_id = int(domain.grid_id)
        offset = Fraction(exp.domain_start_offset_exact(grid_id))
        if offset == 0:
            continue
        label = f"d{grid_id:02d}"
        if offset % (cadence * 3600) != 0:
            raise ValueError(
                f"{label} starts {float(offset) / 3600:g} h into the run, "
                f"between two {source} leads {cadence} h apart; a delayed "
                "nest is prepared from the source's analysis at its own "
                "start time, so it has to start on a lead")
        lead = int(start_lead) + int(offset // 3600)
        valid = exp.domain_start_time(grid_id)
        if (cycle is not None
                and _utc(cycle) + timedelta(hours=lead) != _utc(valid)):
            raise ValueError(
                f"{label} starts at {instant(valid)}, which is not lead "
                f"{lead} of the {instant(cycle)} cycle")
        rows.append({"role": f"nest_start:{label}", "source": str(source),
                     "lead": lead, "valid_time": instant(valid)})
    return sorted(rows, key=lambda row: (row["lead"], row["role"]))


__all__ = ["POSTING_DIRNAME", "POSTING_FAILED_NAME", "POSTING_SCHEDULE_NAME",
           "SCHEDULED", "SCHEDULE_SCHEMA", "declaration", "declarations",
           "expected_at", "grid", "instant", "late_after_minutes", "late_at",
           "lead_rows", "nest_start_needs", "posting", "publication_rule",
           "schedule"]
