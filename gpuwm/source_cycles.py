"""When a source has an initialization, and when its bytes land.

``--cycle latest`` asks one question -- what is the newest init this
source can serve -- and three model names used to answer it.
:func:`gpuwm.fetch.resolve_latest_cycle` branched on gfs/gdas/hrrr and
refused everything else with a sentence about ERA5's publication delay,
which a reader asking for RAP or ICON-EU got verbatim.  In a registry of
thirty-two sources that is the per-model bandaid the arbitrary
acceptance test bans, and it made a reanalysis unaskable for a time it
publishes perfectly well.

The question is answered from DECLARED FACTS instead:

``hours``          the UTC hours of day this producer initializes on.
``delay_hours``    how long after a nominal init its bytes are on a
                   server.  Zero where a completeness PROBE decides
                   publication -- the probe IS the answer there, and a
                   declared delay would only start the walk-back late.
``search_hours``   how far back ``latest`` walks the grid before it
                   gives up and says so.
``record_end``     the last init of a CLOSED archive, or ``None`` for a
                   producer still running.  An archive that ended still
                   has a newest init, and it is not "now minus a delay".

A source with a fetch route already declares its grid there
(``gpuwm.fetch_routes.Route.cycle_hours``, measured against the
producer), so :func:`cycle_grid_for` READS that rather than asking this
module to repeat it: a model added as a route-table row gets ``latest``
with no edit here at all.  What this module carries is the sources whose
schedule nothing else declares, and the shape every answer is given in.

A row that declares no grid and has no route is not on a list of the
unsupported -- :func:`cycle_grid_for` returns ``None`` and the caller
refuses by naming what the row lacks, which is a sentence that stays
true when the row grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


#: The walk-back a source declares nothing better than.  Two days is the
#: retention the operational NCEP servers publish for their own
#: directories, so it is the window in which a probe can still get an
#: answer rather than a guess about how stale a cycle may be.
DEFAULT_SEARCH_HOURS = 48


@dataclass(frozen=True)
class CycleGrid:
    """One source's initialization schedule, as data.

    Constructed from a registry row or derived from a fetch route; the
    resolver never learns which, which is the point.
    """

    hours: tuple[int, ...]
    delay_hours: float = 0.0
    search_hours: int = DEFAULT_SEARCH_HOURS
    basis: str = ""
    record_end: datetime | None = None
    #: Per-cycle-hour forecast horizon, most specific rule first:
    #: ``((cycle_hours_or_None, through_hour), ...)``.  ``None`` matches
    #: any hour, so a trailing ``(None, N)`` is the default rule.  The
    #: shape is the fetch route table's own ``ladders`` shape,
    #: deliberately: that is where a new model declares this, and a
    #: reader who has seen one has seen both.  Empty means the producer
    #: declares no per-cycle variation and no candidate is filtered on
    #: this ground.
    horizons: tuple[tuple[tuple[int, ...] | None, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.hours:
            raise ValueError(
                "a CycleGrid must declare at least one UTC hour; a source "
                "with no cycle concept declares no grid at all, so the "
                "refusal can name the absence instead of an empty table")
        if sorted(set(self.hours)) != list(self.hours):
            raise ValueError(
                f"CycleGrid hours {self.hours} must be sorted and unique")
        if not all(0 <= hour <= 23 for hour in self.hours):
            raise ValueError(
                f"CycleGrid hours {self.hours} must be UTC hours of day")
        if self.delay_hours < 0.0:
            raise ValueError("CycleGrid delay_hours cannot be negative")
        if self.search_hours <= 0:
            raise ValueError(
                "CycleGrid search_hours must be a positive window; zero "
                "would refuse every cycle without looking at one")

    def horizon(self, cycle: datetime) -> int | None:
        """How far this cycle forecasts, or ``None`` if undeclared.

        A cycle that does not reach the end of the requested window is
        not the ``latest`` anything: it is a cycle that cannot serve the
        request.  RAP's 07Z run stops at f021 while its 09Z run reaches
        f051, and HRRR's off-synoptic hours stop at f018 -- filtering on
        a declared horizon is what stops ``latest`` returning one of
        those for a window that needs more.
        """

        for hours, through in self.horizons:
            if hours is None or cycle.hour in hours:
                return through
        return None

    def declaration(self) -> dict[str, object]:
        """This grid as JSON-safe fields, for a manifest or a front end."""

        return {
            "hours": list(self.hours),
            "delay_hours": float(self.delay_hours),
            "search_hours": int(self.search_hours),
            "basis": self.basis,
            "record_end": (None if self.record_end is None
                           else self.record_end.strftime("%Y-%m-%dT%H")),
            "horizons": [[None if hours is None else list(hours), through]
                         for hours, through in self.horizons],
        }

    def snap(self, moment: datetime) -> datetime:
        """The newest grid point at or before ``moment``.

        Hour-by-hour rather than by arithmetic on a period: a grid is a
        SET of hours, not a spacing, and ICON-EU's 00/03/06/09/12/15/18/21
        and GEM-GDPS's 00/12 are both irregular enough that a period
        would have to be inferred and would be wrong for the next
        producer that runs an odd schedule.
        """

        candidate = moment.replace(minute=0, second=0, microsecond=0)
        for _step in range(24):
            if candidate.hour in self.hours:
                return candidate
            candidate -= timedelta(hours=1)
        raise AssertionError(              # pragma: no cover - hours is nonempty
            "a nonempty hour set is reached within one day")

    def newest(self, now: datetime) -> datetime:
        """The newest init this grid says exists, optimistically.

        ``now`` minus the declared publication delay, snapped back onto
        the grid, and never later than a closed archive's last init.
        This is the whole answer where no probe transport exists; where
        one does, it is the first candidate the probe is asked about.
        """

        newest = self.snap(now - timedelta(hours=self.delay_hours))
        if self.record_end is not None and newest > self.record_end:
            return self.snap(self.record_end)
        return newest

    def candidates(self, now: datetime) -> tuple[datetime, ...]:
        """Every cycle ``latest`` may resolve to, newest first."""

        newest = self.newest(now)
        found = [newest]
        while True:
            earlier = self.snap(found[-1] - timedelta(hours=1))
            if (newest - earlier) > timedelta(hours=self.search_hours):
                return tuple(found)
            found.append(earlier)


def route_cycle_grid(source_id: str) -> CycleGrid | None:
    """The grid a source's FETCH ROUTE declares, or ``None``.

    The route table is measured against the producer and is where a new
    model is added, so a row there is the reason ``latest`` needs no
    per-source code: adding ICON, AIFS or a Canadian model is a route
    row, and this reads the ``cycle_hours`` it already had to carry.
    """

    try:
        from gpuwm import fetch_routes

        route = fetch_routes.route_for(source_id)
    except (ImportError, ValueError, KeyError):
        return None
    hours = tuple(sorted({int(hour) for hour in route.cycle_hours}))
    if not hours:
        return None
    # The default host's retention IS the walk-back: it is how far back
    # the server this resolver probes still holds directories, so a
    # candidate older than that cannot be confirmed and asking about it
    # only spends HEADs.  A route that declares no retention takes the
    # module default rather than an unbounded walk.
    retention = next(
        (float(host.retention_hours) for host in route.hosts
         if getattr(host, "retention_hours", None)), None)
    return CycleGrid(
        hours=hours,
        delay_hours=float(route.publication_lag_hours),
        search_hours=int(retention) if retention else DEFAULT_SEARCH_HOURS,
        # The last rung of each ladder rule IS the horizon: the steps are
        # (through_hour, spacing) pairs in increasing order, so the final
        # pair's first element is how far that cycle forecasts.
        horizons=tuple(
            (None if cycle_hours is None else tuple(cycle_hours), steps[-1][0])
            for cycle_hours, steps in route.ladders if steps),
        basis=f"the fetch route table's measured cycle_hours and "
              f"publication_lag_hours for {source_id!r} "
              f"(measured {route.measured})")


def cycle_grid_for(source_id: str) -> CycleGrid | None:
    """The initialization grid for one source id, or ``None``.

    ``None`` means nothing in this product declares when this source
    initializes -- NOT that the source is unsupported.  The caller turns
    that into a refusal naming the missing declaration, so a row which
    later grows a fetch route starts resolving ``latest`` with no edit
    here at all.

    The registry row's own ``cycle_grid`` wins where it declares one:
    that column carries the sources whose schedule the route table
    cannot state (the legacy transports, and a keyed job API like the
    CDS, which has no object to probe and so needs its publication delay
    written down).  Everything else derives.
    """

    try:
        from gpuwm.source_adapters import get_source_adapter

        adapter = get_source_adapter(source_id)
    except (ImportError, ValueError):
        return route_cycle_grid(source_id)
    return adapter.cycle_grid or route_cycle_grid(adapter.source_id)


__all__ = ["DEFAULT_SEARCH_HOURS", "CycleGrid", "cycle_grid_for",
           "route_cycle_grid"]
