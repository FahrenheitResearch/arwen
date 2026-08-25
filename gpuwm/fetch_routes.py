"""The one acquisition engine every table-driven fetch route runs through.

``gpuwm fetch --source`` used to accept four names.  Ten sources had a
runnable packaged profile, a passing 6 h forecast and no way to get their
bytes -- the RRFS arm's 6.3 GiB came down by hand-written ``curl``.  A
capability with no front door is engine-proven, not shipped.

This module closes that, and it closes it as TABLE WORK.  Everything
model-shaped -- which host publishes the bytes, what the key looks like,
which cycles the producer runs, how far each one forecasts and at what
spacing, which downloaded objects are the primaries and which are the
composition's supplement or its cross-source donor -- is rows in the
packaged document ``authorities/rw-wps-fetch-routes.v1.json``.  Nothing
here branches on a model name.  Adding ICON-D2, RAP-AK or a hires Euro
model is a row in that file; if it ever needs a function in this one, the
row grammar was wrong.

Three stages, in order:

* **resolve** -- :func:`resolve_request` turns ``--cycle/--hours`` into an
  ordered, deterministic object list with no network at all, so every
  refusal a request has coming (a cycle the producer does not run, a lead
  past the horizon, a member outside the declared set) is paid for in
  milliseconds rather than after a download.
* **transfer** -- :func:`run_plan` moves the objects through
  :mod:`gpuwm.fetch_pool`, so every table route is parallel by default with
  the same bounded, host-capped, in-order-admission semantics the GFS and
  HRRR routes already have.
* **compose** -- the declared per-lead concatenations (GEFS's disjoint
  ``pgrb2a``+``pgrb2b`` pair, GDPS's one-message-per-file valid time) run
  after the bytes land and verify, producing the exact files
  ``gpuwm prep`` consumes.

The output directory is then a front door, not a pile: it carries the
ordered ``--input-list``, the supplement bindings the composition
declares, ``SHA256SUMS``, and a ``prep-command.txt`` whose bound half runs
as printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import functools
import hashlib
import json
from pathlib import Path
import re
import shlex
from types import MappingProxyType
from typing import Mapping, Sequence

from gpuwm import fetch_endpoints, fetch_pool, source_adapters
from gpuwm.fetch_endpoints import Endpoint
# Imported as a NAME, not as the module: `run_plan` takes a parameter
# called ``progress`` (the route's status-line sink), which would shadow
# a module of that name inside exactly the function that needs it.
from gpuwm import progress as progress_mod
from gpuwm.progress import ByteCounter


#: The packaged acquisition-route document, beside the decode authorities
#: it complements: a mapping says how to READ a source's bytes, a route
#: says where they LIVE.
ROUTE_TABLE_NAME = "rw-wps-fetch-routes.v1.json"
ROUTE_TABLE_SCHEMA = "gpuwm-fetch-routes-v1"

#: SHA-256 of the packaged route table, pinned so a wheel whose data file
#: has drifted from this engine fails loudly at load rather than
#: resolving a key shape nothing was measured against.  Kept in sync by
#: ``tests/test_fetch_routes.py``.
ROUTE_TABLE_SHA256 = (
    "cad041195e608d29af6c6d7f0f74b0b5c40d72faed3564ea55cef5e90be28521"
)

#: Sources whose acquisition predates the route table and keeps its own
#: hand-written transport in :mod:`gpuwm.fetch`: the certified GFS
#: container pair (a NOMADS grib-filter crop plus an S3 full-file route)
#: and HRRR (two hosts, a live-cycle wait mode, an ``.idx`` subsetter).
#: They are named here so the coverage gate can tell "handled elsewhere"
#: from "not handled".
LEGACY_ROUTE_SOURCES = ("gfs", "gdas", "hrrr", "era5")

#: Receipt schemas.
ROUTE_MANIFEST_SCHEMA = "gpuwm-fetch-route-manifest-v1"

#: Every token a route's path template may spell.  A template naming
#: anything else is a table error, caught at load rather than at the
#: first 404.
PATH_TOKENS = frozenset({
    "YYYY", "MM", "DD", "HH", "YYYYMMDD", "YYYYMMDDHH", "YYYYMMDDHHMMSS",
    "F", "FF", "FFF", "MEMBER", "FIELD", "field_lower", "leveltype",
    "LEVEL", "LEVEL_SUFFIX",
})

_TOKEN_RE = re.compile(r"\{([A-Za-z_]+)\}")


# --------------------------------------------------------------------------
# Table loading
# --------------------------------------------------------------------------

def _table_path() -> Path:
    return Path(__file__).with_name("authorities") / ROUTE_TABLE_NAME


def packaged_route_table_sha256() -> str:
    """SHA-256 of the packaged route table as installed."""

    return hashlib.sha256(_table_path().read_bytes()).hexdigest()


def _load_table() -> Mapping[str, object]:
    document = json.loads(_table_path().read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != ROUTE_TABLE_SCHEMA:
        raise ValueError(
            f"{ROUTE_TABLE_NAME} declares schema {schema!r}; this ArWen "
            f"reads {ROUTE_TABLE_SCHEMA!r}")
    return document


def unknown_tokens(template: str) -> tuple[str, ...]:
    """Tokens in ``template`` this engine cannot fill."""

    return tuple(sorted(
        name for name in _TOKEN_RE.findall(template)
        if name not in PATH_TOKENS))


# --------------------------------------------------------------------------
# Row types
# --------------------------------------------------------------------------

#: A route's host row IS an endpoint row: name, base, the retention
#: window it serves, and what it is for.  :mod:`gpuwm.fetch_endpoints`
#: owns the shape and the ladder rules; this alias keeps the older
#: spelling readable where a route talks about "its hosts".
Host = Endpoint


@dataclass(frozen=True)
class FileRow:
    """One family of objects in a route's file set."""

    role: str
    path: str
    primary: bool
    #: ``all`` -- one per requested lead; ``first`` -- only at the window's
    #: first lead (the once-per-cycle invariants a producer publishes at
    #: analysis time alone); ``none`` -- lead-independent.
    leads: str
    axis: str | None
    idx_sidecar: str | None
    #: Leading magic every published object of this family carries.  It is
    #: the cheapest bar that separates a payload from an error page a
    #: proxy served with HTTP 200, and it is declared rather than guessed
    #: from a suffix because ``gec00.t00z.pgrb2a.0p50.f000`` has none.
    magic: str


@dataclass(frozen=True)
class ComposeRow:
    kind: str
    roles: tuple[str, ...]
    name: str
    primary: bool
    why: str


@dataclass(frozen=True)
class DonorRow:
    source: str
    role: str
    leads: tuple[int, ...]
    cycle: str
    why: str


@dataclass(frozen=True)
class Route:
    source_id: str
    label: str
    measured: str
    hosts: tuple[Host, ...]
    host_note: str
    coverage_note: str
    cycle_hours: tuple[int, ...]
    #: MEASURED hours between a nominal cycle time and this producer's
    #: bytes appearing on the ladder head.  It is what lets ``--cycle
    #: latest`` start its walk at a cycle that plausibly exists instead
    #: of HEAD-ing its way down from a cycle nobody has published yet:
    #: publication lag differs by hours between these producers, and a
    #: resolver that assumed one producer's timing spent a probe per
    #: candidate discovering the others'.
    publication_lag_hours: float
    ladders: tuple[tuple[tuple[int, ...] | None, tuple[tuple[int, int], ...]], ...]
    cadences: tuple[int, ...]
    default_cadence: int
    layout: str
    members: Mapping[str, object] | None
    axes: Mapping[str, tuple[Mapping[str, object], ...]]
    files: tuple[FileRow, ...]
    compose: tuple[ComposeRow, ...]
    donors: tuple[DonorRow, ...]
    record_subset_supported: bool
    record_subset_why: str
    prep: Mapping[str, object]

    def host(self, name: str | None) -> Host:
        """One named endpoint, or the head of the ladder.

        The head is what a request with no ``--transport`` starts at;
        which endpoint it IS depends on the cycle's age, so callers
        that have a cycle should ask :func:`endpoint_ladder` instead of
        assuming the table's first row.
        """

        if name is None:
            return self.hosts[0]
        for host in self.hosts:
            if host.name == name:
                return host
        offered = ", ".join(host.name for host in self.hosts)
        raise ValueError(
            f"--transport {name}: --source {self.source_id} publishes on "
            f"{offered}.  {self.host_note or ''}".strip())


def _file_row(raw: Mapping[str, object]) -> FileRow:
    return FileRow(
        role=str(raw["role"]),
        path=str(raw["path"]),
        primary=bool(raw.get("primary", False)),
        leads=str(raw.get("leads", "all")),
        axis=(str(raw["axis"]) if raw.get("axis") else None),
        idx_sidecar=(str(raw["idx_sidecar"]) if raw.get("idx_sidecar")
                     else None),
        magic=str(raw.get("magic", "GRIB")),
    )


def _build_routes() -> Mapping[str, Route]:
    document = _load_table()
    routes: dict[str, Route] = {}
    for source_id, raw in dict(document["routes"]).items():
        ladders = tuple(
            (tuple(entry["cycle_hours"]) if entry.get("cycle_hours") else None,
             tuple((int(step[0]), int(step[1])) for step in entry["steps"]))
            for entry in raw["ladders"])
        routes[source_id] = Route(
            source_id=source_id,
            label=str(raw["label"]),
            measured=str(raw.get("measured", "")),
            hosts=fetch_endpoints.ladder(source_id),
            host_note=str(raw.get("host_note", "")),
            coverage_note=str(raw.get("coverage_note", "")),
            cycle_hours=tuple(int(hour) for hour in raw["cycle_hours"]),
            publication_lag_hours=float(raw["publication_lag_hours"]),
            ladders=ladders,
            cadences=tuple(int(value) for value in raw["cadences"]),
            default_cadence=int(raw["default_cadence"]),
            layout=str(raw.get("layout", "flat")),
            members=(MappingProxyType(dict(raw["members"]))
                     if raw.get("members") else None),
            axes=MappingProxyType({
                name: tuple(MappingProxyType(dict(group)) for group in groups)
                for name, groups in dict(raw.get("axes", {})).items()}),
            files=tuple(_file_row(row) for row in raw["files"]),
            compose=tuple(
                ComposeRow(kind=str(row["kind"]),
                           roles=tuple(str(role) for role in row["roles"]),
                           name=str(row["name"]),
                           primary=bool(row.get("primary", False)),
                           why=str(row.get("why", "")))
                for row in raw.get("compose", [])),
            donors=tuple(
                DonorRow(source=str(row["source"]), role=str(row["role"]),
                         leads=tuple(int(lead) for lead in row["leads"]),
                         cycle=str(row.get("cycle", "same")),
                         why=str(row.get("why", "")))
                for row in raw.get("donors", [])),
            record_subset_supported=bool(
                raw["record_subset"].get("supported", False)),
            record_subset_why=str(raw["record_subset"].get("why", "")),
            prep=MappingProxyType(dict(raw.get("prep", {}))),
        )
    for route in routes.values():
        if not route.hosts:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {route.source_id} names no "
                "endpoint, so there is nowhere to fetch it from")
        for host in route.hosts:
            if not host.why.strip():
                raise ValueError(
                    f"{ROUTE_TABLE_NAME}: route {route.source_id} endpoint "
                    f"{host.name} does not say what it is for; a ladder "
                    "whose refusal cannot name each rung is not a ladder")
        for row in route.files:
            bad = unknown_tokens(row.path)
            if bad:
                raise ValueError(
                    f"{ROUTE_TABLE_NAME}: route {route.source_id} file "
                    f"{row.role} spells unknown token(s) {list(bad)}")
    return MappingProxyType(routes)


def _build_refusals() -> Mapping[str, Mapping[str, object]]:
    document = _load_table()
    return MappingProxyType({
        source_id: MappingProxyType(dict(raw))
        for source_id, raw in dict(document.get("refusals", {})).items()})


_ROUTES = _build_routes()
_REFUSALS = _build_refusals()


def route_ids() -> tuple[str, ...]:
    """Registry ids with a table-driven acquisition route, in table order."""

    return tuple(_ROUTES)


def refusal_ids() -> tuple[str, ...]:
    """Registry ids the table refuses by name, in table order."""

    return tuple(_REFUSALS)


def all_fetchable_sources() -> tuple[str, ...]:
    """Every ``--source`` the fetch front door accepts, sorted."""

    return tuple(sorted(set(LEGACY_ROUTE_SOURCES) | set(_ROUTES)))


def _canonical(source: str) -> str:
    """The registry id for a name or alias, or the name unchanged."""

    try:
        return source_adapters.get_source_adapter(source).source_id
    except ValueError:
        return source.strip().lower().replace("_", "-")


def canonical_source(source: str) -> str:
    """The registry id for a name or alias, for callers outside this module.

    Public because the front doors that ask "is this fetchable" must ask
    it about the SAME name this module dispatches on: an alias answered
    yes here and no there is the drift that makes a printed next-step
    refuse.
    """

    return _canonical(source)


def route_for(source: str) -> Route:
    """The route for ``source``, or the refusal that names why there is none."""

    source_id = _canonical(source)
    route = _ROUTES.get(source_id)
    if route is not None:
        return route
    refusal = _REFUSALS.get(source_id)
    if refusal is not None:
        remedy = ""
        if refusal.get("remedy_source_root"):
            remedy = (
                "  remedy: bring the bytes yourself -- `gpuwm prep --source "
                f"{source_id} --source-root DIR --source-manifest "
                "DIR/SHA256SUMS --source-manifest-sha256 <digest>` is the "
                "designed door for a source this ArWen cannot download.")
        raise ValueError(
            f"--source {source_id}: no fetch route.\n"
            f"  why: {refusal['why']}\n{remedy}".rstrip())
    if source_id in LEGACY_ROUTE_SOURCES:
        raise ValueError(
            f"--source {source_id} has its own transport in gpuwm.fetch and "
            "does not run through the route table")
    try:
        adapter = source_adapters.get_source_adapter(source_id)
    except ValueError as error:
        raise ValueError(str(error)) from error
    # TWO different absences, and they had one sentence between them.
    # "No route" for an unrunnable row means the decode is missing, so a
    # download would buy nothing.  "No route" for a RUNNABLE row means
    # only the transport is missing -- this ArWen reads those bytes and
    # prepares them, it just cannot go and get them -- and telling such
    # a reader "the registry row is not runnable" is a false statement
    # of cause that stops them at a door that is actually open.
    if adapter.runnable:
        why = (
            "no acquisition route is registered for this source, so this "
            f"ArWen has no way to download its bytes.  {adapter.source_id} "
            "IS runnable: bring the files yourself and the prepared route "
            "reads them")
        remedy = (
            "  remedy: `gpuwm prep --source "
            f"{adapter.source_id} --source-root DIR --source-manifest "
            "DIR/SHA256SUMS --source-manifest-sha256 <digest>` is the "
            "designed door for a source this ArWen cannot download.\n")
    else:
        why = (
            f"the registry row is not runnable ({adapter.status.value}); "
            "nothing in this ArWen could read the bytes a download "
            "produced")
        remedy = ""
    raise ValueError(
        f"--source {adapter.source_id}: no fetch route.\n"
        f"  why: {why}.\n"
        f"{remedy}"
        "  see: `gpuwm sources` for what each registered source can do "
        "today, or `gpuwm sources " + adapter.source_id + "` for this "
        "row in full.")


# --------------------------------------------------------------------------
# Cycle and lead grammar
# --------------------------------------------------------------------------

def ladder_for(route: Route, cycle: datetime) -> tuple[int, ...]:
    """Every forecast lead ``cycle`` publishes, in order."""

    steps = None
    for cycle_hours, entry in route.ladders:
        if cycle_hours is None or cycle.hour in cycle_hours:
            steps = entry
            break
    if steps is None:
        raise ValueError(
            f"--source {route.source_id}: no lead ladder for the "
            f"{cycle:%H}Z cycle")
    leads: list[int] = []
    previous = 0
    for index, (through, step) in enumerate(steps):
        start = 0 if index == 0 else previous + step
        leads.extend(range(start, through + 1, step))
        previous = leads[-1]
    return tuple(leads)


def _extended_cycle_hours(route: Route) -> tuple[int, ...]:
    """Cycle hours whose ladder reaches farthest (for the refusal text)."""

    best_hours: tuple[int, ...] = ()
    best_last = -1
    for cycle_hours, steps in route.ladders:
        last = steps[-1][0]
        if last > best_last and cycle_hours:
            best_last, best_hours = last, tuple(cycle_hours)
    return best_hours


def resolve_cycle(route: Route, cycle: datetime) -> datetime:
    """Refuse a cycle hour the producer does not run."""

    if cycle.hour not in route.cycle_hours:
        offered = ", ".join(f"{hour:02d}" for hour in route.cycle_hours)
        raise ValueError(
            f"--source {route.source_id} --cycle {cycle:%Y-%m-%dT%H}: "
            f"{cycle:%H}Z is not a cycle this producer runs.\n"
            f"  it runs: {offered} (UTC).")
    return cycle


def resolve_leads(route: Route, cycle: datetime, hours: int, *,
                  cadence: int | None = None,
                  start_hour: int = 0) -> tuple[int, ...]:
    """The ordered leads a window asks for, checked against the ladder."""

    if hours < 0:
        raise ValueError("--hours cannot be negative")
    cadence = route.default_cadence if cadence is None else int(cadence)
    if cadence not in route.cadences:
        offered = ", ".join(str(value) for value in route.cadences)
        raise ValueError(
            f"--cadence {cadence}: --source {route.source_id} publishes at "
            f"{offered} h spacing (default {route.default_cadence}).")
    ladder = ladder_for(route, cycle)
    last = start_hour + hours
    if last > ladder[-1]:
        extended = _extended_cycle_hours(route)
        reach = ""
        if extended:
            extended_last = max(
                steps[-1][0] for cycle_hours, steps in route.ladders
                if cycle_hours == extended)
            reach = (f"  the {', '.join(f'{h:02d}' for h in extended)}Z "
                     f"cycles reach f{extended_last:03d}.")
        raise ValueError(
            f"--source {route.source_id}: the {cycle:%H}Z cycle forecasts "
            f"through f{ladder[-1]:03d}; this window ends at f{last:03d}.\n"
            f"{reach}".rstrip())
    wanted = [lead for lead in ladder
              if start_hour <= lead <= last
              and (lead - start_hour) % cadence == 0]
    missing = [lead for lead in range(start_hour, last + 1, cadence)
               if lead not in set(ladder)]
    if missing:
        raise ValueError(
            f"--source {route.source_id}: the {cycle:%H}Z cycle does not "
            f"publish f{missing[0]:03d}; its ladder runs "
            f"{_ladder_words(route, cycle)}.")
    if not wanted:
        raise ValueError(
            f"--source {route.source_id}: the requested window resolves to "
            "no forecast lead at all")
    return tuple(wanted)


def _ladder_words(route: Route, cycle: datetime) -> str:
    for cycle_hours, steps in route.ladders:
        if cycle_hours is None or cycle.hour in cycle_hours:
            parts = []
            previous = 0
            for index, (through, step) in enumerate(steps):
                start = 0 if index == 0 else previous + step
                parts.append(
                    f"f{start:03d}..f{through:03d} every {step} h")
                previous = through
            return ", then ".join(parts)
    return ""


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------

def member_tokens(route: Route) -> Mapping[str, str]:
    """``member name -> filename/path token`` for the declared member set.

    Two vocabularies, because the producers use two: GEFS's control is
    ``c00`` in the member grammar and ``gec00`` in the key, while every
    AIGEFS member spells the same string in both.  The table carries the
    pair so neither is inferred from the other.
    """

    if route.members is None:
        return MappingProxyType({})
    default = str(route.members["default"])
    names: dict[str, str] = {
        default: str(route.members.get("default_token", default))}
    low, high = route.members.get("perturbed_range", (1, 0))
    name_fmt = route.members.get("perturbed_name_format")
    token_fmt = route.members.get("perturbed_token_format", name_fmt)
    if name_fmt:
        for number in range(int(low), int(high) + 1):
            names.setdefault(str(name_fmt) % number,
                             str(token_fmt) % number)
    return MappingProxyType(names)


def resolve_member(route: Route, member: str | None) -> tuple[str, str]:
    """``(member_name, filename_token)`` for the requested member."""

    if route.members is None:
        if member is not None:
            raise ValueError(
                f"--member {member}: --source {route.source_id} is not an "
                "ensemble; it publishes one deterministic state.")
        return "", ""
    name = str(route.members["default"]) if member is None else str(member)
    known = member_tokens(route)
    if name not in known:
        listed = tuple(known)
        raise ValueError(
            f"--member {name}: --source {route.source_id} publishes "
            f"{listed[0]} (the control) and {listed[1]}..{listed[-1]} -- "
            f"{len(listed)} members in all.")
    return name, known[name]


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedObject:
    """One published object, and every endpoint that publishes it.

    ``key`` is the host-independent part -- the rendered path template
    -- and it is what makes the ladder possible at all: the NCEP
    operational server and the AWS archive answer the SAME relative key
    with the same bytes (HEAD-verified for every source in the family),
    so falling through hosts is appending one key to another base, not
    re-planning the request.  ``url`` is that key on the ladder's head,
    kept as a field because a plan is read for what it WILL do.
    """

    name: str
    url: str
    relpath: str
    role: str
    lead: int | None
    idx_url: str | None
    key: str = ""

    def urls(self, ladder: Sequence[Endpoint]) -> tuple[str, ...]:
        if not self.key:
            return (self.url,)
        return tuple(endpoint.url(self.key) for endpoint in ladder)


@dataclass(frozen=True)
class ComposePart:
    role: str
    relpath: str


@dataclass(frozen=True)
class ComposeStep:
    kind: str
    name: str
    primary: bool
    lead: int
    parts: tuple[ComposePart, ...]


@dataclass(frozen=True)
class DonorRequest:
    source: str
    role: str
    cycle: datetime
    leads: tuple[int, ...]
    why: str


@dataclass(frozen=True)
class FetchPlan:
    route: Route
    host: Host
    cycle: datetime
    leads: tuple[int, ...]
    member: str
    objects: tuple[PlannedObject, ...]
    compose: tuple[ComposeStep, ...]
    donors: tuple[DonorRequest, ...]
    primary_files: tuple[Path, ...]
    supplement_files: tuple[Path, ...]
    supplement_role: str | None
    member_set: str | None
    out: Path | None = None
    #: The endpoints this cycle may be asked for, in RETENTION order:
    #: the operational server while it still holds the cycle, the
    #: archive behind it.  One entry when ``--transport`` pinned a
    #: host.  Which of them actually moves each object is settled at
    #: transfer time, by availability -- planning stays network-free.
    ladder: tuple[Endpoint, ...] = ()
    #: The host ``--transport`` named, if any.  A typed host is a
    #: decision: it disables fall-through and it IS the request's
    #: identity, where an unpinned request's identity is the ladder.
    pinned_host: str | None = None

    @property
    def source_id(self) -> str:
        return self.route.source_id


def _cycle_context(cycle: datetime) -> dict[str, str]:
    return {
        "YYYY": f"{cycle:%Y}", "MM": f"{cycle:%m}", "DD": f"{cycle:%d}",
        "HH": f"{cycle:%H}", "YYYYMMDD": f"{cycle:%Y%m%d}",
        "YYYYMMDDHH": f"{cycle:%Y%m%d%H}",
        "YYYYMMDDHHMMSS": f"{cycle:%Y%m%d%H}0000",
    }


def _render(template: str, context: Mapping[str, str]) -> str:
    def replace(match: re.Match) -> str:
        token = match.group(1)
        if token not in context:
            raise ValueError(
                f"path template {template!r} wants {{{token}}}, which this "
                "route's axes do not provide")
        return context[token]

    return _TOKEN_RE.sub(replace, template)


def _axis_entries(route: Route, axis: str) -> tuple[dict[str, str], ...]:
    """Expand one declared axis into its per-object token dictionaries."""

    entries: list[dict[str, str]] = []
    for group in route.axes[axis]:
        leveltype = str(group.get("leveltype", ""))
        levels = group.get("levels")
        literal = group.get("level_literal")
        prefix = str(group.get("level_prefix", ""))
        fmt = str(group.get("level_format", "%d"))
        for name in group["names"]:
            if levels:
                for level in levels:
                    text = prefix + (fmt % int(level))
                    entries.append({
                        "FIELD": str(name),
                        "field_lower": str(name).lower(),
                        "leveltype": leveltype,
                        "LEVEL": text,
                        "LEVEL_SUFFIX": f"_{text}",
                    })
            elif literal:
                entries.append({
                    "FIELD": str(name), "field_lower": str(name).lower(),
                    "leveltype": leveltype, "LEVEL": str(literal),
                    "LEVEL_SUFFIX": f"_{literal}",
                })
            else:
                entries.append({
                    "FIELD": str(name), "field_lower": str(name).lower(),
                    "leveltype": leveltype, "LEVEL": "", "LEVEL_SUFFIX": "",
                })
    return tuple(entries)


def _relpath(route: Route, key: str) -> str:
    if route.layout == "upstream":
        return f"upstream/{key}"
    return key.rsplit("/", 1)[-1]


def resolve_mode(source: str, mode: str | None) -> str:
    """The byte transport a table route runs with.

    Full files are the default and the pipeline; record subsetting is an
    opt-in bandwidth saver, and a route that cannot honour it refuses in
    its own words rather than silently degrading.
    """

    route = route_for(source)
    if mode is None or mode == "full-file":
        return "full-file"
    if mode == "idx-subset":
        if route.record_subset_supported:
            return "idx-subset"
        raise ValueError(
            f"--mode idx-subset: --source {route.source_id} takes whole "
            "objects.\n"
            f"  why: {route.record_subset_why}.\n"
            "  remedy: drop --mode (full-file is the default and the "
            "pipeline).")
    if mode == "auto":
        raise ValueError(
            f"--mode auto: --source {route.source_id} has one byte "
            "transport, so there is nothing to probe.  Drop --mode.")
    raise ValueError(f"--mode {mode}: unknown transport")


def endpoint_ladder(route: Route, cycle: datetime, *,
                    host: str | None = None,
                    now: datetime | None = None) -> tuple[Endpoint, ...]:
    """The endpoints ``cycle`` will be asked for, in order.

    The cycle is resolved BEFORE this is called -- that is the whole
    point.  A latest initialization is hours old, so the operational
    server (which published it first) heads the ladder; a cycle older
    than that server's measured window is not asked for there at all,
    and the archive is asked directly.
    """

    return fetch_endpoints.serving_ladder(
        route.source_id, cycle=cycle, now=now, pinned=host)


def resolve_request(source: str, *, cycle: datetime, hours: int,
                    cadence: int | None = None, start_hour: int = 0,
                    host: str | None = None, member: str | None = None,
                    area: str | None = None,
                    out: Path | None = None,
                    now: datetime | None = None) -> FetchPlan:
    """Everything a fetch will do, decided before a single byte moves."""

    route = route_for(source)
    if area is not None:
        raise ValueError(
            f"--area/--point: --source {route.source_id} publishes whole "
            "objects and there is no subsetting service in front of them.\n"
            "  where the crop happens: `gpuwm prep` maps the source onto "
            "your domain, so the namelist geometry is the crop.")
    resolve_cycle(route, cycle)
    leads = resolve_leads(route, cycle, hours, cadence=cadence,
                          start_hour=start_hour)
    if host is not None:
        route.host(host)          # refuses in the route's own words
    ladder = endpoint_ladder(route, cycle, host=host, now=now)
    chosen = ladder[0]
    member_name, member_token = resolve_member(route, member)

    context = _cycle_context(cycle)
    context["MEMBER"] = member_token

    objects: list[PlannedObject] = []
    by_role: dict[str, list[tuple[int, str]]] = {}
    seen: set[str] = set()

    def emit(row: FileRow, lead: int | None) -> None:
        lead_context = dict(context)
        if lead is not None:
            lead_context.update({
                "F": str(lead), "FF": f"{lead:02d}", "FFF": f"{lead:03d}"})
        for entry in (_axis_entries(route, row.axis) if row.axis else ({},)):
            key = _render(row.path, {**lead_context, **entry})
            if key in seen:
                continue
            seen.add(key)
            relpath = _relpath(route, key)
            objects.append(PlannedObject(
                name=relpath, url=chosen.url(key), relpath=relpath,
                role=row.role, lead=lead, key=key,
                idx_url=(chosen.url(f"{key}{row.idx_sidecar}")
                         if row.idx_sidecar else None)))
            by_role.setdefault(row.role, []).append(
                (lead if lead is not None else -1, relpath))

    # Lead-major, so the pool's in-order admitted prefix is a contiguous
    # run of COMPLETE valid times: an interrupted fetch leaves a series a
    # shorter window can still be prepared from, never half of every hour.
    for row in route.files:
        if row.leads == "none":
            emit(row, None)
    for lead in leads:
        for row in route.files:
            if row.leads == "none":
                continue
            if row.leads == "first" and lead != leads[0]:
                continue
            emit(row, lead)

    compose: list[ComposeStep] = []
    for row in route.compose:
        for lead in leads:
            parts = tuple(
                ComposePart(role=role, relpath=relpath)
                for role in row.roles
                for entry_lead, relpath in by_role.get(role, ())
                if entry_lead == lead
                or (entry_lead == leads[0] and lead == leads[0]))
            if not parts:
                continue
            lead_context = dict(context)
            lead_context.update({
                "F": str(lead), "FF": f"{lead:02d}", "FFF": f"{lead:03d}"})
            compose.append(ComposeStep(
                kind=row.kind, name=_render(row.name, lead_context),
                primary=row.primary, lead=lead, parts=parts))

    donors = tuple(
        DonorRequest(source=row.source, role=row.role, cycle=cycle,
                     leads=row.leads, why=row.why)
        for row in route.donors)

    if compose:
        primary = tuple(Path(step.name) for step in compose if step.primary)
    else:
        primary = tuple(
            Path(obj.relpath) for obj in objects
            if any(row.primary and row.role == obj.role for row in route.files))

    supplement_spec = route.prep.get("supplement")
    supplements: tuple[Path, ...] = ()
    supplement_role: str | None = None
    if supplement_spec:
        supplement_role = str(supplement_spec["role"])
        origin = str(supplement_spec["from"])
        select = supplement_spec.get("select")
        if origin == "every_input":
            supplements = primary
        elif origin == "first_input":
            supplements = primary[:1]
        elif origin.startswith("role:"):
            role = origin.split(":", 1)[1]
            supplements = tuple(
                Path(obj.relpath) for obj in objects
                if obj.role == role
                and (select is None or f"_{select}." in obj.relpath
                     or f"_{select}_" in obj.relpath))
        else:
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: route {route.source_id} declares an "
                f"unknown supplement origin {origin!r}")

    return FetchPlan(
        route=route, host=chosen, cycle=cycle, leads=leads,
        member=member_name, objects=tuple(objects), compose=tuple(compose),
        donors=donors, primary_files=primary, supplement_files=supplements,
        supplement_role=supplement_role,
        member_set=(str(route.prep["member_prep"])
                    if route.prep.get("member_prep") else None),
        out=out, ladder=ladder, pinned_host=host)


# --------------------------------------------------------------------------
# Transfer
# --------------------------------------------------------------------------

MANIFEST_NAME = "fetch-manifest.json"
SHA256SUMS_NAME = "SHA256SUMS"
INPUT_LIST_NAME = "inputs.txt"
PREP_COMMAND_NAME = "prep-command.txt"

#: The same bound half, as machine-readable argv tokens.  The text file
#: above is a reader's; this one is a CALLER'S -- ``gpuwm run-plan``'s
#: staged chain composes its preparation from it, so the binding is
#: relayed from the fetch's own artifact instead of being re-derived
#: from a second copy of the route table.  Tokens, not a command
#: string, so no quoting convention has to round-trip Windows paths.
PREP_ARGUMENTS_NAME = "prep-arguments.json"
PREP_ARGUMENTS_SCHEMA = "gpuwm-fetch-prep-arguments-v1"

_USER_AGENT = "gpuwm-fetch/2.5 (+https://github.com/arwenweather)"
_CHUNK = 1 << 20


def _magic_for(plan: FetchPlan, role: str) -> str:
    for row in plan.route.files:
        if row.role == role:
            return row.magic
    return "GRIB"


def _verify_payload(path: Path, *, magic: str, label: str) -> None:
    """The cheapest honest completeness bar for a downloaded object.

    Leading magic separates a payload from an HTML error page a proxy
    served with HTTP 200; the GRIB2 end marker separates a complete
    object from a truncated transfer, which is the failure a
    length-only check misses when the server closes the connection
    early and reports no length at all.
    """

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{label}: the server returned an empty object")
    with path.open("rb") as handle:
        head = handle.read(len(magic))
        if head != magic.encode("ascii"):
            raise ValueError(
                f"{label}: expected a {magic} payload and the first bytes "
                f"are {head!r} -- the host answered with something that is "
                "not this product")
        if magic == "GRIB":
            handle.seek(max(0, size - 4))
            if handle.read(4) != b"7777":
                raise ValueError(
                    f"{label}: the GRIB2 end marker is missing, so the "
                    f"transfer is truncated at {size} bytes")


def _object_token(obj: PlannedObject) -> str | None:
    """What to call one planned object to a person.

    ``f01 atmosphere`` rather than the relpath alone: with six transfers
    in flight, the lead and the role are what tell a reader which part
    of the request is moving, and the legacy routes have printed exactly
    this token for as long as they have printed anything.
    """

    parts = []
    if getattr(obj, "lead", None) is not None:
        parts.append(f"f{int(obj.lead):02d}")
    if getattr(obj, "role", None):
        parts.append(str(obj.role))
    return " ".join(parts) or None


def _admission_reporter(label: str, *, total: int, progress):
    """The per-file completion line, as an ``on_admitted`` callback.

    Named rather than inlined so its TEXT can be pinned by a test.  This
    is the line a reader has been parsing since the serial loop, and the
    start and in-flight lines added around it must not disturb it.
    """

    def landed(index: int, entry: dict) -> None:
        note = ("already present" if entry.get("reused")
                else f"{entry.get('bytes', 0) / (1024 * 1024):.1f} MiB")
        progress(progress_mod.format_transfer_done_line(
            label=label, index=index, total=total,
            name=entry.get("relpath", entry.get("name")), note=note))

    return landed


def _download_object(url: str, dest: Path, *, magic: str, opener=None,
                     timeout: float = 300.0, progress=None,
                     declared_size=None) -> dict:
    """Move one object, verify it, and return its manifest entry.

    ``progress`` is called with the size of each chunk as it lands.  The
    table routes move objects that are hundreds of megabytes each, and
    without this the whole transfer was one silent gap between the
    route's opening line and its manifest (UX finding N10).
    """

    from urllib.request import Request  # local: keeps import cost off load

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    written = 0
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    from gpuwm.nomads_governor import paced_urlopen
    response = paced_urlopen(
        request, timeout=timeout,
        **({"opener": opener} if opener is not None else {}))
    declared = response.headers.get("Content-Length")
    if declared_size is not None and declared is not None:
        # The size the host itself states, the moment it states it.  A
        # HEAD ahead of every transfer would double the request count on
        # exactly the services whose per-request latency the pool exists
        # to hide, so the in-flight line learns the total from the
        # transfer that is already open.
        try:
            declared_size(dest, int(declared))
        except (TypeError, ValueError):          # pragma: no cover
            pass
    with response, part.open("wb") as handle:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            if progress is not None:
                progress(len(chunk))
    if declared is not None and int(declared) != written:
        part.unlink(missing_ok=True)
        raise ValueError(
            f"{dest.name}: the host declared {int(declared)} bytes and "
            f"delivered {written}")
    part.replace(dest)
    _verify_payload(dest, magic=magic, label=dest.name)
    return {"name": dest.name, "bytes": written, "sha256": digest.hexdigest(),
            "url": url}


def _download_along_ladder(plan: FetchPlan, obj: PlannedObject, dest: Path, *,
                           magic: str, fetch, opener, progress,
                           ladder: Sequence[Endpoint] | None = None) -> dict:
    """Move one object, asking each endpoint in turn until one serves.

    The endpoints publish the same key with the same bytes, so a host
    that refuses, throttles, or serves something that does not verify
    is a reason to ask the next one -- not a reason to end the fetch.
    Faults that are not an endpoint's fault (an interrupt, a full disk)
    propagate unchanged, because the next endpoint would fail
    identically and walking the ladder over them would only bury the
    real refusal.

    ``ladder`` is this OBJECT's order, which is the request's ladder
    with any rung that provably already holds the object promoted to
    the head (see :func:`_probe_transfer_ladders`).  It is a reorder,
    never a shorter list, so everything below is unchanged by it.

    When every endpoint fails, the refusal names each one and why.
    """

    ladder = ladder or plan.ladder or (plan.host,)
    attempts: list[tuple[Endpoint, str]] = []
    for position, endpoint in enumerate(ladder):
        url = endpoint.url(obj.key) if obj.key else obj.url
        try:
            entry = fetch(url, dest, magic=magic, opener=opener)
        except BaseException as error:        # noqa: BLE001 - classified
            reason = fetch_endpoints.fault_reason(error)
            if reason is None:
                raise
            attempts.append((endpoint, reason))
            remaining = ladder[position + 1:]
            if not remaining:
                raise ValueError(fetch_endpoints.ladder_refusal(
                    f"fetch {plan.source_id}: {obj.relpath}",
                    attempts)) from error
            dest.with_name(dest.name + ".part").unlink(missing_ok=True)
            progress(
                f"fetch {plan.source_id}: {endpoint.name} did not serve "
                f"{obj.relpath} ({reason}); asking {remaining[0].name}")
            continue
        return {**entry, "endpoint": endpoint.name}
    raise ValueError(
        f"fetch {plan.source_id}: {obj.relpath} has no endpoint to ask")


def _probe_transfer_ladders(
        plan: FetchPlan, objects: Sequence[PlannedObject], *, probe,
        workers: int, progress) -> tuple[dict[str, tuple[Endpoint, ...]],
                                         dict | None]:
    """Ask the throughput rung which of these objects it already has.

    The measured cost this exists to remove: at peak hours the
    operational server paced whole-file transfers at about 3 MB/s per
    file, so a 3.4 GB request took ~20 min where the archive had served
    the same volume in ~3.  The archive is the same bytes under the
    same key; the only reason to pay that is an object the archive does
    not have YET, which is exactly what a HEAD settles.

    One HEAD per object -- milliseconds against a multi-hundred-megabyte
    transfer -- and they run AHEAD of the transfers through the same
    pool, under the same per-host caps, so no probe ever waits behind
    a download.  Probing decides ORDER only: the returned ladder for an
    object is the request's ladder reordered, so every rung is still
    behind the chosen one and a probe that says no (a 404 because the
    mirror lags, a 503 because it is throttling) costs the transfer
    nothing at all.

    Returns ``(per-object ladders, probe receipt)``; both are empty when
    there was nothing to choose between -- a pinned ``--transport``, a
    one-rung ladder, or a source whose ladder head is already its
    quickest host.
    """

    ladder = plan.ladder or (plan.host,)
    if plan.pinned_host is not None or not objects:
        return {}, None
    candidates = fetch_endpoints.transfer_probes(ladder)
    if not candidates:
        return {}, None

    def ask(obj: PlannedObject) -> dict:
        rungs = (fetch_endpoints.transfer_ladder(
            ladder, (obj.key,), probe=probe) if obj.key else ladder)
        return {"relpath": obj.relpath, "ladder": rungs}

    entries, _receipt = fetch_pool.run_transfers(
        [fetch_pool.TransferJob(
            name=obj.relpath, url=candidates[0].url(obj.key),
            action=functools.partial(ask, obj))
         for obj in objects],
        workers=workers)

    ladders = {entry["relpath"]: entry["ladder"] for entry in entries
               if entry["ladder"][0] is not ladder[0]}
    preferred = candidates[0]
    available = len(ladders)
    if available:
        progress(
            f"fetch {plan.source_id}: mirrored: taking the archive for "
            f"throughput -- {available} of {len(objects)} object"
            f"{'' if len(objects) == 1 else 's'} "
            f"{'is' if available == 1 else 'are'} already on "
            f"{preferred.name} ({preferred.host})"
            + (f"; the rest from {ladder[0].name}, which publishes before "
               "the mirrors" if available < len(objects) else ""))
    else:
        progress(
            f"fetch {plan.source_id}: {preferred.name} has not caught up "
            f"with this cycle -- using {ladder[0].name}, which publishes "
            "before the mirrors")
    return ladders, {"endpoint": preferred.name, "objects": len(objects),
                     "available": available}


def _prior_entries(out: Path) -> dict[str, dict]:
    manifest = out / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(entry["relpath"]): entry
            for entry in document.get("files", [])
            if isinstance(entry, dict) and entry.get("relpath")}


#: What an unpinned request records where it used to record one host.
#:
#: The guard below exists to stop two different CYCLES publishing one
#: ``SHA256SUMS``.  Recording the SERVED endpoint as request identity
#: would have made an ordinary fall-through -- the operational server
#: throttling for fifteen minutes, the archive finishing the job --
#: look like exactly that, and refuse the resume that was the remedy.
LADDER_IDENTITY = "ladder"


def _request_identity(plan: FetchPlan) -> dict:
    return {
        "source": plan.source_id,
        "cycle": f"{plan.cycle:%Y-%m-%dT%H}Z",
        "host": plan.pinned_host or LADDER_IDENTITY,
        "member": plan.member,
        "leads": list(plan.leads),
    }


def check_prior_request(out: Path, plan: FetchPlan) -> None:
    """Refuse to publish two different requests into one directory."""

    manifest = out / MANIFEST_NAME
    if not manifest.is_file():
        return
    try:
        prior = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    wanted = _request_identity(plan)
    recorded = prior.get("request", {})

    def differs(key: str) -> bool:
        if recorded.get(key) == wanted[key]:
            return False
        if key != "host":
            return True
        # An unpinned request is compatible with a directory whose
        # prior receipt recorded whichever endpoint happened to serve
        # it -- including one written before this ArWen had a ladder.
        return not (wanted[key] == LADDER_IDENTITY
                    and recorded.get(key) in
                    {host.name for host in plan.route.hosts})

    differing = [key for key in ("source", "cycle", "host", "member")
                 if differs(key)]
    if differing:
        detail = ", ".join(
            f"{key} {recorded.get(key)!r} -> {wanted[key]!r}"
            for key in differing)
        raise ValueError(
            f"--out {out} already holds a different request ({detail}).\n"
            "  remedy: fetch into a different --out, or pass "
            "--force-refetch to move the existing files aside (nothing is "
            "deleted) and re-download this request.\n"
            "  why: one directory publishes one SHA256SUMS and one input "
            "list, and a mixed directory would hand `gpuwm prep` a series "
            "spanning two cycles.")


def _quarantine(out: Path, progress) -> Path | None:
    """Move an existing fetch aside; nothing is deleted."""

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    aside = out / f"quarantine-{stamp}"
    moved = 0
    for entry in sorted(out.iterdir()):
        if entry.name.startswith("quarantine-"):
            continue
        aside.mkdir(parents=True, exist_ok=True)
        entry.replace(aside / entry.name)
        moved += 1
    if moved:
        progress(f"fetch: --force-refetch moved {moved} entr"
                 f"{'y' if moved == 1 else 'ies'} to {aside}")
        return aside
    return None


def run_plan(plan: FetchPlan, *, out: Path, force: bool = False,
             file_workers: int | None = None, progress=print,
             opener=None, downloader=None, probe=None) -> dict:
    """Move the planned objects, compose the primaries, write the receipts.

    Every file rides :mod:`gpuwm.fetch_pool`, so a table route is
    parallel by default with the same bounded, host-capped, in-order
    admission the GFS and HRRR routes have: one failed file still
    refuses by name, and the verified prefix a receipt claims is
    contiguous.

    ``probe`` is the availability question the transfer host is chosen
    with -- ``url -> bool``, defaulting to one governed HEAD.  It runs
    ahead of the transfers, through the same pool, and only for the
    objects this run has still to download.
    """

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if force:
        _quarantine(out, progress)
    else:
        check_prior_request(out, plan)
    prior = _prior_entries(out)
    # ONE byte counter for the whole request, not one per object: the
    # pool keeps several transfers in flight, and six interleaved
    # counters read worse than one.  Injected downloaders keep the
    # signature they always had -- the counter is bound to the default
    # transport only, so a route test's fake is called exactly as before.
    counter = ByteCounter(f"fetch {plan.source_id}")
    # WHICH FILES ARE MOVING, while they are moving.  The counter above
    # says how many bytes the request as a whole has moved and cannot say
    # whose; once the pool put six transfers in flight that became the
    # only per-file signal, arriving at completion.  Default-on: no flag
    # turns this on, and a bare run stops going quiet.
    monitor = progress_mod.TransferMonitor(f"fetch {plan.source_id}")
    fetch = downloader
    if fetch is None:
        # ``declared_size`` is bound to the DEFAULT transport only, for
        # the same reason ``progress`` is: an injected downloader keeps
        # the four-argument signature it has always had, so a route
        # test's fake is called exactly as before.
        fetch = functools.partial(_download_object,
                                  progress=counter.advance,
                                  declared_size=monitor.declare_for_path)

    workers = fetch_pool.resolve_file_workers(file_workers)
    ladder = plan.ladder or (plan.host,)

    # Which objects this run still has to move -- the only ones worth
    # asking a host about.
    reuse: dict[str, dict] = {}
    pending: list[PlannedObject] = []
    for obj in plan.objects:
        known = prior.get(obj.relpath)
        dest = out / obj.relpath
        if (known and dest.is_file()
                and dest.stat().st_size == known.get("bytes")):
            reuse[obj.relpath] = known
        else:
            pending.append(obj)

    # AHEAD of the transfers, not behind them: the throughput rung is
    # asked which of the pending objects it already holds, and each one
    # that it does takes it.  See _probe_transfer_ladders.
    promoted, probe_receipt = _probe_transfer_ladders(
        plan, pending, workers=workers, progress=progress,
        probe=(fetch_endpoints.object_available if probe is None else probe))

    reused = len(reuse)
    jobs = []
    for obj in plan.objects:
        dest = out / obj.relpath
        magic = _magic_for(plan, obj.role)
        known = reuse.get(obj.relpath)
        if known is not None:

            def _reuse(entry=known, relpath=obj.relpath, role=obj.role,
                       lead=obj.lead) -> dict:
                return {**entry, "relpath": relpath, "role": role,
                        "lead": lead, "reused": True}

            jobs.append(fetch_pool.TransferJob(
                name=obj.relpath, url=None, action=_reuse,
                token=_object_token(obj),
                expected_bytes=known.get("bytes")))
            continue

        rungs = promoted.get(obj.relpath, ladder)

        def _get(obj=obj, dest=dest, magic=magic, relpath=obj.relpath,
                 role=obj.role, lead=obj.lead, rungs=rungs) -> dict:
            entry = _download_along_ladder(
                plan, obj, dest, magic=magic, fetch=fetch, opener=opener,
                progress=progress, ladder=rungs)
            return {**entry, "relpath": relpath, "role": role, "lead": lead,
                    "reused": False}

        # The politeness key is the host this object will ACTUALLY be
        # asked first, not the ladder's head: counting a mirrored
        # transfer against the operational server's cap of 2 would
        # throttle the fetch to the pace of the host it just avoided.
        jobs.append(fetch_pool.TransferJob(
            name=obj.relpath, url=rungs[0].url(obj.key) if obj.key
            else obj.url, action=_get,
            token=_object_token(obj),
            # WHERE IT LANDS, so the in-flight byte count can be read off
            # the growing file.  The transport reports its own chunks
            # here, but a route whose copy is owned by something else --
            # the Rust fetch bridge shells out and reports nothing until
            # it exits -- has no other honest source, and this one costs
            # a stat().
            path=dest))

    # WHERE THE BYTES ARE ACTUALLY COMING FROM, not where the ladder
    # starts.  Naming the ladder's head here would contradict the
    # mirrored note printed a moment ago, and an opening line that
    # disagrees with the receipt is worse than no opening line.
    heads: list[Endpoint] = []
    for obj in plan.objects:
        head = promoted.get(obj.relpath, ladder)[0]
        if head not in heads:
            heads.append(head)
    serving = (f"{heads[0].name} ({heads[0].base})" if len(heads) == 1
               else " and ".join(entry.name for entry in heads))
    behind = ", then ".join(entry.name for entry in ladder
                            if entry not in heads)
    progress(
        f"fetch {plan.source_id}: {len(plan.objects)} object"
        f"{'' if len(plan.objects) == 1 else 's'} from {serving}, cycle "
        f"{plan.cycle:%Y-%m-%dT%H}Z, leads "
        f"f{plan.leads[0]:03d}..f{plan.leads[-1]:03d}"
        + (f", member {plan.member}" if plan.member else "")
        + (f"; {behind} behind it" if behind else "")
        + (f"; {reused} already present" if reused else ""))

    # WHAT IT IS DOING, per object, as the verified prefix grows.  The
    # measured shape of this finding: a 792 MB hrrr-prs request printed
    # its opening line and then nothing at all until the manifest, so a
    # slow link and a hung command looked identical (UX finding N10).
    _landed = _admission_reporter(f"fetch {plan.source_id}", total=len(jobs),
                                  progress=progress)

    try:
        entries, receipt = fetch_pool.run_transfers(
            jobs, workers=workers, on_admitted=_landed, monitor=monitor)
    finally:
        monitor.close()
        counter.close()

    composed = _run_compose(plan, out, progress=progress)
    payload = {
        "schema": ROUTE_MANIFEST_SCHEMA,
        "route_table_sha256": packaged_route_table_sha256(),
        "request": _request_identity(plan),
        "label": plan.route.label,
        "mode": "full-file",
        # WHERE THE BYTES CAME FROM.  A receipt that named only the
        # ladder's head would be a claim about intent, not provenance:
        # a fall-through mid-request is normal and must be readable
        # afterwards, per file and in summary.
        "endpoints": _endpoint_receipt(plan, entries, probe=probe_receipt),
        "files": entries,
        "composed": composed,
        "concurrency": receipt,
        "donors": [
            {"source": donor.source, "role": donor.role,
             "cycle": f"{donor.cycle:%Y-%m-%dT%H}Z",
             "leads": list(donor.leads), "why": donor.why}
            for donor in plan.donors],
        "prep": {
            "source": str(plan.route.prep.get("source", plan.source_id)),
            "member_set": plan.member_set,
            "supplement_role": plan.supplement_role,
            "primary_files": [str(path) for path in plan.primary_files],
            "supplement_files": [str(path)
                                 for path in plan.supplement_files],
        },
    }
    _write_json(out / MANIFEST_NAME, payload)
    _write_sha256sums(out, entries, composed)
    return payload


def _endpoint_receipt(plan: FetchPlan,
                      entries: Sequence[Mapping[str, object]],
                      *, probe: Mapping[str, object] | None = None) -> dict:
    """Which endpoints this request considered, and which ones served.

    ``transfer_preference`` is the throughput order the table declares
    and ``probe`` is what the availability question actually answered,
    so a reader can tell "the mirror served because it had the object"
    from "the mirror served because the operational server failed" --
    two very different runs that name the same host.
    """

    ladder = plan.ladder or (plan.host,)
    served: list[str] = []
    for entry in entries:
        name = entry.get("endpoint")
        if isinstance(name, str) and name not in served:
            served.append(name)
    return {
        "pinned": plan.pinned_host,
        "considered": [endpoint.name for endpoint in ladder],
        "transfer_preference": [
            endpoint.name
            for endpoint in fetch_endpoints.transfer_order(ladder)],
        "served": served,
        "probe": (dict(probe) if probe is not None else None),
        "ladder": [
            {"name": endpoint.name, "base": endpoint.base,
             "host": endpoint.host,
             "retention_hours": endpoint.retention_hours,
             "transfer_rank": endpoint.transfer_rank,
             "why": endpoint.why}
            for endpoint in ladder],
    }


def _run_compose(plan: FetchPlan, out: Path, *, progress=print) -> list[dict]:
    composed: list[dict] = []
    for step in plan.compose:
        if step.kind != "concat_per_lead":
            raise ValueError(
                f"{ROUTE_TABLE_NAME}: unknown compose kind {step.kind!r}")
        dest = out / step.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        with dest.open("wb") as handle:
            for part in step.parts:
                data = (out / part.relpath).read_bytes()
                handle.write(data)
                digest.update(data)
                written += len(data)
        _verify_payload(dest, magic="GRIB", label=step.name)
        composed.append({
            "name": step.name, "bytes": written,
            "sha256": digest.hexdigest(), "lead": step.lead,
            "parts": [part.relpath for part in step.parts]})
    if composed:
        progress(f"fetch {plan.source_id}: composed {len(composed)} valid "
                 f"time{'' if len(composed) == 1 else 's'} from "
                 f"{sum(len(item['parts']) for item in composed)} objects")
    return composed


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_sha256sums(out: Path, entries: Sequence[Mapping[str, object]],
                      composed: Sequence[Mapping[str, object]]) -> None:
    lines = [f"{entry['sha256']}  {entry['relpath']}" for entry in entries]
    lines.extend(f"{item['sha256']}  {item['name']}" for item in composed)
    (out / SHA256SUMS_NAME).write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# The handoff: what makes a fetched directory a front door
# --------------------------------------------------------------------------

def write_handoff(plan: FetchPlan, out: Path, *,
                  donor_files: Mapping[str, Path] | None = None
                  ) -> tuple[Path, Path]:
    """Write the ordered ``--input-list`` and the bound prep command.

    A directory of verified bytes is not yet a front door: the caller
    still has to know which of them are the primaries, in what order,
    which one the composition binds as its surface supplement, and under
    which role.  All four are table facts, so they are written down here
    rather than left for a reader to reconstruct -- and the
    ``--input-list`` spelling is what keeps a field-per-file source's
    hundreds of inputs inside the 32 KB Windows command line.
    """

    out = Path(out)
    inputs = out / INPUT_LIST_NAME
    inputs.write_text(
        "".join(f"{(out / path).resolve()}\n" for path in plan.primary_files),
        encoding="utf-8", newline="\n")

    prep_source = str(plan.route.prep.get("source", plan.source_id))
    tokens: list[str] = ["--source", prep_source,
                        "--input-list", str(inputs.resolve())]
    for path in plan.supplement_files:
        binding = (f"{plan.supplement_role}={(out / path).resolve()}"
                   if plan.supplement_role else str((out / path).resolve()))
        tokens += ["--supplement", binding]
    unfetched: list[DonorRequest] = []
    for donor in plan.donors:
        supplied = (donor_files or {}).get(donor.role)
        if supplied is None:
            unfetched.append(donor)
            continue
        tokens += ["--supplement", f"{donor.role}={Path(supplied).resolve()}"]
    tokens += ["--author-input-manifest", str(out.resolve() / "inputs.json")]
    # The text rendering below derives from the SAME tokens, so the two
    # spellings of this handoff cannot disagree.
    arguments = [f"{flag} {_q(value)}"
                 for flag, value in zip(tokens[::2], tokens[1::2])]

    _write_prep_arguments(
        out, plan=plan, prep_source=prep_source, tokens=tokens,
        unfetched=unfetched)

    header = [
        f"# {plan.route.label}",
        f"# cycle {plan.cycle:%Y-%m-%dT%H}Z, leads "
        f"f{plan.leads[0]:03d}..f{plan.leads[-1]:03d}, endpoints "
        + " then ".join(entry.name
                        for entry in (plan.ladder or (plan.host,)))
        + f" (see {MANIFEST_NAME} for which one served each file)",
    ]
    if plan.member:
        header.append(f"# member {plan.member}")
    if plan.member_set:
        header.append("#")
        header.append(
            f"# member identity lives in the PATH, not the filename, so run")
        header.append(
            f"#   gpuwm-member-prep --member-set {plan.member_set} "
            f"--member {plan.member} \\")
        header.append(
            f"#     --cycle {plan.cycle:%Y-%m-%dT%H} --inputs "
            f"{out.resolve() / 'upstream'} --output {out.resolve() / 'members'}")
        header.append(
            "# first, and point --input-list at the verified member tree "
            "it publishes.")
    for donor in unfetched:
        header.append("#")
        header.append(
            f"# STILL NEEDED: --supplement {donor.role}=<a {donor.source} "
            f"{'/'.join(f'f{lead:03d}' for lead in donor.leads)} analysis "
            f"for {plan.cycle:%Y-%m-%dT%H}Z>")
        header.append(f"#   why: {donor.why}")
    header.append("#")
    header.append("# yours to supply: --wps-namelist, --experiment-config,")
    header.append("#                  --geog-root, --output-root")

    body = ["gpuwm prep \\"]
    body.extend(f"  {argument} \\" for argument in arguments[:-1])
    body.append(f"  {arguments[-1]}")

    command = out / PREP_COMMAND_NAME
    command.write_text("\n".join(header + [""] + body) + "\n",
                       encoding="utf-8", newline="\n")
    return inputs, command


def _write_prep_arguments(out: Path, *, plan: "FetchPlan", prep_source: str,
                          tokens: list[str],
                          unfetched: list[DonorRequest]) -> Path:
    """The bound prep handoff as one JSON document a caller composes from.

    ``argv`` is the exact token list ``prep-command.txt`` renders --
    the source binding, the ordered input list, every supplement role
    the route table and the fetched donors decided, and the manifest
    authoring flag.  What is NOT here is exactly what the text file's
    footer says is the caller's: ``--wps-namelist``,
    ``--experiment-config``, ``--geog-root`` and ``--output-root``.
    ``unbound_supplement_roles`` names any donor role this fetch could
    not supply, so a caller refuses before composing a preparation that
    rw-wps would refuse deeper.
    """

    document = {
        "schema": PREP_ARGUMENTS_SCHEMA,
        "source": plan.source_id,
        "prep_source": prep_source,
        "cycle": f"{plan.cycle:%Y-%m-%dT%H}",
        "argv": list(tokens),
        "caller_supplies": ["--wps-namelist", "--experiment-config",
                            "--geog-root", "--output-root"],
        "unbound_supplement_roles": sorted(
            donor.role for donor in unfetched),
        "member": plan.member,
        "member_set": plan.member_set,
    }
    path = out / PREP_ARGUMENTS_NAME
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def _q(value) -> str:
    text = str(value)
    return text if all(ch not in text for ch in ' \t"\'') else shlex.quote(text)


def handoff_lines(plan: FetchPlan, out: Path) -> tuple[str, ...]:
    """The ``next:`` block the fetch front door prints when it finishes."""

    out = Path(out)
    lines = [
        f"fetch {plan.source_id}: next: feed the mapped front door, source "
        "already bound:",
        f"  gpuwm prep --source "
        f"{plan.route.prep.get('source', plan.source_id)} --input-list "
        f"{_q((out / INPUT_LIST_NAME).resolve())}",
        f"  # the supplement bindings and the whole command are written out "
        f"at {(out / PREP_COMMAND_NAME).resolve()}",
        "  # --wps-namelist, --experiment-config, --geog-root and "
        "--output-root are yours.",
    ]
    return tuple(lines)


__all__ = [
    "ComposePart", "ComposeRow", "ComposeStep", "DonorRequest", "DonorRow",
    "Endpoint", "FetchPlan", "FileRow", "Host", "INPUT_LIST_NAME",
    "LADDER_IDENTITY", "LEGACY_ROUTE_SOURCES", "MANIFEST_NAME", "PATH_TOKENS",
    "PREP_ARGUMENTS_NAME", "PREP_ARGUMENTS_SCHEMA",
    "PREP_COMMAND_NAME", "PlannedObject", "ROUTE_MANIFEST_SCHEMA",
    "ROUTE_TABLE_NAME", "ROUTE_TABLE_SCHEMA", "ROUTE_TABLE_SHA256", "Route",
    "SHA256SUMS_NAME", "all_fetchable_sources", "check_prior_request",
    "endpoint_ladder", "handoff_lines", "ladder_for", "member_tokens",
    "packaged_route_table_sha256", "refusal_ids", "resolve_cycle",
    "resolve_leads", "resolve_member", "resolve_mode", "resolve_request",
    "route_for", "route_ids", "run_plan", "unknown_tokens", "write_handoff",
]
