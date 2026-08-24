"""Which endpoints a cycle is asked for, and which one moves the bytes.

Every NCEP source publishes on two hosts that serve byte-identical
objects under byte-identical keys -- verified by HEAD against the live
services: the operational server and the AWS Open Data archive answer
the same relative key with the same ``Content-Length`` for GFS, GDAS,
GEFS, HRRR, RAP and RRFS alike.  So the choice between them is never
about the data.  It is about three things, and only three:

* **Freshness.**  The operational server publishes each object hours
  before any mirror has it.  A run initialized from the newest cycle
  is asking for exactly that.
* **Retention.**  The operational server keeps a bounded window --
  measured by counting the day directories the live tree serves, two
  days for HRRR/RAP/RRFS, four for GEFS, ten for the GFS tree -- and
  the archive keeps everything.
* **Throughput.**  The operational server paces bulk transfers and the
  archive does not.  Measured at peak hours: about 3 MB/s per file
  against the archive serving the same 3.4 GB in roughly a sixth of
  the wall clock.

All three are DATA, declared per endpoint in
``authorities/rw-wps-fetch-routes.v1.json``, and this module is the one
engine that reads them.  Adding a model, or moving a host, is a row
there; nothing here branches on a source name, and the ladder shape is
identical for a route whose transport is table-driven and for the three
whose transport predates the table.

**Two orders over the same rungs.**  :func:`serving_ladder` answers
"which endpoints is this cycle asked for, and in what order" from
retention alone: a latest initialization keeps both, a reanalysis-era
cycle goes straight to the archive without paying for an attempt that
was certain to 404.  :func:`transfer_order` answers a different
question -- "which of them should actually move the bytes" -- from the
declared ``transfer_rank``.

They differ for exactly one reason.  The operational server's
advantage is having the cycle FIRST, and that advantage is spent the
moment the mirror has the same object; what is left after that is
throughput, and the mirror wins it.  So for each requested object
inside the retention window the caller asks the throughput rung
whether it HAS that object -- one HEAD, milliseconds against a
multi-hundred-megabyte transfer -- and :func:`promote` moves it to the
head when it does.  Promotion REORDERS the ladder and never shortens
it: every other rung stays behind the chosen one, so fall-through, the
fault vocabulary and the whole-ladder refusal are unchanged, and a
probe that 404s or errors costs the transfer nothing.

Retention is an optimisation, never a bar.  When NO endpoint's window
covers the age -- a source with no archive behind it -- the whole
ladder is still tried, because a host that MIGHT still hold the cycle
beats a refusal that never asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

#: The packaged acquisition authority.  One document: a source's cycle
#: and lead grammar, its file keys, and -- here -- its endpoints.
TABLE_NAME = "rw-wps-fetch-routes.v1.json"

#: What an availability probe identifies itself as.  Its own string, so
#: a provider reading their logs can tell a HEAD that moved nothing
#: from the transfers that follow it.
PROBE_USER_AGENT = "gpuwm-fetch-probe/2.5 (+https://github.com/arwenweather)"

#: HTTP statuses that mean "ask the next endpoint", each for its own
#: reason: the host is refusing this client (403), does not have this
#: object (404), is rate limiting (429), or is failing/overloaded
#: (5xx).  None of them means the OBJECT is wrong -- the other host
#: publishes the same key -- so every one of them is a reason to move
#: on rather than to end the fetch.
FALLTHROUGH_STATUSES = MappingProxyType({
    403: "the host refused this client",
    404: "the host does not serve this object",
    429: "the host is rate limiting this client",
    500: "the host failed on its side",
    502: "the host's gateway failed",
    503: "the host is unavailable or throttling",
    504: "the host's gateway timed out",
})


def _table_path() -> Path:
    return Path(__file__).with_name("authorities") / TABLE_NAME


@functools.lru_cache(maxsize=1)
def document() -> Mapping[str, object]:
    """The packaged acquisition authority, parsed once."""

    return json.loads(_table_path().read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Endpoint:
    """One rung of a source's ladder.

    ``retention_hours`` is how far back this endpoint serves, measured
    against the live tree; ``None`` is an archive -- effectively
    unbounded.  ``transfer_rank`` is where this rung sits in the
    THROUGHPUT order (lower is quicker), which is a different order
    from the ladder and decides only which host moves the bytes; a row
    that declares none inherits its ladder position, so a source with
    no measured throughput difference transfers in table order.
    ``why`` is the sentence a refusal quotes, so a reader who has just
    been told every endpoint failed also learns what each one was for.
    """

    name: str
    base: str
    retention_hours: float | None
    why: str
    transfer_rank: int = 0

    @property
    def host(self) -> str:
        """The politeness key: this endpoint's lower-cased netloc."""

        return urlsplit(self.base).netloc.lower()

    @property
    def archive(self) -> bool:
        return self.retention_hours is None

    def covers(self, age_hours: float) -> bool:
        """Does this endpoint still serve a cycle ``age_hours`` old?"""

        if self.retention_hours is None:
            return True
        return age_hours <= float(self.retention_hours)

    def url(self, key: str) -> str:
        """The absolute URL for one rendered object key."""

        return f"{self.base}/{key.lstrip('/')}"


def _endpoint(raw: Mapping[str, object], position: int) -> Endpoint:
    retention = raw.get("retention_hours")
    rank = raw.get("transfer_rank")
    return Endpoint(
        name=str(raw["name"]),
        base=str(raw["base"]),
        retention_hours=(None if retention is None else float(retention)),
        why=str(raw.get("why", "")),
        # No declared rank means "transfer in ladder order", which is
        # what every source outside the NCEP family does: the ladder
        # and the transfer order are then the same tuple, and nothing
        # is ever probed.
        transfer_rank=(position if rank is None else int(rank)),
    )


def _rows(source_id: str) -> tuple[Mapping[str, object], ...]:
    routes = dict(document().get("routes", {}))
    if source_id in routes:
        return tuple(dict(routes[source_id])["hosts"])
    legacy = dict(document().get("legacy_ladders", {}))
    raw = legacy.get(source_id)
    if isinstance(raw, list):
        return tuple(raw)
    return ()


@functools.lru_cache(maxsize=None)
def ladder(source_id: str) -> tuple[Endpoint, ...]:
    """``source_id``'s endpoints, in the order they are asked.

    Empty for a source with no declared endpoints at all (ERA5 is a
    manual CDS retrieval; there is no host to prefer).
    """

    rows = _rows(source_id)
    endpoints = tuple(_endpoint(row, position)
                      for position, row in enumerate(rows))
    for row, endpoint in zip(rows, endpoints):
        if row.get("default") and endpoint is not endpoints[0]:
            raise ValueError(
                f"{TABLE_NAME}: {source_id} marks {endpoint.name} as its "
                "default host but lists it behind another endpoint.  The "
                "ladder is its order, so the default must be its head.")
    return endpoints


def has_ladder(source_id: str) -> bool:
    return bool(ladder(source_id))


def endpoint_named(source_id: str, name: str) -> Endpoint:
    for endpoint in ladder(source_id):
        if endpoint.name == name:
            return endpoint
    offered = ", ".join(entry.name for entry in ladder(source_id))
    raise ValueError(
        f"--transport {name}: --source {source_id} publishes on {offered}")


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def cycle_age_hours(cycle: datetime, now: datetime | None = None) -> float:
    """How old ``cycle`` is, in hours, on a naive-UTC clock."""

    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    if cycle.tzinfo is not None:
        cycle = cycle.astimezone(timezone.utc).replace(tzinfo=None)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc).replace(tzinfo=None)
    return (now - cycle).total_seconds() / 3600.0


def serving_ladder(source_id: str, *, cycle: datetime,
                   now: datetime | None = None,
                   pinned: str | None = None) -> tuple[Endpoint, ...]:
    """The endpoints this cycle will actually be asked for, in order.

    A typed ``--transport`` is a DECISION, not a preference: it yields
    that endpoint alone, so an operator who named a host is never
    quietly served from the other one.  Otherwise the ladder is filtered
    to the endpoints whose retention covers the cycle's age -- and, when
    that filter would empty it, kept whole (see the module note:
    retention is an optimisation, never a bar).
    """

    rungs = ladder(source_id)
    if not rungs:
        return ()
    if pinned is not None:
        return (endpoint_named(source_id, pinned),)
    age = cycle_age_hours(cycle, now)
    covering = tuple(entry for entry in rungs if entry.covers(age))
    return covering or rungs


# --------------------------------------------------------------------------
# Which rung moves the bytes: throughput, probed per object
# --------------------------------------------------------------------------

def transfer_order(rungs: Sequence[Endpoint]) -> tuple[Endpoint, ...]:
    """``rungs`` in THROUGHPUT order: quickest bulk host first.

    A second order over the same endpoints, and the only thing it
    decides is which host is asked to move an object it provably has.
    Ties keep ladder order, and a source that declares no rank has the
    two orders equal -- so this is the identity function for every
    publisher outside the NCEP family.
    """

    indexed = list(enumerate(rungs))
    indexed.sort(key=lambda item: (item[1].transfer_rank, item[0]))
    return tuple(entry for _position, entry in indexed)


def transfer_probes(serving: Sequence[Endpoint]) -> tuple[Endpoint, ...]:
    """The rungs worth asking "do you already have this object?".

    Only the rungs strictly quicker than the one the ladder would use
    anyway, in throughput order.  Empty means there is nothing to gain
    -- a one-rung ladder, a pinned ``--transport``, or a source whose
    ladder head is already its quickest host -- and an empty tuple is
    the instruction to probe NOTHING, which is what keeps an
    archive-era cycle and every non-NCEP route exactly as they were.
    """

    if len(serving) < 2:
        return ()
    head = serving[0]
    return tuple(entry for entry in transfer_order(serving)
                 if entry.transfer_rank < head.transfer_rank)


def promote(serving: Sequence[Endpoint],
            endpoint: Endpoint) -> tuple[Endpoint, ...]:
    """``serving`` with ``endpoint`` moved to the head, nothing dropped.

    The concrete breakage this prevents: choosing the mirror by
    REPLACING the ladder would leave a promoted object with one host
    and no fall-through, so a mirror that answered a HEAD and then
    threw a 503 at the transfer would refuse a fetch the operational
    server could have finished.  Promotion is a reorder.
    """

    rest = tuple(entry for entry in serving if entry is not endpoint)
    if len(rest) == len(serving):
        raise ValueError(
            f"{endpoint.name} is not one of the endpoints this cycle is "
            "served by, so it cannot be promoted ahead of them")
    return (endpoint,) + rest


def transfer_ladder(serving: Sequence[Endpoint], keys: Sequence[str], *,
                    probe) -> tuple[Endpoint, ...]:
    """One object's endpoint order: the quickest rung that HAS it, first.

    ``keys`` is everything the object needs from one host at once --
    usually one, but a route whose hour is a PAIR of published objects
    (HRRR's ``wrfnat`` + ``wrfprs``) must not promote a rung that has
    only half of it.  A rung is promoted when it answers for all of
    them.

    This is the whole selection rule, and it is deliberately one
    sequential function: each caller runs it inside its OWN pool, so
    the probes ride the concurrency and per-host caps that caller
    already has rather than a second scheduler invented here.

    A probe that raises is a probe that answered no.  It is not an
    endpoint failure -- nothing is recorded, nothing is spent, and the
    ladder simply stays in retention order with every rung intact.
    """

    for endpoint in transfer_probes(serving):
        try:
            if all(probe(endpoint.url(key)) for key in keys):
                return promote(serving, endpoint)
        except (KeyboardInterrupt, SystemExit, MemoryError):
            raise
        except BaseException:                 # noqa: BLE001 - see docstring
            continue
    return tuple(serving)


def object_available(url: str, *, opener=None, timeout: float = 60.0) -> bool:
    """Does ``url`` exist right now?  One HEAD, governed where it must be.

    True only for a 2xx.  Every other answer -- a 404 because the
    mirror has not caught up, a 503 because it is throttling, a refused
    connection -- is False, because all of them mean the same thing to
    the caller: this rung has not earned the transfer.  None of them is
    an endpoint FAILURE in the ladder's sense, and none is recorded as
    one; a probe that says no simply leaves the ladder in retention
    order, with every rung still behind it.
    """

    from urllib.request import Request
    from gpuwm.nomads_governor import paced_urlopen

    request = Request(url, method="HEAD",
                      headers={"User-Agent": PROBE_USER_AGENT})
    try:
        with paced_urlopen(
                request, timeout=timeout,
                **({"opener": opener} if opener is not None else {})
        ) as response:
            return 200 <= int(response.status) < 300
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except BaseException:                     # noqa: BLE001 - see docstring
        return False


# --------------------------------------------------------------------------
# Host politeness, from the table
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _policy() -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({
        str(host).lower(): MappingProxyType(dict(raw))
        for host, raw in dict(document().get("host_policy", {})).items()})


@functools.lru_cache(maxsize=1)
def host_caps() -> Mapping[str, int]:
    """``netloc -> in-flight request cap`` for every host that declares one.

    A host absent from the table has no cap and keeps whatever pool the
    caller asked for.
    """

    return MappingProxyType({
        host: int(raw["concurrent"])
        for host, raw in _policy().items()
        if raw.get("concurrent") is not None})


def host_cap_why(host: str) -> str:
    """Why ``host`` is capped where it is -- the table's own sentence."""

    return str(_policy().get(host.lower(), {}).get("why", ""))


def host_worker_cap(host: str, workers: int) -> int:
    """How many of ``workers`` may target ``host`` at once."""

    cap = host_caps().get(host.lower())
    if cap is None:
        return workers
    return min(cap, workers)


# --------------------------------------------------------------------------
# Faults: which ones mean "ask the next endpoint"
# --------------------------------------------------------------------------

def fault_reason(error: BaseException) -> str | None:
    """One line naming why this endpoint did not serve, or ``None``.

    ``None`` means the failure is NOT an endpoint's fault and the next
    one would fail identically -- an interrupt, a programming error, a
    full disk.  Those propagate unchanged; walking a ladder over them
    would only multiply the damage and bury the real refusal.

    A ``Retry-After`` the host sent is carried into the sentence.  The
    node-wide governor has already extended its cooldown by it (see
    :func:`gpuwm.nomads_governor.mark_rate_limited`), so by the time
    this is read the host's own retry discipline is spent for this
    request -- what is left to decide is whether to wait fifteen
    minutes or to ask the archive, and the archive has the same bytes.
    """

    if isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
        return None
    if isinstance(error, HTTPError):
        detail = FALLTHROUGH_STATUSES.get(error.code)
        if detail is None:
            return None
        from gpuwm.nomads_governor import retry_after_seconds
        wait = retry_after_seconds(error)
        asked = (f", and asked for Retry-After {wait:g} s" if wait else "")
        return f"HTTP {error.code} -- {detail}{asked}"
    if isinstance(error, URLError):
        return f"the connection failed -- {error.reason}"
    if isinstance(error, TimeoutError):
        return "the connection timed out"
    if isinstance(error, OSError):
        return f"the connection failed -- {error}"
    if isinstance(error, ValueError):
        # A payload that does not verify: an error page served with 200,
        # a truncated object, a declared length the host did not
        # deliver.  The other endpoint publishes the same key, so this
        # is a reason to ask it rather than to end the fetch.
        return f"the object did not verify -- {error}"
    return None


def ladder_refusal(label: str, attempts: Sequence[tuple[Endpoint, str]]
                   ) -> str:
    """The refusal when every endpoint failed: each one, and why.

    The concrete breakage this prevents: a two-host fetch that failed
    on both used to report only the last host's error, so a reader saw
    "403 from the archive" and never learned the operational server had
    been rate limiting them for fifteen minutes.
    """

    lines = [f"{label}: every endpoint refused.  Tried, in order:"]
    for endpoint, reason in attempts:
        lines.append(f"  {endpoint.name} ({endpoint.host}): {reason}")
        if endpoint.why:
            lines.append(f"    what it is for: {endpoint.why}")
    lines.append(
        "  remedy: name one host with --transport to see its own refusal "
        "in full, or pass a cycle inside an endpoint's retention window.")
    return "\n".join(lines)


__all__ = [
    "Endpoint", "FALLTHROUGH_STATUSES", "PROBE_USER_AGENT", "TABLE_NAME",
    "cycle_age_hours", "document", "endpoint_named", "fault_reason",
    "has_ladder", "host_cap_why", "host_caps", "host_worker_cap", "ladder",
    "ladder_refusal", "object_available", "promote", "serving_ladder",
    "transfer_ladder", "transfer_order", "transfer_probes",
]
