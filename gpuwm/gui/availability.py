"""Which data sources hold a start, answered at once and filled in as the hosts answer.

New forecast's When step lists every source beside the calendar.  Each
row's first answer comes from the publication schedule and the archive
bounds, with no server asked, so the list draws the moment it is asked
for.  A start from the last day, and the newest start a source offers,
are then put to the fetch's own object probe
(:func:`gpuwm.source_availability.page_start_check` and
:func:`~gpuwm.source_availability.page_latest`) on a small pool of
background threads, one source per job, each with short per-request
timeouts and a deadline.  Until a row's check comes back the row says
``checking``.

One rule decides what a source offers, for each source and length:

- A start is **confirmed** when a check found it whole, when it is
  earlier than a start found whole (a source publishes its starts in
  order), or when no server is asked about it (a keyed archive, or a
  start more than a day old: the schedule is the whole answer there).
  A row never infers from its own start.
- A start is **due** when it is past the source's usual publication
  delay (:attr:`gpuwm.source_cycles.CycleGrid.usual_delay`, measured).
- **Start** takes a start that is confirmed, or due and not found
  missing by a check (``starts`` ``"now"``).
- **Queue it** also takes a start from the last day that is not
  confirmed (``starts`` ``"queue"``): the forecast is held, saying so,
  until Start would take the start, and then starts in its turn.
- The page **opens on** the newest start Start takes
  (:func:`~gpuwm.source_availability.opening_start`), and calls it the
  **newest run** only once a check confirmed it whole and found every
  start after it (:func:`~gpuwm.source_availability.starts_after`) not yet
  whole.  While a newer start went unheard, nothing is the newest run.

The newest start is asked about only when the chosen start is near it:
for an older start the row shows what an earlier check found, or the
schedule's estimate, and no server is asked at all.  Every check shares
one record of what each object was answered, so the start check and the
newest-start check of the same cycle send each HEAD once.

Answers are kept with the time they were checked, so reopening the page
shows them with their age and asks again only when they are old.  No
page read here waits for a probe, a download, a forecast or another
request; only Start and Queue it wait, briefly, for the check of the one
start they write (:meth:`Board.settle`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any, Callable

#: How long a checked answer is shown without asking again.  A start the
#: host holds stays held; one it does not yet hold is asked again sooner,
#: since the host is publishing it.
FRESH_YES_S = 600.0
FRESH_NO_S = 90.0
#: A check that could not hear a host is tried again after this long.
FRESH_UNCHECKED_S = 60.0
#: Background checks at once, across every source.
WORKERS = 12
#: How long Start and Queue it wait for the check of the one start they write: a source's whole check
#: (``PAGE_SOURCE_DEADLINE_S``) and a little over for a check that waited for a free worker.
SETTLE_S = 20.0


def _utc(now: datetime | None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment


def _words(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y-%m-%dT%H").strftime("%Y-%m-%d %H:%M UTC")


def _hours(value: float | None) -> str:
    return "" if value is None else f"{float(value):g} hour{'' if float(value) == 1 else 's'}"


def _refusal_words(why: str) -> str:
    """The fetch's refusal of a start and length, in the page's words.

    The check asks about the download the run makes, and a start whose
    file ladder the download refuses is answered no from the route table
    before a check starts (:func:`gpuwm.source_availability.availability`),
    so this is the fetch's own sentence for anything else it refuses.
    """

    return f"The download refuses this start and length: {why}"


def _near_now(cycle: str, moment: datetime) -> bool:
    """Whether a start is recent enough (or late enough) that the newest start a host holds bears on it."""

    from gpuwm.source_availability import PUBLICATION_FRONTIER_HOURS, parse_cycle

    try:
        return moment - parse_cycle(cycle) < timedelta(hours=PUBLICATION_FRONTIER_HOURS)
    except ValueError:
        return False


class Heard:
    """What each object was answered, shared by every check on a Board, and the HEADs in flight.

    When the chosen start is the newest one (where the page opens), the
    start check and the newest-start check ask the same objects; each
    asking for itself sent a GEM cycle's final lead twice.  An object
    answered present counts as present for :data:`FRESH_YES_S`, one
    answered missing as missing for :data:`FRESH_NO_S` (the times a row's
    own answer is kept, so asking a row again asks its hosts again), and
    an object a host did not answer is never kept.  A URL another check
    is already asking is waited for rather than sent a second time, and
    the check that waited is handed only an answer still fresh: when the
    HEAD it waited on went unheard, the answer kept from before is the one
    that HEAD was sent to replace, and the waiter hears nothing too.
    """

    #: Past this many kept answers, the ones too old to use are dropped.
    KEEP = 20_000

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._answers: dict[str, tuple[float, bool]] = {}
        self._flight: dict[str, threading.Event] = {}
        self._closed = False

    @staticmethod
    def _fresh(found: bool) -> float:
        return FRESH_YES_S if found else FRESH_NO_S

    def _usable(self, hit: tuple[float, bool] | None) -> bool:
        return hit is not None and self._clock() - hit[0] < self._fresh(hit[1])

    def head(self, url: str) -> bool | None:
        from gpuwm import source_availability

        with self._lock:
            if self._closed:
                return None
            hit = self._answers.get(url)
            if self._usable(hit):
                return hit[1]
            event = self._flight.get(url)
            mine = event is None
            if mine:
                event = self._flight[url] = threading.Event()
        if not mine:
            event.wait(head_bound_s())
            with self._lock:
                hit = self._answers.get(url)
                # Handed the kept answer, a GEM newest start the host did not answer was read from a 200 s old
                # "missing", and the row named a start 12 h older as checked for 10 minutes.
                return hit[1] if self._usable(hit) else None
        found = None
        try:
            # Looked up at call time, so a test's stand-in for quick_head is the one asked.
            found = source_availability.quick_head(url)
        finally:
            with self._lock:
                if found is not None:
                    self._answers[url] = (self._clock(), found)
                    if len(self._answers) > self.KEEP:
                        now = self._clock()
                        self._answers = {key: value for key, value in self._answers.items()
                                         if now - value[0] < self._fresh(value[1])}
                self._flight.pop(url, None)
            event.set()
        return found

    def close(self, wait_s: float) -> None:
        """Send no HEAD from here on, and wait at most ``wait_s`` for the ones already sent to come back."""

        with self._lock:
            self._closed = True
            flying = list(self._flight.values())
        deadline = time.monotonic() + wait_s
        for event in flying:
            event.wait(max(0.0, deadline - time.monotonic()))


def head_bound_s() -> float:
    """One page HEAD's longest wait: its turn under the node's governor, then its timeout."""

    from gpuwm import source_availability

    return source_availability.PAGE_PACE_BUDGET_S + source_availability.PAGE_PROBE_TIMEOUT_S + 1.0


class Board:
    """The checked answers, the checks in flight, and the pool that runs them."""

    def __init__(self, *, workers: int = WORKERS, clock: Callable[[], float] = time.monotonic,
                 session_factory: Callable[[], Any] | None = None) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arwen-gui-probe")
        self._lock = threading.Lock()
        self._done: dict[tuple, tuple[float, dict[str, Any]]] = {}
        self._flight: dict[tuple, Any] = {}
        self._clock = clock
        self._session_factory = session_factory
        # The newest start each (source, last hour) was found whole at, from any check.  Starts publish in order,
        # so it stays good; a later check that heard fewer hosts never moves it back.
        self._known: dict[tuple[str, int], str] = {}
        self._closed = False
        self.heard = Heard(clock)

    # ------------------------------------------------------------ jobs

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from gpuwm.source_availability import ProbeSession

        return ProbeSession(head=self.heard.head)

    def _fresh_for(self, answer: dict[str, Any]) -> float:
        if "unchecked" in (answer.get("basis"), answer.get("state")):
            return FRESH_UNCHECKED_S
        if answer.get("state") == "not-published" or answer.get("missing"):
            return FRESH_NO_S
        return FRESH_YES_S

    def _lookup(self, key: tuple, job: Callable[[], dict[str, Any]], *,
                start: bool = True) -> tuple[dict[str, Any] | None, float | None, bool]:
        """(answer, age in seconds, still checking) for one key; starts the job when there is no fresh answer.

        With ``start`` false nothing is started: the answer already held, however old, or None.
        """

        now = self._clock()
        with self._lock:
            hit = self._done.get(key)
            age = None if hit is None else now - hit[0]
            stale = hit is None or age >= self._fresh_for(hit[1])
            flying = key in self._flight
            if stale and not flying and start and not self._closed:
                self._flight[key] = self._pool.submit(self._run, key, job)
                flying = True
        return (None if hit is None else hit[1]), (None if age is None else round(age)), flying

    def _run(self, key: tuple, job: Callable[[], dict[str, Any]]) -> None:
        try:
            answer = job()
        except Exception as error:  # noqa: BLE001 - a failed check falls back to the schedule and says so
            answer = {"state": "unchecked", "basis": "unchecked", "why": str(error)}
        with self._lock:
            self._done[key] = (self._clock(), answer)
            self._flight.pop(key, None)

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until no check is in flight (for tests and the start-up warm call)."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                # A check close() cancelled before it began is not in flight.
                flying = [future for future in self._flight.values() if not future.done()]
            if not flying:
                return True
            for future in flying:
                try:
                    future.result(timeout=max(0.0, deadline - time.monotonic()))
                except Exception:  # noqa: BLE001
                    pass
        return False

    def settle(self, source: str, cycle: str, hours: float, *, now: datetime | None = None,
               timeout: float = SETTLE_S) -> bool:
        """Wait, at most ``timeout`` seconds, for the check of one start from the last day.

        Only Start and Queue it call this, for the one start they are about
        to write: a start picked while its check was still out passed the
        page and was refused only when its download began.  Nothing is
        started here that :meth:`row` has not started; a start that needs no
        check returns at once.  True when no check of it is still out.
        """

        from gpuwm.source_availability import availability, cached_stop, start_needs_probe

        moment = _utc(now)
        try:
            document = availability(source, float(hours), now=moment, published=cached_stop)
        except (ValueError, KeyError):
            return True
        if not start_needs_probe(document, cycle, now=moment):
            return True
        key = ("start", document["source_id"], cycle, document["last_hour"])
        with self._lock:
            future = self._flight.get(key)
        if future is None:
            return True
        try:
            future.result(timeout=timeout)
        except Exception:  # noqa: BLE001 - still out after the wait: the row says what is known, and why
            return False
        return True

    # ------------------------------------------------------------ what a page reads

    def _learn(self, source_id: str, last_hour: int, cycle: str | None) -> None:
        if not cycle:
            return
        key = (source_id, int(last_hour))
        with self._lock:
            if key not in self._known or cycle > self._known[key]:
                self._known[key] = cycle

    def _found_missing(self, source_id: str, last_hour: int) -> set[str]:
        """Starts whose own check, still fresh, found them not published."""

        now = self._clock()
        with self._lock:
            return {key[2] for key, (stamp, answer) in self._done.items()
                    if key[0] == "start" and key[1] == source_id and key[3] == last_hour
                    and answer.get("state") == "not-published" and now - stamp < FRESH_NO_S}

    def newest(self, source: str, hours: float, *, now: datetime | None = None, ask: bool = True) -> dict[str, Any]:
        """What one source offers for one length, by the rule in this module's docstring.

        ``confirmed``: the newest start a check found whole (or, for a
        source or start no server is asked about, the schedule's own);
        ``missing``: newer starts a check found not yet whole; ``due``: the
        newest start past the usual publication delay; ``opening``: the
        newest start Start takes, which a page opens on; ``newest_run``:
        the opening start once a check confirmed it and found every start
        after it not yet whole, else None.  ``checking`` while the source's
        newest-start check is out; ``newest_basis`` is ``"checked"`` when
        what is known names the newest run or every server answered,
        ``"schedule"`` when no server is asked, ``"checking"`` while the
        check is out, ``"unchecked"`` when a newer start went unheard
        (asked again after :data:`FRESH_UNCHECKED_S`), and ``"unasked"``
        when ``ask`` is false (a start more than a day old) and no earlier
        check is kept.
        """

        from gpuwm.source_availability import availability, cached_stop

        moment = _utc(now)
        try:
            document = availability(source, float(hours), now=moment, published=cached_stop)
        except (ValueError, KeyError):
            return {"confirmed": None, "missing": [], "due": None, "opening": None, "newest_run": None,
                    "newest_basis": "schedule", "checking": False, "checked_age_s": None,
                    "usual_delay_hours": None}
        return self._newest(document, moment, ask=ask)

    def _newest(self, document: dict[str, Any], moment: datetime, *, ask: bool = True) -> dict[str, Any]:
        """:meth:`newest` for one availability document.  With ``ask`` false no check is started: the answer is
        what an earlier check found, and ``newest_basis`` is ``"unasked"`` when there is none."""

        from gpuwm.source_availability import latest_needs_probe, opening_start, page_latest, starts_after

        base = {"due": document.get("due_start"), "usual_delay_hours": document.get("usual_delay_hours")}
        candidate = document.get("latest_candidate")
        if not latest_needs_probe(document, now=moment):
            # No server is asked: the schedule is the whole answer, so its newest start is confirmed.
            opening = opening_start(document, candidate)
            return {**base, "confirmed": candidate, "missing": [], "opening": opening,
                    "newest_run": opening if opening and opening == candidate else None,
                    "newest_basis": "schedule", "checking": False, "checked_age_s": None}
        source_id, last_hour = document["source_id"], int(document["last_hour"])
        answer, age, checking = self._lookup(
            ("latest", source_id, last_hour, candidate),
            lambda: page_latest(document, now=moment, session=self._session()), start=ask)
        if answer is not None and answer.get("downloads"):
            self._learn(source_id, last_hour, answer.get("latest"))
        missing = set((answer or {}).get("missing") or ()) | self._found_missing(source_id, last_hour)
        with self._lock:
            confirmed = self._known.get((source_id, last_hour))
        if confirmed:
            missing = {start for start in missing if start > confirmed}
        opening = opening_start(document, confirmed, missing)
        # The start found whole is the newest run only once every start after it was found not yet whole: a start
        # found while a newer one went unheard may sit behind published starts.
        known = bool(confirmed) and all(start in missing for start in starts_after(document, confirmed))
        # A start more than a day old asks no server about today's runs: with no earlier answer kept, nothing is
        # being checked, and the row says so rather than "checking".
        basis = ("checked" if known or (answer or {}).get("basis") == "checked"
                 else "checking" if checking else "unasked" if answer is None and not ask
                 else "checking" if answer is None else "unchecked")
        return {**base, "confirmed": confirmed, "missing": sorted(missing), "opening": opening,
                "newest_run": opening if known and opening == confirmed else None,
                "newest_basis": basis, "checking": checking, "checked_age_s": age}

    def row(self, source: str, name: str, cycle: str, hours: float, *,
            now: datetime | None = None) -> dict[str, Any]:
        """One source's row for one start and length, never waiting on a server.

        ``state`` is what the row reads ("yes" has it, "unknown" may have
        it, "no"), ``starts`` what takes the start ("now": Start does;
        "queue": only Queue it does, and holds the forecast until Start
        would; "no": neither), and the source's newest-start answer
        (:meth:`newest`) rides along for the calendar and the quick picks.
        """

        from gpuwm.source_availability import (availability, bounds_unread, cached_stop, page_start_check,
                                               published_stop, start_is_due, start_needs_probe, verdict)
        from gpuwm.source_adapters import get_source_adapter

        moment = _utc(now)
        try:
            document = availability(source, float(hours), now=moment, published=cached_stop)
        except (ValueError, KeyError) as error:
            return {"id": source, "name": name, "state": "unknown", "why": str(error), "reason": str(error),
                    "starts": "now", "earliest": None, "cycle_hours": [], "analysis": False,
                    "checking": False, "basis": "schedule", "checked_age_s": None,
                    "confirmed": None, "missing": [], "due": None, "opening": None, "newest_run": None,
                    "newest_basis": "schedule", "newest_checking": False, "usual_delay_hours": None}
        checking = False
        # A provider's own bounds document is read in the background; until then the schedule bounds the row.
        try:
            windows = [w for w in get_source_adapter(source).archive_windows if bounds_unread(w)]
        except Exception:  # noqa: BLE001
            windows = []
        for window in windows:
            _, _, flying = self._lookup(("bounds", window.bounds_url),
                                        lambda window=window: {"stop": str(published_stop(window, timeout=5.0)),
                                                               "basis": "checked"})
            checking = checking or flying
        # Today's newest start is put to the probe only for a start near it.  For an older start the row carries
        # what an earlier probe found and asks no server: a 1996 date asked every source about today's runs (91 s),
        # and a 2021 event's fit asked NOMADS about today's HRRR (22 s).
        newest = self._newest(document, moment, ask=_near_now(cycle, moment))
        answer = verdict(document, cycle, now=moment)
        probe = start_needs_probe(document, cycle, now=moment)
        if answer["state"] == "no" and probe and not document.get("analysis"):
            # The schedule's boundary says "not published yet" of every start inside its measured lag; where the
            # start's objects can be asked, the check decides instead, so a start published early is taken and one
            # still being made can be queued for.
            inside = verdict({**document, "latest_candidate": None}, cycle, now=moment)
            if inside["state"] == "yes":
                answer = inside
        basis, age, note, inferred = "schedule", None, None, None
        starts = "no" if answer["state"] == "no" else "now"
        display = document.get("display_name") or source
        last = int(document["last_hour"])
        if answer["state"] == "yes" and probe:
            key = ("start", document["source_id"], cycle, last)
            checked, age, flying = self._lookup(
                key, lambda: {**page_start_check(document, cycle, now=moment, session=self._session()),
                              "basis": "probe"})
            checking = checking or flying
            state = None if checked is None else checked.get("state")
            confirmed = newest["confirmed"]
            if state == "published":
                basis = "checked"
                self._learn(document["source_id"], last, cycle)
            elif state == "refused":
                # The fetch's own resolver refuses this start and length, so the run's download would refuse it
                # too: this row's answer is no, in the fetch's words, and the other rows are untouched.
                basis, starts = "checked", "no"
                reason = _refusal_words(str(checked.get("why") or ""))
                answer = {"state": "no", "reason": reason, "why": reason}
            elif (state == "not-published" or cycle in newest["missing"]) and confirmed and cycle < confirmed:
                # A later start is whole and a source publishes its starts in order, so the publisher skipped this
                # one: waiting will not bring it, and a run queued for it waited for data that never came.
                basis, starts = "checked", "no"
                reason = (f"Not published through hour {last}, and the later {_words(confirmed)} start is whole, "
                          "so waiting will not bring it.")
                answer = {"state": "no", "reason": reason,
                          "why": (f"{display} did not publish the {_words(cycle)} start through hour {last}, and its "
                                  f"later {_words(confirmed)} start is whole, so waiting will not bring it."),
                          "fix": f"Pick {_words(confirmed)}, the newer start that is whole, or another source."}
            elif state == "not-published" or cycle in newest["missing"]:
                basis = "checked"
                starts = "queue"
                reason = f"Not published through hour {last} yet. Queue it waits for it and starts it once it is."
                answer = {"state": "unknown", "reason": reason,
                          "why": f"{display} has not published the {_words(cycle)} start through hour {last} yet."}
            else:
                # Its own check is out (None) or heard no host ("unchecked"): what is known of other starts decides.
                basis = "checking" if checked is None else "unchecked"
                if confirmed and cycle <= confirmed:
                    basis = "checked"
                    if cycle < confirmed:
                        inferred = confirmed
                        note = f"{_words(confirmed)} was found whole, so this earlier start is published too."
                elif start_is_due(document, cycle):
                    if basis == "unchecked":
                        note = ("The data server did not answer in time. This start is older than this source "
                                "usually takes to publish a whole run (about "
                                f"{_hours(document.get('usual_delay_hours'))}), so it can start.")
                else:
                    starts = "queue"
                    reason = ("Not confirmed yet: the data server did not answer in time. Queue it waits for it "
                              "and starts it once a check confirms it." if basis == "unchecked" else
                              "Checking whether it is published. Queue it can wait for it.")
                    answer = {"state": "unknown", "reason": reason,
                              "why": (f"Whether {display} has published the {_words(cycle)} start is not confirmed "
                                      "yet: " + ("its data server did not answer in time." if basis == "unchecked"
                                                 else "its check has not answered yet."))}
        row = {"id": source, "name": name, **answer, "starts": starts, "earliest": document["earliest"],
               "cycle_hours": document["cycle_hours"], "analysis": document["analysis"],
               "checking": checking, "basis": basis, "checked_age_s": age,
               "confirmed": newest["confirmed"], "missing": newest["missing"], "due": newest["due"],
               "opening": newest["opening"], "newest_run": newest["newest_run"],
               "newest_basis": newest["newest_basis"], "newest_checking": newest["checking"],
               "usual_delay_hours": newest["usual_delay_hours"]}
        if note:
            row["note"] = note
        if inferred:
            row["inferred_from"] = inferred
        return row

    def close(self, wait_s: float | None = None) -> None:
        """Start no check and send no HEAD from here on, and wait (at most one HEAD's time) for the running ones.

        A page server's checks outlived it: 126 HEADs went out after server_close, sent through whatever HEAD was
        in place by then, a later test's stand-in or the real hosts.  A check already running ends within one
        HEAD, since every HEAD it asks after this is answered "not heard" without being sent.
        """

        deadline = time.monotonic() + (head_bound_s() if wait_s is None else float(wait_s))
        with self._lock:
            self._closed = True
        # The checks not yet begun never begin; the running ones hear nothing more from here on.
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.heard.close(max(0.0, deadline - time.monotonic()))
        with self._lock:
            for key in [key for key, future in self._flight.items() if future.cancelled()]:
                del self._flight[key]
            running = list(self._flight.values())
        if running:
            futures_wait(running, timeout=max(0.0, deadline - time.monotonic()))


__all__ = ["Board", "FRESH_NO_S", "FRESH_UNCHECKED_S", "FRESH_YES_S", "Heard", "SETTLE_S", "head_bound_s"]
