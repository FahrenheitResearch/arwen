"""Saying that something is still happening, on one shared mechanism.

Two measured findings from the 2.5.0 persona walks live here, and they
are the same finding twice.

**N9** -- ``gpuwm setup`` spent 15.5 s of its 16.2 s in unbroken silence
while 315 MiB of Thompson tables came down.  Nothing was wrong; nothing
said so either, and a terminal that has printed nothing for a quarter of
a minute is indistinguishable from a hang.

**N10** -- fetch feedback was inverted.  The 792 MB table route printed
its opening line and then nothing at all until the manifest, while the
420 KB GFS route printed a line per file that *block-buffered* through a
pipe: 9.1 s of a 9.8 s fetch arrived in the log at once, at the end.

So: a byte counter with a throttle, and one call that makes a process's
own stdout flush per line.

The counter is on **stderr** deliberately.  ``gpuwm setup`` captures each
step's stdout so it can print one status line per step and replay the
whole text on a refusal (:func:`gpuwm.setup_cli._run_step`); a counter
written to stdout would be captured with it and reach the reader only
after the download it reports on had finished -- which is the silence
this module exists to end.

It also behaves differently on a terminal and in a log, because the two
readers want different things.  On a terminal the update rewrites one
line every fifth of a second, which is a moving number.  Redirected, a
rewritten line is nonsense, so the update is a new line and the throttle
is long: a handful of lines across a long transfer, not ten thousand.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path

#: Terminal cadence: fast enough to read as motion, slow enough that the
#: formatting is never the cost of the transfer.
TTY_INTERVAL_S = 0.2

#: Redirected cadence.  Every update is a permanent line in somebody's
#: log, so the bar for writing one is much higher.
LOG_INTERVAL_S = 5.0

#: Terminal cadence for the CONSOLIDATED in-flight line, which summarises
#: several files rather than showing one moving number.  Slower than
#: :data:`TTY_INTERVAL_S` on purpose: a summary that changes five times a
#: second reads as noise, and the number a reader wants off it (are we
#: moving, and how fast) is stable over seconds, not frames.
TRANSFER_TTY_INTERVAL_S = 5.0

#: Redirected cadence for the same line.  Three times sparser than the
#: terminal's, because every update here is a permanent line in a log and
#: a parallel fetch that runs for an hour must not write seven hundred of
#: them.
TRANSFER_LOG_INTERVAL_S = 15.0

_MIB = 1024.0 * 1024.0
_KIB = 1024.0
_GIB = 1024.0 * 1024.0 * 1024.0


PREP_EVENT_PREFIX = "GPUWM_PREP_EVENT "
PREP_EVENT_SCHEMA = "gpuwm.prep-stage.v1"

#: Set in a program's environment by a parent that reads step records off
#: its output (``gpuwm go``'s stages, :func:`gpuwm.go_cli._run_stage`).  A
#: preparation host in that program (:mod:`gpuwm.prep_output`) then passes
#: each step line its preparer writes on to its own stderr, where the
#: parent reads it; without it the host keeps the line to its log, since
#: its stderr is a person's terminal.
PREP_EVENT_PARENT_ENV = "GPUWM_PREP_EVENT_PARENT"


@contextlib.contextmanager
def prep_stage(stage: str, *, label: str | None = None,
               backend: str | None = None, count: int | None = None,
               index: int | None = None, grid_id: int | None = None,
               stderr: bool = True):
    """Report an actual preparation operation on stderr, leaving JSON stdout.

    The small event envelope is shared with front-door child-output readers.
    Stage names are open; event tags are started/finished/failed. No device,
    forecast module, or output directory is needed to report preparation.
    The yielded dictionary can carry an outcome/reason on completion, so an
    optional file export can finish without claiming it produced files.

    A step taken grid by grid says ``index`` of ``count`` as its place
    among the grids it builds, and the grid's own id in ``grid_id``.

    ``stderr=False`` reports through :func:`emit_event` alone.  The stderr
    line is for a parent that reads a preparer PROGRAM's output
    (``gpuwm prep`` and ``gpuwm go`` turn it into words); a step taken in
    the process that holds the run, like the experiment builder, is heard
    by that process's own listener instead, and its stderr is the person's
    terminal: ``gpuwm run --no-supervise`` and ``gpuwm run-plan`` printed
    each such step as a raw JSON line.
    """
    import json

    fields = {"schema": PREP_EVENT_SCHEMA, "stage": stage,
              "label": label or stage.replace("_", " ")}
    fields.update({key: value for key, value in {
        "backend": backend, "count": count, "index": index,
        "grid_id": grid_id}.items()
        if value is not None})

    def emit(event, **details):
        payload = {**fields, "event": event, **details}
        if stderr:
            print(PREP_EVENT_PREFIX + json.dumps(
                payload, sort_keys=True,
                allow_nan=False), file=sys.stderr, flush=True)
        # Native run hosts may retain the existing preparation receipt as
        # metadata. The operation and its output are unchanged.
        emit_event("warning", **prep_record_event(payload))

    started = time.perf_counter()
    emit("started")
    completion = {}
    try:
        yield completion
    except BaseException as error:
        emit("failed", elapsed_seconds=time.perf_counter() - started,
             error_type=type(error).__name__, error=str(error))
        raise
    else:
        emit("finished", elapsed_seconds=time.perf_counter() - started,
             **{key: completion[key] for key in ("outcome", "reason")
                if key in completion})


def prep_progress(stage: str, *, label: str, done: int, count: int) -> None:
    """Say how far a counted preparation step has got, on the same envelope.

    :func:`prep_stage` says when a step starts and ends; a step that
    builds many things in turn (the boundary times after the start state)
    was silent in between, so a page watching a run could say only which
    step was open for minutes at a time.  This is the ``progress`` action
    of that envelope: ``index`` things of ``count`` are done.  A reader
    that knows only the started/finished/failed actions passes it over.
    """
    import json

    payload = {"schema": PREP_EVENT_SCHEMA, "stage": stage, "label": label,
               "event": "progress", "index": int(done), "count": int(count)}
    print(PREP_EVENT_PREFIX + json.dumps(payload, sort_keys=True, allow_nan=False),
          file=sys.stderr, flush=True)
    emit_event("warning", **prep_record_event(payload))


def prep_record_event(record: dict) -> dict:
    """The fields of the run event one preparation step record rides on.

    A ``warning`` with code ``preparation_progress``: what a run page reads a
    step from.  Shared by the step reporters above and by every parent that
    reads a preparer program's step lines and says them again on its run's
    stream, so a step heard in the process and a step read off a program's
    output land as the same record.  A failure's own text stays with the
    preparer's output and its log.
    """

    public = {key: value for key, value in record.items() if key != "error"}
    return {"code": "preparation_progress", "phase": "prepare",
            "preparation": public, "message": str(record.get("label", ""))}


def relay_prep_record(record: dict) -> None:
    """Say a step record read off a preparer program's output to this process's listeners.

    A preparer that runs as its own program (``gpuwm prep``'s adapter, the
    HRRR nest preparation) reports each step through :func:`emit_event` in
    ITS process, where nobody listens; its parent reads the step's
    ``GPUWM_PREP_EVENT`` line and says it here, so the listener that puts it
    on the run's stream (:func:`gpuwm.runplan._preparation_relay`) hears it.
    Without this a run page on the staged route showed no step and no
    boundary-time count for the whole preparation.
    """

    emit_event("warning", **prep_record_event(record))


def line(text: str, *, stream=None) -> None:
    """One status line, flushed.

    ``print`` alone is not enough through a pipe: CPython block-buffers
    a non-tty stdout, so status lines a command emits over ten seconds
    arrive in the reader's log together, at exit (UX finding N10).
    """

    print(text, file=sys.stdout if stream is None else stream, flush=True)


def format_elapsed(seconds: float) -> str:
    """Elapsed model or wall time, with hours allowed to exceed 24."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


#: ``writing:`` phases whose work is not a write, in a reader's words.  The
#: read-back of a new checkpoint is timed as a write, sized by the file it
#: reads (:meth:`gpuwm.supervisor.RuntimeHeartbeat.__call__`), and the
#: progress line said "writing verify checkpoint" while the run was reading
#: back what it had already written.
_WRITE_PHASE_WORDS = {"verify-checkpoint": "checking checkpoint"}


def write_phase_words(phase: str) -> str:
    """What one ``writing:<phase>`` heartbeat is doing, as a progress line says it."""

    words = _WRITE_PHASE_WORDS.get(phase)
    if words is not None:
        return words
    return "writing " + phase.replace("-", " ").replace("_", " ")


class ForecastProgress:
    """Show validated supervisor heartbeats without replaying worker logs.

    Stage changes appear immediately. An unchanged stage still reports its
    elapsed wall time, so a long preparation does not look like a dead CLI.
    A terminal gets at most one periodic line every five seconds; redirected
    logs get one every thirty. Output failures never interrupt CUDA work.
    """

    def __init__(self, *, stream=None):
        self.stream = sys.stderr if stream is None else stream
        try:
            terminal = self.stream.isatty()
        except (AttributeError, OSError, ValueError):
            terminal = False
        self.interval = 5.0 if terminal else 30.0
        self.started = time.monotonic()
        self.last_print = float("-inf")
        self.phase = None
        self.enabled = True

    def write(self, message: str) -> None:
        if self.enabled:
            try:
                line(message, stream=self.stream)
            except (OSError, ValueError):
                self.enabled = False

    def __call__(self, heartbeat) -> None:
        now = time.monotonic()
        phase = (heartbeat.started_at_utc, heartbeat.status)
        if phase == self.phase and now - self.last_print < self.interval:
            return
        self.phase, self.last_print = phase, now
        if heartbeat.status == "integrating":
            message = (f"Forecast: {format_elapsed(heartbeat.model_elapsed_seconds)} "
                       f"simulated; step {heartbeat.outer_step}")
        elif heartbeat.status == "complete":
            # A terminal heartbeat alone cannot establish a successful exit.
            message = "Finishing forecast"
        elif heartbeat.status.startswith("finalizing:"):
            stage = heartbeat.status.removeprefix("finalizing:")
            message = "Finishing: " + stage.replace("-", " ").replace("_", " ")
        elif heartbeat.status.startswith("writing:"):
            stage = heartbeat.status.removeprefix("writing:")
            message = (f"Forecast: {format_elapsed(heartbeat.model_elapsed_seconds)} "
                       "simulated; " + write_phase_words(stage))
        elif heartbeat.status == "failed":
            message = "Forecast worker reported a failure; reading diagnostics"
        elif heartbeat.status.startswith("waiting:"):
            wait = heartbeat.wait or {}
            on = heartbeat.status.removeprefix("waiting:")
            message = (f"Forecast: {format_elapsed(heartbeat.model_elapsed_seconds)} "
                       f"simulated; waiting on the {on}")
            if wait.get("lead") is not None:
                message += f" (lead {wait['lead']}, late at {wait.get('late_at')})"
        else:
            stage = heartbeat.status.removeprefix("preparing:")
            label = {
                "launch": "starting worker", "worker-start": "starting worker",
                "prepare-case": "loading and preparing inputs",
                "build-domain-tree": "initializing domains",
                "resolve-schedule": "resolving forecast times",
                "resolve-terrain-clock": "reading the terrain for the time step",
                "cold-start-wrfout": "writing initial output",
                "restore-checkpoint": "restoring checkpoint",
            }.get(stage, stage.replace("-", " ").replace("_", " "))
            message = f"Preparing: {label}"
        self.write(f"{message} | elapsed {format_elapsed(now - self.started)}")


def line_buffer_stdout() -> None:
    """Make THIS process flush its own stdout on every newline.

    Called once at each front door, so every ``print`` in the product --
    including the thirty-odd ``progress=print`` defaults in the fetch
    family -- streams when the reader has redirected the command into a
    file or a pipe.  Never raises: a stream that cannot be reconfigured
    (a replaced ``sys.stdout``, a ``StringIO`` under a test harness, a
    detached stream on a Windows service) keeps whatever buffering it
    has, and the command runs exactly as before.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (ValueError, OSError):                # pragma: no cover
            continue


def _mib(count: int) -> str:
    return f"{count / _MIB:.1f} MiB"


class ByteCounter:
    """A throttled byte counter for one transfer, or one set of them.

    ``total`` is optional and is the *pinned* size where one exists --
    the Thompson assets carry an exact byte count in the contract every
    run enforces, so the percentage is true from the first chunk without
    trusting a ``Content-Length`` header.  Without a total the counter
    says what has moved, which is still the difference between "working"
    and "hung".

    Thread-safe: the table routes move several objects at once through
    :mod:`gpuwm.fetch_pool`, and one counter over the whole request
    reads better than six interleaved ones.
    """

    def __init__(self, label: str, total: int | None = None, *,
                 stream=None, interval: float | None = None,
                 enabled: bool = True) -> None:
        self.label = label
        self.total = total if total and total > 0 else None
        self.moved = 0
        self._stream = stream
        self._enabled = enabled
        self._interval = interval
        self._lock = threading.Lock()
        self._last = 0.0
        self._said = False

    # -- the stream, resolved late so redirect_stderr and capsys work --
    def _out(self):
        return sys.stderr if self._stream is None else self._stream

    def _tty(self) -> bool:
        try:
            return bool(self._out().isatty())
        except (AttributeError, ValueError):         # pragma: no cover
            return False

    def _every(self) -> float:
        if self._interval is not None:
            return self._interval
        return TTY_INTERVAL_S if self._tty() else LOG_INTERVAL_S

    def _render(self) -> str:
        if self.total is None:
            return f"{self.label}: {_mib(self.moved)} moved"
        percent = 100.0 * self.moved / self.total
        return (f"{self.label}: {_mib(self.moved)} / {_mib(self.total)} "
                f"({percent:.0f}%)")

    def _emit(self, text: str, *, final: bool) -> None:
        stream = self._out()
        try:
            if self._tty() and not final:
                stream.write("\r" + text)
            elif self._tty():
                stream.write("\r" + text + "\n")
            else:
                stream.write(text + "\n")
            stream.flush()
        except (ValueError, OSError):                # pragma: no cover
            self._enabled = False

    def advance(self, count: int) -> None:
        """Add ``count`` bytes; print when the throttle allows."""

        if not self._enabled or count <= 0:
            return
        with self._lock:
            self.moved += count
            now = time.monotonic()
            # The FIRST chunk always speaks.  A throttle that waits for
            # its own interval before the first line reproduces the
            # finding in miniature: nothing at all for the first five
            # seconds of every transfer.
            if self._said and now - self._last < self._every():
                return
            self._last = now
            self._said = True
            text = self._render()
        self._emit(text, final=False)

    def close(self, note: str = "done") -> None:
        """One last line with the final count, and the line ended."""

        if not self._enabled:
            return
        with self._lock:
            if not self._said and self.moved == 0:
                return
            text = f"{self._render()} {note}".rstrip()
        self._emit(text, final=True)


# ---------------------------------------------------------------------------
# Per-file visibility while several transfers are in flight
# ---------------------------------------------------------------------------

#: Every event tag :class:`TransferMonitor` will ever emit.  A consumer
#: switching on the tag can be exhaustive against this tuple.
TRANSFER_EVENTS = ("fetch_started", "fetch_progress", "fetch_completed")

#: Sinks the ambient :func:`event_sink` context has installed.  A LIST
#: and not a single slot: a run-plan run and a test harness may both want
#: the stream, and the second one must not unhook the first.
_event_sinks: list = []
_event_sink_lock = threading.Lock()


def emit_event(event: str, **fields) -> None:
    """Offer one transfer event to every installed ambient sink.

    Telemetry never fails a fetch: a sink that raises is dropped for that
    one record and the transfer carries on.  The alternative -- a
    disconnected Studio taking a download with it -- is not a trade
    anyone would make.
    """

    with _event_sink_lock:
        sinks = tuple(_event_sinks)
    for sink in sinks:
        try:
            sink(event, **fields)
        except Exception:            # noqa: BLE001 - see the docstring
            pass


def event_sinks_installed() -> bool:
    """Whether any run is listening, for emitters with a cost to measure."""

    with _event_sink_lock:
        return bool(_event_sinks)


@contextlib.contextmanager
def event_sink(sink):
    """Route transfer events to ``sink`` for the duration of the block.

    The fetch family runs several layers below whoever owns a run's event
    stream, and its call sites already carry signatures with a dozen
    keyword arguments.  Threading a stream handle through all of them
    would touch every route to deliver one thing that is genuinely
    ambient: "somebody is watching this process".  So the sink is
    installed around the fetch instead, by the one caller that has the
    stream -- :mod:`gpuwm.runplan` runs ``gpuwm fetch``'s own handler
    IN-PROCESS, so an ambient registration reaches it exactly.
    """

    with _event_sink_lock:
        _event_sinks.append(sink)
    try:
        yield sink
    finally:
        with _event_sink_lock:
            try:
                _event_sinks.remove(sink)
            except ValueError:                   # pragma: no cover
                pass


def _size(count: int) -> str:
    """A byte count at a unit a person can hold in their head."""

    count = int(count)
    if count >= _GIB:
        return f"{count / _GIB:.1f} GiB"
    if count >= _MIB:
        return f"{count / _MIB:.1f} MiB"
    if count >= _KIB:
        return f"{count / _KIB:.1f} KiB"
    return f"{count} B"


def format_transfer_done_line(*, label: str, index: int, total: int,
                              name: str, note: str) -> str:
    """One finished file, in the grammar the fetch routes already print.

    PINNED TEXT.  This is the line a reader has been parsing since the
    serial loop, and the start and progress lines added around it are
    additions to the stream, not a replacement for it -- so this
    formatter exists to be tested rather than to be improved.
    """

    return f"{label}: [{int(index) + 1}/{int(total)}] {name} {note}".rstrip()


class _Transfer:
    """One file's live state, as the monitor knows it."""

    __slots__ = ("name", "token", "host", "expected", "path", "seen",
                 "final", "started", "done", "failed", "cancelled")

    def __init__(self, name, token, host, expected, path, *,
                 started: bool = True):
        self.name = name
        self.token = token
        self.host = host
        self.expected = expected
        self.path = None if path is None else Path(path)
        self.seen = 0
        self.final: int | None = None
        # False only for a file the request declared up front (see
        # TransferMonitor.begin) that no worker has picked up yet.
        self.started = started
        self.done = False
        self.failed = False
        self.cancelled = False

    def moved(self) -> int:
        """Bytes this file has moved, from the best source available.

        Three sources, in order of trust: the final count the transfer
        reported, the running count the transport handed over, and the
        size of the file growing on disk under its own name or the one
        decoration that is conventional.

        A transport that stages under a name of its OWN choosing is not
        reachable from here -- there is nothing about this file to match
        it on.  That case is answered a level up, by what the
        destination directory gained: see
        :meth:`TransferMonitor._directory_gain`.
        """

        if self.final is not None:
            return self.final
        on_disk = 0
        if self.path is not None:
            for candidate in (self.path.with_name(self.path.name + ".part"),
                              self.path.with_name(self.path.name + ".download"),
                              self.path):
                try:
                    on_disk = max(on_disk, candidate.stat().st_size)
                except OSError:
                    continue
        return max(self.seen, on_disk)


class TransferMonitor:
    """Says which files are moving, while they are moving.

    THE REGRESSION THIS EXISTS FOR.  When every fetch went parallel
    through :mod:`gpuwm.fetch_pool`, the per-file feedback of the serial
    loop went with it: six files moved at once and each said exactly one
    thing, at completion.  A user driving the Studio front end watched
    roughly three minutes of silence and then a burst of finished lines,
    and reported it -- a slow link and a hung command had become the
    same picture again, which is the finding :class:`ByteCounter` was
    written for and this class is the concurrent case of.

    Two surfaces, from one set of facts:

    * **stderr**, for a person.  One START line per file as its transfer
      begins, then a single consolidated line at a steady cadence that
      rewrites itself on a terminal and is appended sparsely to a log.
      Completion lines are NOT this class's -- the routes still print
      their own, unchanged, through :func:`format_transfer_done_line`.
    * **the run event stream**, for Studio, through the ambient
      :func:`event_sink`: ``fetch_started``, ``fetch_progress`` and
      ``fetch_completed``, one per file, flattened the way the run-plan
      stream's own events are.

    Every line is written with ONE ``write`` call, so a line from a
    worker thread can never appear inside a line from another.
    """

    def __init__(self, label: str, *, stream=None, events=None,
                 interval: float | None = None, enabled: bool = True,
                 ticker: bool = True, clock=time.monotonic) -> None:
        self.label = label
        self._stream = stream
        self._events = events
        self._interval = interval
        self._enabled = enabled
        self._clock = clock
        self._lock = threading.Lock()
        self._files: dict[str, _Transfer] = {}
        self._order: list[str] = []
        # True once the request has said up front which files it will
        # move (see `begin`); only then is there a whole-request count
        # to put on the event stream.
        self._declared = False
        self._baselines: dict[Path, dict[str, int]] = {}
        self._first_start: float | None = None
        self._last_said = 0.0
        self._said = False
        self._pending_newline = False
        self._want_ticker = ticker
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the stream, resolved late so redirect_stderr and capsys work --
    def _out(self):
        return sys.stderr if self._stream is None else self._stream

    def _tty(self) -> bool:
        try:
            return bool(self._out().isatty())
        except (AttributeError, ValueError):     # pragma: no cover
            return False

    def interval(self) -> float:
        """Seconds between consolidated lines, for this stream."""

        if self._interval is not None:
            return self._interval
        return (TRANSFER_TTY_INTERVAL_S if self._tty()
                else TRANSFER_LOG_INTERVAL_S)

    # -- output -------------------------------------------------------

    def _write(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            self._out().write(text)
            self._out().flush()
        except (ValueError, OSError):            # pragma: no cover
            self._enabled = False

    def _say_line(self, text: str) -> None:
        """One whole line, after closing any rewritten line in progress."""

        prefix = "\n" if self._pending_newline else ""
        self._pending_newline = False
        self._write(prefix + text + "\n")

    def _emit(self, event: str, **fields) -> None:
        fields = {"label": self.label, **fields}
        if self._events is not None:
            try:
                self._events(event, **fields)
            except Exception:        # noqa: BLE001 - telemetry never fails
                pass
        emit_event(event, **fields)

    # -- the three moments --------------------------------------------

    def start(self, name: str, *, token: str | None = None,
              host: str | None = None, expected_bytes: int | None = None,
              path=None, on_disk: bool = False) -> None:
        """One file's transfer has begun.  Says so immediately.

        ``on_disk`` is a file already at its destination that is checked
        before anything moves, and its line says that instead of naming
        a host it is not asking.
        """

        record = _Transfer(name, token, host,
                           int(expected_bytes) if expected_bytes else None,
                           path)
        with self._lock:
            declared = self._files.get(name)
            if declared is None:
                self._order.append(name)
            elif record.expected is None:
                record.expected = declared.expected
            self._files[name] = record
            self._arm_directory(record.path)
            if self._first_start is None:
                self._first_start = self._clock()
            text = self._start_line(record, on_disk=on_disk)
        self._say_line(text)
        # Idempotent, and normally a no-op: the pool calls `begin` before
        # any worker exists.  Kept here so a caller driving the monitor
        # without the pool still gets a moving line.
        self._ensure_ticker()
        self._emit("fetch_started", file=name, token=token, host=host,
                   expected_bytes=record.expected)

    def declare(self, name: str, expected_bytes: int | None) -> None:
        """The size the host declared, learned after the line was printed.

        A HEAD ahead of every transfer would double the request count on
        exactly the services whose per-request latency the pool exists to
        hide, so the expected size is taken from the transfer's own
        ``Content-Length`` when the route has no cheaper source.  The
        start line has already gone out by then; the consolidated line
        picks the total up on its next tick, which is the line the
        number actually matters on.
        """

        if not expected_bytes:
            return
        with self._lock:
            record = self._files.get(name)
            if record is not None and record.expected is None:
                record.expected = int(expected_bytes)

    def declare_for_path(self, path, expected_bytes: int | None) -> None:
        """:meth:`declare`, addressed by destination rather than by name.

        The transport knows where it is writing and not what the route
        called the file, and the two spellings differ (a relpath with
        forward slashes against a platform path).  Matching on the path
        the job already carries avoids inventing a third.
        """

        if not expected_bytes:
            return
        target = Path(path)
        with self._lock:
            for record in self._files.values():
                if record.path is not None and record.path == target:
                    if record.expected is None:
                        record.expected = int(expected_bytes)
                    return

    def observe(self, name: str, count: int) -> None:
        """``count`` more bytes have landed for ``name``.

        The running total for a file whose transport reports its chunks.
        A transport that reports nothing is not a problem: see
        :meth:`_Transfer.moved`.
        """

        if count <= 0:
            return
        with self._lock:
            record = self._files.get(name)
            if record is not None:
                record.seen += int(count)

    def relay(self, name: str):
        """A ``(received, total)`` sink for one transfer attempt of ``name``.

        For a transport that reports a RUNNING count for its object rather
        than chunks -- the Rust backbone, which holds the object in memory
        until it is whole, so nothing grows on disk to be counted.  Each
        report becomes :meth:`observe` of what is new since the last one
        and :meth:`declare` of the size.  One relay per attempt: a
        backbone asked again counts from zero, and the bytes it moves the
        second time are moved all the same.
        """

        last = 0

        def relay(received: int, total: int | None = None) -> None:
            nonlocal last
            if total:
                self.declare(name, total)
            if received > last:
                self.observe(name, received - last)
                last = received

        return relay

    def finish(self, name: str, *, size: int | None = None,
               seconds: float | None = None, host: str | None = None,
               failed: bool = False, cancelled: bool = False) -> None:
        """One file's transfer has ended, well or badly.

        A FAILED file is finished too, and says so, but it is not DONE.
        A monitor that only heard about successes would leave the
        consolidated line counting a file that stopped moving minutes
        ago as in flight; one that counted a failure as done told a
        reader the request was closer to finished than it was.

        ``cancelled`` is a file the pool stopped because ANOTHER file
        failed the request: it did not complete, and it is not the
        failure either.
        """

        with self._lock:
            record = self._files.get(name)
            if record is None:
                record = _Transfer(name, None, host, None, None)
                self._files[name] = record
                self._order.append(name)
            record.started = True
            record.done = True
            record.failed = bool(failed) and not cancelled
            record.cancelled = bool(cancelled)
            if size is not None:
                record.final = int(size)
            elif record.final is None:
                record.final = record.moved()
            if host is not None:
                record.host = host
            moved = record.final
            token = record.token
        extra = {"cancelled": True} if cancelled else {}
        self._emit("fetch_completed", file=name, token=token,
                   host=record.host, bytes=moved,
                   seconds=(None if seconds is None
                            else round(float(seconds), 6)),
                   failed=bool(failed or cancelled), **extra)

    def stopped(self, name: str, *, in_flight: int, queued: int) -> None:
        """Say that ``name`` failed the request and the rest was stopped.

        Without it the log goes from a run of healthy progress lines
        straight to a refusal, and the files that were cut short say
        nothing at all.
        """

        if not in_flight and not queued:
            return
        parts = []
        if in_flight:
            parts.append(f"stopping the {in_flight} "
                         f"file{'' if in_flight == 1 else 's'} in flight")
        if queued:
            parts.append(f"not starting the {queued} "
                         f"file{'' if queued == 1 else 's'} still queued")
        self._say_line(f"{self.label}: {name} failed, so the request "
                       f"cannot complete; {' and '.join(parts)}")

    # -- the consolidated line ----------------------------------------

    def _start_line(self, record: _Transfer, *,
                    on_disk: bool = False) -> str:
        head = f"{self.label}: "
        if record.token:
            head += f"{record.token}: "
        if on_disk:
            return f"{head}{record.name} is already on disk; checking it here"
        parts = []
        if record.host:
            parts.append(str(record.host))
        if record.expected:
            parts.append(f"{_size(record.expected)} expected")
        tail = f" ({', '.join(parts)})" if parts else ""
        return f"{head}{record.name} starting{tail}"

    # -- what the destination gained, for transports that say nothing --

    def _arm_directory(self, path) -> None:
        """Record what a destination already held, before anything moved.

        Called under the lock, from :meth:`start`.  Taken ONCE per
        directory and never refreshed: a baseline that moved with the
        transfer would subtract the very bytes it is meant to count.
        """

        if path is None:
            return
        directory = Path(path).parent
        if directory in self._baselines:
            return
        sizes: dict[str, int] = {}
        try:
            for entry in directory.iterdir():
                try:
                    sizes[entry.name] = entry.stat().st_size
                except OSError:
                    continue
        except OSError:
            pass
        self._baselines[directory] = sizes

    def _directory_gain(self, directory: Path) -> int:
        """Bytes this destination holds that it did not hold at arming.

        WHAT THIS IS FOR, and it was measured rather than assumed.  The
        Rust fetch backbone stages each object under a name of its own
        choosing -- a bare UUID, in the system temp directory -- and
        moves it into place only when the object is whole.  Watched
        against the real ``rw_fetch.exe``, the destination directory
        stayed EMPTY for the whole of a 26 s transfer and then gained a
        146 MB file at once, so a per-file ``stat`` on ``<final>`` or
        ``<final>.part`` has nothing to find and the consolidated line
        read ``0 B`` until the last file landed.

        The gain is per NAME, so a destination that already held a
        reused object does not report it as freshly moved, and a file
        that grows in place is counted for its growth only.

        ITS LIMIT: for a backbone that stages OUTSIDE the destination,
        this still reads zero while a single object is in flight; the
        in-flight count of such a transfer comes from the backbone's own
        progress output instead (see :meth:`relay`).  What this does
        recover is every file that has actually landed -- including one
        the backbone named differently from the name this route asked
        for, which no per-file stat could match.
        """

        baseline = self._baselines.get(directory)
        if baseline is None:
            return 0
        gained = 0
        try:
            entries = list(directory.iterdir())
        except OSError:
            return 0
        for entry in entries:
            try:
                size = entry.stat().st_size
            except OSError:
                continue
            before = baseline.get(entry.name, 0)
            if size > before:
                gained += size - before
        return gained

    def _snapshot(self):
        """(done, failed, total, moved, expected-or-None, in-flight records).

        ``total`` is every file the request declared up front (see
        :meth:`begin`), not only the ones that have started: counting
        started files made the denominator grow as the fetch ran (0 of
        6, 6 of 12, 16 of 22, 35 of 38), so a reader could not tell how
        much was left.  ``done`` counts files that COMPLETED; a failed
        file is counted as failed, and a file the pool stopped because
        another one failed is neither.
        """

        records = [self._files[name] for name in self._order]
        # Grouped by destination, and the two accounts are combined with
        # `max` rather than summed: the directory's gain and the files'
        # own counts describe the SAME bytes from two sides, so adding
        # them would report a transfer at twice its size.  Whichever
        # sees more is the better floor.
        grouped: dict[Path, list[_Transfer]] = {}
        moved = 0
        for record in records:
            if record.path is None:
                moved += record.moved()
            else:
                grouped.setdefault(record.path.parent, []).append(record)
        for directory, group in grouped.items():
            moved += max(sum(record.moved() for record in group),
                         self._directory_gain(directory))
        done = sum(1 for record in records if record.done
                   and not record.failed and not record.cancelled)
        failed = sum(1 for record in records if record.failed)
        expected = None
        if records and all(record.expected or record.done
                           for record in records):
            expected = sum(record.expected if record.expected
                           else (record.final or 0) for record in records)
        flight = [record for record in records
                  if record.started and not record.done]
        return done, failed, len(records), moved, expected, flight

    def _progress_line(self, done, total, moved, expected, elapsed,
                       failed=0) -> str:
        volume = (_size(moved) if expected is None
                  else f"{_size(moved)} of {_size(expected)}")
        rate = (f"{moved / elapsed / _MIB:.1f} MiB/s aggregate"
                if elapsed > 0.0 else "starting")
        failures = f", {failed} failed" if failed else ""
        return (f"{self.label}: {done} of {total} files done{failures}, "
                f"{volume}, {rate}")

    @staticmethod
    def _acquisition(done, total, moved) -> dict:
        """The whole-request count, for a front end's files line.

        The block the ERA5 route already publishes, so the consumers
        that fold it (the TUI's and the remote worker's pipeline
        progress, which the desktop draws as "Files: done / total")
        show the real total without learning a new shape.
        """

        return {"schema": "arwen.acquisition-progress.v1",
                "files_total": int(total), "files_completed": int(done),
                "transferred_bytes": int(moved)}

    def tick(self, *, force: bool = False) -> None:
        """Sample every transfer and say where the request is, if it is time."""

        if not self._enabled:
            return
        now = self._clock()
        with self._lock:
            if not self._files:
                return
            if (self._said and not force
                    and now - self._last_said < self.interval()):
                return
            self._last_said = now
            self._said = True
            done, failed, total, moved, expected, flight = self._snapshot()
            elapsed = (now - self._first_start
                       if self._first_start is not None else 0.0)
            text = self._progress_line(done, total, moved, expected, elapsed,
                                       failed=failed)
            moving = [(record.name, record.moved(), record.expected)
                      for record in flight]
            whole = (self._acquisition(done, total, moved)
                     if self._declared else None)
        if self._tty():
            self._write("\r" + text)
            self._pending_newline = True
        else:
            self._write(text + "\n")
        for name, bytes_moved, record_expected in moving:
            self._emit("fetch_progress", file=name, bytes=bytes_moved,
                       expected_bytes=record_expected)
        if whole is not None:
            self._emit("fetch_progress", acquisition=whole)

    # -- the thread that makes the line appear without a caller -------

    def begin(self, files=()) -> None:
        """Learn the whole request, then start the ticker, before any transfer.

        ``files`` is ``(name, expected_bytes-or-None)`` for every file the
        request will move, in order.  Declared HERE, before any worker
        exists, so the very first consolidated line says out of how many
        (``0 of 38 files done`` rather than ``0 of 6``) and neither a
        queued nor a failed file can make the denominator move.  Sizes
        known up front count toward the expected total; while any file's
        size is unknown the line states the bytes moved without one.

        WHY THE TICKER IS NOT STARTED LAZILY BY THE FIRST ``start``.
        Creating a thread costs real time, and paying it inside whichever
        worker happened to call first makes that worker late relative to
        its siblings -- which reorders the transfers themselves.  A route
        test that pinned the order its objects were asked for caught
        exactly that.  Started once, up front, no worker pays it.
        """

        with self._lock:
            for name, expected in files:
                if name in self._files:
                    continue
                self._order.append(name)
                self._files[name] = _Transfer(
                    name, None, None, int(expected) if expected else None,
                    None, started=False)
                self._declared = True
        self._ensure_ticker()

    def _ensure_ticker(self) -> None:
        if not self._want_ticker:
            return
        with self._lock:
            if self._thread is not None:
                return
            # Assigned INSIDE the lock, before the thread is started: two
            # workers reaching an unguarded `is None` check together
            # would each start a ticker, and the second one would never
            # be joined by `close`.
            self._thread = threading.Thread(
                target=self._run_ticker, name="gpuwm-fetch-progress",
                daemon=True)
            self._thread.start()

    def _run_ticker(self) -> None:
        # A SIDE THREAD, and it has to be one.  The pool's jobs are
        # opaque callables -- the whole point of that design is that the
        # pool adds no bars to a route's transport -- so there is no
        # per-chunk seam here to hang a cadence off.  The alternative is
        # exactly the silence being fixed.
        period = max(0.05, min(self.interval(), 1.0))
        while not self._stop.wait(period):
            try:
                self.tick()
            except Exception:        # noqa: BLE001 - telemetry never fails
                return

    def close(self) -> None:
        """Stop the ticker and end any line left open on a terminal."""

        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if self._pending_newline:
            self._pending_newline = False
            self._write("\n")
        if self._declared:
            # The last word on the whole request, so a front end that
            # folds the latest count does not keep the one from the final
            # tick, up to a cadence before the last file landed.
            with self._lock:
                done, _failed, total, moved, _expected, _flight = (
                    self._snapshot())
                self._declared = False
            self._emit("fetch_progress",
                       acquisition=self._acquisition(done, total, moved))

    def __enter__(self) -> "TransferMonitor":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


__all__ = ["ByteCounter", "ForecastProgress", "format_elapsed", "LOG_INTERVAL_S", "TRANSFER_EVENTS",
           "TRANSFER_LOG_INTERVAL_S", "TRANSFER_TTY_INTERVAL_S",
           "TTY_INTERVAL_S", "TransferMonitor", "emit_event", "event_sink",
           "format_transfer_done_line", "line", "line_buffer_stdout", "prep_progress", "prep_record_event",
           "prep_stage", "relay_prep_record"]
