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


def line(text: str, *, stream=None) -> None:
    """One status line, flushed.

    ``print`` alone is not enough through a pipe: CPython block-buffers
    a non-tty stdout, so status lines a command emits over ten seconds
    arrive in the reader's log together, at exit (UX finding N10).
    """

    print(text, file=sys.stdout if stream is None else stream, flush=True)


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
                 "final", "done", "failed")

    def __init__(self, name, token, host, expected, path):
        self.name = name
        self.token = token
        self.host = host
        self.expected = expected
        self.path = None if path is None else Path(path)
        self.seen = 0
        self.final: int | None = None
        self.done = False
        self.failed = False

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
              path=None) -> None:
        """One file's transfer has begun.  Says so immediately."""

        record = _Transfer(name, token, host,
                           int(expected_bytes) if expected_bytes else None,
                           path)
        with self._lock:
            if name not in self._files:
                self._order.append(name)
            self._files[name] = record
            self._arm_directory(record.path)
            if self._first_start is None:
                self._first_start = self._clock()
            text = self._start_line(record)
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

    def finish(self, name: str, *, size: int | None = None,
               seconds: float | None = None, host: str | None = None,
               failed: bool = False) -> None:
        """One file's transfer has ended, well or badly.

        A FAILED file is finished too, and says so.  A monitor that only
        heard about successes would leave the consolidated line counting
        a file that stopped moving minutes ago as in flight, which is the
        same lie about progress in a smaller box.
        """

        with self._lock:
            record = self._files.get(name)
            if record is None:
                record = _Transfer(name, None, host, None, None)
                self._files[name] = record
                self._order.append(name)
            record.done = True
            record.failed = bool(failed)
            if size is not None:
                record.final = int(size)
            elif record.final is None:
                record.final = record.moved()
            if host is not None:
                record.host = host
            moved = record.final
            token = record.token
        self._emit("fetch_completed", file=name, token=token,
                   host=record.host, bytes=moved,
                   seconds=(None if seconds is None
                            else round(float(seconds), 6)),
                   failed=bool(failed))

    # -- the consolidated line ----------------------------------------

    def _start_line(self, record: _Transfer) -> str:
        head = f"{self.label}: "
        if record.token:
            head += f"{record.token}: "
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

        HONEST ABOUT ITS LIMIT: for a backbone that stages OUTSIDE the
        destination, this still reads zero while a single object is in
        flight.  What it does recover is every file that has actually
        landed -- including one the backbone named differently from the
        name this route asked for, which no per-file stat could match --
        so a window of several files stops reporting nothing until the
        end.
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
        """(done, total, moved, expected-or-None, in-flight records)."""

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
        done = sum(1 for record in records if record.done)
        expected = None
        if records and all(record.expected or record.done
                           for record in records):
            expected = sum(record.expected if record.expected
                           else (record.final or 0) for record in records)
        flight = [record for record in records if not record.done]
        return done, len(records), moved, expected, flight

    def _progress_line(self, done, total, moved, expected, elapsed) -> str:
        volume = (_size(moved) if expected is None
                  else f"{_size(moved)} of {_size(expected)}")
        rate = (f"{moved / elapsed / _MIB:.1f} MiB/s aggregate"
                if elapsed > 0.0 else "starting")
        return (f"{self.label}: {done} of {total} files done, "
                f"{volume}, {rate}")

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
            done, total, moved, expected, flight = self._snapshot()
            elapsed = (now - self._first_start
                       if self._first_start is not None else 0.0)
            text = self._progress_line(done, total, moved, expected, elapsed)
            moving = [(record.name, record.moved(), record.expected)
                      for record in flight]
        if self._tty():
            self._write("\r" + text)
            self._pending_newline = True
        else:
            self._write(text + "\n")
        for name, bytes_moved, record_expected in moving:
            self._emit("fetch_progress", file=name, bytes=bytes_moved,
                       expected_bytes=record_expected)

    # -- the thread that makes the line appear without a caller -------

    def begin(self) -> None:
        """Start the ticker, ON THE CALLER'S THREAD, before any transfer.

        WHY IT IS NOT STARTED LAZILY BY THE FIRST ``start``.  Creating a
        thread costs real time, and paying it inside whichever worker
        happened to call first makes that worker late relative to its
        siblings -- which reorders the transfers themselves.  A route
        test that pinned the order its objects were asked for caught
        exactly that.  Started once, up front, no worker pays it.
        """

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

    def __enter__(self) -> "TransferMonitor":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


__all__ = ["ByteCounter", "LOG_INTERVAL_S", "TRANSFER_EVENTS",
           "TRANSFER_LOG_INTERVAL_S", "TRANSFER_TTY_INTERVAL_S",
           "TTY_INTERVAL_S", "TransferMonitor", "emit_event", "event_sink",
           "format_transfer_done_line", "line", "line_buffer_stdout"]
