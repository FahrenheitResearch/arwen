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

import sys
import threading
import time

#: Terminal cadence: fast enough to read as motion, slow enough that the
#: formatting is never the cost of the transfer.
TTY_INTERVAL_S = 0.2

#: Redirected cadence.  Every update is a permanent line in somebody's
#: log, so the bar for writing one is much higher.
LOG_INTERVAL_S = 5.0

_MIB = 1024.0 * 1024.0


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


__all__ = ["ByteCounter", "LOG_INTERVAL_S", "TTY_INTERVAL_S",
           "line", "line_buffer_stdout"]
