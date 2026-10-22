"""Bounded live subprocess output for CLI hosts, without shell execution."""

from __future__ import annotations

import codecs
import contextlib
import contextvars
import io
import os
import signal
import subprocess
import threading
import time


OUTPUT_CHUNK_SIZE = 16384

#: How often the launcher looks at a child it is waiting for.
WAIT_POLL_SECONDS = 1.0

#: How long a child whose main thread has already died may take to finish
#: exiting before the launcher stops waiting for it.  On Linux a killed
#: process is reaped only once EVERY one of its threads has left the
#: kernel; one thread stuck in uninterruptible sleep (memory reclaim of a
#: ~170 GB process, a device flush) keeps waitpid from returning while the
#: main thread already shows as a zombie.  A 1 km CONUS preparation child
#: was SIGKILLed that way and its `gpuwm prep` parent sat in waitpid for
#: 47 minutes, printing "Initialize remaining forcing states (N s total)"
#: the whole time.  Ordinary teardown of even a very large process takes
#: seconds, so a minute of a dead main thread is a dead child.
DEAD_CHILD_GRACE_SECONDS = 60.0

#: How long the launcher waits, once the child has ended, for both pipe
#: drains to reach end of file.  A surviving grandchild, or a child thread
#: that has not finished exiting, can hold the write ends open forever; the
#: drains then stay behind as daemon threads and the launcher returns.
DRAIN_JOIN_SECONDS = 30.0


def leader_exit_state(pid):
    """``(state, exit_code)`` of a Linux process's main thread, or ``None``.

    Fields 3 and 52 of ``/proc/<pid>/stat``.  ``exit_code`` is the raw
    wait status the kernel recorded when that thread exited (0 while it
    runs).  ``None`` where there is no ``/proc`` (Windows, macOS) or the
    process is already gone.
    """
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as handle:
            text = handle.read().decode("ascii", "replace")
    except (OSError, ValueError):
        return None
    # The command name may hold spaces or parentheses; it ends at the
    # last ")".  What follows starts at field 3.
    fields = text[text.rfind(")") + 2:].split()
    if not fields:
        return None
    try:
        exit_code = int(fields[49])
    except (IndexError, ValueError):
        exit_code = 0  # kernel older than 3.5: no field 52
    return fields[0], exit_code


def _program_name(command):
    words = [str(word) for word in command]
    if "-m" in words[:-1]:
        return words[words.index("-m") + 1]
    return os.path.basename(words[0]) if words else "child"


#: Linux signal names by number.  The raw wait status comes from Linux
#: ``/proc``, so its numbers are Linux numbers; the host's own ``signal``
#: table cannot name them on Windows (no SIGKILL there), which made the
#: same status read "signal 9" on one platform and "SIGKILL" on another.
_LINUX_SIGNAL_NAMES = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL", 6: "SIGABRT",
    7: "SIGBUS", 8: "SIGFPE", 9: "SIGKILL", 10: "SIGUSR1", 11: "SIGSEGV",
    12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
}


def _death_words(exit_code):
    """Popen-convention return code and a sentence for a raw wait status."""
    number = exit_code & 0x7F
    if number:
        name = _LINUX_SIGNAL_NAMES.get(number)
        if name is None:
            try:
                name = signal.Signals(number).name
            except ValueError:
                name = f"signal {number}"
        hint = (", often the out-of-memory killer" if number == 9 else "")
        return -number, f"was killed by signal {number} ({name}{hint})"
    status = (exit_code >> 8) & 0xFF
    return status, f"exited with status {status}"


def wait_for_child(process, command, *, report=None,
                   grace=None, poll=None, leader_state=None,
                   clock=time.monotonic):
    """Wait for ``process``; return its code, or the code of a dead child.

    Same result as ``process.wait()`` for a child that exits.  For a
    child whose main thread is a zombie with a fatal signal or a non-zero
    status (``/proc/<pid>/stat``) while its exit has not completed for
    ``grace`` seconds, return that code (-N for signal N, as Popen does)
    and say so through ``report``.  A zombie main thread with status 0 is
    not taken as dead: a program may end its main thread and run on in
    others.  The unreaped child is handed to a daemon thread that reaps
    it whenever the kernel lets go of it.
    """
    grace = DEAD_CHILD_GRACE_SECONDS if grace is None else grace
    poll = WAIT_POLL_SECONDS if poll is None else poll
    leader_state = leader_exit_state if leader_state is None else leader_state
    dead_since = None
    while True:
        try:
            return process.wait(timeout=poll)
        except subprocess.TimeoutExpired:
            pass
        state = leader_state(process.pid)
        if state is None or state[0] not in ("Z", "X") or not state[1]:
            dead_since = None
            continue
        now = clock()
        if dead_since is None:
            dead_since = now
            continue
        if now - dead_since < grace:
            continue
        code, words = _death_words(state[1])
        threading.Thread(target=process.wait, name="reap-dead-child",
                         daemon=True).start()
        if report is not None:
            report(f"{_program_name(command)} (pid {process.pid}) {words} "
                   f"and has not finished exiting after {now - dead_since:.0f} s; "
                   f"stopped waiting for it (exit {code}).\n")
        return code


def text_chunks(text):
    """Bound temporary encoding/tail buffers even for in-process large writes."""
    for offset in range(0, len(text), OUTPUT_CHUNK_SIZE):
        yield text[offset:offset + OUTPUT_CHUNK_SIZE]


class AdapterOutputError(RuntimeError):
    """The child ran, but its output could not be retained or delivered."""


class DiagnosticLog:
    """Raise the first log failure once; keep the error-report path usable.

    Both adapter streams share this log. After a write/flush failure, further
    log writes are disabled so reporting that failure cannot recurse through
    the same broken destination. The owning CLI still reports failure.
    """
    def __init__(self, stream):
        self.stream = stream
        self.failure = None

    @property
    def closed(self):
        return self.stream.closed

    def _failed(self, error):
        if self.failure is None:
            self.failure = error
            raise AdapterOutputError(str(error)) from error

    def write(self, text):
        if self.failure is None:
            try:
                return self.stream.write(text)
            except (OSError, ValueError) as error:
                self._failed(error)
        return len(text)

    def flush(self):
        if self.failure is None and not self.closed:
            try:
                self.stream.flush()
            except (OSError, ValueError) as error:
                self._failed(error)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            self.stream.close()
        except (OSError, ValueError) as closing_error:
            # A cleanup failure must not replace an earlier launch/IO error.
            if error_type is None:
                self._failed(closing_error)
        return False


#: Where a host that owns a diagnostic log parks the two streams the
#: children it launches must be delivered to.
#:
#: It lives HERE, beside the draining implementation, and not in the
#: front door that opens it, because a front door is also a program:
#: ``python -m gpuwm.source_cli`` executes that file once as
#: ``__main__`` and a second time under its package name when a library
#: imports it, and the two copies then hold two registries.  The copy
#: that launches the preparation read its own empty one, fell back to an
#: inherited-handle launch, and the log the run had already advertised
#: on screen ("Details: ...") stayed zero bytes for the whole run, with
#: the child's refusal on the terminal instead of in the file the
#: message named.  A module nothing runs as a program has one copy, so
#: both copies of any door share this.
ADAPTER_OUTPUT = contextvars.ContextVar("adapter_output", default=None)


@contextlib.contextmanager
def redirect_adapter_output(stdout, stderr):
    """Deliver the output of launched children to these two streams."""
    token = ADAPTER_OUTPUT.set((stdout, stderr))
    try:
        yield
    finally:
        ADAPTER_OUTPUT.reset(token)


def run_adapter_command(command, *, env=None, cwd=None):
    """Launch a child, streamed to the host's log when one is open."""
    streams = ADAPTER_OUTPUT.get()
    if streams is None:
        return subprocess.run(command, check=False, env=env, cwd=cwd)
    return run_streamed(command, *streams, env=env, cwd=cwd)


def run_streamed(command, stdout, stderr, *, env=None, cwd=None):
    """Drain both pipes concurrently; deliver flushed chunks before child exit.

    Receipts can be large or contain no newlines. Fixed-size reads and an
    incremental decoder keep memory bounded and preserve split UTF-8 characters.
    A failing destination is remembered while both pipes continue draining.

    Neither waiting step can block forever: a child that died but cannot
    finish exiting is reported dead (:func:`wait_for_child`), and drains
    still open :data:`DRAIN_JOIN_SECONDS` after the child ended (a
    grandchild or an exiting thread holds the pipes) are left behind.
    """
    failures = []
    environment = os.environ.copy() if env is None else dict(env)
    environment["PYTHONIOENCODING"] = "utf-8"
    note_lock = threading.Lock()

    def note(text):
        # Said on the child's error stream, so a host that summarizes the
        # child's last stderr lines (gpuwm prep) shows it as the cause.
        with note_lock:
            try:
                stderr.write(text)
                stderr.flush()
            except Exception as error:
                failures.append(error)

    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment,
                               cwd=cwd)

    def drain(pipe, destination):
        decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")("replace"), translate=True)
        failed = False
        try:
            while True:
                block = pipe.read1(OUTPUT_CHUNK_SIZE)
                text = decoder.decode(block, final=not block)
                if text and not failed:
                    try:
                        destination.write(text)
                        destination.flush()
                    except Exception as error:
                        failures.append(error)
                        failed = True
                if not block:
                    break
        except Exception as error:
            failures.append(error)
        finally:
            pipe.close()

    workers = [threading.Thread(target=drain, args=pair, daemon=True)
               for pair in ((process.stdout, stdout), (process.stderr, stderr))]
    for worker in workers:
        worker.start()
    notes = []
    try:
        code = wait_for_child(process, command, report=notes.append)
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            wait_for_child(process, command, report=notes.append)
        raise
    finally:
        deadline = time.monotonic() + DRAIN_JOIN_SECONDS
        for worker in workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        held = [name for worker, name in zip(workers, ("stdout", "stderr"))
                if worker.is_alive()]
        if held:
            notes.append(
                f"{_program_name(command)} (pid {process.pid}) ended, but its "
                f"{' and '.join(held)} stayed open {DRAIN_JOIN_SECONDS:.0f} s "
                f"longer (a process it started, or one of its threads, still "
                f"holds {'them' if len(held) > 1 else 'it'}); output after "
                f"this point is not kept.\n")
        for text in notes:
            note(text)
    if failures:
        raise AdapterOutputError(str(failures[0])) from failures[0]
    return subprocess.CompletedProcess(command, code)
