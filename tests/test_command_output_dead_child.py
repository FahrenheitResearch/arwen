"""A launcher whose child died must not wait on it forever.

Origin: a 1 km CONUS preparation child (``gpuwm.mapped_direct``) was
SIGKILLed; its main thread showed as a zombie with exit signal 9 while the
process could not finish exiting, and the ``gpuwm prep`` parent sat in
waitpid for 47 minutes printing its progress heartbeat.  A grandchild that
keeps the output pipes open blocked the pipe drains the same way.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from gpuwm import command_output
from gpuwm.command_output import leader_exit_state, run_streamed, wait_for_child


class _Sink:
    def __init__(self):
        self.parts = []

    def write(self, text):
        self.parts.append(text)
        return len(text)

    def flush(self):
        pass

    @property
    def text(self):
        return "".join(self.parts)


class _NeverEnds:
    """A Popen stand-in whose wait never returns on its own."""

    pid = 4242

    def __init__(self, finish_after=None, code=0):
        self.calls = 0
        self.finish_after = finish_after
        self.code = code

    def wait(self, timeout=None):
        self.calls += 1
        if timeout is None:
            # The reaper thread a dead child is handed to; it must never
            # be the launcher's own wait.
            time.sleep(3600)
        if self.finish_after is not None and self.calls >= self.finish_after:
            return self.code
        raise subprocess.TimeoutExpired("child", timeout)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 10.0
        return self.now


def test_a_child_killed_by_a_signal_that_cannot_finish_exiting_is_reported_dead():
    said = []
    code = wait_for_child(_NeverEnds(), ["python", "-m", "gpuwm.mapped_direct"],
                          report=said.append, grace=60.0, poll=0.0,
                          leader_state=lambda pid: ("Z", 9), clock=_Clock())
    assert code == -9
    assert said == [
        "gpuwm.mapped_direct (pid 4242) was killed by signal 9 (SIGKILL, often "
        "the out-of-memory killer) and has not finished exiting after 60 s; "
        "stopped waiting for it (exit -9).\n"]


def test_a_nonzero_exit_status_that_cannot_finish_exiting_is_reported():
    said = []
    code = wait_for_child(_NeverEnds(), ["/opt/bin/adapter", "--x"],
                          report=said.append, grace=60.0, poll=0.0,
                          leader_state=lambda pid: ("Z", 3 << 8), clock=_Clock())
    assert code == 3
    assert said[0].startswith("adapter (pid 4242) exited with status 3 and has not")


@pytest.mark.parametrize("state", [("S", 0), ("R", 0), ("D", 0), ("Z", 0), None])
def test_a_live_child_or_a_main_thread_that_ended_cleanly_is_waited_for(state):
    # ("Z", 0) is a main thread that ended on purpose while others run on;
    # None is a platform without /proc.  Neither is a dead child.
    said = []
    process = _NeverEnds(finish_after=50, code=7)
    code = wait_for_child(process, ["child"], report=said.append, grace=0.0,
                          poll=0.0, leader_state=lambda pid: state,
                          clock=_Clock())
    assert (code, said, process.calls) == (7, [], 50)


def test_the_grace_restarts_when_the_main_thread_is_seen_alive_again():
    states = iter([("Z", 9), ("S", 0)] * 3 + [("Z", 9)] * 100)
    said = []
    clock = _Clock()
    code = wait_for_child(_NeverEnds(), ["child"], report=said.append,
                          grace=25.0, poll=0.0,
                          leader_state=lambda pid: next(states), clock=clock)
    assert code == -9
    # First zombie sighting after the flapping starts the clock; the next
    # three 10 s ticks pass the 25 s grace.
    assert "after 30 s" in said[0]


def test_proc_stat_is_read_past_a_command_name_holding_parentheses(tmp_path, monkeypatch):
    fields = ["Z"] + ["0"] * 48 + ["9", "0"]
    line = "123 (a) (b c) " + " ".join(fields) + "\n"
    stat = tmp_path / "stat"
    stat.write_text(line)
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) == "/proc/123/stat":
            return real_open(stat, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    assert leader_exit_state(123) == ("Z", 9)


linux = pytest.mark.skipif(not sys.platform.startswith("linux"),
                           reason="needs Linux /proc and raw thread exit")

_SYS_EXIT = {"x86_64": 60, "aarch64": 93}.get(platform.machine())


def _child_whose_main_thread_exits(status, then):
    """A child whose MAIN thread leaves with ``status`` (raw exit, not
    exit_group) while another thread runs ``then``: the main thread is a
    zombie whose exit has not completed, as for a killed process with a
    stuck thread."""
    return textwrap.dedent(f"""
        import ctypes, os, sys, threading, time
        def other():
            {then}
        threading.Thread(target=other).start()
        print("pid", os.getpid(), flush=True)
        time.sleep(0.3)
        ctypes.CDLL(None, use_errno=True).syscall({_SYS_EXIT}, {status})
    """)


@linux
@pytest.mark.skipif(_SYS_EXIT is None, reason="unknown exit syscall number")
def test_a_child_that_cannot_finish_exiting_does_not_hold_the_launcher(monkeypatch):
    monkeypatch.setattr(command_output, "DEAD_CHILD_GRACE_SECONDS", 1.0)
    monkeypatch.setattr(command_output, "WAIT_POLL_SECONDS", 0.1)
    monkeypatch.setattr(command_output, "DRAIN_JOIN_SECONDS", 0.5)
    out, err = _Sink(), _Sink()
    started = time.monotonic()
    result = run_streamed(
        [sys.executable, "-c", _child_whose_main_thread_exits(3, "time.sleep(120)")],
        out, err)
    took = time.monotonic() - started
    pid = int(out.text.split()[1])
    try:
        assert result.returncode == 3
        assert took < 20, took
        assert f"(pid {pid}) exited with status 3 and has not finished exiting" in err.text
        assert "stayed open" in err.text
        assert leader_exit_state(pid)[0] == "Z"
    finally:
        os.kill(pid, signal.SIGKILL)


@linux
@pytest.mark.skipif(_SYS_EXIT is None, reason="unknown exit syscall number")
def test_a_main_thread_that_ends_cleanly_is_not_taken_for_a_dead_child(monkeypatch):
    monkeypatch.setattr(command_output, "DEAD_CHILD_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(command_output, "WAIT_POLL_SECONDS", 0.1)
    out, err = _Sink(), _Sink()
    result = run_streamed(
        [sys.executable, "-c",
         _child_whose_main_thread_exits(0, "time.sleep(2); print('late', flush=True); os._exit(5)")],
        out, err)
    assert result.returncode == 5
    assert "late" in out.text
    assert "has not finished exiting" not in err.text


@linux
def test_a_grandchild_holding_the_pipes_does_not_hold_the_launcher(monkeypatch):
    monkeypatch.setattr(command_output, "DRAIN_JOIN_SECONDS", 1.0)
    out, err = _Sink(), _Sink()
    script = ("import subprocess, sys\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
              "print('grandchild', p.pid, flush=True)\n")
    started = time.monotonic()
    result = run_streamed([sys.executable, "-c", script], out, err)
    took = time.monotonic() - started
    grandchild = int(out.text.split()[1])
    try:
        assert result.returncode == 0
        assert took < 20, took
        assert "stdout and stderr stayed open 1 s longer" in err.text
    finally:
        os.kill(grandchild, signal.SIGKILL)


def test_an_ordinary_child_keeps_its_output_and_code():
    out, err = _Sink(), _Sink()
    result = run_streamed([sys.executable, "-c",
                           "import sys; print('x' * 70000); "
                           "print('e', file=sys.stderr); sys.exit(4)"], out, err)
    assert result.returncode == 4
    assert out.text == "x" * 70000 + "\n"
    assert err.text == "e\n"
