"""The pool's task dispatch, against fake workers -- no GPU, no model.

``MemberPool`` is the only piece of the concurrent path that is pure
parent-side scheduling: which worker gets which trajectory, and how the
answers are gathered.  The GPU half (``run_member_leg``) is the same
function the serial path calls, so the scheduling is what a change here
can actually break -- a dropped trajectory, a result attributed to the
wrong member, or a deadlock when there are more trajectories than
workers.

The fake worker speaks the real protocol on stdin/stdout and reports
which worker served each task, so uneven work and widths that do not
divide the trajectory count are exercised for real rather than argued
about.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools.da_member_leg import MemberPool


FAKE_WORKER = textwrap.dedent('''
    import json, os, sys, time
    sys.stdout.write(json.dumps({"ready": True}) + "\\n")
    sys.stdout.flush()
    me = os.environ.get("FAKE_WORKER_ID", "?")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        task = json.loads(line)
        if task.get("stop"):
            break
        # Uneven service times, so a width that does not divide the
        # trajectory count really does leave workers idle at different
        # moments.
        time.sleep(float(task.get("sleep", 0.0)))
        sys.stdout.write(json.dumps({
            "ok": True, "name": task["name"], "entry": {"served_by": me},
        }) + "\\n")
        sys.stdout.flush()
''')


class _FakePool(MemberPool):
    """A MemberPool whose workers are the script above."""

    def __init__(self, width, tmp_path):
        self.width = int(width)
        self.workdir = Path(tmp_path)
        self.workdir.mkdir(parents=True, exist_ok=True)
        script = self.workdir / "fake_worker.py"
        script.write_text(FAKE_WORKER, encoding="utf-8")
        self.procs = []
        for index in range(self.width):
            env = dict(os.environ, FAKE_WORKER_ID=str(index),
                       PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.Popen(
                [sys.executable, str(script)], env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
            assert json.loads(proc.stdout.readline())["ready"]
            self.procs.append(proc)


def _tasks(count, sleeps=None):
    return [{"name": str(i), "sleep": (sleeps or {}).get(i, 0.0)}
            for i in range(count)]


@pytest.mark.parametrize("width,count", [(1, 1), (2, 5), (3, 5), (4, 11),
                                         (5, 3), (2, 11)])
def test_every_trajectory_comes_back_exactly_once(width, count, tmp_path):
    """No trajectory may be dropped, duplicated or misattributed.

    Widths that do not divide the trajectory count are the interesting
    ones: 5 over 3 workers and 11 over 2 both leave a partial final
    batch.
    """
    pool = _FakePool(width, tmp_path)
    try:
        results = pool.run_leg(_tasks(count))
    finally:
        pool.close()
    assert sorted(results, key=int) == [str(i) for i in range(count)]
    for name, payload in results.items():
        assert payload["name"] == name, "a result was misattributed"


def test_a_slow_trajectory_does_not_lose_the_others(tmp_path):
    """One long member must not strand the rest of the leg."""
    pool = _FakePool(3, tmp_path)
    try:
        results = pool.run_leg(
            _tasks(7, sleeps={0: 0.35, 4: 0.25}))
    finally:
        pool.close()
    assert len(results) == 7
    assert set(results) == {str(i) for i in range(7)}


def test_workers_are_reused_across_legs(tmp_path):
    """The pool is persistent: a second leg reuses the same processes.

    Re-spawning per leg would pay a CUDA context and a kernel compile
    again every 900 simulated seconds, which is most of what the
    concurrency was worth.
    """
    pool = _FakePool(2, tmp_path)
    try:
        pids = [proc.pid for proc in pool.procs]
        first = pool.run_leg(_tasks(4))
        second = pool.run_leg(_tasks(4))
        assert [proc.pid for proc in pool.procs] == pids
        assert set(first) == set(second) == {"0", "1", "2", "3"}
        for payload in list(first.values()) + list(second.values()):
            assert payload["entry"]["served_by"] in {"0", "1"}
    finally:
        pool.close()


def test_more_workers_than_trajectories_is_not_a_deadlock(tmp_path):
    pool = _FakePool(6, tmp_path)
    try:
        results = pool.run_leg(_tasks(2))
    finally:
        pool.close()
    assert set(results) == {"0", "1"}


def test_a_dead_worker_is_named_not_an_anonymous_broken_pipe(tmp_path):
    """A worker that died must produce a diagnostic, not a bare EPIPE.

    Found by this test: handing a trajectory to a dead worker raised
    OSError(EINVAL) from the stdin write, which names neither the worker
    nor the trajectory it was about to run.  An ensemble that loses a
    member deserves to say which one.
    """
    pool = _FakePool(2, tmp_path)
    try:
        pool.procs[0].kill()
        pool.procs[0].wait(timeout=10)
        with pytest.raises(RuntimeError) as raised:
            pool.run_leg(_tasks(4))
        message = str(raised.value)
        assert "member worker 0" in message
        assert "trajectory" in message
    finally:
        for proc in pool.procs:
            if proc.poll() is None:
                try:
                    proc.stdin.write(json.dumps({"stop": True}) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                proc.wait(timeout=10)


def test_all_workers_are_used_when_the_leg_is_wide_enough(tmp_path):
    """Concurrency has to actually reach every worker.

    A dispatch bug that always refilled worker 0 would still return the
    right answers, just serially -- and the wall clock is the only thing
    that would notice.  This makes it a test.
    """
    pool = _FakePool(3, tmp_path)
    try:
        results = pool.run_leg(_tasks(9, sleeps={i: 0.05 for i in range(9)}))
    finally:
        pool.close()
    served = {payload["entry"]["served_by"] for payload in results.values()}
    assert served == {"0", "1", "2"}, f"only workers {served} did any work"
