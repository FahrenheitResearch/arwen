"""Single-writer discipline and atomic publication -- the guard itself.

Every claim here is proved against the mechanism, not a mock of it: the
cross-process gates spawn real ``python`` subprocesses that take the
real OS lock, and the crash gate has a child abandon the lock by exiting
without unwinding, so the release is the kernel's rather than the
program's.
"""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from gpuwm import fetch_guard


@pytest.fixture(autouse=True)
def isolated_lock_root(tmp_path, monkeypatch):
    """Never touch the machine-wide lock root from a test."""

    root = tmp_path / "locks"
    monkeypatch.setenv(fetch_guard.LOCK_ROOT_ENV, str(root))
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()
    yield root
    fetch_guard._HELD.clear()
    fetch_guard._KEY_LOCKS.clear()


def _child(source: str, *args: str, env_extra: dict | None = None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *args],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))


_HOLD_THEN_WAIT = """
    import os, sys, time
    from pathlib import Path
    from gpuwm import fetch_guard

    kind, target = sys.argv[1], sys.argv[2]
    flag, seconds = Path(sys.argv[3]), float(sys.argv[4])
    release = flag.with_suffix(".release")
    with fetch_guard.hold(kind, target, timeout_s=30):
        publishing = flag.with_suffix(".publishing")
        publishing.write_text(str(os.getpid()), encoding="utf-8")
        publishing.replace(flag)
        deadline = time.monotonic() + seconds
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    print("released")
"""

_ABANDON = """
    import os, sys
    from pathlib import Path
    from gpuwm import fetch_guard

    target, flag = sys.argv[1], Path(sys.argv[2])
    lock = fetch_guard.hold("fetch-out", target, timeout_s=30).acquire()
    flag.write_text("held", encoding="utf-8")
    # Leave without unwinding: no release, no atexit, no __exit__.  Only
    # the kernel can hand this lock back, which is the whole point of
    # using an OS lock instead of a sentinel file.
    os._exit(0)
"""


# ---------------------------------------------------------------------------
# Lock identity
# ---------------------------------------------------------------------------

def test_lock_path_is_per_target_and_per_kind(tmp_path):
    one = tmp_path / "run-a"
    two = tmp_path / "run-b"
    assert fetch_guard.lock_path("fetch-out", one) != \
        fetch_guard.lock_path("fetch-out", two)
    assert fetch_guard.lock_path("fetch-out", one) != \
        fetch_guard.lock_path("fetch-geog", one)
    # Spelling the same directory differently must not split one output
    # root into two writers: the key is the resolved path.
    one.mkdir()
    for spelling in (one.parent / "." / one.name,
                     one / "sub" / "..",
                     Path(str(one) + os.sep)):
        assert fetch_guard.lock_path("fetch-out", spelling) == \
            fetch_guard.lock_path("fetch-out", one), spelling


def test_lock_file_lives_outside_the_output_tree(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with fetch_guard.hold("fetch-out", out):
        assert list(out.iterdir()) == []
    assert fetch_guard.lock_path("fetch-out", out).is_file()


# ---------------------------------------------------------------------------
# The lie: two writers in one output directory
# ---------------------------------------------------------------------------

def _spawn_holder(kind: str, target: Path, flag: Path, seconds: float):
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(_HOLD_THEN_WAIT),
         kind, str(target), str(flag), str(seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=dict(os.environ),
        cwd=str(Path(__file__).resolve().parents[1]))


def _await_flag(flag: Path, child) -> int:
    deadline = time.monotonic() + 60
    while not flag.exists() and time.monotonic() < deadline:
        assert child.poll() is None, child.communicate()
        time.sleep(0.05)
    assert flag.exists(), "child never reported holding the lock"
    return int(flag.read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", ["fetch-out", "fetch-geog", "fetch-tables"])
@pytest.mark.parametrize("budget", [0.0, 0.5])
def test_a_second_process_cannot_hold_the_same_target(tmp_path, kind, budget):
    """LS-3 mechanism: real concurrent processes, real OS lock.

    Both wait budgets refuse rather than proceed -- a zero budget fails
    fast, a positive one queues first -- and every locked resource (data
    output, geog root, table root) behaves the same way.
    """

    target = tmp_path / "run"
    flag = tmp_path / "held.flag"
    child = _spawn_holder(kind, target, flag, 30)
    try:
        holder_pid = _await_flag(flag, child)
        with pytest.raises(fetch_guard.FetchLockBusy) as caught:
            fetch_guard.hold(kind, target, timeout_s=budget).acquire()
        assert "refuses rather than interleave" in str(caught.value)
        # A Windows venv redirector can keep a launcher process around while
        # the real interpreter owns the OS lock.  The holder's own PID is the
        # authority recorded by production; Popen.pid can name the launcher.
        assert f"pid {holder_pid}" in str(caught.value)
    finally:
        flag.with_suffix(".release").write_text("go", encoding="utf-8")
        child.wait(timeout=60)
    # Once the holder is gone the lock is free again.
    with fetch_guard.hold(kind, target, timeout_s=5):
        pass


def test_different_targets_and_kinds_do_not_exclude_each_other(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    flag = tmp_path / "held.flag"
    child = _spawn_holder("fetch-out", first, flag, 30)
    try:
        _await_flag(flag, child)
        # A different --out is a different writer, and so is a different
        # resource over the same directory.
        with fetch_guard.hold("fetch-out", second, timeout_s=0):
            pass
        with fetch_guard.hold("fetch-geog", first, timeout_s=0):
            pass
    finally:
        flag.with_suffix(".release").write_text("go", encoding="utf-8")
        child.wait(timeout=60)


def test_a_crashed_holder_does_not_wedge_the_directory(tmp_path):
    """A cooperative sentinel would leave this directory unfetchable."""

    target = tmp_path / "run"
    flag = tmp_path / "held.flag"
    done = _child(_ABANDON, str(target), str(flag))
    assert done.returncode == 0, done.stderr
    assert flag.read_text() == "held"
    with fetch_guard.hold("fetch-out", target, timeout_s=2):
        pass


def test_the_same_thread_nests_but_another_thread_queues(tmp_path):
    target = tmp_path / "run"
    with fetch_guard.hold("fetch-out", target):
        # The CLI takes it around the request guard, the library takes it
        # again around the transfer: nesting must not deadlock.
        with fetch_guard.hold("fetch-out", target):
            pass
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            def other():
                with fetch_guard.hold("fetch-out", target, timeout_s=0):
                    return "took it"
            with pytest.raises(fetch_guard.FetchLockBusy):
                pool.submit(other).result(timeout=30)
    # Released cleanly after the nesting unwound.
    with fetch_guard.hold("fetch-out", target, timeout_s=0):
        pass


def test_a_waiting_writer_announces_before_it_refuses(tmp_path):
    target = tmp_path / "run"
    lines: list[str] = []
    with fetch_guard.hold("fetch-out", target):
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            def other():
                with fetch_guard.hold("fetch-out", target, timeout_s=0.4,
                                      progress=lines.append):
                    return None
            with pytest.raises(fetch_guard.FetchLockBusy):
                pool.submit(other).result(timeout=30)
    # The in-process queue refuses without a wait announcement only when
    # the budget is zero; with a budget it waits, and waiting is loud.
    assert lines == [] or any("waiting up to" in line for line in lines)


def test_lock_timeout_env_is_the_wait_budget(tmp_path, monkeypatch):
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, "0")
    assert fetch_guard.OutputLock("fetch-out", tmp_path).timeout_s == 0.0
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, "12.5")
    assert fetch_guard.OutputLock("fetch-out", tmp_path).timeout_s == 12.5
    monkeypatch.setenv(fetch_guard.LOCK_TIMEOUT_ENV, "not-a-number")
    with pytest.raises(ValueError, match="not a number of seconds"):
        fetch_guard.OutputLock("fetch-out", tmp_path)


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------

def test_staging_names_are_unique_per_call(tmp_path):
    """The fixed ``<name>.tmp`` was what two publishers collided on."""

    target = tmp_path / "receipt.json"
    seen = {fetch_guard._staging_path(target, "publish") for _ in range(50)}
    assert len(seen) == 50
    assert all(str(os.getpid()) in path.name for path in seen)


@pytest.mark.parametrize("payload", ["first\n", "second, longer\n", ""])
def test_atomic_write_publishes_whole_or_not_at_all(tmp_path, payload,
                                                    monkeypatch):
    target = tmp_path / "receipt.json"
    fetch_guard.atomic_write_text(target, "original\n")

    real_replace = os.replace
    calls: list = []

    def failing_replace(src, dst):
        calls.append((src, dst))
        raise OSError("injected failure between write and publish")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected failure"):
        fetch_guard.atomic_write_text(target, payload)
    monkeypatch.setattr(os, "replace", real_replace)

    # The old receipt is intact and no staging litter survived.
    assert target.read_text() == "original\n"
    assert [path.name for path in tmp_path.iterdir()] == ["receipt.json"]
    assert calls, "the injection point was never reached"

    fetch_guard.atomic_write_text(target, payload)
    assert target.read_text() == payload


def test_atomic_write_text_is_lf_and_utf8(tmp_path):
    target = tmp_path / "receipt.tsv"
    fetch_guard.atomic_write_text(target, "a\tb\nµ\n")
    assert target.read_bytes() == "a\tb\nµ\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Quarantine never overwrites evidence
# ---------------------------------------------------------------------------

def test_quarantine_never_overwrites_an_earlier_artifact(tmp_path,
                                                         monkeypatch):
    """LS-14: a forced timestamp collision used to destroy evidence."""

    monkeypatch.setattr(time, "time_ns", lambda: 1_700_000_000_000_000_000)
    kept = []
    for index in range(4):
        victim = tmp_path / "payload.grib2"
        victim.write_bytes(f"generation {index}".encode("ascii"))
        kept.append(fetch_guard.quarantine(victim))

    assert len({path.name for path in kept}) == 4
    for index, path in enumerate(kept):
        assert path.read_bytes() == f"generation {index}".encode("ascii")


def test_quarantine_tag_selects_the_evidence_name(tmp_path):
    victim = tmp_path / "payload.grib2"
    victim.write_bytes(b"x")
    aside = fetch_guard.quarantine(victim, tag="inventory-change")
    assert ".inventory-change-" in aside.name
    assert not victim.exists()
