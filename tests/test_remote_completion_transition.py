"""Real Linux wrapper/runner exits; barriers order publication, not sleeps.

The runner emits protocol-only bytes, never weather. Queue entries are already
processed/evicted so this test cannot invoke a renderer or another viewer worker.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import select
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gpuwm import remote_artifacts as ra, remote_native_plots as plots
from gpuwm import remote_preparation_v2 as preparation, remote_processed as legacy
from gpuwm import remote_processed_v2 as viewer, remote_worker as rw

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open")
    or not hasattr(signal, "pidfd_send_signal"), reason="Linux pidfd ownership contract")
WATCHDOG = 10  # Deadlock guard only; never establishes the interleaving.

RUNNER = r'''
import json, os, socket, sys, time
from pathlib import Path
from gpuwm import remote_worker as rw
record = rw._record(Path(sys.argv[1]))
root = Path(record["outdir"])
prepared = sys.argv[4] == "prepared"
started = rw._now()
manifest = {"schema": "gpuwm.run-manifest.v1", "route": "prepared" if prepared else "experiment",
    "run_id": "fixture-native-run", "pid": os.getpid(), "started_at_utc": started,
    "run_dir": str(root), "outputs_dir": str(root), "plan_source": record["snapshot_plan"],
    "plan_sha256": record["plan_sha256"], "events_path": str(root / "events.jsonl")}
rw._write(root / "run-manifest.json", manifest)
frame = root / "wrfout_d01_fixture"
frame.write_bytes(b"protocol-only committed bytes, not weather")
base = {"schema_version": "gpuwm.run-plan.event.v1", "emitted_unix_ms": int(time.time()*1000)}
events = [
    {**base, "sequence": 1, "event": "resolved_plan", "config_source": record["snapshot_config"],
     "config_sha256": record["config_sha256"]},
    {**base, "sequence": 2, "event": "output_committed", "domain": 1, "path": str(frame),
     "size_bytes": frame.stat().st_size, "valid_time": "2026-09-07T18:00:00Z"},
    {**base, "sequence": 3, "event": "completed", "dry_run": False,
     "run_dir": str(root), "receipt_path": str(root / "run-manifest.json"),
     "outputs_committed": 1, "receipts": {}, "summary": {"executed": True}}]
if prepared:
    producer = root / "chain" / "run-20260910-050000Z_i202609071800Z"
    producer.mkdir(parents=True)
    (producer.parent / "latest-run.txt").write_text(producer.name+"\n")
    native_start = rw._now()
    emitted = int(time.time()*1000)
    inner = {**manifest, "route": "experiment", "run_id": "inner-native-run",
        "started_at_utc": native_start, "run_dir": str(producer), "outputs_dir": str(producer),
        "events_path": str(producer / "events.jsonl"),
        "plan_source": "gpuwm go "+record["snapshot_config"], "plan_sha256": "f"*64}
    rw._write(producer / "run-manifest.json", inner)
    destination = producer / frame.name
    frame.rename(destination)
    events[1]["path"] = str(destination)
    native_events = [{**row, "emitted_unix_ms": emitted} for row in events]
    native_events[-1].update(run_dir=str(producer), receipt_path=str(producer / "run-manifest.json"))
    (producer / "events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in native_events))
    events[-1]["emitted_unix_ms"] = int(time.time()*1000)
(root / "events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
with socket.socket(socket.AF_UNIX) as ready:
    ready.connect(sys.argv[2])
    ready.sendall(b"R")
    assert ready.recv(1) == b"X"
raise SystemExit(int(sys.argv[3]))
'''

WRAPPER = r'''
import os, sys
from pathlib import Path
from types import SimpleNamespace
from gpuwm import remote_worker as rw
notice, release = int(sys.argv[2]), int(sys.argv[3])
receipt = len(sys.argv) < 5 or sys.argv[4] != "no-runner-receipt"
write, popen = rw._write, rw.subprocess.Popen
class Child:
    def __init__(self, *args, **kwargs):
        self.real = popen(*args, **kwargs)
        os.write(notice, b"S")
        assert os.read(release, 1) == b"S"  # Test-only live-runner setup barrier.
    def __getattr__(self, name):
        return getattr(self.real, name)
    def poll(self):
        return self.real.poll()
clock, first = rw.time, True
def poll_sleep(seconds):
    global first
    if first:
        first = False
        os.write(notice, b"P")  # The first existing live-child loop has executed.
    clock.sleep(seconds)
rw.time = SimpleNamespace(monotonic=clock.monotonic, sleep=poll_sleep)
def barrier_write(path, value):
    if path.name == "runner.json" and not receipt:
        return  # This runner exited before the wrapper could record a receipt.
    if path.name == "result.json":
        os.write(notice, b"E")  # Actual run_worker has reaped the runner.
        assert os.read(release, 1) == b"X"
    write(path, value)
rw.subprocess.Popen, rw._write = Child, barrier_write
raise SystemExit(rw.run_worker(Path(sys.argv[1]), os.environ[rw.TOKEN_ENV]))
'''


def read_byte(fd, expected):
    assert select.select([fd], [], [], WATCHDOG)[0], "wrapper barrier did not arrive"
    assert os.read(fd, 1) == expected


def job_inputs(tmp_path, name):
    """One durable start-plan job's captured inputs, as the RPC door saves them."""
    store = rw._store(tmp_path, create=True)
    directory = store / name
    directory.mkdir(mode=0o700)
    inputs = directory / "inputs"
    inputs.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    config = inputs / "case.toml"
    config.write_text('[experiment]\nstart_time="2026-09-07T18:00:00Z"\n'
        'run_seconds=3600\nrestart_interval_s=0\n[[domain]]\ngrid_id=1\nhistory_interval_s=900\n')
    plan = inputs / "plan.json"
    rw._write(plan, {"schema": "gpuwm.run-plan.v1", "name": "fixture", "route": "experiment",
                     "config": {"path": "case.toml"}, "output_root": str(output)})
    return directory, output, config, plan


def job_record(tmp_path, directory, output, config, plan, argv):
    digest = rw._file_sha(config)
    record = {"schema": "gpuwm.remote.job.v1", "id": directory.name,
        "token": secrets.token_hex(32), "created_at": rw._now(), "action": "start-plan",
        "config": str(config), "snapshot_config": str(config), "snapshot_plan": str(plan),
        "plan_sha256": rw._file_sha(plan), "config_sha256": digest, "snapshot_sha256": digest,
        "snapshot_inputs": {"case.toml": digest, "plan.json": rw._file_sha(plan)},
        "outdir": str(output), "runtime": {}, "cwd": str(tmp_path),
        "source": {"config_path": "/desktop/original.toml", "config_sha256": "b" * 64},
        "argv": argv}
    rw._write(directory / "job.json", record)
    return record


@pytest.fixture
def transition(tmp_path):
    processes, descriptors, sockets = [], [], []
    directory, output, config, plan = job_inputs(tmp_path, "completion-fixture")
    # Short socket path: pytest's own tmp_path may exceed AF_UNIX's 108 bytes.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="arwen-completion-") as socket_dir:
        socket_path = str(Path(socket_dir) / "runner.sock")
        listener = socket.socket(socket.AF_UNIX)
        sockets.append(listener)
        listener.bind(socket_path)
        listener.listen(1)
        listener.settimeout(WATCHDOG)
        record = job_record(tmp_path, directory, output, config, plan,
            [sys.executable, "-c", RUNNER, str(directory), socket_path, "0", "experiment"])
        c = SimpleNamespace(workspace=tmp_path, directory=directory, record=record,
                            output=output, config=config, plan=plan, processes=processes)

        def launch(exit_code=0, *, prepared=False, receipt=True):
            record["argv"][-2:] = [str(exit_code), "prepared" if prepared else "experiment"]
            if prepared:
                rewrite(plan, lambda value: value.update(route="prepared"))
                record["plan_sha256"] = record["snapshot_inputs"]["plan.json"] = rw._file_sha(plan)
            rw._write(directory / "job.json", record)
            notice_r, notice_w = os.pipe()
            release_r, release_w = os.pipe()
            descriptors.extend([notice_r, release_w])
            c.release_w, c.published, c.in_gap = release_w, False, False
            environment = {**os.environ, rw.TOKEN_ENV: record["token"],
                           "PYTHONPATH": str(Path(rw.__file__).resolve().parents[1])}
            log = (directory / "test-wrapper.log").open("wb")
            try:
                c.wrapper = subprocess.Popen(
                    [sys.executable, "-c", WRAPPER, str(directory), str(notice_w), str(release_r)]
                    + ([] if receipt else ["no-runner-receipt"]),
                    env=environment, pass_fds=(notice_w, release_r),
                    stdout=log, stderr=log, start_new_session=True)
            finally:
                log.close()
                os.close(notice_w)
                os.close(release_r)
            processes.append(c.wrapper)
            read_byte(notice_r, b"S")
            runner, _ = listener.accept()
            sockets.append(runner)
            runner.settimeout(WATCHDOG)
            assert runner.recv(1) == b"R"
            c.manifest_path = output / "run-manifest.json"
            c.manifest = rw._json(c.manifest_path)
            c.runner_pid = c.manifest["pid"]
            assert rw._has_token(c.runner_pid, record["token"])
            c.runner_identity = rw._process(c.runner_pid)
            os.write(release_w, b"S")
            read_byte(notice_r, b"P")
            runner.sendall(b"X")
            read_byte(notice_r, b"E")
            # This is the requested gap, using the real wrapper's status.
            assert rw._process(c.runner_pid) is None
            assert not rw._has_token(c.runner_pid, record["token"])
            assert rw._status(directory)["state"] == "running"
            assert not (directory / "result.json").exists()
            assert (directory / "runner.json").exists() is receipt
            c.in_gap = True
            c.parent_events = output / "events.jsonl"
            c.producer = output
            if prepared:
                c.pointer = output / "chain" / "latest-run.txt"
                c.producer = c.pointer.parent / c.pointer.read_text().strip()
            c.events = c.producer / "events.jsonl"
            c.event = json.loads(c.events.read_bytes().splitlines()[1])
            c.frame = Path(c.event["path"])
            selection = viewer._selection()
            # Normal already-derived entry: no test fake of the scheduler.
            entry = {"schema": viewer.SCHEMA, "job_id": record["id"], "state": "evicted",
                     "sequence": 2, "domain": 1, **selection,
                     "commit_sha256": ra._sha(c.events.read_bytes().splitlines(keepends=True)[1])}
            path = viewer._entry_path(viewer._root(tmp_path), record["id"], 2, selection)
            rw._write(path, entry)
            c.release_w = release_w
            c.published = False
            return c

        def publish():
            if c.in_gap and not c.published and c.wrapper.poll() is None:
                os.write(c.release_w, b"X")
                c.published = True
                assert c.wrapper.wait(timeout=WATCHDOG) == 0

        c.launch, c.publish = launch, publish
        try:
            yield c
        finally:
            if hasattr(c, "wrapper"):
                publish()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=WATCHDOG)
            for sock in sockets:
                sock.close()
            for fd in descriptors:
                os.close(fd)


def preparation_receipt(c):
    return viewer._directory(viewer._root(c.workspace), c.record["id"]) / "preparation.json"


def plots_receipt(c):
    return plots._root(c.workspace, c.record["id"]) / "status.json"


def viewer_receipt(c):
    return viewer._directory(viewer._root(c.workspace), c.record["id"]) / "status.json"


@contextmanager
def watching(c, monkeypatch, *, cancellation=False, module=preparation, start=None, receipt=preparation_receipt):
    """Drive one watcher's real loop on a fake monotonic clock.

    ``module`` is the watcher under test: its own ``time`` is the clock its
    completion wait uses, so the same barrier drives background preparation,
    the plot gallery and the compact viewer queue.
    """
    notices, permits = queue.Queue(), queue.Queue()
    clock = SimpleNamespace(now=1000.0, wall=time.time())
    def sleep(seconds):
        notices.put(("wait", seconds))
        command = permits.get(timeout=WATCHDOG)
        if command == "stop":
            raise RuntimeError("test teardown")
        clock.now += seconds
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.wall, sleep=sleep))
    # os.nice is irrelevant and would persistently change the pytest thread.
    monkeypatch.setattr(module.os, "nice", lambda _value: None)
    cancelled = threading.Event()
    options = {"cancel": SimpleNamespace(is_set=cancelled.is_set, wait=sleep)} if cancellation else {}
    if start is None:
        start = lambda **kwargs: preparation.worker(c.workspace, c.record["id"], **kwargs)
    def run():
        try:
            notices.put(("done", start(**options)))
        except BaseException as error:
            notices.put(("raised", repr(error)))
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    def notice():
        kind, value = notices.get(timeout=WATCHDOG)
        path = receipt(c)
        return kind, value, rw._json(path) if path.exists() else None
    watch = SimpleNamespace(clock=clock, resume=lambda: permits.put("go"), notice=notice,
                            cancel=cancelled.set, receipt=lambda: receipt(c))
    try:
        yield watch
    finally:
        permits.put("stop")
        thread.join(timeout=WATCHDOG)
        assert not thread.is_alive(), "preparation worker did not release its lease"


def expect_wait(w):
    kind, seconds, receipt = w.notice()
    assert kind == "wait", (kind, seconds, receipt)
    assert 0 < seconds <= preparation.POLL_SECONDS
    assert receipt is None or not receipt["done"]
    return seconds


def expect_exit(w, code):
    kind, actual, receipt = w.notice()
    assert (kind, actual) == ("done", code), (kind, actual, receipt)
    return receipt


def expect_done(w, code):
    receipt = expect_exit(w, code)
    assert receipt["done"]
    return receipt


@pytest.mark.parametrize("prepared", [False, True], ids=["experiment", "hosted"])
def test_runner_exit_before_wrapper_result_settles(transition, monkeypatch, prepared):
    c = transition.launch(prepared=prepared)
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])
        # Multiple full revalidations while the real wrapper cannot publish.
        w.resume()
        expect_wait(w)
        c.publish()
        assert rw._status(c.directory)["state"] == "completed"
        w.resume()
        receipt = expect_done(w, 0)
        assert receipt["state"] == "complete" and receipt["evicted"] == 1


def rewrite(path, change):
    value = rw._json(path)
    change(value)
    rw._write(path, value)


def changed(c, which):
    job = c.directory / "job.json"
    if which == "manifest":
        rewrite(c.manifest_path, lambda value: value.update(run_id="another-run"))
    elif which == "manifest_missing":
        c.manifest_path.unlink()
    elif which == "plan":
        c.plan.write_bytes(c.plan.read_bytes() + b" ")
    elif which == "plan_coherent":
        c.plan.write_bytes(c.plan.read_bytes() + b" ")
        digest = rw._file_sha(c.plan)
        def binding(value):
            value["plan_sha256"] = digest
            value["snapshot_inputs"]["plan.json"] = digest
        rewrite(job, binding)
        rewrite(c.manifest_path, lambda value: value.update(plan_sha256=digest))
    elif which == "source":
        rewrite(job, lambda value: value["source"].update(config_sha256="c" * 64))
    elif which == "config":
        c.config.write_bytes(c.config.read_bytes() + b"\n")
    elif which == "runner_receipt":
        rewrite(c.directory / "runner.json", lambda value: value["identity"].update(start_ticks="0"))
    elif which == "runner_receipt_missing":
        (c.directory / "runner.json").unlink()
    elif which == "commit":
        rows = [json.loads(line) for line in c.events.read_bytes().splitlines()]
        rows[1]["geometry"] = {"changed": True}
        c.events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif which == "frame":
        before = c.frame.stat()
        c.frame.write_bytes(b"X" * before.st_size)
        # Same bytes count and restored mtime still changes ctime.
        os.utime(c.frame, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif which == "frame_missing":
        c.frame.unlink()
    elif which == "frame_symlink":
        elsewhere = c.workspace / "not-an-owned-output"
        elsewhere.write_bytes(c.frame.read_bytes())
        c.frame.unlink()
        c.frame.symlink_to(elsewhere)
    elif which == "token":
        rewrite(job, lambda value: value.update(token="e" * 64))
    elif which == "wrapper_identity":
        rewrite(c.directory / "started.json", lambda value: value["identity"].update(start_ticks="0"))
    else:
        raise AssertionError(which)


@pytest.mark.parametrize("settled", [False, True], ids=["during-wait", "at-settlement"])
@pytest.mark.parametrize("which", ["manifest", "manifest_missing", "plan", "plan_coherent", "source", "config",
    "runner_receipt", "runner_receipt_missing", "commit", "frame", "frame_missing", "frame_symlink", "token", "wrapper_identity"])
def test_changed_evidence_never_retries_or_mutates_queue(transition, monkeypatch, which, settled):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        w.resume()
        expect_wait(w)  # Establish a second unchanged full validation first.
        if settled:
            c.publish()
        changed(c, which)
        w.resume()
        receipt = expect_done(w, 2)
        assert receipt["state"] == "failed"
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])


def test_wrapper_failure_overrides_native_completion(transition, monkeypatch):
    c = transition.launch(exit_code=1)
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        c.publish()
        assert rw._status(c.directory)["state"] == "failed"
        w.resume()
        receipt = expect_done(w, 2)
        assert "success" in receipt["error"]
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])


def test_wrapper_loss_is_not_another_completion_retry(transition, monkeypatch):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        c.wrapper.kill()
        c.wrapper.wait(timeout=WATCHDOG)
        assert rw._status(c.directory)["state"] == "lost"
        w.resume()
        receipt = expect_done(w, 2)
        assert "ownership" in receipt["error"]


@pytest.mark.parametrize("own_token", [False, True], ids=["wrong-token", "same-token-new-process"])
def test_live_replacement_process_is_not_completion(transition, monkeypatch, own_token):
    c = transition.launch()
    environment = dict(os.environ)
    environment.pop(rw.TOKEN_ENV, None)
    if own_token:
        environment[rw.TOKEN_ENV] = c.record["token"]
    stranger = subprocess.Popen([sys.executable, "-c", "import sys;print('R',flush=True);sys.stdin.buffer.read(1)"],
        env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    c.processes.append(stranger)
    assert select.select([stranger.stdout], [], [], WATCHDOG)[0]
    assert stranger.stdout.readline() == b"R\n"
    try:
        with watching(c, monkeypatch) as w:
            expect_wait(w)
            rewrite(c.manifest_path, lambda value: value.update(pid=stranger.pid))
            rewrite(c.directory / "runner.json", lambda value: value.update(identity=rw._process(stranger.pid)))
            w.resume()
            receipt = expect_done(w, 2)
            assert "live" in receipt["error"] or "settle" in receipt["error"]
    finally:
        stranger.stdin.close()
        stranger.stdout.close()
        stranger.wait(timeout=WATCHDOG)


def test_initial_live_wrong_token_has_no_retry(transition, monkeypatch):
    c = transition.launch()
    # The pytest process is live and never inherits this job's unique token.
    assert not rw._has_token(os.getpid(), c.record["token"])
    rewrite(c.manifest_path, lambda value: value.update(pid=os.getpid()))
    rewrite(c.directory / "runner.json", lambda value: value.update(identity=rw._process(os.getpid())))
    with watching(c, monkeypatch) as w:
        receipt = expect_done(w, 2)
        assert "live" in receipt["error"]


@pytest.mark.parametrize("how", ["stop", "foreign-stop", "local"])
def test_cancellation_terminates_completion_wait(transition, monkeypatch, how):
    c = transition.launch()
    with watching(c, monkeypatch, cancellation=how == "local") as w:
        expect_wait(w)
        if how == "local":
            w.cancel()
        else:
            rw._write(c.directory / "stop.json", {
                "token": c.record["token"] if how == "stop" else "f" * 64, "requested_at": rw._now()})
        w.resume()
        receipt = expect_exit(w, 2)
        assert "cancel" in receipt["error"]
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])
        # This job's own stop request, and its watcher's own shutdown, end the
        # wait without a terminal receipt: ensure() returns early forever on a
        # done receipt, so a cancel must not poison this job's preparation.
        # A stop marker owned by another job is tampering and stays terminal.
        foreign = how == "foreign-stop"
        assert receipt["done"] is foreign
        assert receipt["state"] == ("failed" if foreign else "cancelled")
        preparation.ensure(c.workspace, c.record["id"])
        assert (preparation_receipt(c).parent / "preparation.log").exists() is not foreign


def test_a_cancelled_preparation_is_relaunchable_where_a_failed_one_is_not(transition, monkeypatch):
    c = transition.launch()
    launched = []
    monkeypatch.setattr(preparation.subprocess, "Popen", lambda *args, **kwargs: launched.append(args) or None)
    for state, done in (("cancelled", False), ("failed", True)):
        rw._write(preparation_receipt(c), {"schema": preparation.SCHEMA, "job_id": c.record["id"],
            "state": state, "done": done, "error": "fixture", "updated_unix_ms": 0})
        preparation.ensure(c.workspace, c.record["id"])
    assert len(launched) == 1


def test_shutdown_signal_cancels_the_watcher_instead_of_killing_it(monkeypatch):
    delivered = {}
    def fake_worker(workspace, job, *, cancel):
        signal.raise_signal(signal.SIGTERM)
        delivered["set"] = cancel.wait(WATCHDOG)
        return 0
    monkeypatch.setattr(preparation, "worker", fake_worker)
    monkeypatch.setattr(rw, "_workspace", lambda request: Path(request["workspace"]))
    previous = signal.getsignal(signal.SIGTERM)
    try:
        assert preparation.main(["--workspace", ".", "--job", "any-job"]) == 0
    finally:
        signal.signal(signal.SIGTERM, previous)
    # SIGTERM inside a validation would leave the operator a stale receipt and
    # a lease; the wired event unwinds the loop through the cancelled receipt.
    assert delivered == {"set": True}


@pytest.mark.parametrize("late_success", [False, True])
def test_one_monotonic_deadline_cannot_reset_or_be_beaten_by_late_success(transition, monkeypatch, late_success):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        delay = expect_wait(w)
        for _ in range(3):
            w.resume()
            delay = expect_wait(w)
        w.clock.now = 1000.0 + preparation.COMPLETION_SECONDS - delay
        w.clock.wall -= 1_000_000_000  # Wall-clock reversal cannot extend the budget.
        if late_success:
            c.publish()
        w.resume()
        receipt = expect_done(w, 2)
        assert w.clock.now == 1000.0 + preparation.COMPLETION_SECONDS
        assert "deadline" in receipt["error"]


def test_deadline_is_checked_after_terminal_validation_before_queue_mutation(transition, monkeypatch):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        original = ra._completion_evidence
        def slow_validation(record, state, *args, **kwargs):
            result = original(record, state, *args, **kwargs)
            if state["state"] == "completed":
                w.clock.now = 1000.0 + preparation.COMPLETION_SECONDS
            return result
        monkeypatch.setattr(ra, "_completion_evidence", slow_validation)
        c.publish()
        w.resume()
        assert "deadline" in expect_done(w, 2)["error"]
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])


@pytest.mark.parametrize("wall_jump", [-1e10, 1e10])
def test_wall_clock_jump_does_not_expire_valid_transition(transition, monkeypatch, wall_jump):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        w.clock.wall += wall_jump
        c.publish()
        w.resume()
        assert expect_done(w, 0)["state"] == "complete"


@pytest.mark.parametrize("damage", ["no-completed", "dry-run", "failed", "partial", "wrong-resolved"])
def test_missing_runner_alone_is_never_completion_proof(transition, monkeypatch, damage):
    c = transition.launch()
    rows = [json.loads(line) for line in c.events.read_bytes().splitlines()]
    if damage == "no-completed":
        rows.pop()
    elif damage == "dry-run":
        rows[-1]["dry_run"] = True
    elif damage == "failed":
        rows[-1]["event"] = "failed"
    elif damage == "wrong-resolved":
        rows[0]["config_sha256"] = "c" * 64
    c.events.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if damage == "partial":
        with c.events.open("ab") as stream:
            stream.write(b'{"incomplete"')
    with watching(c, monkeypatch) as w:
        expect_done(w, 2)  # No wait notification is permitted.


def test_success_published_between_status_and_binding_still_settles(transition, monkeypatch):
    c = transition.launch()
    original = rw._status
    def publish_after_status(directory):
        state = original(directory)
        if not c.published:
            c.publish()
        return state
    monkeypatch.setattr(rw, "_status", publish_after_status)
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        w.resume()
        assert expect_done(w, 0)["state"] == "complete"


def test_missing_runner_receipt_names_the_completion_race(transition):
    # The real wrapper, with a runner that exits before its receipt is written.
    c = transition.launch(receipt=False)
    state = rw._status(c.directory)
    with pytest.raises(ra.ProducerCompletionUnprovable) as error:
        ra.bound_manifest(c.record, state, job_directory=c.directory)
    message = str(error.value)
    assert not isinstance(error.value, ra.ProducerCompletionPending)
    # Not a bare FileNotFoundError path: it names the race, says it is not
    # tampering, and points at the refusal that already covers the window.
    assert "runner receipt" in message and "completion race" in message
    assert "not tampering" in message and "terminal state" in message
    # No completion privilege is granted here, and no authority is published.
    with pytest.raises(ValueError, match="belongs to this active remote job") as ordinary:
        ra.bound_manifest(c.record, state)
    assert not isinstance(ordinary.value, ra.ProducerCompletionUnprovable)


def test_off_linux_the_transition_names_the_platform_breakage(transition, monkeypatch):
    c = transition.launch()
    class WithoutPidfd:
        def __getattr__(self, name):
            if name == "pidfd_open":
                raise AttributeError(name)
            return getattr(os, name)
    monkeypatch.setattr(ra, "os", WithoutPidfd())
    state = rw._status(c.directory)
    with pytest.raises(ValueError) as error:
        ra.bound_manifest(c.record, state, job_directory=c.directory)
    message = str(error.value)
    assert not isinstance(error.value, AttributeError)
    assert isinstance(error.value, ra.ProducerCompletionUnprovable)
    assert "Linux pidfd process handles" in message and "terminal state" in message


def published_plot(c, state="failed"):
    """A frame whose gallery receipt already exists: no renderer is reachable."""
    root = plots._root(c.workspace, c.record["id"])
    authority = ra._sha(c.events.read_bytes().splitlines(keepends=True)[1])
    value = {"schema": plots.SCHEMA, "state": state, "job_id": c.record["id"], "domain": 1,
             "sequence": 2, "commit_sha256": authority}
    if state == "failed":
        value["error"] = "fixture: this frame's compact store was retired before plots ran"
    rw._write(plots._receipt(root, 2), value)
    return root


def test_plots_worker_waits_for_the_wrapper_instead_of_a_terminal_receipt(transition, monkeypatch):
    c = transition.launch()
    root = published_plot(c)
    with watching(c, monkeypatch, module=plots, receipt=plots_receipt,
                  start=lambda **options: plots.worker(c.workspace, c.record["id"], **options)) as w:
        expect_wait(w)
        # run_worker launches this watcher on every start-plan job and _status
        # surfaces its receipt as native_plots: a terminal failure here is what
        # the user reads, and plots.ensure would then refuse to relaunch.
        assert not (root / "status.json").exists()
        w.resume()
        expect_wait(w)
        c.publish()
        w.resume()
        receipt = expect_done(w, 0)
        assert receipt["state"] == "complete_with_errors" and receipt["failed"] == 1
        assert receipt["simulation_state"] == "completed"


def test_plots_worker_cancellation_leaves_the_gallery_relaunchable(transition, monkeypatch):
    c = transition.launch()
    published_plot(c)
    with watching(c, monkeypatch, module=plots, receipt=plots_receipt, cancellation=True,
                  start=lambda **options: plots.worker(c.workspace, c.record["id"], **options)) as w:
        expect_wait(w)
        w.cancel()
        w.resume()
        receipt = expect_exit(w, 2)
        assert receipt["state"] == "cancelled" and receipt["done"] is False
        assert "cancel" in receipt["error"]


def test_plots_door_waits_with_the_watcher_instead_of_refusing(transition, monkeypatch):
    c = transition.launch()
    monkeypatch.setattr(preparation, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(plots, "ensure", lambda *args, **kwargs: None)
    request = {"schema": "gpuwm.remote.request.v1", "action": "native-plots",
               "workspace": str(c.workspace), "job": c.record["id"], "domain": 1, "sequence": 2}
    value = plots.catalog(request, c.workspace)
    assert value["waiting"] and value["producer_completing"] and "panels" not in value
    c.publish()
    settled = plots.catalog(request, c.workspace)
    assert settled["waiting"] and "producer_completing" not in settled
    assert settled["run_id"] == c.manifest["run_id"]


def queued_frame(c, monkeypatch):
    """One queued frame whose entry is already retired: no conversion runs."""
    processor = {"path": "fixture-native", "stamp": [1, 2, 3, 4, 5]}
    monkeypatch.setattr(legacy, "_processor_identity", lambda: processor)
    root, selection = viewer._root(c.workspace), viewer._selection()
    authority = ra._sha(c.events.read_bytes().splitlines(keepends=True)[1])
    rw._write(viewer._entry_path(root, c.record["id"], 2, selection),
              {"schema": viewer.SCHEMA, "job_id": c.record["id"], "state": "failed", "sequence": 2,
               "domain": 1, **selection, "commit_sha256": authority, "processor": processor,
               "error": "fixture: this frame was already attempted by this processor"})
    viewer._save_queue(root, c.record["id"], [{**selection, "domain": 1, "sequence": 2,
        "run_id": c.manifest["run_id"], "commit_sha256": authority}])
    return root


def test_viewer_worker_keeps_the_queue_through_the_runner_exit_window(transition, monkeypatch):
    c = transition.launch()
    root = queued_frame(c, monkeypatch)
    with watching(c, monkeypatch, module=viewer, receipt=viewer_receipt,
                  start=lambda **options: viewer.worker(c.workspace, **options)) as w:
        expect_wait(w)
        # The pre-patch worker wrote a terminal failed receipt here and then
        # emptied this queue, destroying the user's selected frames.
        assert [row["sequence"] for row in viewer._queue(root, c.record["id"])] == [2]
        assert not viewer_receipt(c).exists()
        w.resume()
        w.clock.now += ra.COMPLETION_POLL_SECONDS
        expect_wait(w)
        c.publish()
        w.resume()
        w.clock.now += ra.COMPLETION_POLL_SECONDS
        assert expect_exit(w, 0)["state"] == "idle"
        assert not viewer._queue(root, c.record["id"])


def test_viewer_worker_cancellation_keeps_the_queued_frames(transition, monkeypatch):
    c = transition.launch()
    root = queued_frame(c, monkeypatch)
    with watching(c, monkeypatch, module=viewer, receipt=viewer_receipt, cancellation=True,
                  start=lambda **options: viewer.worker(c.workspace, **options)) as w:
        expect_wait(w)
        w.cancel()
        w.resume()
        receipt = expect_exit(w, 2)
        assert receipt["state"] == "cancelled" and receipt["done"] is False
        assert [row["sequence"] for row in viewer._queue(root, c.record["id"])] == [2]


def test_viewer_door_waits_with_the_watcher_instead_of_refusing(transition):
    c = transition.launch()
    request = {"schema": "gpuwm.remote.request.v1", "action": "processed-frame-v2",
               "workspace": str(c.workspace), "job": c.record["id"], "domain": 1, "sequence": 2}
    value = viewer.catalog(request, c.workspace, start=False)
    assert value["waiting"] and value["producer_completing"] and value["state"] == "waiting_for_output"
    c.publish()
    settled = viewer.catalog(request, c.workspace, start=False)
    assert "producer_completing" not in settled and settled["run_id"] == c.manifest["run_id"]


WATCHERS = {
    "preparation": (preparation, preparation_receipt, lambda c, monkeypatch: None,
                    lambda c: lambda **options: preparation.worker(c.workspace, c.record["id"], **options)),
    "plots": (plots, plots_receipt, lambda c, monkeypatch: published_plot(c),
              lambda c: lambda **options: plots.worker(c.workspace, c.record["id"], **options)),
    "viewer": (viewer, viewer_receipt, lambda c, monkeypatch: queued_frame(c, monkeypatch),
               lambda c: lambda **options: viewer.worker(c.workspace, **options)),
}


@pytest.mark.parametrize("name", list(WATCHERS))
def test_a_runner_without_a_receipt_keeps_every_watcher_pending(transition, monkeypatch, name):
    """An unprovable window is a pending job, never a terminal receipt.

    This runner exits before its wrapper records a receipt, so the window can
    never be proved for this job. Its refusal sends the reader to the same way
    out as the rest of the window, retrieving the frames once the job reports a
    terminal state; a terminal done receipt here closed that way out, because
    ensure() refuses to relaunch a watcher whose receipt says done.
    """
    module, receipt, prepare_case, start = WATCHERS[name]
    c = transition.launch(receipt=False)
    prepare_case(c, monkeypatch)
    with watching(c, monkeypatch, module=module, receipt=receipt, start=start(c)) as w:
        expect_wait(w)
        pending = rw._json(receipt(c))
        assert pending["done"] is False and pending["state"] == "waiting_for_producer_completion"
        assert "completion race" in pending["error"] and "terminal state" in pending["error"]
        w.resume()
        w.clock.now += ra.COMPLETION_POLL_SECONDS
        expect_wait(w)
        assert rw._json(receipt(c))["done"] is False
        # The way out the refusal names: the wrapper settles, the job reports a
        # terminal state, and this watcher finishes the work it kept.
        c.publish()
        w.resume()
        w.clock.now += ra.COMPLETION_POLL_SECONDS
        assert expect_exit(w, 0)["done"] is True


def test_a_pending_window_receipt_still_relaunches_its_watcher(transition, monkeypatch):
    """The gallery and the background map preparation reopen for this job.

    ensure() refuses to relaunch a watcher whose receipt says done, which is
    what made the terminal receipt close the way out. The receipt this window
    writes is the loop's own, taken from the running loop and restored after
    the test harness stops it.
    """
    c = transition.launch(receipt=False)
    published_plot(c)
    for module, receipt, start in (
            (preparation, preparation_receipt,
             lambda **options: preparation.worker(c.workspace, c.record["id"], **options)),
            (plots, plots_receipt,
             lambda **options: plots.worker(c.workspace, c.record["id"], **options))):
        with watching(c, monkeypatch, module=module, receipt=receipt, start=start) as w:
            expect_wait(w)
            pending = rw._json(receipt(c))
            assert pending["done"] is False
        rw._write(receipt(c), pending)
        launched = []
        monkeypatch.setattr(module.subprocess, "Popen",
                            lambda *args, **kwargs: launched.append(args[0]))
        module.ensure(c.workspace, c.record["id"])
        assert launched, module.__name__


def test_a_window_without_a_receipt_ends_when_its_wrapper_does(transition, monkeypatch):
    """The pending wait is bounded by the wrapper whose liveness defines it.

    No timer bounds this wait, because the window exists only while the durable
    status reads running, and that state is read from the wrapper's own live
    process identity on every pass. A wrapper that dies without publishing a
    result ends the window as an ownership failure, which is terminal.
    """
    c = transition.launch(receipt=False)
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        assert rw._json(preparation_receipt(c))["done"] is False
        c.wrapper.kill()
        assert c.wrapper.wait(timeout=WATCHDOG) != 0
        w.resume()
        assert "ownership" in expect_done(w, 2)["error"]


@pytest.mark.parametrize("settled", [False, True], ids=["during-wait", "at-settlement"])
def test_a_receipt_that_vanishes_under_a_held_wait_is_not_the_race(transition, monkeypatch, settled):
    """Evidence this wait already hashed cannot become a reason to keep waiting."""
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        (c.directory / "runner.json").unlink()
        if settled:
            c.publish()
        w.resume()
        error = expect_done(w, 2)["error"]
        assert "already hashed by this wait" in error and "completion race" in error


def test_a_door_refusal_that_is_not_the_transition_keeps_its_own_wording(transition):
    c = transition.launch()
    # In the same window, but without the receipt the transition needs: the
    # door must fall back to the refusal it has always raised here.
    (c.directory / "runner.json").unlink()
    request = {"schema": "gpuwm.remote.request.v1", "action": "processed-frame-v2",
               "workspace": str(c.workspace), "job": c.record["id"], "domain": 1, "sequence": 2}
    with pytest.raises(ValueError, match="belongs to this active remote job") as error:
        viewer.catalog(request, c.workspace, start=False)
    assert not isinstance(error.value, ra.ProducerCompletionPending)


def test_direct_readers_still_refuse_the_transition(transition):
    c = transition.launch()
    state = rw._status(c.directory)
    with pytest.raises(ValueError, match="belongs") as error:
        ra.bound_manifest(c.record, state)
    assert not isinstance(error.value, ra.ProducerCompletionPending)
    with pytest.raises(ra.ProducerCompletionPending):
        ra.bound_manifest(c.record, state, job_directory=c.directory)


def test_permission_failure_is_not_missing_process_proof(transition, monkeypatch):
    c = transition.launch()
    def denied(_pid, _flags):
        raise PermissionError("cannot inspect process")
    monkeypatch.setattr(ra.os, "pidfd_open", denied)
    with watching(c, monkeypatch) as w:
        assert "cannot inspect" in expect_done(w, 2)["error"]


def test_commits_changing_between_job_read_and_settlement_are_refused(transition, monkeypatch):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        c.publish()
        original = preparation.legacy._job
        def interleave(*args, **kwargs):
            record, state, bound, commits = original(*args, **kwargs)
            # Even a transient divergent read may not be admitted against a
            # restored on-disk proof. Exercise the commits-vs-proof comparison.
            commits[0][0]["geometry"] = {"other": True}
            return record, state, bound, commits
        monkeypatch.setattr(preparation.legacy, "_job", interleave)
        w.resume()
        assert "commits changed" in expect_done(w, 2)["error"]


@pytest.mark.parametrize("settled", [False, True], ids=["during-wait", "at-settlement"])
@pytest.mark.parametrize("which", ["parent-manifest", "producer-manifest", "pointer", "parent-resolved", "producer-resolved"])
def test_hosted_bindings_are_revalidated(transition, monkeypatch, settled, which):
    c = transition.launch(prepared=True)
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        if settled:
            c.publish()
        if which == "parent-manifest":
            rewrite(c.manifest_path, lambda value: value.update(run_id="new-parent"))
        elif which == "producer-manifest":
            rewrite(c.producer / "run-manifest.json", lambda value: value.update(run_id="new-producer"))
        elif which == "pointer":
            c.pointer.unlink()
        else:
            path = c.parent_events if which == "parent-resolved" else c.events
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            rows[0]["config_sha256"] = "0" * 64
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        w.resume()
        expect_done(w, 2)
        assert not viewer._queue(viewer._root(c.workspace), c.record["id"])


@pytest.mark.parametrize("result", [
    {"token": "0" * 64}, {"state": "failed"}, {"state": "stopped"},
    {"exit_code": 1}, {"exit_code": True}, {"error": "cleanup failed"}])
def test_conflicting_terminal_result_is_never_success(transition, monkeypatch, result):
    c = transition.launch()
    with watching(c, monkeypatch) as w:
        expect_wait(w)
        c.publish()
        rewrite(c.directory / "result.json", lambda value: value.update(result))
        w.resume()
        expect_done(w, 2)


def test_wrapper_receipt_binds_the_original_job_not_a_rewritten_one(transition, monkeypatch):
    c = transition.launch()
    receipt = rw._json(c.directory / "runner.json")
    assert receipt["identity"] == c.runner_identity
    assert receipt["owner"] == rw._json(c.directory / "started.json")["identity"]
    assert receipt["job_sha256"] == rw._file_sha(c.directory / "job.json")
    changed(c, "plan_coherent")  # Before any consumer has observed the transition.
    with watching(c, monkeypatch) as w:
        assert "ownership" in expect_done(w, 2)["error"]


SURVIVOR = r"""
import signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
"""

MEASURED_RUNNER = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from gpuwm import remote_worker as rw
record = rw._record(Path(sys.argv[1]))
root = Path(record["outdir"])
manifest = {"schema": "gpuwm.run-manifest.v1", "route": "experiment", "run_id": "settle-native-run",
    "pid": os.getpid(), "started_at_utc": rw._now(), "run_dir": str(root), "outputs_dir": str(root),
    "plan_source": record["snapshot_plan"], "plan_sha256": record["plan_sha256"],
    "events_path": str(root / "events.jsonl")}
rw._write(root / "run-manifest.json", manifest)
frame = root / "wrfout_d01_fixture"
frame.write_bytes(b"protocol-only committed bytes, not weather")
base = {"schema_version": "gpuwm.run-plan.event.v1", "emitted_unix_ms": int(time.time()*1000)}
events = [
    {**base, "sequence": 1, "event": "resolved_plan", "config_source": record["snapshot_config"],
     "config_sha256": record["config_sha256"]},
    {**base, "sequence": 2, "event": "output_committed", "domain": 1, "path": str(frame),
     "size_bytes": frame.stat().st_size, "valid_time": "2026-09-07T18:00:00Z"},
    {**base, "sequence": 3, "event": "completed", "dry_run": False, "run_dir": str(root),
     "receipt_path": str(root / "run-manifest.json"), "outputs_committed": 1,
     "receipts": {}, "summary": {"executed": True}}]
(root / "events.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
if sys.argv[2] == "descendants":
    # An owned descendant that ignores the first two stages: the wrapper pays
    # the whole escalation ladder after this runner is already gone.
    subprocess.Popen([sys.executable, "-c", sys.argv[3]], stdin=subprocess.DEVNULL)
    time.sleep(1)
(root / "runner-exit.json").write_text(json.dumps({"exit_unix_ns": time.time_ns()}))
raise SystemExit(0)
'''

PLAIN_WRAPPER = r'''
import os, sys
from pathlib import Path
from gpuwm import remote_worker as rw
raise SystemExit(rw.run_worker(Path(sys.argv[1]), os.environ[rw.TOKEN_ENV]))
'''


@pytest.mark.parametrize("mode,floor", [("prompt", 0.0), ("descendants", 5.0)], ids=["prompt", "full-ladder"])
def test_measured_wrapper_settlement_fits_the_completion_budget(tmp_path, mode, floor):
    """Observe the interval COMPLETION_SECONDS budgets, on the real clock.

    Measures the wall interval between the runner's last instruction and the
    wrapper's publication of result.json: the receipt's st_mtime_ns minus a
    CLOCK_REALTIME stamp the runner writes immediately before SystemExit. Both
    stamps come from one host's CLOCK_REALTIME, so the difference is the window
    a reader waits through plus the runner's own last write. Nothing here is
    faked: this is the one case in this file that runs on the real clock, so
    the budget's margin is observed rather than asserted.

    The full-ladder case leaves an owned descendant that ignores SIGINT and
    SIGTERM, which is the worst case the budget exists for: three cleanup
    stages of three seconds plus a one second child wait.
    """
    directory, output, config, plan = job_inputs(tmp_path, "settle-fixture")
    record = job_record(tmp_path, directory, output, config, plan,
        [sys.executable, "-c", MEASURED_RUNNER, str(directory), mode, SURVIVOR])
    environment = {**os.environ, rw.TOKEN_ENV: record["token"],
                   "PYTHONPATH": str(Path(rw.__file__).resolve().parents[1])}
    log = (directory / "test-wrapper.log").open("wb")
    try:
        wrapper = subprocess.Popen([sys.executable, "-c", PLAIN_WRAPPER, str(directory)],
            env=environment, stdout=log, stderr=log, start_new_session=True)
    finally:
        log.close()
    try:
        assert wrapper.wait(timeout=6 * WATCHDOG) == 0, (directory / "test-wrapper.log").read_text()
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=WATCHDOG)
    result = directory / "result.json"
    assert rw._json(result)["state"] == "completed"
    exited = json.loads((output / "runner-exit.json").read_text())["exit_unix_ns"]
    settle = (result.stat().st_mtime_ns - exited) / 1e9
    print(f"measured runner-exit to result.json [{mode}]: {settle:.3f} s "
          f"(budget {preparation.COMPLETION_SECONDS} s)")
    assert floor < settle < preparation.COMPLETION_SECONDS
