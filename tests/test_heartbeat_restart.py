"""The heartbeat's restart record: a new attempt in the same worker.

A head-bound forecast whose streamed interval moved its terrain clock sets
its attempt aside and runs again on the sealed preparation, in the same
process (the tree runner; the single-domain head-bound runner reuses the
same record).  Its step and model time go back to zero, and a supervisor
that holds each record to the one before it read that as a worker going
backward: `gpuwm go`'s ``ForecastWatchdog`` stopped the stage with 124.

``RuntimeHeartbeat.restarting`` publishes ``preparing:restart``, and it and
every record after it carry ``restart = {attempt, reason}``, which
``supervisor._heartbeat_regression`` takes as the start of a new attempt.
"""

from __future__ import annotations

import os

import pytest

from gpuwm import supervisor
from gpuwm.forecast_supervisor import ForecastHeartbeat, ForecastWatchdog


def _base(**overrides):
    fields = dict(schema=supervisor.HEARTBEAT_SCHEMA, run_id="r",
                  config_digest="d", pid=1, started_at_utc="s",
                  model_elapsed_seconds=0.0, outer_step=0,
                  last_durable_wrfout=None, last_checkpoint=None)
    fields.update(overrides)
    return fields


def _record(second, status, **overrides):
    return supervisor.Heartbeat(
        updated_at_utc=f"2026-09-30T00:00:{second:02d}+00:00",
        status=status, **_base(**overrides))


def _second(reason="the clock moved"):
    return {"attempt": 2, "reason": reason}


def test_the_restart_record_is_declared_and_round_trips(tmp_path):
    record = _record(1, supervisor.RESTART_STATUS, restart=_second())
    path = supervisor.write_heartbeat(tmp_path / "run-progress.json", record)
    assert supervisor.read_heartbeat(path) == record
    # A first attempt's records keep their published field set.
    assert "restart" not in _record(1, "integrating").as_dict()
    with pytest.raises(ValueError, match="declares the attempt"):
        _record(1, supervisor.RESTART_STATUS)
    for bad in ({"attempt": 1, "reason": "x"}, {"attempt": True, "reason": "x"},
                {"attempt": "2", "reason": "x"}, {"attempt": 2}):
        with pytest.raises(ValueError, match="restart"):
            _record(1, "integrating", restart=bad)


def test_a_new_attempt_may_start_again_from_zero():
    last = _record(1, "integrating", outer_step=50,
                   model_elapsed_seconds=3600.0)
    for status in (supervisor.RESTART_STATUS,
                   "preparing:restore-prepared-domain-tree",
                   "waiting:preparation", "integrating"):
        wait = ({"on": "preparation", "lead": None, "expected_at": None,
                 "late_at": None, "since_utc": "x"}
                if status.startswith("waiting:") else None)
        current = _record(2, status, restart=_second(), wait=wait)
        assert supervisor._heartbeat_regression(last, current) is None


def test_the_records_of_one_attempt_still_go_forward():
    restarted = _record(1, "integrating", outer_step=10, restart=_second())
    back = _record(2, "integrating", outer_step=9, restart=_second())
    assert "outer_step moved backward" in supervisor._heartbeat_regression(
        restarted, back)
    assert "status moved backward" in supervisor._heartbeat_regression(
        restarted, _record(2, "preparing:first-step", outer_step=10,
                           restart=_second()))
    # An attempt never goes back to an earlier one, and a record without
    # the declaration after one with it is an earlier attempt.
    assert "attempt moved backward" in supervisor._heartbeat_regression(
        restarted, _record(2, "integrating", outer_step=11))
    third = _record(3, supervisor.RESTART_STATUS,
                    restart={"attempt": 3, "reason": "again"})
    assert supervisor._heartbeat_regression(restarted, third) is None
    assert "attempt moved backward" in supervisor._heartbeat_regression(
        third, _record(4, "integrating", restart=_second()))
    # A published end is still the end.
    done = _record(1, "complete", outer_step=50)
    assert "terminal status" in supervisor._heartbeat_regression(
        done, _record(2, supervisor.RESTART_STATUS, restart=_second()))


def test_the_runtime_heartbeat_publishes_the_restart(tmp_path):
    path = tmp_path / "run-progress.json"
    heartbeat = supervisor.RuntimeHeartbeat(
        path, run_id="r", config_sha256="d", started_at_utc="s")
    wrfout = tmp_path / "wrfout_d01"
    wrfout.write_bytes(b"frame")
    heartbeat(model_elapsed_seconds=600.0, outer_step=40,
              last_durable_wrfout=wrfout, last_checkpoint=None)
    assert heartbeat.attempt == 1
    first = supervisor.read_heartbeat(path)
    supervisor.restart_attempt(heartbeat, "the clock moved")
    restart = supervisor.read_heartbeat(path)
    assert restart.status == supervisor.RESTART_STATUS
    assert restart.restart == {"attempt": 2, "reason": "the clock moved"}
    assert (restart.outer_step, restart.model_elapsed_seconds,
            restart.last_durable_wrfout) == (0, 0.0, None)
    assert supervisor._heartbeat_regression(first, restart) is None
    # Every later record declares the attempt it belongs to.
    heartbeat.waiting("preparation", since_utc="x")
    assert supervisor.read_heartbeat(path).restart["attempt"] == 2
    heartbeat.waited()
    assert supervisor.read_heartbeat(path).status == supervisor.RESTART_STATUS
    heartbeat.preparing("restore-prepared-domain-tree")
    heartbeat(model_elapsed_seconds=0.0, outer_step=0,
              last_durable_wrfout=None, last_checkpoint=None)
    assert supervisor.read_heartbeat(path).restart["attempt"] == 2
    supervisor.restart_attempt(heartbeat, "again")
    assert supervisor.read_heartbeat(path).restart == {
        "attempt": 3, "reason": "again"}


def test_a_callback_without_the_hook_is_left_alone():
    supervisor.restart_attempt(object(), "the clock moved")
    heard = []
    supervisor.restart_attempt(
        type("Observer", (), {"restarting": lambda self, why: heard.append(
            why)})(), "the clock moved")
    assert heard == ["the clock moved"]


def test_a_new_attempt_enters_again_under_the_forecast_watchdog(tmp_path):
    """The new attempt's first step beat is entry, as the process's was."""

    outdir = tmp_path / "run"
    outdir.mkdir()
    clock = type("Clock", (), {"now": 0.0,
                               "__call__": lambda self: self.now})()
    watchdog = ForecastWatchdog(
        ["python", "-m", "gpuwm.prepared_domain_tree_forecast",
         "--outdir", str(outdir)], clock=clock)
    heartbeat = ForecastHeartbeat(
        outdir / supervisor.HEARTBEAT_NAME, run_id=watchdog.run_id,
        config_sha256=watchdog.digest, started_at_utc=watchdog.started_at)
    statuses = []

    def beat(step):
        heartbeat(model_elapsed_seconds=10.0 * step, outer_step=step,
                  last_durable_wrfout=None, last_checkpoint=None)
        statuses.append(supervisor.read_heartbeat(heartbeat.path).status)
        assert watchdog.check(os.getpid()) is None
        clock.now += 1.0

    for step in range(4):
        beat(step)
    heartbeat.restarting("the clock moved")
    assert watchdog.check(os.getpid()) is None
    # Nothing to time while the new attempt prepares, however long.
    clock.now += 600.0
    assert watchdog.check(os.getpid()) is None
    for step in range(3):
        beat(step)
    assert statuses == ["preparing:first-step", "integrating", "integrating",
                        "integrating", "preparing:first-step", "integrating",
                        "integrating"]


# -- gpuwm run's supervisor: a new attempt is timed as preparation ----------


def _restarted(status, step=0, **fields):
    return {"status": status, "step": step,
            "restart": {"attempt": 2, "reason": "the clock moved"}, **fields}


def _restart_script(silent_polls):
    # Each poll moves the fake clock 61 s, so two silent polls pass the
    # 120 s step bound the first attempt's steps set.
    return [[
        {"status": "integrating", "step": 1},
        {"status": "integrating", "step": 2},
        _restarted(supervisor.RESTART_STATUS),
        *({} for _ in range(silent_polls)),
        _restarted("preparing:restore-prepared-domain-tree"),
        {}, {},
        _restarted("integrating", 1),
        _restarted("complete", 3, exit=0),
    ]]


def test_the_run_supervisor_times_a_new_attempt_as_a_preparation(
        monkeypatch, tmp_path):
    """``supervise_experiment`` takes a restart as a new preparation.

    THE BREAKAGE: ``integrating_seen`` stayed set from the first attempt,
    so the second attempt's restore, silent for longer than the step
    bound, was stopped as "worker integrating heartbeat became stale"
    while the worker recovered as designed.
    """

    from test_supervisor import _install_scripted_supervisor

    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, _restart_script(3), clock_step=61.0)
    result = supervisor.supervise_experiment(
        config, tmp_path / "out", max_restarts=0, poll_seconds=0.05)
    assert result.attempts == 1 and len(processes) == 1
    assert processes[0].terminated is False
    assert result.heartbeat.status == "complete"
    assert result.heartbeat.restart == {"attempt": 2,
                                        "reason": "the clock moved"}


def test_the_run_supervisor_still_bounds_a_new_attempts_preparation(
        monkeypatch, tmp_path):
    """CONTROL: the new attempt's preparation has the preparation's bound."""

    from test_supervisor import _install_scripted_supervisor

    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, _restart_script(4), clock_step=61.0)
    with pytest.raises(supervisor.SupervisorError,
                       match="preparation timed out in preparing:restart"):
        supervisor.supervise_experiment(
            config, tmp_path / "out", max_restarts=0, poll_seconds=0.05,
            prep_timeout_seconds=200.0)
    assert processes[0].terminated


# -- run-plan's observers: renders halted, heartbeat told, renders re-armed --


def _renders_sleep(monkeypatch, seconds):
    import subprocess
    import sys

    from gpuwm import first_products, go_cli, render

    monkeypatch.setattr(render, "announce_missing_basemap",
                        lambda *a, **k: None)
    monkeypatch.setattr(go_cli, "render_command", lambda *a, **k: [
        sys.executable, "-S", "-c", f"import time; time.sleep({seconds})"])
    if os.name == "nt":
        monkeypatch.setattr(first_products, "_own_group_options", lambda: {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW})


def _render_process(worker):
    import time

    from gpuwm import first_products

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with first_products._RUNNING_LOCK:
            process = first_products._RUNNING.get(worker._thread.ident)
        if process is not None:
            return process
        time.sleep(0.01)
    raise AssertionError("the analysis render never started")


@pytest.mark.parametrize("through_go", [False, True])
def test_a_run_plan_restart_ends_its_renders_and_arms_new_ones(
        tmp_path, monkeypatch, through_go):
    """``RunObserver.restarting``, directly and through go's observer.

    The runner calls :func:`supervisor.restart_attempt` on its observer
    before it moves the attempt's folder aside.  Under run-plan and
    ``gpuwm go`` that observer is ``RunObserver`` (behind ``_GoObserver``
    for go), which must tell the heartbeat, end the renders reading the
    attempt's frames and wait for their processes (a picture file open
    in the folder stops the move on Windows), arm fresh renders for the
    new attempt and say the restart on the event stream.
    """

    import json
    from datetime import datetime

    from gpuwm.runplan import EventStream, RunObserver, _GoObserver

    _renders_sleep(monkeypatch, 30)
    heartbeat = supervisor.RuntimeHeartbeat(
        tmp_path / supervisor.HEARTBEAT_NAME, run_id="r", config_sha256="d",
        started_at_utc="s")
    observer = RunObserver(EventStream(tmp_path / "events.jsonl",
                                       mirror=None), heartbeat=heartbeat)
    observer.arm_first_products({"run": tmp_path, "render": tmp_path / "png",
                                 "render_products": "t2"})
    frame = tmp_path / "wrfout_d01_2026-09-30_00_00_00"
    frame.write_bytes(b"attempt one history fixture")
    observer.output_committed(domain=1, valid_time=datetime(2026, 9, 30),
                              path=frame)
    heartbeat(model_elapsed_seconds=0.0, outer_step=0,
              last_durable_wrfout=frame, last_checkpoint=None)
    early, live = observer.first_products, observer.live_products
    assert early is not None and live is not None
    process = _render_process(early)
    try:
        assert process.poll() is None and observer.outputs_committed == 1
        supervisor.restart_attempt(
            _GoObserver(observer) if through_go else observer,
            "the clock moved")
        assert process.poll() is not None, (
            "the restart left the attempt's render drawing")
        assert early._ended.ended and not live.running
        # Fresh renders for the new attempt's frames, none running yet.
        assert observer.first_products not in (None, early)
        assert observer.live_products not in (None, live)
        assert observer.outputs_committed == 0
        record = supervisor.read_heartbeat(heartbeat.path)
        assert record.status == supervisor.RESTART_STATUS
        assert record.restart == {"attempt": 2, "reason": "the clock moved"}
        warnings = [json.loads(line) for line in (
            tmp_path / "events.jsonl").read_text(
                encoding="utf-8").splitlines()]
        warnings = [event for event in warnings
                    if event.get("event") == "warning"]
        assert [event["code"] for event in warnings] == ["forecast_restarted"]
        assert warnings[0]["reason"] == "the clock moved"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        observer.stop_live_products(halt=True)
