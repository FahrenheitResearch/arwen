"""An interruption request survives a failed write and is consumed once."""
from dataclasses import replace

import pytest

from gpuwm.core.restart_request import RESTART_REQUEST_ENV, RestartRequest


def test_unconfigured_request_does_no_filesystem_work(monkeypatch):
    monkeypatch.delenv(RESTART_REQUEST_ENV, raising=False)
    request = RestartRequest()
    assert request.path is None
    assert not request.pending()
    request.acknowledge()


@pytest.mark.parametrize("periodic", [False, True])
def test_request_runs_at_synchronized_boundary_without_duplicate_periodic_write(
        tmp_path, monkeypatch, periodic):
    from test_model import _model
    from gpuwm.core.clock import build_schedule, resolve_clock
    from gpuwm.core.model import PERIOD_BEGIN, execute_experiment

    exp, model = _model()
    if periodic:
        exp = replace(exp, restart_interval_s=6.0)
        model.schedule = build_schedule(exp, resolve_clock(exp))
        clocks = model.schedule.clock.clocks()
        for node in model.walk_parent_first():
            node.clock = clocks[node.cfg.grid_id]
    path = tmp_path / "restart.request"
    path.write_text("interruption")
    monkeypatch.setenv(RESTART_REQUEST_ENV, str(path))
    monkeypatch.setattr("gpuwm.core.dycore.step", lambda *a, **k: None)
    monkeypatch.setattr("gpuwm.core.model._ask_the_checkpoints_question", lambda n: None)
    writes = []

    def write(tree, ticks):
        assert tree._runtime_status.schedule_cursor == PERIOD_BEGIN
        assert tree._runtime_status.pending_feedback == 0
        assert {node.clock.ticks for node in tree.walk_parent_first()} == {ticks}
        writes.append(ticks)

    execute_experiment(model, restart_handler=write, validate_state=False,
                       pool_trim_per_period=False)
    assert writes == (list(range(6, 61, 6)) if periodic else [6])
    assert not path.exists()


def test_requested_restart_event_retains_measured_duration(tmp_path):
    from test_progress_log import START, make_log
    from gpuwm.progress_log import STEP_LOG_FILENAME, read_step_log

    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"restart")
    log, _ = make_log(tmp_path)
    log.restart_written(domain=1, valid_time=START, path=checkpoint,
                        wall_seconds=1.23456789)
    log.close(status="SUCCESS")
    record = next(r for r in read_step_log(tmp_path / STEP_LOG_FILENAME)
                  if r["event"] == "restart_written")
    assert record["wall_seconds"] == 1.234568
    assert record["size_bytes"] == 7


def test_pending_request_is_acknowledged_only_after_success(tmp_path, monkeypatch):
    path = tmp_path / "restart.request"
    monkeypatch.setenv(RESTART_REQUEST_ENV, str(path))
    request = RestartRequest()
    assert not request.pending()
    path.write_text("spot interruption")
    assert request.pending()
    # A writer exception never calls acknowledge, so a new boundary can
    # retry and the worker cannot mistake the request for a durable file.
    assert RestartRequest().pending()
    request.acknowledge()
    assert not path.exists()
    assert not request.pending()
