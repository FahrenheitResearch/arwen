"""Retry starts and failures are durable and visible before replay begins."""
import json
from types import SimpleNamespace

import pytest

from gpuwm.core.health import HealthCheckError
from gpuwm.io import restart
from gpuwm.stability_recovery import NestedHealthRecovery
from test_stability_recovery import _checkpoint, _health_error


def _events(capsys):
    prefix = "gpuwm stability recovery: "
    return [json.loads(line[len(prefix):])
            for line in capsys.readouterr().out.splitlines()
            if line.startswith(prefix)]


def test_retry_attempt_is_durable_and_visible_before_restore(
        monkeypatch, tmp_path, capsys):
    model, exp, checkpoint = _checkpoint(monkeypatch, tmp_path)
    notices = []
    observer = SimpleNamespace(
        restarting=lambda reason: None,
        warn=lambda code, message, **record: notices.append((code, record)))
    recovery = NestedHealthRecovery(model=model, experiment=exp,
        output_directory=tmp_path, observer=observer)
    restore = restart.restore_tree_restart
    seen = []

    def checked_restore(path, tree, **kwargs):
        receipt = json.loads(recovery.receipt_path.read_text())
        assert receipt["status"] == "RETRYING"
        assert receipt["failures"][0]["phase"] == "post-d01-sync.d02"
        assert receipt["failures"][0]["variable"] == "w"
        assert receipt["attempts"][0]["status"] == "VALIDATING_RESTORE"
        assert receipt["attempts"][0]["checkpoint"] == str(checkpoint.resolve())
        seen.extend(_events(capsys))
        assert seen[-1]["phase"] == "validating_restore"
        assert seen[-1]["retry"] == 1
        assert notices[-1][1]["recovery"]["phase"] == "validating_restore"
        return restore(path, tree, **kwargs)

    monkeypatch.setattr(restart, "restore_tree_restart", checked_restore)
    calls = []

    def execute(_active):
        calls.append(1)
        if len(calls) == 1:
            model.root.clock.ticks = 3700
            model.root.clock.elapsed_seconds = 3700
            raise _health_error()
        return "replayed"

    assert recovery.run(execute) == "replayed"
    seen.extend(_events(capsys))
    assert [(event["retry"], event["phase"]) for event in seen] == [
        (1, "validating_restore"), (1, "resumed")]
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["status"] == "RECOVERED"
    assert len(receipt["attempts"]) == 1


def test_failed_restore_preserves_visible_retry_and_cause(
        monkeypatch, tmp_path, capsys):
    from gpuwm.prepared_domain_tree_forecast import _write_failed_run_receipt

    model, exp, _ = _checkpoint(monkeypatch, tmp_path)
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    original_configs = {gid: node.cfg for gid, node in model.nodes_by_grid_id.items()}

    def refused_restore(*args, **kwargs):
        raise ValueError("checkpoint changed after manifest validation")

    monkeypatch.setattr(restart, "restore_tree_restart", refused_restore)
    error = _health_error()

    def execute(_active):
        model.root.clock.ticks = 3700
        model.root.clock.elapsed_seconds = 3700
        raise error

    with pytest.raises(HealthCheckError) as observed:
        recovery.run(execute)
    assert observed.value is error
    assert {gid: node.cfg for gid, node in model.nodes_by_grid_id.items()} == original_configs
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["status"] == "REFUSED"
    assert receipt["attempts"][0]["status"] == "REFUSED"
    assert "checkpoint changed" in receipt["attempts"][0]["refusal"]
    assert receipt["failures"][0]["domain"] == "d02"
    assert receipt["failures"][0]["variable"] == "w"
    events = _events(capsys)
    assert [(event["retry"], event["phase"]) for event in events] == [
        (1, "validating_restore"), (1, "refused")]
    assert "checkpoint changed" in events[-1]["refusal"]
    _write_failed_run_receipt(tmp_path, error)
    failed = json.loads((tmp_path / "evidence" / "failed-run-receipt.json").read_text())
    assert failed["stability_recovery_receipt"] == str(recovery.receipt_path.resolve())


def test_two_retries_have_distinct_ids_and_visible_phases(
        monkeypatch, tmp_path, capsys):
    model, exp, _ = _checkpoint(monkeypatch, tmp_path)
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)

    def execute(_active):
        model.root.clock.ticks = 3700
        model.root.clock.elapsed_seconds = 3700
        raise _health_error()

    with pytest.raises(HealthCheckError):
        recovery.run(execute)
    events = _events(capsys)
    assert [(event["retry"], event["phase"]) for event in events] == [
        (1, "validating_restore"), (1, "resumed"),
        (2, "validating_restore"), (2, "resumed")]
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["maximum_retries"] == 2
    assert len(receipt["attempts"]) == 2
    assert len(receipt["failures"]) == 3
    assert receipt["status"] == "EXHAUSTED"


def test_interrupted_restore_closes_visible_attempt_and_propagates_interrupt(
        monkeypatch, tmp_path, capsys):
    model, exp, _ = _checkpoint(monkeypatch, tmp_path)
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    original_configs = {gid: node.cfg for gid, node in model.nodes_by_grid_id.items()}

    def interrupted_restore(*args, **kwargs):
        raise KeyboardInterrupt("stop during checkpoint restore")

    monkeypatch.setattr(restart, "restore_tree_restart", interrupted_restore)

    def execute(_active):
        model.root.clock.ticks = 3700
        model.root.clock.elapsed_seconds = 3700
        raise _health_error()

    with pytest.raises(KeyboardInterrupt, match="stop during checkpoint restore"):
        recovery.run(execute)
    assert {gid: node.cfg for gid, node in model.nodes_by_grid_id.items()} == original_configs
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["status"] == "INTERRUPTED"
    assert receipt["attempts"][0]["status"] == "INTERRUPTED"
    assert receipt["failures"][0]["domain"] == "d02"
    assert "KeyboardInterrupt" in receipt["terminal_error"]
    assert "stop during checkpoint restore" in receipt["attempts"][0]["terminal_error"]
    assert [(event["retry"], event["phase"]) for event in _events(capsys)] == [
        (1, "validating_restore"), (1, "interrupted")]
