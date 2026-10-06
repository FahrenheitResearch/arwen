"""CPU checkpoint I/O, retry discrimination and owned-output rollback."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from fractions import Fraction
import json
from pathlib import Path
from threading import Condition
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.health import HealthCheckError, validate_fields_cpu
from gpuwm.io import restart
from gpuwm.io.wrfout import PerDomainWrfoutWriters, wrfout_filename
from gpuwm.output_identity import completed_file_record
from gpuwm.progress_log import write_frame_marker
from gpuwm.stability_recovery import NestedHealthRecovery, reduced_clock_policy
from test_restart import _cfg, _sealed_tree_fixture


@dataclass(frozen=True)
class Domain:
    grid_id: int
    parent_id: int
    run: object
    start_time: object = None
    i_parent_start: int = 1
    j_parent_start: int = 1
    spawn: object = None
    retire: object = None
    follow: object = None


@dataclass(frozen=True)
class Experiment:
    start_time: object
    domains: tuple
    relocation: object


def _health_error():
    report = validate_fields_cpu({"w": np.array([239.79], np.float32)},
                                phase="post-d01-sync.d02")
    assert not report.ok
    return HealthCheckError(report)


def _checkpoint(monkeypatch, tmp_path):
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200, payload_seed=4,
        run_overrides={"use_adaptive_time_step": True,
                       "max_time_step": 20, "min_time_step": 5})
    for node in model.walk_parent_first():
        cfg = node.cfg
        node.cfg = Domain(cfg.grid_id, cfg.parent_id, cfg.run,
                          i_parent_start=cfg.i_parent_start,
                          j_parent_start=cfg.j_parent_start)
        node.state.w[...] = 0
        node.clock.step_ticks = 10
        node.clock.dt_fp32 = np.float32(10)
        node.clock.elapsed_seconds = 3600
        node.clock.adaptive_state = None
    checkpoint = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))
    experiment = Experiment(start, tuple(node.cfg for node in
                                        model.walk_parent_first()),
                            SimpleNamespace(enabled=False))

    class WValidator:
        def __init__(self, node):
            self.node = node

        def require_healthy(self, *, phase):
            report = validate_fields_cpu({"w": self.node.state.w}, phase=phase)
            if not report.ok:
                raise HealthCheckError(report)

    import gpuwm.core.health as health
    monkeypatch.setattr(health, "health_validator_for_domain",
                        lambda _model, node: WValidator(node))
    return model, experiment, checkpoint


def test_policy_halves_actual_step_and_caps_the_next_adaptive_leg():
    cfg = _cfg(use_adaptive_time_step=True, max_time_step=30,
               min_time_step=5, target_cfl=1.2, target_hcfl=.7)
    reduced, evidence = reduced_clock_policy(
        cfg, failed_dt=Fraction(621, 100), checkpoint_dt=Fraction(7))
    assert Fraction(reduced.max_time_step, reduced.max_time_step_den) == Fraction(31, 10)
    assert Fraction(reduced.starting_time_step, reduced.starting_time_step_den) == Fraction(31, 10)
    assert Fraction(reduced.min_time_step, reduced.min_time_step_den) == Fraction(31, 10)
    assert reduced.target_cfl == .6 and reduced.target_hcfl == .35
    assert evidence["maximum_step_s"] == [30., 3.1]
    assert reduced.w_damping == cfg.w_damping
    assert reduced.epssm == cfg.epssm


def test_real_two_domain_checkpoint_restores_corruption_and_records_retry(
        monkeypatch, tmp_path):
    model, exp, checkpoint = _checkpoint(monkeypatch, tmp_path)
    history = [{"ticks": 3600}, {"ticks": 3700}]
    notices = []
    observer = SimpleNamespace(restarting=notices.append,
        warn=lambda code, message, **record: notices.append((code, record)))
    recovery = NestedHealthRecovery(model=model, experiment=exp,
        output_directory=tmp_path, history=history, observer=observer)
    calls = []

    def execute(active):
        calls.append(active)
        if len(calls) == 1:
            model.node(2).state.w[...] = np.float32(239.79)
            model.root.clock.ticks = 3700
            model.root.clock.elapsed_seconds = 3700
            raise _health_error()
        np.testing.assert_array_equal(model.node(2).state.w, 0)
        assert model.root.clock.ticks == 3600
        assert model.node(2).coupler.valid is False
        assert active.domains[1].run.max_time_step == 5
        assert model.node(2).cfg.run.target_cfl == exp.domains[1].run.target_cfl / 2
        return "clean-replayed-leg"

    assert recovery.run(execute) == "clean-replayed-leg"
    assert len(calls) == 2
    assert history == [{"ticks": 3600}]
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["status"] == "RECOVERED"
    assert receipt["attempts"][0]["checkpoint"] == str(checkpoint.resolve())
    assert receipt["failures"][0]["domain"] == "d02"
    assert any(isinstance(event, tuple) and event[0] == "stability_retry"
               for event in notices)


def test_retry_is_bounded_and_each_cap_shrinks(monkeypatch, tmp_path):
    model, exp, _ = _checkpoint(monkeypatch, tmp_path)
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    calls = []

    def execute(active):
        calls.append(active.domains[0].run.max_time_step /
                     (active.domains[0].run.max_time_step_den or 1))
        model.root.clock.ticks = 3700
        model.root.clock.elapsed_seconds = 3700
        raise _health_error()

    with pytest.raises(HealthCheckError):
        recovery.run(execute)
    assert calls == [20, 5, 2.5]
    receipt = json.loads(recovery.receipt_path.read_text())
    assert receipt["status"] == "EXHAUSTED"
    assert len(receipt["attempts"]) == 2
    assert len(receipt["failures"]) == 3


@pytest.mark.parametrize("failure", [MemoryError("capacity"),
    ValueError("bad config"), FloatingPointError("untyped nonfinite")])
def test_only_typed_full_state_gate_failures_retry(monkeypatch, tmp_path, failure):
    model, exp, _ = _checkpoint(monkeypatch, tmp_path)
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    calls = []
    def execute(_active):
        calls.append(1)
        raise failure
    with pytest.raises(type(failure)) as observed:
        recovery.run(execute)
    assert observed.value is failure
    assert calls == [1] and recovery.receipt["attempts"] == []


@pytest.mark.parametrize("reason", ["missing", "torn", "fixed", "moving"])
def test_unproven_recovery_fails_closed(monkeypatch, tmp_path, reason):
    model, exp, checkpoint = _checkpoint(monkeypatch, tmp_path)
    if reason == "missing":
        model._last_checkpoint = None
    elif reason == "torn":
        child = restart.tree_restart_members(checkpoint)[2]
        child.unlink()
    elif reason == "fixed":
        exp = replace(exp, domains=tuple(replace(dc,
            run=replace(dc.run, use_adaptive_time_step=False)) for dc in exp.domains))
        for dc in exp.domains:
            model.node(dc.grid_id).cfg = dc
    else:
        exp = replace(exp, relocation=SimpleNamespace(enabled=True))
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    def execute(_active):
        raise _health_error()
    with pytest.raises(HealthCheckError):
        recovery.run(execute)
    assert json.loads(recovery.receipt_path.read_text())["status"] == "REFUSED"


def _writer_shell(tmp_path):
    manager = object.__new__(PerDomainWrfoutWriters)
    manager.output_dir = tmp_path / "wrfout"
    manager.output_dir.mkdir()
    paths = []
    for hour in (0, 1, 2):
        path = manager.output_dir / wrfout_filename(datetime(2026, 10, 4, hour), 2)
        path.write_bytes(f"owned frame {hour}".encode())
        paths.append(path)
        write_frame_marker(tmp_path / "ready", domain=2,
                           valid_time=datetime(2026, 10, 4, hour), path=path)
    writer = SimpleNamespace(paths=paths.copy(), _condition=Condition(),
        _completed_records=[completed_file_record(path) for path in paths])
    writer.completed_records = tuple(writer._completed_records)
    manager._writers = {2: writer}
    manager._archived_paths = []
    manager._archived_records = []
    manager._captured_paths = []
    manager._published_paths = set(paths)
    manager.drain = lambda: None
    return manager, paths


def test_writer_rewind_discards_only_owned_suffix_and_markers(tmp_path):
    manager, paths = _writer_shell(tmp_path)
    unrelated = manager.output_dir / "unrelated"
    unrelated.write_text("preserve")
    receipts = manager.rewind_to_checkpoint(datetime(2026, 10, 4, 1),
        marker_directory=tmp_path / "ready",
        before_delete=lambda records: (
            all(path.exists() for path in paths) or pytest.fail(
                "deletion started before its receipt was recorded")))
    assert [path.exists() for path in paths] == [True, True, False]
    assert unrelated.exists()
    assert manager.paths == tuple(paths[:2])
    assert len(receipts) == 2
    assert receipts[0]["bytes"] == len(b"owned frame 2")
    assert not (tmp_path / "ready" / f"{paths[2].name}.json").exists()
    assert paths[2] not in manager._published_paths


def test_writer_rewind_refuses_modified_output_before_deleting_anything(tmp_path):
    manager, paths = _writer_shell(tmp_path)
    paths[2].write_bytes(b"modified externally")
    with pytest.raises(RuntimeError, match="changed|replaced|wrote"):
        manager.rewind_to_checkpoint(datetime(2026, 10, 4, 0),
                                     marker_directory=tmp_path / "ready")
    assert all(path.exists() for path in paths)


def _counting_coupler(child):
    from gpuwm.core.nest import NestCoupler
    from gpuwm.core.microphysics_transition import resolve_microphysics_transition
    coupler = object.__new__(NestCoupler)
    coupler.child_node = child
    coupler.microphysics_transition = resolve_microphysics_transition(
        child.parent.cfg.run, child.cfg.run)
    coupler.reset_restart_observation()
    coupler._valid = True
    return coupler


def test_actual_restart_resets_force_observation_without_relaxing_coverage(
        monkeypatch, tmp_path):
    from gpuwm.runtime import _write_microphysics_transition_receipt
    model, exp, checkpoint = _checkpoint(monkeypatch, tmp_path)
    child = model.node(2)
    coupler = _counting_coupler(child)
    child.coupler = coupler
    coupler.force_count = 3650
    coupler.first_parent_ticks = 10
    coupler.last_parent_ticks = 3700
    coupler.first_parent_step = 1
    coupler.last_parent_step = 3650
    coupler._last_tables = object()
    coupler._prepared_feedback = object()
    model.root.clock.step_count = 3650
    model.root.clock.ticks = 3700
    restart.restore_tree_restart(checkpoint, model)
    assert coupler.force_count == 0
    assert coupler.first_parent_step is None and coupler.last_parent_step is None
    assert coupler._last_tables is None and coupler._prepared_feedback is None
    assert not coupler.valid
    assert model.root.clock.step_count == 3600
    # Ten replayed parent steps have exactly ten newly observed forces.
    model.root.clock.step_count = 3610
    model.root.clock.ticks = 3700
    model.root.clock.elapsed_seconds = 3700
    coupler.force_count = 10
    coupler.first_parent_ticks = 3610
    coupler.last_parent_ticks = 3700
    coupler.first_parent_step = 3601
    coupler.last_parent_step = 3610
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    path, _sha, edges = _write_microphysics_transition_receipt(
        evidence, model, SimpleNamespace(name="restart-coverage"), resumed=True)
    assert edges[0]["process_force_count"] == 10
    assert edges[0]["force_count_matches_parent_steps"] is True
    assert json.loads(path.read_text())["resumed_process"] is True
    # Removing one FORCE still fails the unchanged full coverage gate.
    coupler.force_count = 9
    with pytest.raises(RuntimeError, match="coverage is incomplete"):
        _write_microphysics_transition_receipt(
            evidence, model, SimpleNamespace(name="restart-coverage"), resumed=True)


def test_retry_receipt_preserves_abandoned_force_observation(monkeypatch, tmp_path):
    model, exp, _checkpoint_path = _checkpoint(monkeypatch, tmp_path)
    child = model.node(2)
    child.coupler = _counting_coupler(child)
    child.coupler.force_count = 11
    child.coupler.first_parent_step = 3601
    child.coupler.last_parent_step = 3611
    child.coupler.first_parent_ticks = 3610
    child.coupler.last_parent_ticks = 3710
    recovery = NestedHealthRecovery(model=model, experiment=exp,
                                    output_directory=tmp_path)
    calls = []
    def execute(_active):
        calls.append(1)
        if len(calls) == 1:
            model.root.clock.ticks = 3710
            raise _health_error()
        assert child.coupler.force_count == 0
        return "replayed"
    assert recovery.run(execute) == "replayed"
    abandoned = recovery.receipt["attempts"][0]["discarded_leg_force_observations"]
    assert abandoned["d02"]["process_force_count"] == 11
