"""Prepared children own their activation analysis, clock and first feedback."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from gpuwm import prepared_domain_tree_forecast as runner
from gpuwm.core.model import execute_experiment
from test_delayed_nest_activation import (
    DELAY_SECONDS, _Writers, _stashing_step, _tree)
from test_model import _Coupler, _HistoryState
from test_prepared_domain_tree_forecast import _sha, _synthetic_prepared_tree


def test_preflight_accepts_owned_dated_child_analysis(tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(
        tmp_path, monkeypatch, delayed=True)
    inputs = runner.preflight_prepared_tree(
        prepared_root=prepared, preparation_receipt_sha256=_sha(receipt),
        experiment_config=config, experiment_config_sha256=_sha(config))
    assert inputs.experiment.domain_start_offset_exact(2) == 3600
    assert inputs.domains[1].cache_reader.header["metadata"]["user"]["initial_valid_time"] \
        == "2026-07-23T01:00:00"


@pytest.mark.parametrize("stamp", [None, "2026-07-23T00:00:00"])
def test_preflight_refuses_missing_or_root_dated_child_analysis(
        tmp_path, monkeypatch, stamp):
    prepared, receipt, config = _synthetic_prepared_tree(
        tmp_path, monkeypatch, delayed=True)
    header_path = prepared / "hierarchy-artifacts/domains/d02/prepared-cache/header.json"
    header = json.loads(header_path.read_text())
    header["metadata"]["user"]["initial_valid_time"] = stamp
    header_path.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="initial_valid_time.*delayed start_time"):
        runner.preflight_prepared_tree(
            prepared_root=prepared, preparation_receipt_sha256=_sha(receipt),
            experiment_config=config, experiment_config_sha256=_sha(config))


def test_child_receipt_must_agree_with_owned_analysis():
    exp, model = _tree(delay_s=DELAY_SECONDS)
    child = model.node(2).cfg
    reader = SimpleNamespace(header={"metadata": {"user": {
        "initial_valid_time": exp.domain_start_time(2).isoformat()}}})
    with pytest.raises(ValueError, match="domain receipt valid_time"):
        runner._validate_delayed_prepared_time(
            exp, child, reader, {"valid_time": exp.start_time.isoformat()})


def test_delayed_child_refuses_a_moving_ancestor(monkeypatch):
    exp, _ = _tree(delay_s=DELAY_SECONDS)
    monkeypatch.setattr("gpuwm.static.corridor.moving_grid_ids",
                        lambda exp: frozenset({1}))
    with pytest.raises(ValueError, match="moving ancestor d01"):
        runner._validate_delayed_prepared_geometry(exp)
    # Its own follower can start after the child exists; only the
    # preparation footprint's ancestors must remain fixed before birth.
    monkeypatch.setattr("gpuwm.static.corridor.moving_grid_ids",
                        lambda exp: frozenset({2}))
    runner._validate_delayed_prepared_geometry(exp)


@pytest.mark.parametrize("feedback", [0, 1])
def test_prepared_activation_uses_exact_clock_and_shared_feedback(
        monkeypatch, feedback):
    from gpuwm.runtime import _submit_tree_history_frame

    exp, model = _tree(delay_s=DELAY_SECONDS, smooth_option=2)
    exp = replace(exp, feedback=feedback)
    model._activation_context = {"experiment": exp}
    monkeypatch.setattr("gpuwm.core.nest.NestCoupler", _Coupler)
    monkeypatch.setattr("gpuwm.core.dycore.step", _stashing_step)
    calls = []
    analysis = _HistoryState()
    case = SimpleNamespace(initial_result=SimpleNamespace(state=analysis))

    def initialize(node, clock):
        calls.append((node.cfg.grid_id, clock.ticks,
                      node.parent.clock.ticks, node._started))
        return SimpleNamespace(grid=node.grid, state=analysis), case

    writers = _Writers()
    model._io_manager = writers
    execute_experiment(
        model, validate_state=False, delayed_child_initializer=initialize,
        skip_feedback_path=feedback == 0,
        history_handler=lambda _tree, node, ticks: _submit_tree_history_frame(
            writers, node, ticks))
    assert calls == [(2, DELAY_SECONDS, DELAY_SECONDS, False)]
    assert model.node(2)._started
    assert model.node(2).state is analysis
    assert model._prepared_by_grid_id[2] is case
    assert analysis.domain_start_offset == DELAY_SECONDS
    coupler = model.node(2).coupler
    assert (coupler.feedback, coupler.smooth_option) == (feedback, 2)
    if feedback:
        assert coupler.calls[:3] == ["prepare", "commit", "finalize"]
    else:
        assert coupler.calls[0] == "force"
        assert "commit" not in coupler.calls
    child_frames = [(ticks, refl) for gid, ticks, refl in writers.frames if gid == 2]
    assert [ticks for ticks, _ in child_frames] == [120, 180, 240, 300]
    assert child_frames[0][1] is None
    assert all(refl is not None for _, refl in child_frames[1:])
