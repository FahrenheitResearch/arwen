"""Checkpoint rollback starts a new FORCE observation without relaxing coverage.

The failure prevented here is stale FORCE counts from an abandoned execution
leg being compared with the restored parent's earlier step count. These tests
exercise the ordinary tree checkpoint reader, independently of automatic retry.
"""

from datetime import timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.microphysics_transition import resolve_microphysics_transition
from gpuwm.core.nest import NestCoupler
from gpuwm.io import restart
from gpuwm.runtime import _write_microphysics_transition_receipt
from test_restart import _rewrite_restart_archive, _sealed_tree_fixture


def _array_bytes(model):
    """All checkpointed atmospheric arrays plus the deterministic setup."""
    return {
        (node.cfg.grid_id, name): getattr(node.state, name).tobytes()
        for node in model.walk_parent_first()
        for name in (*restart.STATE_SERIALIZED_ATTRS,
                     *restart.STATE_SETUP_ARRAYS)
        if isinstance(getattr(node.state, name, None), np.ndarray)
    }


def _abandoned_leg(monkeypatch, tmp_path):
    model, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=3, run_seconds=7200, payload_seed=4)
    for node in model.walk_parent_first():
        node.clock.step_ticks = 10
        node.clock.elapsed_seconds = 3600
        # Held heating is zero at the forced ring in a produced checkpoint.
        # The generic fixture injects arbitrary bytes there; use a valid held
        # field so the reader's existing ring normalization is byte-neutral.
        node.state.h_diabatic[...] = 0
    saved_arrays = _array_bytes(model)
    checkpoint = restart.write_tree_restart(
        tmp_path, model, start + timedelta(seconds=3600))

    child = model.node(2)
    coupler = object.__new__(NestCoupler)
    coupler.child_node = child
    coupler.microphysics_transition = resolve_microphysics_transition(
        child.parent.cfg.run, child.cfg.run)
    # An abandoned leg has counters and caches beyond the saved clock.
    # Initialize these directly so the regression also runs on the pre-fix code.
    coupler.force_count = 3650
    coupler.first_parent_ticks = 10
    coupler.last_parent_ticks = 3700
    coupler.first_parent_step = 1
    coupler.last_parent_step = 3650
    coupler.force_sync_bytes = 1000
    coupler.feedback_sync_bytes = 2000
    coupler.feedback_host_scratch_bytes = 3000
    coupler.feedback_count = 50
    coupler.last_feedback_ticks = 3700
    coupler._last_tables = object()
    coupler._prepared_feedback = object()
    coupler._geometry_bound = True
    coupler.registrations = object()
    coupler._valid = True
    child.coupler = coupler
    for node in model.walk_parent_first():
        node.clock.step_count = 3650
        node.clock.ticks = 3700
        node.clock.elapsed_seconds = 3700
        node.state.w[...] = np.float32(239.79)
    return model, checkpoint, saved_arrays


def test_validated_rollback_discards_stale_force_observation_only(
        monkeypatch, tmp_path):
    model, checkpoint, saved_arrays = _abandoned_leg(monkeypatch, tmp_path)
    coupler = model.node(2).coupler
    registrations = coupler.registrations

    info = restart.restore_tree_restart(checkpoint, model)

    assert info.elapsed_ticks == 3600
    assert model.root.clock.step_count == 3600
    assert coupler.force_count == coupler.feedback_count == 0
    assert coupler.force_sync_bytes == coupler.feedback_sync_bytes == 0
    assert coupler.feedback_host_scratch_bytes == 0
    assert coupler.first_parent_ticks is coupler.last_parent_ticks is None
    assert coupler.first_parent_step is coupler.last_parent_step is None
    assert coupler.last_feedback_ticks is None
    assert coupler._last_tables is coupler._prepared_feedback is None
    assert not coupler.valid
    assert coupler._geometry_bound
    assert coupler.registrations is registrations
    assert _array_bytes(model) == saved_arrays

    # Ten replayed parent steps have ten newly observed forces. The actual
    # finalization gate accepts their receipt after the verified restore.
    model.root.clock.step_count = 3610
    model.root.clock.ticks = 3700
    model.root.clock.elapsed_seconds = 3700
    coupler.force_count = 10
    coupler.first_parent_ticks = 3610
    coupler.last_parent_ticks = 3700
    coupler.first_parent_step = 3601
    coupler.last_parent_step = 3610
    path, _sha, edges = _write_microphysics_transition_receipt(
        tmp_path, model, SimpleNamespace(name="checkpoint-coverage"),
        resumed=True)
    assert edges[0]["process_force_count"] == 10
    assert edges[0]["force_count_matches_parent_steps"] is True
    assert json.loads(path.read_text())["resumed_process"] is True

    # Losing a FORCE still fails the existing finalization invariant.
    coupler.force_count = 9
    with pytest.raises(RuntimeError, match="force coverage is incomplete"):
        _write_microphysics_transition_receipt(
            tmp_path, model, SimpleNamespace(name="checkpoint-coverage"),
            resumed=True)


def test_invalid_checkpoint_cannot_reset_force_observation(
        monkeypatch, tmp_path):
    model, checkpoint, _saved_arrays = _abandoned_leg(monkeypatch, tmp_path)
    coupler = model.node(2).coupler
    observations = vars(coupler).copy()
    before_arrays = _array_bytes(model)
    child_checkpoint = restart.tree_restart_members(checkpoint)[2]

    def break_child_set(_payload, header):
        header["checkpoint_set_id"] = "different-generation"

    _rewrite_restart_archive(
        child_checkpoint, child_checkpoint, break_child_set)
    with pytest.raises(restart.RestartMismatchError,
                       match="mismatched checkpoint sets"):
        restart.restore_tree_restart(checkpoint, model)

    assert vars(coupler) == observations
    assert _array_bytes(model) == before_arrays
    assert all(node.clock.ticks == 3700 and node.clock.step_count == 3650
               for node in model.walk_parent_first())
