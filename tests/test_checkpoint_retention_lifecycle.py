"""Checkpoint retention judges each set by the domains its own header declares.

Breakage these prevent: once a live nest retired, every later checkpoint set
held fewer members than the largest set on disk, so none of them counted as
complete and retention stopped pruning for the rest of the run (nine sets
were left where one was kept).  The same rule let a torn new set, whose
children were written before its root, stand in for an older whole set of
the same size, and the whole set was removed.
"""
from __future__ import annotations

import json

import numpy as np

from gpuwm.resume import discover_checkpoint_sets, retire_superseded_checkpoints


def _generation(directory, hour, declared, present=None):
    """One tree checkpoint set at ``hour``; ``present`` members of ``declared``."""
    for gid in declared if present is None else present:
        path = directory / f"gpuwmrst_d{gid:02d}_2026-09-27_{hour:02d}_00_00__s{hour}.npz"
        header = {"domain_ids": list(declared), "grid_id": gid,
                  "checkpoint_set_id": f"s{hour}"}
        with path.open("wb") as stream:
            np.savez(stream, __gpuwm_restart_header__=np.frombuffer(
                json.dumps(header).encode("utf-8"), dtype=np.uint8))


def _left(directory):
    return sorted(item.set_id for item in discover_checkpoint_sets(directory))


def test_a_retired_nest_does_not_stop_retention(tmp_path):
    _generation(tmp_path, 1, [1, 2])
    _generation(tmp_path, 2, [1])
    _generation(tmp_path, 3, [1])
    retire_superseded_checkpoints(tmp_path, keep=1)
    assert _left(tmp_path) == ["s3"]


def test_a_torn_set_of_the_same_size_never_replaces_a_whole_one(tmp_path):
    _generation(tmp_path, 1, [1])
    # The tree writer publishes the children before the root commit marker.
    _generation(tmp_path, 2, [1, 2], present=[2])
    retire_superseded_checkpoints(tmp_path, keep=1)
    assert _left(tmp_path) == ["s1", "s2"]


def test_keep_two_across_a_retirement_and_a_new_torn_set(tmp_path):
    _generation(tmp_path, 1, [1, 2])
    _generation(tmp_path, 2, [1])
    _generation(tmp_path, 3, [1])
    _generation(tmp_path, 4, [1, 2], present=[2])
    retire_superseded_checkpoints(tmp_path, keep=2)
    assert _left(tmp_path) == ["s2", "s3", "s4"]


def test_an_unreadable_newest_set_is_not_counted(tmp_path):
    _generation(tmp_path, 1, [1])
    (tmp_path / "gpuwmrst_d01_2026-09-27_02_00_00__s2.npz").write_bytes(b"cut short")
    retire_superseded_checkpoints(tmp_path, keep=1)
    assert _left(tmp_path) == ["s1", "s2"]


def test_single_domain_checkpoints_are_whole_by_themselves(tmp_path):
    for hour in (1, 2, 3):
        path = tmp_path / f"gpuwmrst_d01_2026-09-27_{hour:02d}_00_00.npz"
        with path.open("wb") as stream:
            np.savez(stream, __gpuwm_restart_header__=np.frombuffer(
                json.dumps({"grid_id": 1}).encode("utf-8"), dtype=np.uint8))
    retire_superseded_checkpoints(tmp_path, keep=1)
    left = discover_checkpoint_sets(tmp_path)
    assert [item.valid_time.hour for item in left] == [3]


def test_the_tree_writer_keeps_one_set_after_a_nest_retires(tmp_path, monkeypatch):
    """The real scheduler, lifecycle runner, tree writer and retention."""
    from test_lifecycle_restart_identity import EPISODIC, _case, _det_step, _run

    monkeypatch.setattr("gpuwm.core.dycore.step", _det_step)
    monkeypatch.setenv("GPUWM_KEEP_CHECKPOINTS", "1")
    run = _run(_case(retire=EPISODIC["retire"]), tmp_path)
    assert run.runner.spawns_executed == 1
    assert 2 in run.runner.retired_times
    sets = discover_checkpoint_sets(tmp_path)
    assert len(sets) == 1, [(item.valid_time.isoformat(), sorted(item.members))
                            for item in sets]
    assert sorted(sets[0].members) == [1]
