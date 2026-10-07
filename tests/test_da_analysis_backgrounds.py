"""The cycle driver hands the analysis member backgrounds without copies.

A packed member publishes no host mirror of its state; the controller reads
the analysed fields from the member's own leg-end restart on demand, and
stages no second npz of every member unless the replay bundle asks for it.
"""

from __future__ import annotations

import types

import numpy as np

from gpuwm.da.radar_assimilation import CheckpointStateView
from tools.da_cycle_prepared import (analysis_backgrounds, member_background,
                                     release_backgrounds)


def _restart(path, seed):
    rng = np.random.default_rng(seed)
    state = {"u": rng.normal(size=(3, 4, 6)).astype(np.float32),
             "qv": rng.random((3, 4, 5)).astype(np.float32),
             "p": rng.random((3, 4, 5)).astype(np.float32)}
    np.savez(path, **{f"state/{k}": v for k, v in state.items()},
             **{"driver/tsk": np.ones((4, 5), np.float32)})
    return state


def test_packed_result_is_read_lazily_from_its_own_restart(tmp_path):
    state = _restart(tmp_path / "gpuwmrst_d01_a.npz", 1)
    result = types.SimpleNamespace(snapshot=None,
                                   restart=tmp_path / "gpuwmrst_d01_a.npz")
    background = member_background(result)
    assert isinstance(background, CheckpointStateView)
    assert set(background) == set(state)
    assert background.cached_bytes == 0
    assert background["qv"].tobytes() == state["qv"].tobytes()
    assert background.cached_bytes == state["qv"].nbytes
    in_process = types.SimpleNamespace(snapshot=state, restart=None)
    assert member_background(in_process) is state
    # An in-process mirror is used as it is, even an empty one; only a
    # result that crossed the worker boundary (None) reads its restart.
    empty = {}
    assert member_background(types.SimpleNamespace(
        snapshot=empty, restart=tmp_path / "unreadable.npz")) is empty
    held = {0: background, "control": state}
    release_backgrounds(held)
    assert held == {} and background.cached_bytes == 0


def test_no_files_are_staged_unless_the_bundle_asks(tmp_path):
    states = {i: _restart(tmp_path / f"r{i}.npz", i) for i in range(3)}
    views = {i: CheckpointStateView(tmp_path / f"r{i}.npz") for i in range(3)}
    staged = tmp_path / "cycle_001"
    direct = analysis_backgrounds(views, 3, directory=staged, t_end=3600.0,
                                  files=False)
    assert all(direct[i] is views[i] for i in range(3))
    assert not staged.exists()
    paths = analysis_backgrounds(views, 3, directory=staged, t_end=3600.0,
                                 files=True)
    for index, path in paths.items():
        assert path.name == "gpuwmrst_d01_003600.npz"
        with np.load(path) as saved:
            assert sorted(saved.files) == sorted(
                f"state/{k}" for k in states[index])
            for name, value in states[index].items():
                assert saved[f"state/{name}"].tobytes() == value.tobytes()
    release_backgrounds(views)


def test_member_increment_record_keeps_its_float32_bytes(tmp_path):
    from tools.da_cycle_prepared import write_member_increments

    rng = np.random.default_rng(3)
    increments = {"qv": rng.normal(size=(3, 4, 5)) * 1e-4,
                  "u": rng.normal(size=(3, 4, 6)).astype(np.float32)}
    path = write_member_increments(tmp_path / "increments_m000_leg1.npz",
                                   increments)
    with np.load(path) as saved:
        assert sorted(saved.files) == ["qv", "u"]
        for name, value in increments.items():
            assert saved[name].dtype == np.float32
            assert saved[name].tobytes() == np.asarray(
                value).astype(np.float32).tobytes()
    import zipfile
    with zipfile.ZipFile(path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED
                   for info in archive.infolist())


def test_the_cycle_releases_analysis_inputs_before_the_recovery_write():
    """The solver's whole-ensemble output and the member backgrounds are
    released inside the analysis block, after the merge that made
    ``pending`` and before the per-leg recovery generation, and nothing in
    the cycle stages member mirrors to disk except for the replay bundle."""
    import ast
    import inspect

    from tools import da_cycle_prepared

    tree = ast.parse(inspect.getsource(da_cycle_prepared.cycle))
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.setdefault(node.func.attr, []).append(node.lineno)
    merge = max(calls["merge_hotstart_increments"])
    generation = min(calls["write_generation"])
    released = [line for line in calls["release_backgrounds"]
                if merge < line < generation]
    assert released, "member backgrounds outlive the analysis"
    assert "savez" not in calls, "the cycle stages member mirrors itself"
    assert calls["analysis_backgrounds"]
