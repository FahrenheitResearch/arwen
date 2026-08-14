"""``gpuwm resume`` locate/parse tests -- CPU-only, no forecast is run.

The fixture checkpoints are genuine-format NPZ files: the same
``__gpuwm_restart_header__`` JSON member and array-manifest layout
``write_restart`` produces, so ``read_restart_header`` and
``validate_manifest_checkpoint`` -- the REAL machinery -- adjudicate
them.  What is faked is only the payload (two small arrays instead of a
model state); the identity checks that consume the payload belong to
``run --restart`` and are exercised by the restart family, not here.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm.resume import (LATEST, discover_checkpoint_sets,
                          resolve_resume_checkpoint)

_HEADER_KEY = "__gpuwm_restart_header__"


def _write_checkpoint(path, *, grid_id: int, domain_ids=None,
                      corrupt: str | None = None) -> None:
    arrays = {
        "state/u": np.arange(6, dtype=np.float32).reshape(2, 3),
        "state/v": np.zeros((2, 3), dtype=np.float32),
    }
    header = {
        "format_version": 3,
        "grid_id": grid_id,
        "array_manifest": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()},
    }
    if domain_ids is not None:
        header["domain_ids"] = list(domain_ids)
    if corrupt == "manifest":
        # A member the manifest does not declare: manifest-invalid.
        arrays["state/orphan"] = np.ones(2, dtype=np.float32)
    payload = {_HEADER_KEY: np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)}
    payload.update(arrays)
    with path.open("wb") as stream:
        np.savez(stream, **payload)
    if corrupt == "truncate":
        path.write_bytes(path.read_bytes()[:128])


def _single(outdir, instant: str, *, corrupt=None):
    path = outdir / f"gpuwmrst_d01_{instant}.npz"
    _write_checkpoint(path, grid_id=1, corrupt=corrupt)
    return path


def _tree(outdir, instant: str, set_id: str, domains=(1, 2, 3), *,
          declared=None, corrupt_member=None):
    declared = list(domains) if declared is None else list(declared)
    paths = {}
    for gid in domains:
        path = outdir / f"gpuwmrst_d{gid:02d}_{instant}__{set_id}.npz"
        _write_checkpoint(
            path, grid_id=gid, domain_ids=declared,
            corrupt=("manifest" if gid == corrupt_member else None))
        paths[gid] = path
    return paths


def test_discovery_groups_sets_and_sorts_newest_first(tmp_path):
    _single(tmp_path, "1974-04-03_13_00_00")
    _tree(tmp_path, "1974-04-03_15_00_00", "abc123")
    _single(tmp_path, "1974-04-03_14_00_00")
    (tmp_path / "gpuwmrst_d01_notes.txt").write_text("not a checkpoint")
    (tmp_path / "wrfout_d01_1974-04-03_13-00-00").write_bytes(b"x")

    sets = discover_checkpoint_sets(tmp_path)
    assert [s.valid_time.strftime("%H") for s in sets] == ["15", "14", "13"]
    tree = sets[0]
    assert tree.set_id == "abc123"
    assert sorted(tree.members) == [1, 2, 3]
    assert tree.handle.name == "gpuwmrst_d01_1974-04-03_15_00_00__abc123.npz"
    assert sets[1].set_id is None and sorted(sets[1].members) == [1]


def test_latest_takes_the_newest_valid_set(tmp_path):
    _single(tmp_path, "1974-04-03_13_00_00")
    _tree(tmp_path, "1974-04-03_15_00_00", "abc123")
    resolution = resolve_resume_checkpoint(tmp_path, LATEST)
    assert resolution.checkpoint.name == \
        "gpuwmrst_d01_1974-04-03_15_00_00__abc123.npz"
    assert resolution.skipped == ()


def test_latest_skips_an_invalid_newer_set_with_a_reason(tmp_path):
    _single(tmp_path, "1974-04-03_13_00_00")
    # Newest set: one member fails manifest validation (mid-write crash).
    _tree(tmp_path, "1974-04-03_15_00_00", "abc123", corrupt_member=2)
    resolution = resolve_resume_checkpoint(tmp_path)
    assert resolution.checkpoint.name == "gpuwmrst_d01_1974-04-03_13_00_00.npz"
    assert len(resolution.skipped) == 1
    assert "15_00_00" in resolution.skipped[0]


def test_latest_skips_a_torn_tree_set(tmp_path):
    _single(tmp_path, "1974-04-03_13_00_00")
    # Newest set declares d01..d03 but d03 never landed on disk.
    _tree(tmp_path, "1974-04-03_15_00_00", "abc123", domains=(1, 2),
          declared=(1, 2, 3))
    resolution = resolve_resume_checkpoint(tmp_path)
    assert resolution.checkpoint.name == "gpuwmrst_d01_1974-04-03_13_00_00.npz"
    assert "torn set" in resolution.skipped[0]


def test_latest_skips_a_truncated_single_domain_file(tmp_path):
    good = _single(tmp_path, "1974-04-03_13_00_00")
    _single(tmp_path, "1974-04-03_15_00_00", corrupt="truncate")
    resolution = resolve_resume_checkpoint(tmp_path)
    assert resolution.checkpoint == good


def test_no_checkpoints_is_a_clear_refusal(tmp_path):
    with pytest.raises(ValueError, match="no gpuwmrst_d"):
        resolve_resume_checkpoint(tmp_path)


def test_every_set_invalid_lists_every_reason(tmp_path):
    _single(tmp_path, "1974-04-03_13_00_00", corrupt="manifest")
    _single(tmp_path, "1974-04-03_15_00_00", corrupt="truncate")
    with pytest.raises(ValueError) as excinfo:
        resolve_resume_checkpoint(tmp_path)
    message = str(excinfo.value)
    assert "refusing to guess" in message
    assert "15_00_00" in message and "13_00_00" in message


def test_explicit_from_path_passes_through_unvalidated(tmp_path):
    """--from CKPT defers every check to the run machinery it feeds."""
    path = _single(tmp_path, "1974-04-03_13_00_00", corrupt="manifest")
    resolution = resolve_resume_checkpoint(tmp_path, path)
    assert resolution.checkpoint == path
    assert resolution.checkpoint_set is None


def test_explicit_from_path_must_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        resolve_resume_checkpoint(tmp_path, tmp_path / "gone.npz")


def test_resume_parser_carries_runs_supervision_surface(tmp_path):
    """resume parses like run: same supervision flags, plus --from."""
    args = _parse(["resume", "cfg.toml", "--outdir", str(tmp_path),
                   "--from", "latest", "--no-supervise",
                   "--supervisor-max-restarts", "5"])
    assert args.command == "resume"
    assert args.from_checkpoint == "latest"
    assert args.no_supervise is True
    assert args.supervisor_max_restarts == 5
    assert args.health_debug is False
    defaults = _parse(["resume", "cfg.toml"])
    assert defaults.from_checkpoint == "latest"
    assert str(defaults.outdir) == str(cli.Path("out") / "run")


def _parse(argv):
    """Parse through the real gpuwm parser without dispatching."""
    captured = {}

    class _Stop(Exception):
        pass

    original = cli.argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        namespace = original(self, args, namespace)
        captured["args"] = namespace
        raise _Stop()

    cli.argparse.ArgumentParser.parse_args = capture
    try:
        cli.main(argv)
    except _Stop:
        pass
    finally:
        cli.argparse.ArgumentParser.parse_args = original
    return captured["args"]


def test_cli_resume_resolves_then_dispatches_as_run(tmp_path, monkeypatch,
                                                    capsys):
    """End-to-end through cli.main up to the (stubbed) run dispatch."""
    _single(tmp_path, "1974-04-03_13_00_00")
    tree = _tree(tmp_path, "1974-04-03_15_00_00", "abc123")
    config = tmp_path / "exp.toml"
    config.write_text("[experiment]\n")  # sniffed as experiment-shaped

    seen = {}

    def fake_load(path, **kwargs):
        # **kwargs for the same reason as every other double of this
        # loader: this test is about resume RESOLVING a restart and
        # dispatching as a run, not about the loader's options.
        from types import SimpleNamespace
        seen["config"] = path
        return SimpleNamespace(name="stub-exp"), object()

    def fake_run(exp, data, outdir, *, restart=None, health_debug=False):
        seen["restart"] = restart
        from types import SimpleNamespace
        return SimpleNamespace(wrfout_paths=[], completed_seconds=0.0,
                               nan_free=True)

    import gpuwm.case_data as case_data
    import gpuwm.runtime as runtime
    monkeypatch.setattr(case_data, "load_experiment_case", fake_load)
    monkeypatch.setattr(runtime, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "is_experiment_toml", lambda path: True)

    rc = cli.main(["resume", str(config), "--outdir", str(tmp_path),
                   "--no-supervise"])
    assert rc == 0
    assert seen["restart"] == tree[1]
    out = capsys.readouterr().out
    assert re.search(r"resume: continuing from .*abc123\.npz", out)


# --- tie-break determinism ---------------------------------------------
#
# Two checkpoint sets can land on one model instant (a supervisor retry
# writes a fresh set id at the same model clock).  When their mtimes also
# tie -- second-resolution or a coarsening filesystem -- the selection
# used to fall out of Path.glob discovery order, so which checkpoint a
# resume continued from was a property of the filesystem.  Both creation
# orders must now choose the same set.


@pytest.mark.parametrize("creation_order",
                         [("aaa111", "bbb222"), ("bbb222", "aaa111")])
def test_tied_sets_at_one_instant_resolve_by_set_id(tmp_path,
                                                    creation_order):
    instant = "1974-04-03_15_00_00"
    for set_id in creation_order:
        _tree(tmp_path, instant, set_id)
    stamp = 1_500_000_000_000_000_000
    for path in tmp_path.glob("gpuwmrst_d*.npz"):
        os.utime(path, ns=(stamp, stamp))

    sets = discover_checkpoint_sets(tmp_path)
    assert [entry.set_id for entry in sets] == ["bbb222", "aaa111"]
    assert resolve_resume_checkpoint(tmp_path, LATEST).checkpoint.name == \
        f"gpuwmrst_d01_{instant}__bbb222.npz"


def test_a_subsecond_newer_set_wins_over_its_predecessor(tmp_path):
    """Nanosecond mtimes: 'newer' is not rounded away inside one second."""
    instant = "1974-04-03_15_00_00"
    # The lexicographically SMALLER set id is the newer one, so a set-id
    # tie-break alone would pick the wrong set and only mtime resolution
    # can carry this.
    older = _tree(tmp_path, instant, "zzz999")
    newer = _tree(tmp_path, instant, "aaa111")
    for path in older.values():
        os.utime(path, ns=(1_500_000_000_100_000_000,) * 2)
    for path in newer.values():
        os.utime(path, ns=(1_500_000_000_900_000_000,) * 2)

    assert [entry.set_id for entry in discover_checkpoint_sets(tmp_path)] == \
        ["aaa111", "zzz999"]
