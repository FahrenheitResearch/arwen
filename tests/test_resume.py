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
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm.resume import (LATEST, discover_checkpoint_sets,
                          resolve_resume_checkpoint)

_HEADER_KEY = "__gpuwm_restart_header__"


def _write_checkpoint(path, *, grid_id: int, domain_ids=None,
                      corrupt: str | None = None, written_mode=None) -> None:
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
    if written_mode is not None:
        # The writer's provenance stamp; absent by default, because a
        # checkpoint written before it existed must resume unchanged.
        header["written_mode"] = dict(written_mode)
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
          declared=None, corrupt_member=None, written_mode=None):
    declared = list(domains) if declared is None else list(declared)
    paths = {}
    for gid in domains:
        path = outdir / f"gpuwmrst_d{gid:02d}_{instant}__{set_id}.npz"
        _write_checkpoint(
            path, grid_id=gid, domain_ids=declared,
            corrupt=("manifest" if gid == corrupt_member else None),
            written_mode=written_mode)
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
    tree = _tree(tmp_path, "1974-04-03_15_00_00", "abc123",
                 written_mode={"mode": "resident", "shape": [2, 3]})
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
    # The notes reach the operator on the same stream as the continuation
    # line: the mode this run resolves to and the road the file names.
    assert ("resume: this checkpoint was WRITTEN resident and this run "
            "resolves [tiles] to resident") in out


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


# --- which memory road wrote the checkpoint ----------------------------
#
# restart x memory mode is a FREE combination and must stay one:
# ``streaming.identity_payload_entry`` contributes nothing to the restart
# identity on purpose, so a checkpoint written streamed resumes resident
# and one written resident resumes streamed, which is the operation that
# lets a forecast outgrowing its card continue on the same card.  The
# stamp below is provenance for the operator and never a condition.


def test_the_restart_writer_stamps_the_road_it_wrote_on():
    from gpuwm.io import restart

    cfg = SimpleNamespace(nx=41, ny=37)
    resident = restart.written_mode_note(restart.RESIDENT_WRITTEN_MODE, cfg)
    assert resident == {"mode": "resident", "shape": [37, 41]}
    streamed = restart.written_mode_note(
        restart.STREAMED_WRITTEN_MODE, cfg, store="host")
    assert streamed == {"mode": "streamed", "shape": [37, 41], "store": "host"}
    with pytest.raises(ValueError, match="written mode must be"):
        restart.written_mode_note("swapped", cfg)

    assert restart.header_written_mode({"written_mode": resident}) == "resident"
    assert restart.header_written_mode({"written_mode": streamed}) == "streamed"
    # A file that names no road, written before the stamp existed or by
    # the streamed writer, which does not stamp yet, says nothing, and
    # saying nothing is never a refusal.
    assert restart.header_written_mode({"producer": {}}) is None
    assert restart.header_written_mode({"written_mode": "resident"}) is None
    assert restart.header_written_mode("not a header") is None


# --- what the resume discloses about this run's memory mode ------------
#
# Nothing here refuses, clamps or downgrades anything; the resume states
# the fact so the operator does not have to derive it from two modules.


def _tiles_config(tmp_path, mode: str, *, name="exp.toml"):
    config = tmp_path / name
    config.write_text(
        "[experiment]\nname = \"resume-mode\"\n\n"
        f"[tiles]\nmode = \"{mode}\"\n", encoding="utf-8")
    return config


def test_resume_states_this_run_resolved_memory_mode(tmp_path):
    resident_cfg = _tiles_config(tmp_path, "off")
    _single(tmp_path, "1974-04-03_13_00_00")

    resolution = resolve_resume_checkpoint(tmp_path, LATEST,
                                           config=resident_cfg)
    assert len(resolution.notes) == 1
    note = resolution.notes[0]
    assert "resident" in note
    assert "mode-independent" in note
    assert "written either way" in note and "resumes either way" in note

    streamed_cfg = _tiles_config(tmp_path, "on", name="streamed.toml")
    streamed = resolve_resume_checkpoint(tmp_path, LATEST,
                                         config=streamed_cfg).notes
    assert len(streamed) == 1
    assert "streamed" in streamed[0]
    assert "mode-independent" in streamed[0]

    # The note is disclosure, so it never becomes a condition: the same
    # checkpoint resolves under either mode, to the same file.
    assert resolve_resume_checkpoint(tmp_path, LATEST,
                                     config=streamed_cfg).checkpoint == \
        resolve_resume_checkpoint(tmp_path, LATEST,
                                  config=resident_cfg).checkpoint

    # CONTROL: no config, no note, and the resolution is otherwise the
    # same object it always was.
    assert resolve_resume_checkpoint(tmp_path, LATEST).notes == ()


def test_the_memory_mode_note_declines_to_guess_rather_than_refusing(tmp_path):
    """An unreadable config costs a note, never the resume."""
    from gpuwm.resume import resume_memory_mode_note

    assert resume_memory_mode_note(tmp_path / "absent.toml") is None
    broken = tmp_path / "broken.toml"
    broken.write_text("this is not = = toml\n", encoding="utf-8")
    assert resume_memory_mode_note(broken) is None
    _single(tmp_path, "1974-04-03_13_00_00")
    assert resolve_resume_checkpoint(tmp_path, LATEST,
                                     config=broken).notes == ()


def test_the_memory_mode_note_reads_per_domain_overrides_too(tmp_path):
    """A tree whose grids disagree is not reported as if they agreed."""
    from gpuwm.resume import resume_memory_mode_note

    config = tmp_path / "mixed.toml"
    config.write_text(
        "[experiment]\nname = \"mixed\"\n\n"
        "[tiles]\nmode = \"on\"\n\n"
        "[[domain]]\ngrid_id = 1\n\n"
        "[[domain]]\ngrid_id = 2\ntiles = { mode = \"off\" }\n",
        encoding="utf-8")
    note = resume_memory_mode_note(config)
    assert note is not None
    assert "per-domain mix" in note
    assert "mode-independent" in note


def test_the_resume_states_the_road_the_checkpoint_was_written_on(tmp_path):
    """The sentence resume_memory_mode_note could not say by itself.

    Reading the experiment can only ever report the mode of the run doing
    the READING.  The written mode comes off the file, so a checkpoint
    left by a run whose logs are gone still says which road it died on.
    """
    from gpuwm.resume import resume_written_mode_note

    streamed_header = {"written_mode": {"mode": "streamed",
                                        "shape": [37, 41], "store": "host"}}
    resident_cfg = _tiles_config(tmp_path, "off", name="resident.toml")

    note = resume_written_mode_note(
        tmp_path / "ckpt.npz", read_header=lambda path: streamed_header,
        config=resident_cfg)
    assert "WRITTEN streamed" in note
    assert "resolves [tiles] to resident" in note
    assert "mode-independent" in note

    # No config: the file's own half still gets said.
    alone = resume_written_mode_note(
        tmp_path / "ckpt.npz", read_header=lambda path: streamed_header)
    assert "WRITTEN streamed" in alone

    # A file with no stamp, and a file that cannot be read at all, each
    # cost the note and never the resume.
    assert resume_written_mode_note(
        tmp_path / "ckpt.npz", read_header=lambda path: {}) is None

    def unreadable(path):
        raise ValueError("unreadable header")

    assert resume_written_mode_note(
        tmp_path / "ckpt.npz", read_header=unreadable) is None


def test_the_resolution_carries_both_halves_of_the_memory_mode_sentence(tmp_path):
    """Written mode and resolved mode, side by side, on the resolution."""
    stamped = tmp_path / "gpuwmrst_d01_1974-04-03_13_00_00.npz"
    _write_checkpoint(stamped, grid_id=1,
                      written_mode={"mode": "streamed", "shape": [37, 41],
                                    "store": "host"})
    config = _tiles_config(tmp_path, "off")

    notes = resolve_resume_checkpoint(tmp_path, LATEST, config=config).notes
    assert len(notes) == 2
    assert "this run resolves [tiles] to resident" in notes[0]
    assert "WRITTEN streamed" in notes[1]

    # An explicit --from path is the same door and says the same thing.
    explicit = resolve_resume_checkpoint(tmp_path, stamped, config=config).notes
    assert explicit == notes

    # Disclosure, never a condition: the same file resolves either way.
    assert resolve_resume_checkpoint(tmp_path, LATEST, config=config).checkpoint \
        == resolve_resume_checkpoint(tmp_path, LATEST).checkpoint
    assert len(resolve_resume_checkpoint(tmp_path, LATEST).notes) == 1
