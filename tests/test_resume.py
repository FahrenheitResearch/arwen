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
from pathlib import Path
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
    from gpuwm import capabilities

    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))
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


# --- the experiment argument -------------------------------------------
#
# The reported invocation: a desktop terminal built the resume command
# out of the run's NAME and ran it from the install folder, so the engine
# resolved "<name>.child" against that folder and found nothing, while
# the file beside it was "<name>.child.toml" and the run directory it had
# just read a checkpoint out of held the config the run was made from.


def _config(path, text="[grid]\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_argument_as_typed_is_taken_first(tmp_path, monkeypatch):
    """A working invocation resolves on rung one and touches nothing else."""
    from gpuwm.resume import resolve_resume_experiment

    run = tmp_path / "run"
    _config(run / "child.toml")
    typed = _config(tmp_path / "exp.toml")
    monkeypatch.chdir(tmp_path)

    resolution = resolve_resume_experiment("exp.toml", run)
    assert resolution.path == Path("exp.toml")
    assert resolution.tried == ()
    assert resolution.note is None


def test_the_hidden_toml_extension_is_the_second_rung(tmp_path, monkeypatch):
    """THE REPORT: the file manager hid the extension the argument needs."""
    from gpuwm.resume import resolve_resume_experiment

    install = tmp_path / "install"
    run = tmp_path / "run"
    run.mkdir()
    _config(install / "downscale-child.child.toml")
    monkeypatch.chdir(install)

    resolution = resolve_resume_experiment("downscale-child.child", run)
    assert resolution.path == Path("downscale-child.child.toml")
    assert resolution.tried == ("downscale-child.child: does not exist",)
    assert ".toml extension" in resolution.note


def test_the_argument_is_read_against_the_run_directory(tmp_path, monkeypatch):
    """Rungs three and four: the same name, against --outdir."""
    from gpuwm.resume import resolve_resume_experiment

    run = tmp_path / "run"
    _config(run / "case")
    monkeypatch.chdir(tmp_path)
    assert resolve_resume_experiment("case", run).path == run / "case"

    plain = tmp_path / "run2"
    _config(plain / "case.toml")
    resolution = resolve_resume_experiment("case", plain)
    assert resolution.path == plain / "case.toml"
    assert "--outdir" in resolution.note


def test_the_run_directory_answers_with_what_it_ran(tmp_path, monkeypatch):
    """Rung five, once for each document a run route writes."""
    from gpuwm.resume import resolve_resume_experiment

    monkeypatch.chdir(tmp_path)
    child = tmp_path / "child-run"
    _config(child / "child.toml")
    assert resolve_resume_experiment("absent", child).path == (
        child / "child.toml")

    prepared = tmp_path / "prepared-run"
    _config(prepared / "experiment.toml")
    assert resolve_resume_experiment("absent", prepared).path == (
        prepared / "experiment.toml")

    supervised = tmp_path / "supervised-run"
    _config(supervised / "captured-config-0001.toml")
    resolution = resolve_resume_experiment("absent", supervised)
    assert resolution.path == supervised / "captured-config-0001.toml"
    assert "recorded for itself" in resolution.note


def test_the_newest_capture_is_the_one_a_resume_reads(tmp_path, monkeypatch):
    """A directory resumed twice holds two captures; the last run's wins."""
    import os

    from gpuwm.resume import resolve_resume_experiment

    monkeypatch.chdir(tmp_path)
    run = tmp_path / "run"
    first = _config(run / "captured-config-aaa.toml")
    second = _config(run / "captured-config-bbb.toml")
    os.utime(first, ns=(1_000_000_000_000, 1_000_000_000_000))
    os.utime(second, ns=(2_000_000_000_000, 2_000_000_000_000))
    assert resolve_resume_experiment("absent", run).path == second

    os.utime(first, ns=(3_000_000_000_000, 3_000_000_000_000))
    assert resolve_resume_experiment("absent", run).path == first


def test_an_absolute_argument_is_not_tried_twice(tmp_path, monkeypatch):
    """Rungs three and four ARE rungs one and two for an absolute path."""
    from gpuwm.resume import resolve_resume_experiment

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    missing = tmp_path / "nowhere" / "case"
    with pytest.raises(ValueError) as excinfo:
        resolve_resume_experiment(missing, run)
    message = str(excinfo.value)
    assert message.count(str(missing) + ":") == 1
    assert str(run / "child.toml") in message


def test_the_refusal_names_every_path_it_tried(tmp_path, monkeypatch):
    """Nothing to load, and the reader is told everywhere it looked."""
    from gpuwm.resume import RUN_RECORD_NAMES, resolve_resume_experiment

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        resolve_resume_experiment("case.child", run)
    message = str(excinfo.value)
    for expected in ("case.child: does not exist",
                     "case.child.toml: does not exist",
                     str(run / "case.child"),
                     str(run / "case.child.toml")):
        assert expected in message
    for name in RUN_RECORD_NAMES:
        assert str(run / name) in message
    assert "gpuwm domain" in message


def test_a_rung_that_is_the_wrong_kind_says_which_kind(tmp_path, monkeypatch):
    """The ladder and the loader use one vocabulary for "not a config"."""
    from gpuwm.experiment import config_path_kind
    from gpuwm.resume import resolve_resume_experiment

    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "case.child").mkdir()
    (run / "child.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        resolve_resume_experiment("case.child", run)
    message = str(excinfo.value)
    assert "case.child: is a directory" in message
    assert f"{run / 'child.toml'}: is empty" in message
    # The same two words the loader would have used on the same paths.
    assert config_path_kind(tmp_path / "case.child") == "is a directory"
    assert config_path_kind(run / "child.toml") == "is empty"


def test_an_empty_config_is_still_refused_by_its_own_loader(tmp_path):
    """The kind helper did not cost readable_config_path its two cases."""
    from gpuwm.experiment import readable_config_path

    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        readable_config_path(empty)
    with pytest.raises(ValueError, match="is a directory"):
        readable_config_path(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        readable_config_path(tmp_path / "absent.toml")
    good = tmp_path / "good.toml"
    good.write_text("[grid]\n", encoding="utf-8")
    assert readable_config_path(good) == good


def test_a_path_whose_kind_cannot_be_decided_carries_a_remedy_too(
        tmp_path, monkeypatch):
    """The fourth kind was the one with nowhere to go.

    A directory, a missing file and a zero-byte file each come back with
    what to do next.  The fourth -- the one where ``is_file()`` ITSELF
    failed, which is a symbolic-link loop, a permission wall or a mount
    that is gone -- came back as the path, the errno and a full stop, so
    the reader with the least to go on got the least.
    """
    import errno as errno_module
    from pathlib import Path

    from gpuwm.experiment import readable_config_path

    target = tmp_path / "loop.toml"

    def refuse(self):
        if Path(self) == target:
            raise OSError(errno_module.ELOOP,
                          "Too many levels of symbolic links")
        return False

    monkeypatch.setattr(Path, "is_file", refuse)
    with pytest.raises(ValueError) as excinfo:
        readable_config_path(target)
    message = str(excinfo.value)
    # What it was instead, in the operating system's own words ...
    assert "Too many levels of symbolic links" in message
    # ... and the way out, in the shape the other three kinds carry.
    assert "remedy:" in message
    assert "gpuwm domain" in message


# --- a downscaled child is a run nothing continues ---------------------


def _child_run(outdir, *, finished, frames=2, result="PASS",
               records_config=True):
    """A child's run directory.

    ``records_config`` is the difference between the two doors that
    make one: the ``--point`` derivation writes the configuration it
    derived into ``--out`` as ``child.toml``, and a
    ``--child-config`` run is handed its configuration from outside
    the run directory and records none.  The same run either way.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if records_config:
        (outdir / "child.toml").write_text("[grid]\n", encoding="utf-8")
    (outdir / "downscale-plan.json").write_text("{}", encoding="utf-8")
    for index in range(frames):
        (outdir / f"wrfout_d02_1974-04-03_12_0{index}_00").write_bytes(b"x")
    if finished:
        (outdir / "report.json").write_text(
            json.dumps({"result": result, "child_steps": 30,
                        "final_restart": "gpuwmrst_d02.npz"}),
            encoding="utf-8")
    return outdir


def test_a_child_run_directory_is_recognised_by_what_the_run_wrote(tmp_path):
    """The configuration beside a run is a record, not the marker."""
    from gpuwm.offline_child import CHILD_REPORT_PIPELINE
    from gpuwm.resume import offline_child_run_at

    assert offline_child_run_at(tmp_path / "absent") is None

    # A file of that name is a file of that name wherever a reader keeps
    # one, so it still names no route on its own.
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "child.toml").write_text("[grid]\n", encoding="utf-8")
    assert offline_child_run_at(bare) is None

    planned = _child_run(tmp_path / "planned", finished=False)
    child = offline_child_run_at(planned)
    assert child is not None and child.finished is False
    assert child.result is None and len(child.frames) == 2
    assert child.config == planned / "child.toml"

    # The runner door writes no plan document; its report names the route.
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "report.json").write_text(
        json.dumps({"result": "PASS", "child_steps": 4}), encoding="utf-8")
    assert offline_child_run_at(runner) is not None

    # And a report that names its own pipeline is enough by itself,
    # which is the one key both of a child's outcomes write.
    named = tmp_path / "named"
    named.mkdir()
    (named / "report.json").write_text(
        json.dumps({"result": "PASS", "pipeline": CHILD_REPORT_PIPELINE}),
        encoding="utf-8")
    assert offline_child_run_at(named) is not None

    # A report.json written by another route is not a child.  This one
    # is the prepared-forecast failure document, key for key.
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text(
        json.dumps({"schema": "gpuwm.report.v1", "status": "FAIL",
                    "error": "boom", "error_type": "ValueError"}),
        encoding="utf-8")
    assert offline_child_run_at(other) is None


def test_a_finished_child_is_told_there_is_nothing_to_resume(tmp_path):
    """The reported case: the child reached its last frame."""
    from gpuwm.resume import (offline_child_resume_refusal,
                              offline_child_run_at)
    from gpuwm.explain import render

    outdir = _child_run(tmp_path / "child-run", finished=True, frames=4)
    child = offline_child_run_at(outdir)
    assert child.finished and child.result == "PASS"

    action = render(offline_child_resume_refusal(child), explain=False)
    assert "nothing to resume" in action
    assert "reached its last frame" in action
    assert "draw the 4 frame(s)" in action
    assert f"gpuwm render {outdir / 'wrfout_d*'} --series" in action
    assert str(outdir / "render") in action


def test_an_unfinished_child_is_re_run_not_continued(tmp_path):
    """No report means it stopped inside the forecast; downscale is the door."""
    from gpuwm.resume import (offline_child_resume_refusal,
                              offline_child_run_at)
    from gpuwm.explain import render

    outdir = _child_run(tmp_path / "child-run", finished=False, frames=1)
    action = render(
        offline_child_resume_refusal(offline_child_run_at(outdir)),
        explain=False)
    assert "`gpuwm resume` cannot continue" in action
    assert "run `gpuwm downscale` again" in action
    assert "draw the 1 frame(s)" in action

    # A child that wrote no frame at all is only re-run.
    empty = tmp_path / "empty-run"
    empty.mkdir()
    (empty / "child.toml").write_text("[grid]\n", encoding="utf-8")
    (empty / "downscale-plan.json").write_text("{}", encoding="utf-8")
    bare = render(
        offline_child_resume_refusal(offline_child_run_at(empty)),
        explain=False)
    assert "gpuwm downscale" in bare
    assert "frame(s)" not in bare


def _blown_up_report(outdir, *, pictures=0):
    """``report.json`` for a child that went non-finite, from its writer.

    Composed by the product code, not by this test: the invariant under
    test is what ``gpuwm resume`` reads back out of whatever the failure
    path actually writes, so a hand-built dict here would pin the test
    to itself.
    """
    import numpy as np

    from gpuwm.core.dycore import decode_stability_record
    from gpuwm.offline_child_run import (_ChildProgress,
                                         _publish_failure_report,
                                         child_health_log_fields,
                                         describe_nonfinite_child)

    # The terminal row through the PRODUCT's own decoder, like the report
    # around it: a hand-written ``{"w_max": None}`` is a row
    # ``child_health_log_fields`` cannot produce, and a fixture that
    # composes itself is pinned to itself.
    gone = child_health_log_fields(decode_stability_record(
        np.array([np.inf, np.nan, np.nan, 0.0, 0.0, np.nan, 0.0, 0.0],
                 dtype=np.float64), cfg=None))
    progress = _ChildProgress()
    progress.outdir = outdir
    capsule = describe_nonfinite_child(
        step=6624, total_steps=69120, model_seconds=2760.0,
        run_seconds=28800.0, cadence_seconds=60.0,
        trend=[{"step": 6480, "model_seconds": 2700.0,
                "w_max": 22.971, "w_max_state": "measured",
                "cfl": 0.20854, "cfl_state": "measured"},
               {"step": 6624, "model_seconds": 2760.0,
                "w_max": gone["w_max"], "w_max_state": gone["w_max_state"],
                "cfl": gone["cfl"], "cfl_state": gone["cfl_state"]}],
        survey={"surveyed": ["W", "U", "V", "T"],
                "fields": [{"field": "W", "carrier": "w",
                            "shape": [50, 798, 798], "size": 31840200,
                            "count": 1,
                            "bounding_box": {"k": [12, 12], "j": [401, 401],
                                             "i": [388, 388]},
                            "edges": [],
                            "cell": {"k": 12, "j": 401, "i": 388}}]})
    # The dict `keep_early_render` hands the publisher, in its own
    # shape: a count, whether the tree could be read, and the banner
    # standing over whatever is in it.
    return _publish_failure_report(
        progress, capsule,
        kept={"pictures": pictures, "pictures_error": None,
              "render": str(Path(outdir) / "png"),
              "banner": (None if not pictures
                         else str(Path(outdir) / "png"
                                  / "DID-NOT-FINISH.txt"))})


def test_a_child_that_blew_up_is_re_run_and_told_why(tmp_path):
    """THE INVARIANT: the verdict says whether a child finished.

    A child that stops being finite publishes ``report.json`` as surely
    as one that reaches its last frame, so reading the DOCUMENT as the
    end-of-forecast marker tells a reader whose child died at model
    second 2760 of 28800 that there is nothing left to integrate, and
    withholds the one remedy they need.
    """
    from gpuwm.explain import render
    from gpuwm.resume import (offline_child_resume_refusal,
                              offline_child_run_at)

    outdir = _child_run(tmp_path / "child-run", finished=False, frames=2)
    _blown_up_report(outdir)

    child = offline_child_run_at(outdir)
    # The document is there, and it does not say the child finished.
    assert child.report is not None and child.result == "FAIL"
    assert child.finished is False

    action = render(offline_child_resume_refusal(child), explain=False)
    assert "run `gpuwm downscale` again" in action
    assert "nothing to resume" not in action
    assert "reached its last frame" not in action
    assert "draw the 2 frame(s)" in action
    # And the capsule the run already wrote is quoted back, so the
    # reader learns why it stopped without opening the report.
    assert "The child blew up:" in action
    assert "at one cell, (k=12, j=401, i=388)" in action


def test_a_blown_up_child_is_recognised_without_a_plan_document(tmp_path):
    """The runner door writes no plan, and a failure report has no steps."""
    from gpuwm.resume import offline_child_run_at

    outdir = tmp_path / "runner-child"
    outdir.mkdir()
    (outdir / "child.toml").write_text("[grid]\n", encoding="utf-8")
    _blown_up_report(outdir)

    child = offline_child_run_at(outdir)
    assert child is not None and child.finished is False
    assert child.failure.startswith("The child blew up:")


def test_a_child_that_recorded_no_config_is_still_a_child(tmp_path):
    """THE REPORTED SHAPE: the configuration was handed in from outside.

    Only the ``--point`` derivation writes ``child.toml`` into
    ``--out``.  A ``--child-config`` run -- which is every run a desktop
    or a script composes the configuration for -- leaves the plan
    document, its frames and its report, and read by the configuration
    file it was not a child at all: the capsule under its own report
    was never quoted, and the reader was sent to look for checkpoints
    that a child never writes for this purpose.
    """
    from gpuwm.explain import render
    from gpuwm.resume import (offline_child_resume_refusal,
                              offline_child_run_at)

    outdir = _child_run(tmp_path / "child-run", finished=False, frames=2,
                        records_config=False)
    _blown_up_report(outdir)
    assert not (outdir / "child.toml").exists()

    child = offline_child_run_at(outdir)
    assert child is not None and child.config is None
    assert child.result == "FAIL" and child.finished is False

    action = render(offline_child_resume_refusal(child), explain=False)
    assert "The child blew up:" in action
    assert "run `gpuwm downscale` again" in action
    assert "draw the 2 frame(s)" in action

    # And with neither the plan document nor the configuration, which
    # is that same run under the runner door: the capsule in the report
    # is the whole of what says a child wrote this directory.
    runner = tmp_path / "runner-child"
    runner.mkdir()
    _blown_up_report(runner)
    named = offline_child_run_at(runner)
    assert named is not None and named.config is None
    assert named.failure.startswith("The child blew up:")


def test_the_refusal_names_why_a_child_cannot_be_continued(tmp_path):
    """The explain half carries the breakage, not just the remedy."""
    from gpuwm.resume import (offline_child_resume_refusal,
                              offline_child_run_at)
    from gpuwm.explain import render

    outdir = _child_run(tmp_path / "child-run", finished=True)
    why = render(offline_child_resume_refusal(offline_child_run_at(outdir)),
                 explain=True)
    assert "parent history archive" in why
    assert "lateral boundaries" in why
    assert "no flag that continues a child" in why


# --- the two doors, through the front door -----------------------------


def test_cli_resume_loads_the_config_the_run_recorded(tmp_path, monkeypatch,
                                                      capsys):
    """THE REPORTED INVOCATION, end to end, to the (stubbed) run dispatch."""
    from gpuwm import capabilities

    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))
    _single(tmp_path, "1974-04-03_13_00_00")
    install = tmp_path / "install"
    install.mkdir()
    (tmp_path / "captured-config-0001.toml").write_text(
        "[experiment]\n", encoding="utf-8")

    seen = {}

    def fake_load(path, **kwargs):
        seen["config"] = path
        return SimpleNamespace(name="stub-exp"), object()

    def fake_run(exp, data, outdir, *, restart=None, health_debug=False):
        seen["restart"] = restart
        return SimpleNamespace(wrfout_paths=[], completed_seconds=0.0,
                               nan_free=True)

    import gpuwm.case_data as case_data
    import gpuwm.runtime as runtime
    monkeypatch.setattr(case_data, "load_experiment_case", fake_load)
    monkeypatch.setattr(runtime, "run_experiment", fake_run)
    monkeypatch.setattr(cli, "is_experiment_toml", lambda path: True)
    monkeypatch.chdir(install)

    rc = cli.main(["resume", "somerun.child", "--outdir", str(tmp_path),
                   "--no-supervise"])
    assert rc == 0
    assert Path(seen["config"]) == tmp_path / "captured-config-0001.toml"
    out = capsys.readouterr().out
    assert "is not a readable configuration file" in out
    assert "recorded for itself" in out
    assert "continuing from" in out


def test_cli_resume_refuses_a_child_run_directory(tmp_path, monkeypatch,
                                                  capsys):
    """And the run directory is asked before the argument is judged."""
    from gpuwm import capabilities

    installed = capabilities.is_installed
    monkeypatch.setattr(capabilities, "is_installed",
                        lambda module: module == "cupy" or installed(module))
    outdir = _child_run(tmp_path / "child-run", finished=True, frames=3)
    _single(outdir, "1974-04-03_13_00_00")

    monkeypatch.chdir(tmp_path)
    rc = cli.main(["resume", "somerun.child", "--outdir", str(outdir),
                   "--no-supervise"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nothing to resume" in err
    assert "gpuwm render" in err
    # The argument was never the subject: no ladder, no loader refusal.
    assert "does not exist; pass the experiment .toml" not in err


def test_cli_resume_refuses_a_blown_up_child_that_recorded_no_config(
        tmp_path, monkeypatch, capsys):
    """The door itself, on the directory the reported run left behind.

    A child that blew up and recorded no configuration used to reach
    the checkpoint resolution, which answered "no gpuwmrst_d*.npz
    checkpoint files" -- true, and not what happened to the run.
    """
    outdir = _child_run(tmp_path / "child-run", finished=False, frames=1,
                        records_config=False)
    _blown_up_report(outdir)

    monkeypatch.chdir(tmp_path)
    rc = cli.main(["resume", "child-settings.toml", "--outdir", str(outdir),
                   "--no-supervise"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "The child blew up:" in err
    assert "run `gpuwm downscale` again" in err
    assert "at one cell, (k=12, j=401, i=388)" in err
    # The directory answered before the argument was judged, so neither
    # the checkpoint ladder nor the experiment ladder ran.
    assert "checkpoint files" not in err
    assert "is not a readable configuration file" not in err
