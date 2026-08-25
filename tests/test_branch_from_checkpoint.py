"""``gpuwm branch`` -- seed a NEW run from an existing checkpoint.

CPU-only.  No forecast is integrated here: every assertion is about
what the door DECIDES and what it WRITES, which is exactly the half a
machine without a card can adjudicate.

The trust anchor the spec asks for -- "an unchanged branch produces
frames byte-identical to the continued original" -- is split in two
deliberately, and neither half is faked:

* the CPU half, proven here, is that an unchanged branch hands the run
  machinery byte-identical INPUTS: the same restart identity payload
  (:func:`gpuwm.core.model.restart_identity_payload`, the definition
  ``experiment_fingerprint`` hashes) and the same checkpoint file plain
  ``gpuwm resume`` would have resolved.  Same state in, same identity,
  same code path.
* the FRAME half needs a GPU and real forcing bytes, so it is
  ``test_frames_match_the_continued_original_on_a_card`` below: an
  env-gated node-campaign row that SKIPS when the case is not staged
  rather than asserting something weaker and calling it proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

import gpuwm.cli as cli
from gpuwm.branch import (BRANCH_MANIFEST_NAME, BRANCH_RECEIPT_NAME,
                          BRANCHED_CONFIG_NAME, emit_experiment_toml,
                          prepare_branch)
from gpuwm.core.model import restart_identity_payload
from gpuwm.experiment import load_experiment
from gpuwm.resume import resolve_resume_checkpoint

_HEADER_KEY = "__gpuwm_restart_header__"

#: A two-domain experiment with a tracker: enough surface for both
#: halves of the split -- the ``[relocation]`` bounds a branch MAY
#: change and the per-domain geometry/physics it may not.
BASE = """\
[experiment]
name = "synth"
start_time = 1974-04-03T12:00:00
run_seconds = 3600.0
restart_interval_s = 900.0

[shared]
nz = 8
ztop = 12000.0

[relocation]
enabled = true
grid_id = 2
max_move_parent_cells = 4
min_overlap_fraction = 0.5
cadence_seconds = 900.0

[relocation.follow]
field = "uh"
threshold = 25.0
fallback_threshold = 40.0
search_margin_cells = 15
min_shift_cells = 2
max_shift_cells = 4
cooldown_seconds = 900.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 100
ny = 80
time_step = 60
dx = 12000.0
history_interval_s = 900.0

[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 40
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
e_we = 61
e_sn = 61
history_interval_s = 900.0
"""


def _write_checkpoint(path: Path, *, grid_id: int, domain_ids=None) -> None:
    """A genuine-format restart NPZ: real header, real array manifest."""
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
    payload = {_HEADER_KEY: np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8)}
    payload.update(arrays)
    with path.open("wb") as stream:
        np.savez(stream, **payload)


@pytest.fixture()
def source_run(tmp_path):
    """A finished-looking run directory: config, wrfout, checkpoint set."""
    run = tmp_path / "source-run"
    run.mkdir()
    config = tmp_path / "exp.toml"
    config.write_text(BASE, newline="\n")
    instant = "1974-04-03_12_15_00"
    for gid in (1, 2):
        _write_checkpoint(run / f"gpuwmrst_d{gid:02d}_{instant}__set1.npz",
                          grid_id=gid, domain_ids=[1, 2])
    (run / "wrfout_d01_1974-04-03_12-15-00").write_bytes(b"frame")
    (run / "progress.jsonl").write_text('{"event": "run_start"}\n',
                                        newline="\n")
    return config, run


def _tree_state(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"):
            hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()}


# --- the emitter contract ------------------------------------------------


def test_the_emitter_round_trips_an_experiment_config(source_run):
    config, _ = source_run
    raw = tomllib.loads(config.read_text())
    assert tomllib.loads(emit_experiment_toml(raw)) == raw


# --- the trust anchor, CPU half -----------------------------------------


def test_an_unchanged_branch_reparses_to_the_source_config(source_run,
                                                           tmp_path):
    config, run = source_run
    plan = prepare_branch(config=config, from_run=run,
                          outdir=tmp_path / "branch-run")
    assert tomllib.loads(plan.config_path.read_text()) == \
        tomllib.loads(config.read_text())


def test_an_unchanged_branch_keeps_the_restart_identity_byte_identical(
        source_run, tmp_path):
    config, run = source_run
    plan = prepare_branch(config=config, from_run=run,
                          outdir=tmp_path / "branch-run")
    parent = restart_identity_payload(load_experiment(config))
    branched = restart_identity_payload(load_experiment(plan.config_path))
    assert branched == parent
    assert plan.receipt["restart_identity"]["equal"] is True
    assert (plan.receipt["restart_identity"]["payload_sha256"]
            == plan.receipt["restart_identity"]["parent_payload_sha256"])


def test_an_unchanged_branch_takes_the_checkpoint_plain_resume_would(
        source_run, tmp_path):
    config, run = source_run
    plan = prepare_branch(config=config, from_run=run,
                          outdir=tmp_path / "branch-run")
    assert plan.checkpoint == resolve_resume_checkpoint(
        run, "latest", config=config).checkpoint


# --- pinned settings refuse BY NAME with the pinned reason ---------------


@pytest.mark.parametrize("setting", [
    "name=other",
    "start_time=1974-04-03T18:00:00",
    "shared.nz=12",
    "domain.1.nx=200",
    "domain.1.time_step=30",
    "domain.1.run.mp_physics=10",
    "domain.2.i_parent_start=44",
])
def test_a_pinned_setting_is_refused_by_name(source_run, tmp_path, setting):
    config, run = source_run
    name = setting.split("=")[0]
    with pytest.raises(ValueError) as caught:
        prepare_branch(config=config, from_run=run,
                       outdir=tmp_path / "branch-run", settings=[setting])
    message = str(caught.value)
    assert f"`{name}`" in message
    # The gate names the concrete breakage, not just "refused".
    assert "restart identity" in message
    assert "restart_identity_payload" in message
    # ... and the way out.
    assert "run_seconds" in message and "relocation." in message
    assert not (tmp_path / "branch-run").exists()


def test_the_follow_refusal_points_at_the_branchable_tracker_bounds(
        source_run, tmp_path):
    """The one split a what-if screen will get wrong if we do not say it.

    ``[relocation.follow]`` is tracker BOUNDS and branchable;
    ``[[domain]] [follow]`` is where a child sits and binds.
    """
    config, run = source_run
    with pytest.raises(ValueError) as caught:
        prepare_branch(config=config, from_run=run,
                       outdir=tmp_path / "branch-run",
                       settings=["domain.2.follow.level_hpa=500"])
    message = str(caught.value)
    assert "`domain.2.follow.level_hpa`" in message
    assert "[relocation.follow]" in message


def test_an_unknown_setting_is_refused_by_name(source_run, tmp_path):
    config, run = source_run
    with pytest.raises(ValueError) as caught:
        prepare_branch(config=config, from_run=run,
                       outdir=tmp_path / "branch-run",
                       settings=["run_second=7200"])
    assert "`run_second`" in str(caught.value)


def test_a_legacy_run_table_config_is_refused_with_the_reason(source_run,
                                                              tmp_path):
    config, run = source_run
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[run]\ncase = "something"\n', newline="\n")
    with pytest.raises(ValueError, match="legacy"):
        prepare_branch(config=legacy, from_run=run,
                       outdir=tmp_path / "branch-run")
    assert not (tmp_path / "branch-run").exists()


def test_a_setting_without_an_equals_sign_is_refused(source_run, tmp_path):
    config, run = source_run
    with pytest.raises(ValueError, match="run_seconds"):
        prepare_branch(config=config, from_run=run,
                       outdir=tmp_path / "branch-run",
                       settings=["run_seconds"])


# --- legal overrides apply and are stamped ------------------------------


def test_legal_overrides_apply_and_are_stamped_in_the_new_provenance(
        source_run, tmp_path):
    config, run = source_run
    plan = prepare_branch(
        config=config, from_run=run, outdir=tmp_path / "branch-run",
        settings=["run_seconds=7200.0",
                  "relocation.follow.threshold=40.0",
                  "domain.2.history_interval_s=1800.0",
                  "output.preset=minimal"])
    raw = tomllib.loads(plan.config_path.read_text())
    assert raw["experiment"]["run_seconds"] == 7200.0
    assert raw["relocation"]["follow"]["threshold"] == 40.0
    assert raw["domain"][1]["history_interval_s"] == 1800.0
    assert raw["output"]["preset"] == "minimal"
    # It still loads, and it still resumes the same checkpoint.
    branched = load_experiment(plan.config_path)
    assert branched.run_seconds == 7200.0
    assert restart_identity_payload(branched) == \
        restart_identity_payload(load_experiment(config))

    stamped = {row["setting"]: row for row in plan.receipt["overrides"]}
    assert set(stamped) == {"run_seconds", "relocation.follow.threshold",
                            "domain.2.history_interval_s", "output.preset"}
    assert stamped["run_seconds"]["from"] == 3600.0
    assert stamped["run_seconds"]["to"] == 7200.0
    assert stamped["relocation.follow.threshold"]["from"] == 25.0
    assert stamped["output.preset"]["from"] is None
    assert stamped["output.preset"]["to"] == "minimal"


def test_an_illegal_value_for_a_changeable_setting_leaves_no_run_dir(
        source_run, tmp_path):
    """A refusal must not poison its own --outdir for the retry.

    ``output`` IS changeable, so the name table passes it through and
    the schema owner is what refuses -- which happens after the branch
    config has been written at its final path.  The corrected retry has
    to find an empty target, not a half-made run folder.
    """
    config, run = source_run
    target = tmp_path / "branch-run"
    with pytest.raises(ValueError, match="preset"):
        prepare_branch(config=config, from_run=run, outdir=target,
                       settings=["output.preset=nonsense"])
    assert not target.exists()


def test_a_pre_existing_empty_target_survives_a_refusal(source_run,
                                                        tmp_path):
    config, run = source_run
    target = tmp_path / "branch-run"
    target.mkdir()
    with pytest.raises(ValueError, match="preset"):
        prepare_branch(config=config, from_run=run, outdir=target,
                       settings=["output.preset=nonsense"])
    assert target.is_dir() and not list(target.iterdir())


def test_the_receipt_and_manifest_record_the_parent_checkpoint(source_run,
                                                               tmp_path):
    config, run = source_run
    plan = prepare_branch(config=config, from_run=run,
                          outdir=tmp_path / "branch-run")
    receipt = json.loads((plan.outdir / BRANCH_RECEIPT_NAME).read_text())
    assert receipt["schema"] == "gpuwm-branch-receipt/v1"
    assert Path(receipt["parent"]["run_directory"]) == run.resolve()
    assert Path(receipt["parent"]["checkpoint"]["path"]) == \
        plan.checkpoint.resolve()
    assert Path(receipt["parent"]["config"]["path"]) == config.resolve()
    assert receipt["parent"]["config"]["sha256"] == \
        hashlib.sha256(config.read_bytes()).hexdigest()

    manifest = json.loads((plan.outdir / BRANCH_MANIFEST_NAME).read_text())
    assert manifest["schema"] == "gpuwm-branch-manifest/v1"
    members = {row["grid_id"]: row for row in manifest["members"]}
    assert sorted(members) == [1, 2]
    for grid_id, row in members.items():
        member = run / Path(row["path"]).name
        assert row["sha256"] == \
            hashlib.sha256(member.read_bytes()).hexdigest()
        assert row["bytes"] == member.stat().st_size
    assert manifest["valid_time"] == "1974-04-03T12:15:00"


# --- the source run is read-only ----------------------------------------


def test_the_branch_never_writes_into_the_source_run(source_run, tmp_path):
    config, run = source_run
    before = _tree_state(run)
    prepare_branch(config=config, from_run=run,
                   outdir=tmp_path / "branch-run",
                   settings=["run_seconds=7200.0"])
    assert _tree_state(run) == before


def test_a_target_inside_the_source_run_is_refused(source_run):
    config, run = source_run
    with pytest.raises(ValueError, match="inside"):
        prepare_branch(config=config, from_run=run,
                       outdir=run / "what-if")
    assert not (run / "what-if").exists()


def test_the_source_run_itself_is_refused_as_a_target(source_run):
    config, run = source_run
    with pytest.raises(ValueError, match="inside|same"):
        prepare_branch(config=config, from_run=run, outdir=run)


def test_a_non_empty_target_is_refused(source_run, tmp_path):
    config, run = source_run
    target = tmp_path / "branch-run"
    target.mkdir()
    (target / "wrfout_d01_1974-04-03_12-00-00").write_bytes(b"someone else")
    with pytest.raises(ValueError, match="not empty"):
        prepare_branch(config=config, from_run=run, outdir=target)


def test_declared_input_paths_are_rebased_onto_the_source_directory(
        tmp_path):
    """A branched config lives elsewhere, so relative inputs must move.

    ``load_experiment`` resolves ``[case_data]`` paths against the
    config FILE's directory.  Copy the text unchanged into a new run
    folder and every relative declaration silently points at nothing.
    """
    home = tmp_path / "case"
    (home / "data").mkdir(parents=True)
    (home / "geog").mkdir()
    forcing = home / "data" / "forcing.nc"
    forcing.write_bytes(b"bytes")
    (home / "Vtable.SOURCE").write_text("vtable\n", newline="\n")
    (home / "namelist.wps").write_text("&share\n/\n", newline="\n")
    config = home / "exp.toml"
    config.write_text(BASE + "\n" + "\n".join([
        "[case_data]",
        'forcing = "data/forcing.nc"',
        'vtable = "Vtable.SOURCE"',
        'wps_namelist = "namelist.wps"',
        'geog_root = "geog"',
        "sfcp_to_sfcp = true",
        'output_title = "synth"',
        "",
    ]), newline="\n")
    run = tmp_path / "source-run"
    run.mkdir()
    _write_checkpoint(run / "gpuwmrst_d01_1974-04-03_12_15_00.npz",
                      grid_id=1)

    plan = prepare_branch(config=config, from_run=run,
                          outdir=tmp_path / "branch-run")
    raw = tomllib.loads(plan.config_path.read_text())
    assert Path(raw["case_data"]["forcing"]) == forcing.resolve()
    rebased = {row["setting"] for row in plan.receipt["rebased_paths"]}
    assert rebased == {"case_data.forcing", "case_data.vtable",
                       "case_data.wps_namelist", "case_data.geog_root"}
    # The relocated config still loads its declared case from the new
    # location, which is the whole point of the rebase.
    from gpuwm.case_data import load_experiment_case
    load_experiment_case(plan.config_path, require_met_inputs=False)


# --- the CLI door --------------------------------------------------------


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


def test_branch_parser_carries_runs_supervision_surface(tmp_path):
    args = _parse(["branch", "cfg.toml", "--from-run", str(tmp_path),
                   "--outdir", str(tmp_path / "new"),
                   "--set", "run_seconds=7200",
                   "--no-supervise", "--supervisor-max-restarts", "5"])
    assert args.command == "branch"
    assert args.from_checkpoint == "latest"
    assert args.settings == ["run_seconds=7200"]
    assert args.no_supervise is True
    assert args.supervisor_max_restarts == 5
    assert args.health_debug is False
    assert args.prepare_only is False


def test_cli_branch_dispatches_as_run_into_the_new_outdir(
        source_run, tmp_path, monkeypatch, capsys):
    config, run = source_run
    target = tmp_path / "branch-run"
    seen = {}

    def fake_load(path, **kwargs):
        from types import SimpleNamespace
        seen["config"] = Path(path)
        return SimpleNamespace(name="stub-exp"), object()

    def fake_run(exp, data, outdir, *, restart=None, health_debug=False):
        from types import SimpleNamespace
        seen["restart"] = restart
        seen["outdir"] = Path(outdir)
        return SimpleNamespace(wrfout_paths=[], completed_seconds=0.0,
                               nan_free=True)

    import gpuwm.case_data as case_data
    import gpuwm.runtime as runtime
    monkeypatch.setattr(case_data, "load_experiment_case", fake_load)
    monkeypatch.setattr(runtime, "run_experiment", fake_run)

    rc = cli.main(["branch", str(config), "--from-run", str(run),
                   "--outdir", str(target), "--set", "run_seconds=7200.0",
                   "--no-supervise"])
    assert rc == 0
    assert seen["config"] == (target / BRANCHED_CONFIG_NAME)
    assert seen["outdir"] == target
    assert seen["restart"].name.startswith("gpuwmrst_d01_")
    assert seen["restart"].parent == run
    out = capsys.readouterr().out
    assert "branch: " in out


def test_cli_branch_prepare_only_stops_before_the_run(source_run, tmp_path,
                                                      monkeypatch):
    config, run = source_run
    target = tmp_path / "branch-run"

    def refuse(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("--prepare-only must not integrate")

    import gpuwm.runtime as runtime
    monkeypatch.setattr(runtime, "run_experiment", refuse)

    rc = cli.main(["branch", str(config), "--from-run", str(run),
                   "--outdir", str(target), "--prepare-only"])
    assert rc == 0
    assert (target / BRANCHED_CONFIG_NAME).is_file()
    assert (target / BRANCH_RECEIPT_NAME).is_file()
    assert (target / BRANCH_MANIFEST_NAME).is_file()


def test_cli_branch_prints_a_pinned_refusal_as_one_sentence(source_run,
                                                            tmp_path,
                                                            capsys):
    config, run = source_run
    rc = cli.main(["branch", str(config), "--from-run", str(run),
                   "--outdir", str(tmp_path / "branch-run"),
                   "--set", "domain.1.run.mp_physics=10"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("gpuwm branch: ")
    assert "`domain.1.run.mp_physics`" in err
    assert "Traceback" not in err


# --- the frame half: a card and real bytes, on the node campaign ---------

#: Point this at a staged run directory whose config, checkpoint and
#: forcing are all present, and this row runs the real comparison the
#: spec's phase-5 proof names.  Absent, it SKIPS: a frame equality
#: nobody integrated is not evidence of anything.
BRANCH_TRUST_ANCHOR_ENV = "GPUWM_BRANCH_TRUST_ANCHOR_RUN"


@pytest.mark.gpu
@pytest.mark.slow
def test_frames_match_the_continued_original_on_a_card(tmp_path):
    staged = os.environ.get(BRANCH_TRUST_ANCHOR_ENV, "")
    if not staged:
        pytest.skip(
            f"{BRANCH_TRUST_ANCHOR_ENV} is unset: the frame half of the "
            "trust anchor needs a card and real forcing bytes, so it "
            "runs on the node campaign, not here")
    import shutil

    source = Path(staged)
    config = source / "config.toml"
    assert config.is_file(), f"{config} must be the run's config"
    checkpoint = resolve_resume_checkpoint(
        source, "latest", config=config).checkpoint

    # Arm A continues the original in a copy of it, so the staged run is
    # left exactly as it was and can arm the next attempt.  Arm B branches
    # the untouched original from the SAME checkpoint, with no overrides.
    continued = tmp_path / "continued"
    shutil.copytree(source, continued)
    branched = tmp_path / "branched"
    assert cli.main(["resume", str(config), "--outdir", str(continued),
                     "--from", str(continued / checkpoint.name)]) == 0
    assert cli.main(["branch", str(config), "--from-run", str(source),
                     "--from", str(checkpoint),
                     "--outdir", str(branched)]) == 0

    # The branch directory starts empty, so every frame in it was written
    # after the checkpoint; each must byte-equal its counterpart in the
    # continued arm.
    produced = sorted(branched.glob("wrfout_d*"))
    assert produced, "the branch wrote no history frames"
    for frame in produced:
        counterpart = continued / frame.name
        assert counterpart.is_file(), frame.name
        assert frame.read_bytes() == counterpart.read_bytes(), frame.name
