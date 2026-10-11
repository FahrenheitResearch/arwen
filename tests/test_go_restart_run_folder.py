"""`gpuwm go CONFIG --restart CHECKPOINT` resumes a custom-domain run folder.

A `gpuwm go` run of a `gpuwm domain` configuration binds its prepared
bundle to the authority stage's generated experiment config and WPS
namelist, so the resume needed --prepared-root, --wps-namelist and the
generated config in place of the user's own; with the user's config the
restore refused "supplied experiment_config differs from the portable
source manifest".  These tests hold the run-folder resolution that closes
it, and the refusals it keeps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gpuwm import go_cli


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_folder(tmp_path: Path) -> dict:
    """A go run folder as the authority stage and the forecast leave it."""

    config = tmp_path / "dom.toml"
    config.write_text("[experiment]\nname = \"dom\"\n", encoding="utf-8")
    wps = tmp_path / "dom.namelist.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    folder = tmp_path / "go" / "run-20261009-000000Z_i202610081200Z"
    authority = folder / "authority"
    authority.mkdir(parents=True)
    generated = authority / "experiment.toml"
    generated.write_text("[experiment]\nname = \"dom\"\nmap_proj = 1\n",
                         encoding="utf-8")
    generated_wps = authority / "namelist.wps"
    generated_wps.write_text("&share\n/\n", encoding="utf-8")
    (authority / "authority-receipt.json").write_text(json.dumps({
        "base": {
            "experiment_config": {"path": str(config), "sha256": _sha(config)},
            "wps_namelist": {"path": str(wps), "sha256": _sha(wps)}},
        "generated": {
            "experiment_config": {"path": str(generated),
                                  "sha256": _sha(generated)},
            "wps_namelist": {"path": str(generated_wps),
                             "sha256": _sha(generated_wps)}},
    }), encoding="utf-8")
    (folder / "prepared").mkdir()
    (folder / "run").mkdir()
    checkpoint = folder / "run" / "gpuwmrst_d01_2026-10-08_13_00_00__abc.npz"
    checkpoint.write_bytes(b"checkpoint")
    return {"config": config, "folder": folder, "generated": generated,
            "generated_wps": generated_wps, "checkpoint": checkpoint}


def _args(**values) -> argparse.Namespace:
    return argparse.Namespace(**{"restart": None, "prepared_root": None,
                                 "wps_namelist": None, **values})


def test_a_go_checkpoint_resumes_from_its_own_run_folder(tmp_path):
    run = _run_folder(tmp_path)
    args = _args(restart=run["checkpoint"])
    bound = go_cli._checkpoint_run_bindings(args, run["config"])
    assert bound == run["generated"]
    assert Path(args.prepared_root) == run["folder"] / "prepared"
    assert Path(args.wps_namelist) == run["generated_wps"]
    # The generated config itself is the same run's configuration too.
    args = _args(restart=run["checkpoint"])
    assert go_cli._checkpoint_run_bindings(args, run["generated"]) \
        == run["generated"]


def test_explicit_flags_stay_as_given(tmp_path):
    run = _run_folder(tmp_path)
    wps = tmp_path / "mine.wps"
    wps.write_text("&share\n/\n", encoding="utf-8")
    args = _args(restart=run["checkpoint"], wps_namelist=wps)
    assert go_cli._checkpoint_run_bindings(args, run["config"]) \
        == run["generated"]
    assert args.wps_namelist == wps
    other = tmp_path / "other-prepared"
    other.mkdir()
    args = _args(restart=run["checkpoint"], prepared_root=other)
    assert go_cli._checkpoint_run_bindings(args, run["config"]) is None
    assert args.prepared_root == other and args.wps_namelist is None


def test_an_edited_configuration_is_refused_by_name(tmp_path):
    run = _run_folder(tmp_path)
    run["config"].write_text("[experiment]\nname = \"edited\"\n",
                             encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="is not that configuration"):
        go_cli._checkpoint_run_bindings(
            _args(restart=run["checkpoint"]), run["config"])


def test_edited_authority_files_are_refused_by_name(tmp_path):
    run = _run_folder(tmp_path)
    run["generated"].write_text("[experiment]\nname = \"x\"\n",
                                encoding="utf-8")
    with pytest.raises(go_cli.GoRefusal, match="no longer match"):
        go_cli._checkpoint_run_bindings(
            _args(restart=run["checkpoint"]), run["config"])


def test_a_checkpoint_outside_a_go_run_folder_is_left_to_the_run_plan(tmp_path):
    run = _run_folder(tmp_path)
    loose = tmp_path / "gpuwmrst_d01_2026-10-08_13_00_00__abc.npz"
    loose.write_bytes(b"checkpoint")
    args = _args(restart=loose)
    assert go_cli._checkpoint_run_bindings(args, run["config"]) is None
    assert args.prepared_root is None
    assert go_cli._checkpoint_run_bindings(_args(), run["config"]) is None
