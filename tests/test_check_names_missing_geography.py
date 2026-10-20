"""``gpuwm check`` names geography the config reads that WPS_GEOG lacks.

Breakage (2.9.2 box door test): ``gpuwm check conus.toml --devices 2`` on
the HRRR region recipe passed, and ``gpuwm go`` of the same file refused
before its fetch for the unstaged ``orogwd3_10m`` indexes.  check now asks
the same function go refuses by and exits 6 when the memory legs pass.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm import cli


def _args(tmp_path, prepared_root=None):
    config = tmp_path / "exp.toml"
    config.write_text("[experiment]\nname = 'x'\n")
    return argparse.Namespace(config=str(config), prepared_root=prepared_root,
                              ingest_preflight_handler=lambda a: 0)


def test_check_names_the_geography_go_refuses(monkeypatch, capsys, tmp_path):
    seen = {}

    def refusal(geog_root, *, config=None):
        seen["config"] = config
        return ("the selected static fields require missing geography "
                "indexes in /g: orogwd3_10m/index\n  remedy: gpuwm fetch-geog "
                f"--config {config}")

    monkeypatch.setattr("gpuwm.go_cli.geography_refusal", refusal)
    monkeypatch.setattr(cli, "check_main", lambda a: 0)
    code = cli.combined_check_main(_args(tmp_path))
    err = capsys.readouterr().err
    assert code == cli.EXIT_GEOGRAPHY_MISSING
    assert "orogwd3_10m/index" in err
    assert f"gpuwm fetch-geog --config {seen['config']}" in err


@pytest.mark.parametrize("memory_code", [1, 6])
def test_a_harder_memory_verdict_wins(monkeypatch, capsys, tmp_path, memory_code):
    monkeypatch.setattr("gpuwm.go_cli.geography_refusal",
                        lambda geog_root, *, config=None: "missing x")
    monkeypatch.setattr(cli, "check_main", lambda a: memory_code)
    args = _args(tmp_path)
    assert cli.combined_check_main(args) == memory_code
    assert not args._geography_missing_only
    assert "BLOCKER" in capsys.readouterr().err


def test_staged_geography_is_silent(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("gpuwm.go_cli.geography_refusal",
                        lambda geog_root, *, config=None: None)
    monkeypatch.setattr(cli, "check_main", lambda a: 0)
    assert cli.combined_check_main(_args(tmp_path)) == 0
    assert "BLOCKER" not in capsys.readouterr().err


def test_a_prepared_bundle_is_not_asked(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("gpuwm.go_cli.geography_refusal",
                        lambda geog_root, *, config=None: "missing x")
    monkeypatch.setattr(cli, "check_main", lambda a: 0)
    assert cli.combined_check_main(_args(tmp_path, prepared_root="/p")) == 0
    assert "BLOCKER" not in capsys.readouterr().err


def test_domain_without_staged_geography_creates_config_and_names_next_step(tmp_path):
    """A fresh install can create the file fetch-geog needs next."""
    config = tmp_path / "fresh.toml"
    env = dict(os.environ, GPUWM_CASE_DATA_ROOT=str(tmp_path / "case-data"),
               GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="")
    done = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "domain", "--point=35.3,-97.5",
         "--card", "24gb", "--ladder", "12", "--source", "gfs",
         "--cycle", "2026-07-29T18", "--hours", "6", "--out", str(config),
         "--physics-profile", "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1"],
        capture_output=True, text=True, env=env, timeout=60)
    printed = done.stdout + done.stderr
    assert done.returncode == 0, printed
    assert config.is_file()
    assert not (tmp_path / "case-data" / "WPS_GEOG").exists()
    assert printed.count(f"gpuwm fetch-geog --config {config}") == 1
    from gpuwm.geog_assets import geography_for_run

    for dataset in geography_for_run(config)["datasets"]:
        assert printed.count(f"{dataset}/index") == 1
    assert "next step" in printed
    assert "gpuwm check FAILED" not in printed
    assert "gpuwm check: PASS (rc 0)" not in printed
@pytest.mark.parametrize("declared", ["staged", "${TEST_WOFS_GEOG}"])
def test_check_uses_the_configured_geography_root(monkeypatch, tmp_path, declared):
    args = _args(tmp_path)
    staged = tmp_path / "staged"
    monkeypatch.setenv("TEST_WOFS_GEOG", str(staged))
    Path(args.config).write_text(
        "[experiment]\nname = 'x'\n[case_data]\ngeog_root = '" + declared + "'\n")
    seen = []
    monkeypatch.setattr("gpuwm.go_cli.geography_refusal",
                        lambda root, **kwargs: seen.append(root))
    assert cli._check_geography(args) is None
    assert seen == [staged.resolve()]
