"""``gpuwm go --dry-run`` refuses an unusable WPS_GEOG tree, as the launch does.

The breakage: the dry run claims to validate the route, and both of its
branches (the prepared stage composer the GFS route takes, and the
registered launch the HRRR and ICON routes take) printed the plan and
exited 0 for ``--geog-root`` naming a directory that does not exist, while
the real launch of the same command refused with ``geography_refusal``
("the staged WPS_GEOG tree is not usable") before its fetch.  Measured on
the 2.8.8 candidate on node-1 with a GFS config: dry run rc 0, launch
refused.
"""
from __future__ import annotations

import pytest

from gpuwm import go_cli
from gpuwm.cli import main

#: The real check, taken before tests/conftest.py's autouse pin answers
#: the default root as staged, for the default-root negative control.
REAL_GEOGRAPHY_REFUSAL = go_cli.geography_refusal

_POINT = {"gfs": "--point=39,-98", "hrrr": "--point=39,-98",
          "icon-eu": "--point=50.1,8.7"}


def _emit(tmp_path, source):
    config = tmp_path / f"{source} area.toml"
    assert main(["domain", _POINT[source], "--card", "16gb", "--ladder", "12",
                 "--source", source, "--cycle", "2026-08-18T06",
                 "--hours", "3", "--out", str(config)]) == 0
    return config


def _staged(root):
    from gpuwm.geog_assets import geog_datasets

    geog = root / "WPS_GEOG"
    for name in geog_datasets():
        (geog / name).mkdir(parents=True, exist_ok=True)
        (geog / name / "index").write_text("", encoding="utf-8")
    return geog


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
def test_the_dry_run_refuses_a_geog_root_that_does_not_exist(
        tmp_path, capsys, source):
    config = _emit(tmp_path, source)
    capsys.readouterr()
    missing = tmp_path / "never-fetched" / "WPS_GEOG"
    rc = main(["go", str(config), "--dry-run", "--geog-root", str(missing),
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 2, captured.out + captured.err
    assert "the staged WPS_GEOG tree is not usable" in captured.err
    assert str(missing) in captured.err
    assert "gpuwm fetch-geog" in captured.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
def test_the_dry_run_passes_a_staged_geog_root(tmp_path, capsys, source):
    """Non-vacuity: the same command over a tree shaped like a real one."""
    config = _emit(tmp_path, source)
    geog = _staged(tmp_path / "geog")
    rc = main(["go", str(config), "--dry-run", "--geog-root", str(geog),
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "WPS_GEOG tree is not usable" not in captured.err


def test_the_dry_run_refuses_an_absent_default_root(
        tmp_path, capsys, monkeypatch):
    """The default root is asked too; the suite's pin is put back here."""
    monkeypatch.setattr(go_cli, "geography_refusal", REAL_GEOGRAPHY_REFUSAL)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    config = _emit(tmp_path, "gfs")
    capsys.readouterr()
    rc = main(["go", str(config), "--dry-run",
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 2, captured.out + captured.err
    assert "the staged WPS_GEOG tree is not usable" in captured.err
    assert str(tmp_path / "case-data" / "WPS_GEOG") in captured.err
