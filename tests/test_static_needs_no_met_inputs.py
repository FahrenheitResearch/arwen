"""`gpuwm static` builds geography without a met cycle on disk.

``runtime.write_static`` reads ``geog_root``, the WPS namelist and the
projection.  It never opens the forcing GRIB or the Vtable -- and yet
through 2.3.2 the shared ``[case_data]`` loader required both to exist
before it would hand the config over, so "build 30 m terrain for this
domain" refused until a whole forecast cycle had been downloaded.  That
is a gate on bytes the command does not read, and it sat directly across
the documented high-resolution terrain path.

What is scoped is only the EXISTENCE check.  Both keys are still
required declarations, because the config still describes a whole case,
and `gpuwm run` still refuses at the front door when its forcing is
absent -- a forecast that starts without forcing is a forecast that fails
an hour in.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from gpuwm.case_data import build_case_data


def _table(tmp_path: Path, forcing: str = "absent.grib") -> dict:
    geog = tmp_path / "WPS_GEOG"
    geog.mkdir(exist_ok=True)
    (tmp_path / "namelist.wps").write_text("&share\n/\n", encoding="utf-8")
    return {
        "forcing": [forcing],
        "vtable": "absent.Vtable",
        "wps_namelist": "namelist.wps",
        "geog_root": "WPS_GEOG",
        "sfcp_to_sfcp": True,
        "output_title": "t",
    }


def test_a_run_still_refuses_absent_forcing(tmp_path):
    """The default is unchanged and stays strict."""
    with pytest.raises(ValueError, match="does not exist"):
        build_case_data(_table(tmp_path), source="t.toml",
                        base_dir=tmp_path)


def test_a_static_build_accepts_absent_forcing_and_vtable(tmp_path):
    data = build_case_data(_table(tmp_path), source="t.toml",
                           base_dir=tmp_path, require_met_inputs=False)
    assert data.forcing == (tmp_path / "absent.grib",)
    assert data.vtable == tmp_path / "absent.Vtable"


def test_a_static_build_still_requires_what_it_actually_reads(tmp_path):
    """Scoped, not disabled: geog_root and the namelist stay required."""
    table = _table(tmp_path)
    (tmp_path / "namelist.wps").unlink()
    with pytest.raises(ValueError, match="wps_namelist"):
        build_case_data(table, source="t.toml", base_dir=tmp_path,
                        require_met_inputs=False)

    table = _table(tmp_path)
    table["geog_root"] = "no-such-tree"
    with pytest.raises(ValueError, match="geog_root"):
        build_case_data(table, source="t.toml", base_dir=tmp_path,
                        require_met_inputs=False)


def test_the_declarations_are_still_mandatory(tmp_path):
    """Relaxing existence must not relax the schema."""
    table = _table(tmp_path)
    del table["forcing"]
    with pytest.raises(ValueError, match="missing required key"):
        build_case_data(table, source="t.toml", base_dir=tmp_path,
                        require_met_inputs=False)


def test_an_unmatched_forcing_glob_is_empty_not_fatal_for_static(tmp_path):
    """A declared pattern that matches nothing expands to nothing.

    It must not invent a path that does not exist, and it must not
    refuse a command that never reads the files.
    """
    data = build_case_data(_table(tmp_path, "cycle-*.grib"),
                           source="t.toml", base_dir=tmp_path,
                           require_met_inputs=False)
    assert data.forcing == ()

    with pytest.raises(ValueError, match="matched no files"):
        build_case_data(_table(tmp_path, "cycle-*.grib"),
                        source="t.toml", base_dir=tmp_path)


def test_the_static_command_asks_for_the_relaxation(tmp_path):
    """The wiring, not just the capability: cli.py must pass it.

    A parameter nothing sets is a parameter that fixes nothing.
    """
    source = (Path(__file__).resolve().parents[1] / "gpuwm" / "cli.py"
              ).read_text(encoding="utf-8")
    assert 'require_met_inputs=args.command != "static"' in source, (
        "gpuwm/cli.py no longer scopes the met-input existence check to "
        "non-static commands; `gpuwm static` is demanding a forcing GRIB "
        "it never reads again")
