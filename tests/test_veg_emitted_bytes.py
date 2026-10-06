"""Generic vegetation emission retains original pre-lane configuration bytes."""
from datetime import datetime
from pathlib import Path
import tomllib

import pytest

from gpuwm.domain_wizard import render_config
from gpuwm.experiment import build_experiment_from_config_tables
from gpuwm.namelist_import import import_namelists
from test_audit_parent_generic_bytes import assert_old_bytes, controls


VEG_AUDIT_PARENT = "1077c6e8abe502aab59de103d7e6d70dbb8588a1"
SWITCHES = ("usemonalb", "rdlai2d", "fractional_seaice")


def assert_generic_runtime_off(text):
    experiment = build_experiment_from_config_tables(
        tomllib.loads(text), source="generic vegetation byte control",
        base_dir=Path.cwd())
    for domain in experiment.domains:
        assert domain.run.usemonalb is False
        assert domain.run.rdlai2d is False
        assert domain.run.fractional_seaice == 0


@pytest.mark.parametrize("case", ("ruc_monthly_omitted", "ruc_monthly_false0"))
def test_generic_vegetation_namelist_bytes_match_original_parent(case, tmp_path):
    matrix, pins = controls(VEG_AUDIT_PARENT)
    documents = matrix["namelists"][case]
    wps, inp = tmp_path / "namelist.wps", tmp_path / "namelist.input"
    wps.write_text(documents["wps"], encoding="utf-8")
    inp.write_text(documents["input"], encoding="utf-8")
    emitted, _ = import_namelists(wps, inp)
    shared = tomllib.loads(emitted)["shared"]
    if case == "ruc_monthly_omitted":
        assert set(SWITCHES).isdisjoint(shared)
    else:
        # These explicit false booleans were emitted by the old importer.
        # Fractional sea-ice zero was omitted by that historical importer.
        assert shared["usemonalb"] is False and shared["rdlai2d"] is False
        assert "fractional_seaice" not in shared
    assert_old_bytes(VEG_AUDIT_PARENT, pins, "namelist/" + case, emitted)
    assert_generic_runtime_off(emitted)


@pytest.mark.parametrize("source", ("gfs", "era5", "rap", "rrfs"))
def test_generic_vegetation_recipe_bytes_match_original_parent(source):
    matrix, pins = controls(VEG_AUDIT_PARENT)
    recipe = matrix["recipes"]
    emitted = render_config(
        name=recipe["name"], start_time=datetime.fromisoformat(recipe["start_time"]),
        hours=recipe["hours"], projection=recipe["projection"],
        dims=[tuple(row) for row in recipe["dims"]], ratios=tuple(recipe["ratios"]),
        root_dx_m=recipe["root_dx_m"], fetch_hints={"source": source},
        case_data=None, profile=recipe["profile"],
    )
    assert set(SWITCHES).isdisjoint(tomllib.loads(emitted)["shared"])
    assert_old_bytes(VEG_AUDIT_PARENT, pins, "recipe/" + source, emitted)
    assert_generic_runtime_off(emitted)
