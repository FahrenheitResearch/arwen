"""Operational shortwave options are selected by the named source doors."""
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.config import radiation_scheme_ids
from gpuwm.physics_source_defaults import namelist_physics_defaults


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/source_requests"
RECIPE = ROOT / "configs/recipes/hrrr_configuration_clock.toml"


@pytest.mark.parametrize("name", ("hrrr_wrf.nl", "hrrr_wrf.nl.c18c"))
def test_named_source_import_selects_the_operational_radiation(tmp_path, name):
    from gpuwm.namelist_import import import_namelists

    text = (FIXTURE / "hrrr_wrf.nl.c18c").read_text()
    text = text.replace("swint_opt                           = 0,",
                        "swint_opt                           = 1,")
    text = text.replace("aer_opt                             = 0,",
                        "aer_opt                             = 3,")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    imported, _ = import_namelists(
        FIXTURE / "hrrr_namelist.wps.c18", path)
    run = build_experiment(tomllib.loads(imported), source="shortwave source").root.run
    assert run.mp_physics == 28
    assert radiation_scheme_ids(run) == (4, 4)
    assert run.ra_rrtmg_variant == "rrtmg_legacy"
    assert run.rrtmg_cloud_optics_form == "noaa_wrf39"
    assert run.radt == 15.0
    assert run.aer_opt == 3
    assert run.swint_opt == 1


def test_shipped_configuration_recipe_reaches_the_real_planner(tmp_path, capsys):
    from gpuwm.cli import main

    path = tmp_path / RECIPE.name
    path.write_bytes(RECIPE.read_bytes())
    run = load_experiment(path).root.run
    assert run.mp_physics == 28
    assert run.mp28_aerosol_source == "analysis"
    assert run.use_rap_aero_icbc is True
    assert radiation_scheme_ids(run) == (4, 4)
    assert run.ra_rrtmg_variant == "rrtmg_legacy"
    assert run.rrtmg_cloud_optics_form == "noaa_wrf39"
    assert run.radt == 15.0
    assert run.aer_opt == 3
    assert run.swint_opt == 1
    before = path.read_bytes()
    assert main(["go", str(path), "--dry-run"]) == 0
    capsys.readouterr()
    assert path.read_bytes() == before


def test_generic_import_door_keeps_its_radiation_defaults():
    assert namelist_physics_defaults("namelist.input") == {}
    assert namelist_physics_defaults("ordinary.toml") == {}


def test_generated_namelist_retains_analyzed_aerosol_request():
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.namelist_import import parse_namelist_text

    exp = load_experiment(RECIPE)
    for stock in (False, True):
        ph = parse_namelist_text(render_namelist_input(exp, stock=stock))["physics"]
        assert ph["use_aero_icbc"] == [True]
        assert ph["use_rap_aero_icbc"] == [True]


def test_analyzed_aerosol_rows_are_absent_for_an_ordinary_request():
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.namelist_import import parse_namelist_text

    exp = load_experiment(ROOT / "configs/hrrr_native_quick_demo.toml")
    ph = parse_namelist_text(render_namelist_input(exp))["physics"]
    # The demo runs the fork MYNN (bl_mynn_version = "gsd_41", filled from
    # its source's generation row since lane/mynn-exact), whose emission
    # always states use_aero_icbc; for an ordinary request it states off.
    assert ph.get("use_aero_icbc", [False]) == [False]
    assert "use_rap_aero_icbc" not in ph


def test_explicit_modern_radiation_keeps_its_cloud_form(tmp_path):
    from gpuwm.namelist_import import import_namelists
    from gpuwm.physics_source_defaults import with_physics_selector_comment

    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(with_physics_selector_comment(
        (FIXTURE / "hrrr_wrf.nl.c18c").read_text(),
        {"bl_mynn_version": "wrf_461"}), encoding="utf-8")
    imported, _ = import_namelists(
        FIXTURE / "hrrr_namelist.wps.c18", path,
        rrtmg_variant="rte-rrtmgp")
    run = build_experiment(tomllib.loads(imported), source="modern override").root.run
    assert run.ra_rrtmg_variant == "rte-rrtmgp"
    assert run.bl_mynn_version == "wrf_461"
    assert run.rrtmg_cloud_optics_form == "wrf_461"


def test_explicit_modern_radiation_refuses_the_unported_aerosol(tmp_path):
    from gpuwm.namelist_import import import_namelists, NamelistRefusal

    text = (FIXTURE / "hrrr_wrf.nl.c18c").read_text().replace(
        "aer_opt                             = 0,",
        "aer_opt                             = 3,")
    path = tmp_path / "hrrr_wrf.nl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(NamelistRefusal, match="legacy RRTMG"):
        import_namelists(FIXTURE / "hrrr_namelist.wps.c18", path,
                         rrtmg_variant="rte-rrtmgp")
