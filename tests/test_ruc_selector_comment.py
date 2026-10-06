"""Carried RUC choices survive content heuristics and real route generation."""
from pathlib import Path
import json
import tomllib

import pytest

from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.namelist_import import import_namelists, parse_namelist
from gpuwm.physics_source_defaults import (
    namelist_physics_defaults, read_physics_selector_comment,
    recipe_physics_defaults, with_physics_selector_comment)
from test_namelist_import import _pair
from test_ruc_fork_configuration_doors import _ruc_input


FORMS = {
    "ruc_irrigation": ("wrf_461", "wrf_45"),
    "ruc_snow": ("wrf_461", "wrf_45"),
    "ruc_qvg_cold_start": ("wrf", "air"),
    "ruc_2m_diagnostic": ("flux", "log_profile"),
}


@pytest.mark.parametrize("key,value", [(key, value) for key, values in FORMS.items() for value in values])
def test_explicit_ruc_generation_wins_over_fork_content(tmp_path, key, value):
    wps, inp = _pair(tmp_path, inp=_ruc_input(fork=True))
    inp.write_text(with_physics_selector_comment(inp.read_text(), {key: value}), encoding="utf-8")
    text, report = import_namelists(wps, inp)
    exp = build_experiment(tomllib.loads(text), source="carried RUC selector")
    assert all(getattr(domain.run, key) == value for domain in exp.domains)
    assert any(row.key == key and row.value == value for row in report.defaults_applied)


@pytest.mark.parametrize("key", list(FORMS))
@pytest.mark.parametrize("value", [True, "unknown"])
def test_ruc_generation_metadata_refuses_invalid_types_and_values(tmp_path, key, value):
    with pytest.raises(ValueError, match="physics selector comment"):
        with_physics_selector_comment("&physics\n/\n", {key: value})
    wps, inp = _pair(tmp_path, inp=_ruc_input(fork=False))
    inp.write_text("! gpuwm-physics-selectors-v1: " + json.dumps({key: value}) + "\n" + inp.read_text())
    with pytest.raises(ValueError, match="physics selector comment"):
        import_namelists(wps, inp)


def _public_experiment(tmp_path, settings, *, factor):
    wps, inp = _pair(tmp_path, inp=_ruc_input(fork=False))
    text, _ = import_namelists(wps, inp)
    raw = tomllib.loads(text)
    raw["shared"].update(settings)
    if factor:
        raw["shared"].update(diff_6th_form="noaa_wrf39", diff_6th_factor2=0.04,
                              upper_wind_limiter_form="wrf_461")
    return raw, build_experiment(raw, source="declared RUC route")


@pytest.mark.parametrize("factor", [False, True])
@pytest.mark.parametrize("settings", [
    {"ruc_irrigation": "wrf_461", "ruc_snow": "wrf_461",
     "ruc_qvg_cold_start": "wrf", "ruc_2m_diagnostic": "flux"},
    {"ruc_irrigation": "wrf_45", "ruc_snow": "wrf_45",
     "ruc_qvg_cold_start": "air", "ruc_2m_diagnostic": "log_profile"},
    {"ruc_irrigation": "wrf_461", "ruc_snow": "wrf_45",
     "ruc_qvg_cold_start": "wrf", "ruc_2m_diagnostic": "log_profile"},
])
def test_actual_generated_namelist_preserves_all_four_ruc_runconfig_choices(tmp_path, settings, factor):
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.hrrr_route_inputs import render_namelist_input

    raw, exp = _public_experiment(tmp_path, settings, factor=factor)
    wps, inp = tmp_path / "emitted.wps", tmp_path / "emitted.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, tmp_path / "emitted.toml"),
                   encoding="utf-8")
    inp.write_text(render_namelist_input(exp), encoding="utf-8")
    carried = read_physics_selector_comment(inp.read_text())
    if factor:
        assert carried["ruc_irrigation"] == settings["ruc_irrigation"]
        assert carried["ruc_snow"] == settings["ruc_snow"]
        assert carried["upper_wind_limiter_form"] == "wrf_461"
        assert parse_namelist(inp)["dynamics"]["diff_6th_factor2"] == [0.04] * len(exp.domains)
    elif all(settings[key] == values[0] for key, values in FORMS.items()):
        # No fork content and all generic selectors.  The route reads its
        # pair under the hrrr recipe, which fills the fork forms wherever
        # the comment is silent, so the generic forms are carried (an
        # unmarked comment was read back as the fork's forms).  The
        # comment states only the experiment's own values.
        assert {key: carried[key] for key in FORMS} == settings
        assert carried == {key: getattr(exp.root.run, key) for key in carried}
    text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(text), source="actual RUC reimport")
    for domain in restored.domains:
        assert {key: getattr(domain.run, key) for key in FORMS} == settings
        assert domain.run.ruc_soilprop == "wrf_45"
    # The route's own reading, under its recipe request, says the same.
    text, _ = import_namelists(wps, inp, request_source="hrrr",
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    routed = build_experiment(tomllib.loads(text), source="route RUC reimport")
    for domain in routed.domains:
        assert {key: getattr(domain.run, key) for key in FORMS} == settings
    if factor:
        assert restored.root.run.upper_wind_limiter_form == "wrf_461"


def test_recipe_defaults_do_not_turn_a_filename_into_a_content_signature():
    named = namelist_physics_defaults("hrrr_wrf.nl")
    assert "ruc_irrigation" not in named and "ruc_snow" not in named
    for source in ("hrrr", "hrrr-prs", "hrrr-native"):
        defaults = recipe_physics_defaults(source)
        assert defaults["ruc_irrigation"] == defaults["ruc_snow"] == "wrf_45"
    for source in ("gfs", "era5", "rap", "rrfs"):
        assert recipe_physics_defaults(source) == {}


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "hrrr-native"])
def test_bare_source_recipe_and_real_generated_ruc_values_select_the_fork(tmp_path, capsys, source):
    from gpuwm.cli import main
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.hrrr_route_inputs import render_namelist_input

    config = tmp_path / "native.toml"
    assert main([
        "domain", "--point=35.2,-97.4", "--card", "24gb", "--ladder", "12-3",
        "--source", source, "--cycle", "2026-10-02T21", "--hours", "1", "--out", str(config),
    ]) == 0
    capsys.readouterr()
    exp = load_experiment(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert all(domain.run.sf_surface_physics == 3 for domain in exp.domains)
    assert all(domain.run.ruc_irrigation == domain.run.ruc_snow == "wrf_45" for domain in exp.domains)
    for domain in exp.domains:
        expected = (5, 5) if domain.parent_id == 0 else (3, 3)
        assert (domain.run.v_sca_adv_order, domain.run.v_mom_adv_order) == expected
    wps, inp = tmp_path / "generated.wps", tmp_path / "generated.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, config), encoding="utf-8")
    inp.write_text(render_namelist_input(exp), encoding="utf-8")
    text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(text), source="source-selected actual RUC input")
    assert all(domain.run.ruc_irrigation == domain.run.ruc_snow == "wrf_45" for domain in restored.domains)
    for domain in restored.domains:
        expected = (5, 5) if domain.parent_id == 0 else (3, 3)
        assert (domain.run.v_sca_adv_order, domain.run.v_mom_adv_order) == expected
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before


def test_existing_ruc_clock_recipe_selects_qualified_forms_without_replacing_clock():
    recipe = Path(__file__).resolve().parents[1] / "configs/recipes/hrrr_configuration_clock.toml"
    exp = load_experiment(recipe)
    run = exp.root.run
    assert run.sf_surface_physics == 3
    assert run.ruc_irrigation == run.ruc_snow == "wrf_45"
    assert (run.dt, run.time_step_sound, run.terrain_clock) == (20.0, 6, "pinned")
