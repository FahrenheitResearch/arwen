"""Sun-angle albedo selection and explicit radiation choices at real doors."""
from pathlib import Path
from datetime import datetime
import tomllib

import pytest

from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.namelist_import import import_namelists
from test_namelist_import import INPUT_TEXT, _pair


def _input(tmp_path, *, name="namelist.input", settings="", mp=55):
    text = INPUT_TEXT.replace("mp_physics = 55, 55", f"mp_physics = {mp}, {mp}")
    text = text.replace("&physics\n", "&physics\n" + settings)
    wps, original = _pair(tmp_path, inp=text)
    path = tmp_path / name
    if path != original:
        path.write_bytes(original.read_bytes())
    return wps, path


@pytest.mark.parametrize("name", ["hrrr_wrf.nl", "hrrr_wrf.nl.c18c"])
def test_named_source_albedo_default_is_enabled(tmp_path, name):
    text, report = import_namelists(*_input(tmp_path, name=name))
    exp = build_experiment(tomllib.loads(text), source="named source")
    assert all(domain.run.alb_sol == 1 for domain in exp.domains)
    assert any(row.key == "alb_sol" and row.value == 1 for row in report.defaults_applied)


def test_named_source_explicit_off_wins_over_the_albedo_default(tmp_path):
    text, report = import_namelists(*_input(
        tmp_path, name="hrrr_wrf.nl", settings=" alb_sol = 0,\n"))
    assert "alb_sol" not in tomllib.loads(text)["shared"]
    exp = build_experiment(tomllib.loads(text), source="explicit source OFF")
    assert all(domain.run.alb_sol == 0 for domain in exp.domains)
    assert not any(row.key == "alb_sol" for row in report.defaults_applied)
    assert any(row.section == "physics" and row.key == "alb_sol" for row in report.translated)


def test_named_source_explicit_microphysics_zero_defaults_remain_zero(tmp_path):
    text, _ = import_namelists(*_input(
        tmp_path, name="hrrr_wrf.nl", settings=(
            " mp_zero_out = 0,\n mp_zero_out_all = 0,\n mp_zero_out_thresh = 1e-8,\n")))
    run = build_experiment(tomllib.loads(text), source="explicit source zero").root.run
    assert (run.mp_zero_out, run.mp_zero_out_all, run.mp_zero_out_thresh) == (0, 0, 1e-8)


def test_unnamed_omission_and_explicit_albedo_off_keep_identical_bytes(tmp_path):
    absent, _ = import_namelists(*_input(tmp_path))
    off, _ = import_namelists(*_input(tmp_path, settings=" alb_sol = 0,\n"))
    assert absent == off
    assert "alb_sol" not in tomllib.loads(off)["shared"]
    run = build_experiment(tomllib.loads(off), source="ordinary source").root.run
    assert run.alb_sol == 0 and run.ra_rrtmg_variant == "rte-rrtmgp"


@pytest.mark.parametrize("source", ["hrrr", "hrrr-prs", "hrrr-native"])
@pytest.mark.parametrize("explicit_mp8", [False, True], ids=["bare-gsd-mp28", "explicit-modern-mp8"])
def test_bare_source_recipe_and_generated_namelist_select_albedo(tmp_path, capsys, source, explicit_mp8):
    from gpuwm.cli import main
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.hrrr_route_inputs import render_namelist_input

    from gpuwm.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    config = tmp_path / "native.toml"
    profile_args = (["--physics-profile", THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID]
                    if explicit_mp8 else [])
    assert main([
        "domain", "--point=35.2,-97.4", "--card", "24gb", "--ladder", "12-3",
        "--source", source, "--cycle", "2026-10-02T21", "--hours", "1", "--out", str(config),
        *profile_args,
    ]) == 0
    capsys.readouterr()
    exp = load_experiment(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    assert all(domain.run.alb_sol == 1 for domain in exp.domains)
    # Every bare HRRR alias selects the same declared source composition.
    bare_gsd = not explicit_mp8
    expected_mp = 28 if bare_gsd else 8
    expected_mynn = "gsd_41" if bare_gsd else "wrf_461"
    expected_radiation = "rrtmg_legacy" if bare_gsd else "rte-rrtmgp"
    assert all(domain.run.mp_physics == expected_mp for domain in exp.domains)
    assert all(domain.run.bl_mynn_version == expected_mynn for domain in exp.domains)
    assert all(domain.run.bl_mynn_cloud_tendency_form == "wrf_461" for domain in exp.domains)
    assert all(domain.run.ra_rrtmg_variant == expected_radiation for domain in exp.domains)
    wps = tmp_path / "emitted.namelist.wps"
    namelist = tmp_path / "emitted.namelist.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, config), encoding="utf-8")
    namelist.write_text(render_namelist_input(exp), encoding="utf-8")
    text, _ = import_namelists(wps, namelist,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(text), source="actual generated input")
    assert all(domain.run.alb_sol == 1 for domain in restored.domains)
    assert all(domain.run.mp_physics == expected_mp for domain in restored.domains)
    assert all(domain.run.bl_mynn_version == expected_mynn for domain in restored.domains)
    assert all(domain.run.bl_mynn_cloud_tendency_form == "wrf_461" for domain in restored.domains)
    assert all(domain.run.ra_rrtmg_variant == expected_radiation for domain in restored.domains)
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before


def test_coupled_factor_preserves_explicit_legacy_upper_limiter_on_round_trip(tmp_path):
    from gpuwm.companion_domains import candidate_wps_text
    from gpuwm.hrrr_route_inputs import render_namelist_input
    from gpuwm.domain_wizard import render_config
    from gpuwm.experiment import build_experiment_from_config_tables
    from gpuwm.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID

    raw = tomllib.loads(render_config(
        name="source-request", start_time=datetime(2026, 10, 2, 21), hours=1,
        projection={"map_proj": "lambert", "ref_lat": 38.5, "ref_lon": -97.5,
                    "truelat1": 38.5, "truelat2": 38.5, "stand_lon": -97.5},
        dims=[(50, 50)], ratios=(), root_dx_m=3000,
        fetch_hints={"source": "hrrr"}, case_data=None,
        profile=THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID))
    raw["shared"]["upper_wind_limiter_form"] = "wrf_461"
    exp = build_experiment_from_config_tables(
        raw, source="explicit legacy limiter", base_dir=tmp_path)
    assert exp.root.run.diff_6th_factor2 == 0.04
    wps = tmp_path / "generated.wps"
    inp = tmp_path / "generated.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, tmp_path / "generated.toml"),
                   encoding="utf-8")
    inp.write_text(render_namelist_input(exp), encoding="utf-8")
    text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(text), source="carried actual limiter")
    assert restored.root.run.diff_6th_form == "noaa_wrf39"
    assert restored.root.run.upper_wind_limiter_form == "wrf_461"
    assert restored.root.run.diff_6th_factor2 == 0.04


def test_omitted_variant_with_explicit_aerosol_three_selects_legacy_before_validation(tmp_path):
    text, report = import_namelists(*_input(tmp_path, mp=28, settings=" aer_opt = 3,\n"))
    exp = build_experiment(tomllib.loads(text), source="requested aerosol optics")
    assert all(domain.run.aer_opt == 3 and domain.run.ra_rrtmg_variant == "rrtmg_legacy"
               for domain in exp.domains)
    assert any(row.key == "ra_rrtmg_variant" and row.value == "rrtmg_legacy"
               for row in report.defaults_applied)


def test_explicit_modern_variant_with_aerosol_three_is_refused(tmp_path):
    with pytest.raises(ValueError, match="legacy RRTMG"):
        import_namelists(*_input(tmp_path, mp=28, settings=" aer_opt = 3,\n"),
                         rrtmg_variant="rte-rrtmgp")


def test_generic_omitted_variant_keeps_the_explicit_modern_bytes(tmp_path):
    absent, _ = import_namelists(*_input(tmp_path))
    modern, _ = import_namelists(*_input(tmp_path), rrtmg_variant="rte-rrtmgp")
    assert absent == modern


@pytest.mark.parametrize("value", ["3.0", ".true.", "'3'"])
def test_aerosol_option_requires_integer_values(tmp_path, value):
    with pytest.raises(ValueError, match="aer_opt.*integer"):
        import_namelists(*_input(tmp_path, mp=28, settings=f" aer_opt = {value},\n"))


@pytest.mark.parametrize("explicit_modern", [False, True])
def test_cli_preserves_variant_omission_and_explicit_modern(tmp_path, capsys, explicit_modern):
    from gpuwm.cli import main

    wps, inp = _input(tmp_path, mp=28, settings=" aer_opt = 3,\n")
    output = tmp_path / "resolved.toml"
    args = ["import-namelist", str(wps), str(inp), "--output", str(output)]
    if explicit_modern:
        args += ["--rrtmg-variant", "rte-rrtmgp"]
    status = main(args)
    captured = capsys.readouterr()
    if explicit_modern:
        assert status == 2 and "legacy RRTMG" in captured.err
        assert not output.exists()
    else:
        assert status == 0
        run = load_experiment(output).root.run
        assert run.aer_opt == 3 and run.ra_rrtmg_variant == "rrtmg_legacy"


def test_shipped_gsd_full_recipe_selects_solar_albedo_and_keeps_corrected_cloud_form():
    path = Path(__file__).resolve().parents[1] / "configs/recipes/hrrr_v4_gsd41.toml"
    exp = load_experiment(path)
    assert exp.root.run.alb_sol == 1
    assert exp.root.run.bl_mynn_version == "gsd_41"
    assert exp.root.run.bl_mynn_cloud_tendency_form == "wrf_461"
    assert exp.root.run.mp_physics == 28
    assert exp.root.run.ra_rrtmg_variant == "rrtmg_legacy"
