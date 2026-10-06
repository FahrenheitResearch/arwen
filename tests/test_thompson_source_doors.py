"""Bare HRRR recommendations execute fork MP28; explicit MP8 stays explicit."""
import tomllib

import pytest

from gpuwm.cli import main
from gpuwm.companion_domains import candidate_wps_text
from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.hrrr_route_inputs import render_namelist_input
from gpuwm.namelist_import import import_namelists
from gpuwm.physics_compat import THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID, single_domain_runtime_switches
from gpuwm.source_adapters import get_source_adapter


STAGED_SOURCE_PROFILE = "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1"


def assert_staged_mynn_is_preserved(cfg):
    assert cfg.bl_mynn_version == "gsd_41"
    assert cfg.bl_mynn_mixlength == 2
    assert cfg.bl_mynn_gsd41_unsquared_qtke is False
    assert cfg.bl_mynn_cloud_tendency_form == "wrf_461"
    assert cfg.mynn_sfclay_variant == "gsl_wrf39"
    assert cfg.scalar_pblmix == 1
    assert cfg.aer_init_opt == cfg.wif_input_opt == 1
    assert cfg.ra_rrtmg_variant == "rrtmg_legacy"


def emit(tmp_path, capsys, source, profile=None):
    config = tmp_path / "native.toml"
    args = ["domain", "--point=35.2,-97.4", "--card", "24gb", "--ladder", "12-3",
            "--source", source, "--cycle", "2026-10-02T21", "--hours", "1", "--out", str(config)]
    if profile is not None:
        args += ["--physics-profile", profile]
    assert main(args) == 0
    capsys.readouterr()
    return config, load_experiment(config), tomllib.loads(config.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_bare_hrrr_recommendation_executes_both_thompson_fork_forms(tmp_path, capsys, source):
    config, exp, raw = emit(tmp_path, capsys, source)
    recommendation = get_source_adapter(source).default_physics_profile
    assert recommendation == STAGED_SOURCE_PROFILE
    declared = single_domain_runtime_switches(recommendation)
    assert declared["mp_physics"] == 28
    assert declared["thompson_version"] == declared["thompson_fork_snow_fall"] == "wrf_39_noaa"
    for domain in exp.domains:
        assert domain.run.mp_physics == 28
        assert domain.run.thompson_version == domain.run.thompson_fork_snow_fall == "wrf_39_noaa"
        assert_staged_mynn_is_preserved(domain.run)
    wps, inp = tmp_path / "emitted.wps", tmp_path / "emitted.input"
    wps.write_text(candidate_wps_text(raw, exp, exp, config), encoding="utf-8")
    inp.write_text(render_namelist_input(exp), encoding="utf-8")
    text, _ = import_namelists(wps, inp,
        rrtmg_variant=exp.root.run.ra_rrtmg_variant,
        rrtmg_compatibility=exp.root.run.wrf_rrtmg_compatibility)
    restored = build_experiment(tomllib.loads(text), source="actual bare Thompson reimport")
    for domain in restored.domains:
        assert domain.run.mp_physics == 28
        assert domain.run.thompson_version == domain.run.thompson_fork_snow_fall == "wrf_39_noaa"
        assert_staged_mynn_is_preserved(domain.run)
    before = config.read_bytes()
    assert main(["go", str(config), "--dry-run"]) == 0
    capsys.readouterr()
    assert config.read_bytes() == before


@pytest.mark.parametrize("source", ("hrrr", "hrrr-prs", "hrrr-native"))
def test_explicit_mp8_hrrr_profile_does_not_receive_inert_or_invalid_fork_selectors(tmp_path, capsys, source):
    _, exp, raw = emit(tmp_path, capsys, source, THOMPSON_MYNN_RUC_RTE_RRTMGP_PROFILE_ID)
    assert "thompson_version" not in raw["shared"]
    assert "thompson_fork_snow_fall" not in raw["shared"]
    for domain in exp.domains:
        assert domain.run.mp_physics == 8
        assert domain.run.thompson_version == "wrf_461"
        assert domain.run.thompson_fork_snow_fall == "blend"
        assert domain.run.bl_mynn_version == "wrf_461"
        assert domain.run.bl_mynn_cloud_tendency_form == "wrf_461"
