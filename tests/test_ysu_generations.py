"""Public selection and historical restart semantics of the YSU fix."""
from dataclasses import asdict, replace
from pathlib import Path
import json

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.ysu_generations import YSU_GENERATIONS, ysu_generation


def config(**kwargs):
    return RunConfig(nx=8, ny=8, nz=40, dx=3000., dy=3000., ztop=10000.,
                     dt=10., run_seconds=20., **kwargs)


def test_corrected_default_and_registry_rows():
    cfg = config()
    assert cfg.ysu_version == "wrf_480"
    registry = json.loads((Path(__file__).parents[1] / "gpuwm/physics_registry_v2.json").read_text())
    row = registry["parameters"]["ysu_version"]
    assert row["default"] == cfg.ysu_version
    assert set(row["enum"]) == set(YSU_GENERATIONS)
    assert row["read_when"] == {"bl_pbl_physics": 1}
    generations = registry["components"]["pbl"]["options"]["ysu"]["numerical_generations"]
    assert generations["options"] == YSU_GENERATIONS


@pytest.mark.parametrize("version", YSU_GENERATIONS)
def test_explicit_versions_validate(version):
    validate_run_config(config(ysu_version=version))


def test_bad_generation_is_rejected():
    with pytest.raises(ValueError, match="ysu_version"):
        validate_run_config(config(ysu_version="wrf_undefined"))


def test_selector_comment_roundtrip():
    from gpuwm.physics_source_defaults import (with_physics_selector_comment,
                                                read_physics_selector_comment)
    for version in YSU_GENERATIONS:
        text = with_physics_selector_comment("&physics\n bl_pbl_physics=1,\n/\n",
                                             {"ysu_version": version})
        assert read_physics_selector_comment(text) == {"ysu_version": version}


def test_restart_cannot_flip_and_absent_header_means_legacy():
    from gpuwm.io.restart import _require_config_match, configuration_echo
    cfg = config(bl_pbl_physics=1, sf_sfclay_physics=1, ysu_version="wrf_461")
    old = asdict(cfg)
    old.pop("ysu_version")
    _require_config_match(old, cfg, "historical restart")
    assert "ysu_version" not in configuration_echo(cfg)
    new = replace(cfg, ysu_version="wrf_480")
    assert configuration_echo(new)["ysu_version"] == "wrf_480"
    with pytest.raises(Exception, match="ysu_version"):
        _require_config_match(old, new, "historical restart")
    with pytest.raises(Exception, match="ysu_version"):
        _require_config_match(asdict(new), cfg, "corrected restart")
    _require_config_match(asdict(new), new, "corrected restart")


def test_inactive_ysu_keeps_historical_restart_identity():
    from gpuwm.io.restart import _require_config_match, configuration_echo
    cfg = config(bl_pbl_physics=0)
    old = asdict(cfg)
    old.pop("ysu_version")
    _require_config_match(old, cfg, "PBL-off restart")
    assert "ysu_version" not in configuration_echo(cfg)


def test_generation_paths_keep_legacy_source_and_noftz_is_scoped():
    from gpuwm.core.kernels import module_options
    assert "--ftz=false" in module_options(ysu_generation("wrf_480")["kernel_module"])
    # wrf_461 is 2.8.9's no-FMA YSU unit, unchanged.
    assert module_options(ysu_generation("wrf_461")["kernel_module"]) == ("-std=c++17", "--fmad=false")


def test_preflight_prices_the_selected_generation():
    from types import SimpleNamespace
    from gpuwm.core.preflight import domain_kernel_modules
    for version in YSU_GENERATIONS:
        cfg = config(bl_pbl_physics=1, sf_sfclay_physics=1, ysu_version=version)
        modules = domain_kernel_modules(SimpleNamespace(run=cfg, grid_id=1), prices_refl=False)
        selected = ysu_generation(version)["kernel_module"]
        assert selected in modules
        assert ("ysu" in modules) == (version == "wrf_461")


def test_legacy_reproduces_289_in_both_arithmetic_modes():
    """wrf_461 is 2.8.9's YSU kernel and restart identity, default and strict:
    a wrf_461 run (or a 2.8.9 checkpoint continued under it) must not move."""
    from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES
    from gpuwm.ysu_generations import ysu_kernel_selection, ysu_algorithm_identity
    assert ysu_kernel_selection("wrf_461") == ("ysu", ())
    assert ysu_kernel_selection("wrf_461", strict=True) == ("ysu", ())
    assert ysu_algorithm_identity("wrf_461") == PBL_ALGORITHM_IDENTITIES[1]
    assert ysu_algorithm_identity("wrf_461", strict=True) == PBL_ALGORITHM_IDENTITIES[1]
    assert ysu_kernel_selection("wrf_480", strict=True) == ("ysu_480", ())
    assert ysu_algorithm_identity("wrf_480") != ysu_algorithm_identity("wrf_461")
