"""Configuration requests choose the fork; data-source choices do not."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import pytest

from gpuwm.experiment import load_experiment
from gpuwm.namelist_import import (OPERATIONAL_FORK_THOMPSON_DEFAULTS,
    import_namelists, operational_fork_thompson_defaults, parse_namelist)

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "tests/data/hrrr_wrf_v4_1_21.nl"
RECIPE = ROOT / "configs/recipes/hrrr_configuration_cut.toml"


def test_untouched_operational_namelist_requests_both_fork_selectors():
    assert hashlib.sha256(FORK.read_bytes()).hexdigest() == (
        "50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694")
    assert operational_fork_thompson_defaults(parse_namelist(FORK)) == {
        "thompson_version": "wrf_39_noaa",
        "thompson_fork_snow_fall": "wrf_39_noaa"}


def test_fork_signature_with_another_microphysics_scheme_adds_no_selectors():
    assert operational_fork_thompson_defaults({
        "physics": {"mp_physics": [8], "alb_sol": [1]}}) == {}


def test_importer_emits_the_fork_selectors_for_a_supported_fork_configuration(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    inp = INPUT_TEXT.replace(" mp_physics = 55, 55,",
                             " mp_physics = 28, 28,\n bl_mynn_tkebudget = 0,")
    text, _ = import_namelists(*_pair(tmp_path, inp=inp), name="fork")
    path = tmp_path / "fork.toml"
    path.write_text(text)
    exp = load_experiment(path)
    for domain in exp.domains:
        assert {key: getattr(domain.run, key)
                for key in OPERATIONAL_FORK_THOMPSON_DEFAULTS} == (
                    OPERATIONAL_FORK_THOMPSON_DEFAULTS)


def test_nonzero_fork_budget_still_names_the_missing_output_package(tmp_path):
    from test_namelist_import import INPUT_TEXT, _pair
    inp = INPUT_TEXT.replace(" mp_physics = 55, 55,",
                             " mp_physics = 28, 28,\n bl_mynn_tkebudget = 1,")
    with pytest.raises(ValueError, match="writes no TKE budget terms"):
        import_namelists(*_pair(tmp_path, inp=inp), name="fork")


def test_non_fork_import_is_byte_identical_to_the_previous_importer(tmp_path):
    from test_namelist_import import _pair
    control = os.environ.get("GPUWM_CONFIGURATION_DOOR_BASE")
    if not control:
        # d191b05b3, lane/286-fork-thompson: preserve the pre-ruling importer
        # as exact LF bytes. The 7df301d1b gate bundle predates this commit,
        # so the control must travel with the tests rather than require Git.
        path = ROOT / "tests/fixtures/thompson/namelist_import_d191b05b3.py"
    else:
        path = Path(control) / "gpuwm/namelist_import.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        'fe8bd9f884c6c3ea4eb9a3c1a5680116ae00114a8cf71325dccbc9f89d0e47cd')
    spec = importlib.util.spec_from_file_location("before_configuration_doors", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    paths = _pair(tmp_path)
    actual, report = import_namelists(*paths, name="same")
    expected, old_report = module.import_namelists(*paths, name="same")
    assert actual.encode() == expected.encode()
    assert report.format() == old_report.format()


@pytest.mark.parametrize("name", ["hrrr_configuration_cut.toml",
                                 "conus_hrrr_configuration.toml"])
def test_recipe_resolves_the_requested_fork_values(name):
    exp = load_experiment(RECIPE.parent / name)
    assert exp.root.run.mp_physics == 28
    assert {key: getattr(exp.root.run, key)
            for key in OPERATIONAL_FORK_THOMPSON_DEFAULTS} == (
                OPERATIONAL_FORK_THOMPSON_DEFAULTS)


def test_existing_hrrr_data_demos_keep_the_previous_generation():
    for name in ("hrrr_native_quick_demo", "hrrr_native_3km_demo",
                 "hrrr_prs_demo", "hrrr_prs_3km_demo"):
        cfg = load_experiment(ROOT / "configs" / (name + ".toml")).root.run
        assert cfg.thompson_version == "wrf_461"
        assert cfg.thompson_fork_snow_fall == "blend"


def test_recipe_is_consumed_by_the_real_go_planner(tmp_path):
    from _staged_geog import staged_case_data_env

    # The dry run asks the default geography, as the launch does.
    env = staged_case_data_env(tmp_path / "case-data", base=dict(
        os.environ, CUDA_VISIBLE_DEVICES="-1", GPUWM_NO_LOCAL_GPU="1"))
    result = subprocess.run([sys.executable, "-m", "gpuwm.cli", "go",
                             str(RECIPE), "--dry-run"], cwd=ROOT,
                            capture_output=True, text=True, timeout=90,
                            env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rap-native" in result.stdout
