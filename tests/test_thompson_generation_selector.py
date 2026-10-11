"""Generation selection must use producer metadata and preserve overrides."""
from types import SimpleNamespace

import pytest


def item(title, mp=8):
    return SimpleNamespace(global_attributes={"TITLE": title}, mp_physics=mp)


@pytest.mark.parametrize("title", ("OUTPUT FROM WRF V4.4 MODEL", "OUTPUT FROM REAL_EM V4.4 PREPROCESSOR", "output from wrf v4.4.0 model"))
def test_known_generation_is_selected_from_title(title):
    from gpuwm.thompson_generations import select_classic_generation
    assert select_classic_generation({1:item(title)})[0] == "wrf_44"


@pytest.mark.parametrize("title", ("OUTPUT FROM WRF V4.6.1 MODEL", "", "OUTPUT FROM REAL_EM V4.7.1 PREPROCESSOR"))
def test_other_inputs_keep_default(title):
    from gpuwm.thompson_generations import select_classic_generation
    assert select_classic_generation({1:item(title)})[0] == "wrf_461"


def test_runtime_title_takes_priority_over_preprocessor_title():
    from gpuwm.thompson_generations import select_classic_generation
    selected, receipt = select_classic_generation({1:item("OUTPUT FROM REAL_EM V4.2.2 PREPROCESSOR")}, runtime_titles=["OUTPUT FROM WRF V4.4 MODEL"])
    assert selected == "wrf_44"
    assert receipt["source"] == "runtime TITLE"


def test_override_is_preserved():
    from gpuwm.thompson_generations import select_classic_generation
    selected, receipt = select_classic_generation({1:item("OUTPUT FROM WRF V4.4 MODEL")}, override="wrf_461")
    assert selected == "wrf_461"
    assert receipt["source"] == "explicit override"


def test_aerosol_generation_is_not_rebound():
    from gpuwm.thompson_generations import select_classic_generation
    assert select_classic_generation({1:item("OUTPUT FROM WRF V4.4 MODEL", 28)})[0] is None


def test_inconsistent_model_generations_are_refused():
    from gpuwm.thompson_generations import select_classic_generation
    with pytest.raises(ValueError, match="different Thompson generations"):
        select_classic_generation({1:item("OUTPUT FROM WRF V4.4 MODEL"),2:item("OUTPUT FROM WRF V4.6.1 MODEL")})


def test_unknown_override_is_refused():
    from gpuwm.thompson_generations import select_classic_generation
    with pytest.raises(ValueError, match="unknown classic Thompson generation"):
        select_classic_generation({1:item("OUTPUT FROM WRF V4.4 MODEL")}, override="invalid")


def test_generated_source_has_a_default_and_a_header_capacity_tier():
    from gpuwm.core.kernels import module_source, module_source_int_defines
    default = module_source("thompson_column")
    tier = module_source_int_defines("thompson_column",(("THOMPSON_COLUMN_KMAX",128),))
    assert "struct ThompsonColumn" in default
    assert tier.index("#define THOMPSON_COLUMN_KMAX 128") < tier.index("struct ThompsonColumn")


def test_real_parameter_power_is_not_rewritten_to_integer_power():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/"tools/thompson_generation_oracle/translate.py"
    spec = importlib.util.spec_from_file_location("generation_translate",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    symbols = {"x":module.lake.Variable("x","float",[]),"b":module.lake.Variable("b","float",[],initial="3.0",const=True)}
    assert module.expr("x**b",symbols) == "tc_pow(x,b)"
    assert module.expr("x**3",symbols) == "tc_pow(x,3)"


@pytest.mark.parametrize("levels,expected", ((49,50016),(64,50016),(65,100000),(128,100000),(129,199968),(256,199968)))
def test_preflight_prices_the_capacity_that_will_launch(levels, expected):
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core.preflight import kernel_local_frame_bytes
    cfg = RunConfig(nx=8,ny=8,nz=levels,dx=1000.,dy=1000.,ztop=20000.,dt=1.,run_seconds=2.,mp_physics=8,thompson_version="wrf_44")
    assert kernel_local_frame_bytes(experiment_from_run_config(cfg, datetime(2020,1,1)))["thompson_column"] == expected


def test_generation_table_price_counts_only_the_uploaded_records():
    from gpuwm.core.preflight import thompson_generation_coefficient_bytes
    cfg = SimpleNamespace(mp_physics=8,thompson_version="wrf_44")
    # 5 rain-graupel, 12 rain-snow, 6 freezing, 5 auxiliary records.
    assert thompson_generation_coefficient_bytes(cfg) == 373927424


def test_neutral_directory_omitted_by_checkpoint_can_resume(tmp_path):
    from gpuwm.config import RunConfig
    from gpuwm.io.restart import configuration_echo, _require_config_match
    cfg = RunConfig(nx=8,ny=8,nz=4,dx=1000.,dy=1000.,ztop=10000.,dt=1.,run_seconds=2.)
    stored = configuration_echo(cfg)
    assert "thompson_table_dir" not in stored
    _require_config_match(stored,cfg,tmp_path/"checkpoint.npz")


def test_checkpoint_still_refuses_a_changed_generation_directory(tmp_path):
    from dataclasses import replace
    from gpuwm.config import RunConfig
    from gpuwm.io.restart import configuration_echo, _require_config_match, RestartMismatchError
    cfg = RunConfig(nx=8,ny=8,nz=4,dx=1000.,dy=1000.,ztop=10000.,dt=1.,run_seconds=2.,mp_physics=8,thompson_version="wrf_44",thompson_table_dir="original-tables")
    with pytest.raises(RestartMismatchError,match="thompson_table_dir"):
        _require_config_match(configuration_echo(cfg),replace(cfg,thompson_table_dir="different-tables"),tmp_path/"checkpoint.npz")
