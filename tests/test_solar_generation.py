"""Solar generation selection and restart safety."""
from dataclasses import replace
import pytest
from gpuwm.config import RunConfig
from gpuwm.io.restart import configuration_echo, _require_config_match

def config():
    return RunConfig(nx=8, ny=8, nz=4, dx=3000., dy=3000., ztop=10000., dt=5., run_seconds=10.)


def test_restart_generation_is_bound():
    cfg = config()
    legacy = replace(cfg, solar_generation='wrf_461')
    assert configuration_echo(cfg)['solar_generation'] == 'wrf_480'
    assert 'solar_generation' not in configuration_echo(legacy)
    _require_config_match(configuration_echo(legacy), legacy, 'legacy-checkpoint')
    with pytest.raises(Exception, match='COSZEN and HRANG'):
        _require_config_match(configuration_echo(legacy), cfg, 'legacy-checkpoint')
    with pytest.raises(Exception, match='interpolation fit'):
        _require_config_match(configuration_echo(cfg), legacy, 'corrected-checkpoint')


@pytest.mark.parametrize('value', ['wrf_470', '', None, 480])
def test_unknown_generation_refuses(value):
    from gpuwm.core.solar import solar_generation
    with pytest.raises(ValueError, match='COSZEN'):
        solar_generation(value)


def test_selector_comment_roundtrip():
    from gpuwm.physics_source_defaults import with_physics_selector_comment, read_physics_selector_comment
    for generation in ('wrf_461', 'wrf_480'):
        comment = with_physics_selector_comment('', {'solar_generation': generation})
        assert read_physics_selector_comment(comment)['solar_generation'] == generation


def test_namelist_import_uses_generic_default(tmp_path):
    from test_namelist_import import WPS_TEXT, INPUT_TEXT
    from gpuwm.namelist_import import import_namelists
    from gpuwm.experiment import load_experiment
    wps, inp, out = (tmp_path/name for name in ('namelist.wps','namelist.input','experiment.toml'))
    wps.write_text(WPS_TEXT)
    inp.write_text(INPUT_TEXT)
    text,_report = import_namelists(wps,inp)
    out.write_text(text)
    exp = load_experiment(out)
    assert {domain.run.solar_generation for domain in exp.domains} == {'wrf_480'}


def test_midpoint_keeps_requested_radt_separate_from_call_cadence():
    from gpuwm.core.solar import solar_period_seconds
    cfg = replace(config(),radt=7.25)
    assert solar_period_seconds(cfg,440.) == 435.
    assert solar_period_seconds(replace(cfg,solar_generation='wrf_461'),440.) == 440.
