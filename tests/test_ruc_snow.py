"""Snow lineage validation, configuration loading and identity boundaries.

A prepared input is reusable with either snow lineage.  A resumed forecast
must keep the lineage that wrote its checkpoint.
"""

from dataclasses import asdict, replace

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.ruc_tier import ruc_snow_form
from gpuwm.experiment import load_experiment


def _cfg(**overrides):
    values = dict(nx=24, ny=24, nz=8, dx=3000.0, dy=3000.0,
                  ztop=10000.0, dt=6.0, run_seconds=60.0,
                  sf_surface_physics=3, num_soil_layers=9,
                  sf_sfclay_physics=1, bl_pbl_physics=1, bldt=0.0,
                  moist=True, mp_physics=6)
    values.update(overrides)
    return RunConfig(**values)


def _experiment(tmp_path, snow_line):
    path = tmp_path / 'snow.toml'
    path.write_text(f'''[experiment]
name = "snow-lineage"
start_time = 2026-01-01T00:00:00
run_seconds = 60.0
restart_interval_s = 0.0
[shared]
nz = 8
ztop = 10000.0
sf_surface_physics = 3
num_soil_layers = 9
sf_sfclay_physics = 1
bl_pbl_physics = 1
bldt = 0.0
moist = true
mp_physics = 6
{snow_line}
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 24
ny = 24
time_step = 6
dx = 3000.0
history_interval_s = 60.0
''', encoding='utf-8')
    return load_experiment(path)


@pytest.mark.parametrize('bad', ['wrf_46', '', 0, None, True, 'WRF_45'])
def test_unknown_snow_lineages_are_refused_before_a_run(bad):
    with pytest.raises(ValueError, match='ruc_snow'):
        ruc_snow_form(bad)
    with pytest.raises(ValueError, match='ruc_snow'):
        validate_run_config(_cfg(ruc_snow=bad))


@pytest.mark.parametrize('line, expected', [
    ('', 'wrf_461'),
    ('ruc_snow = "wrf_45"', 'wrf_45'),
    ('ruc_snow = "wrf_461"', 'wrf_461'),
])
def test_shared_toml_loads_the_default_and_both_named_lineages(tmp_path, line, expected):
    cfg = _experiment(tmp_path, line).root.run
    assert cfg.sf_surface_physics == 3
    assert cfg.ruc_snow == expected
    validate_run_config(cfg)


@pytest.mark.parametrize('name', ['wrf_46', 'WRF_45'])
def test_shared_toml_refuses_unknown_snow_lineages(tmp_path, name):
    with pytest.raises(ValueError, match='ruc_snow'):
        _experiment(tmp_path, f'ruc_snow = "{name}"')


@pytest.mark.parametrize('stored_name, live_name', [
    ('wrf_45', 'wrf_461'), ('wrf_461', 'wrf_45'),
])
def test_restart_refuses_both_snow_lineage_flips(stored_name, live_name):
    from gpuwm.io.restart import _require_config_match, configuration_echo

    stored = _cfg(ruc_snow=stored_name)
    live = replace(stored, ruc_snow=live_name)
    _require_config_match(configuration_echo(stored), stored, 'checkpoint')
    with pytest.raises(ValueError, match='ruc_snow'):
        _require_config_match(configuration_echo(stored), live, 'checkpoint')


def test_an_omitted_selector_binds_the_current_default_only():
    from gpuwm.io.restart import (
        _configuration_digest_values, _require_config_match, configuration_echo)

    default = _cfg()
    stored = asdict(default)
    stored.pop('ruc_snow')
    _require_config_match(stored, default, 'checkpoint')
    assert _configuration_digest_values(stored) == _configuration_digest_values(asdict(default))
    assert 'ruc_snow' not in configuration_echo(default)
    other = replace(default, ruc_snow='wrf_45')
    assert configuration_echo(other)['ruc_snow'] == 'wrf_45'
    assert _configuration_digest_values(stored) != _configuration_digest_values(asdict(other))
    with pytest.raises(ValueError, match='ruc_snow'):
        _require_config_match(stored, other, 'checkpoint')


def test_prepared_input_reuse_accepts_a_snow_flip_but_keeps_grid_identity():
    from gpuwm.ingest.prepared_cache import (
        compare_prepared_domain_config, effective_prepared_domain_config)

    cfg = _cfg()
    cached = {'run': asdict(cfg)}
    live = {'run': asdict(replace(cfg, ruc_snow='wrf_45'))}
    assert compare_prepared_domain_config(cached, live) == ([], [])
    assert effective_prepared_domain_config(cached) == effective_prepared_domain_config(live)
    # A grid change still changes preparation, even beside an inert snow flip.
    moved_grid = {'run': asdict(replace(cfg, ruc_snow='wrf_45', dx=4000.0))}
    assert compare_prepared_domain_config(cached, moved_grid) == ([], ['run.dx'])
