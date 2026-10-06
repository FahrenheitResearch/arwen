"""The qualified RUC source keeps prescribed fields through each door."""
from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from gpuwm.config import RunConfig
from gpuwm.domain_wizard import experiment_from_text
from gpuwm.experiment import build_experiment, load_experiment
from gpuwm.hrrr_prepared_bundle import render_wps_namelist
from gpuwm.hrrr_route_inputs import render_namelist_input
from gpuwm.namelist_import import import_namelists, parse_namelist_text
from gpuwm.physics_source_defaults import (
    recipe_physics_defaults, with_physics_selector_comment)
from test_namelist_import import _pair
from test_physics_source_defaults import _recipe
from test_ruc_fork_configuration_doors import _ruc_input


QUALIFIED = {
    'ruc_soilprop': 'wrf_45', 'ruc_irrigation': 'wrf_45',
    'ruc_snow': 'wrf_45', 'ruc_qvg_cold_start': 'air',
    'ruc_2m_diagnostic': 'log_profile', 'rdlai2d': True, 'usemonalb': True,
}


@pytest.mark.parametrize('recipe', (
    'conus_hrrr_configuration.toml', 'hrrr_configuration_cut.toml',
    'hrrr_v4_gsd41.toml', 'hrrr_configuration_clock.toml',
))
def test_shipped_ruc_recipes_load_the_complete_qualified_set(recipe):
    path = Path(__file__).resolve().parents[1] / 'configs' / 'recipes' / recipe
    shared = tomllib.loads(path.read_text(encoding='utf-8'))['shared']
    assert {key: shared[key] for key in QUALIFIED} == QUALIFIED
    for key, value in QUALIFIED.items():
        assert type(shared[key]) is type(value), key
    exp = load_experiment(path)
    for domain in exp.domains:
        assert domain.run.sf_surface_physics == 3
        assert {key: getattr(domain.run, key) for key in QUALIFIED} == QUALIFIED


@pytest.mark.parametrize('source', ('hrrr', 'hrrr-native', 'hrrr-prs'))
def test_authored_ruc_source_selects_the_complete_qualified_set(source):
    requested = recipe_physics_defaults(source)
    assert {key: requested[key] for key in QUALIFIED} == QUALIFIED
    exp = experiment_from_text(_recipe(source), source='source request')
    for key, value in QUALIFIED.items():
        assert getattr(exp.root.run, key) == value


def test_generic_requests_and_global_ruc_defaults_stay_unchanged():
    for source in ('gfs', 'era5', 'rrfs'):
        assert recipe_physics_defaults(source) == {}
    for key, value in (
            ('ruc_qvg_cold_start', 'wrf'), ('ruc_2m_diagnostic', 'flux'),
            ('rdlai2d', False), ('usemonalb', False)):
        assert RunConfig.__dataclass_fields__[key].default == value


@pytest.mark.parametrize('prescribed', (True, False))
def test_native_ruc_companion_preserves_standard_bools_and_legacy_choices(
        tmp_path, prescribed):
    raw = tomllib.loads(_recipe('hrrr-native'))
    raw['shared'].update(rdlai2d=prescribed, usemonalb=prescribed,
                         ruc_qvg_cold_start='wrf', ruc_2m_diagnostic='flux',
                         ra_lw_physics=4, ra_sw_physics=4)
    # The domain builder consumes the experiment tables, while the source
    # route owns the companion fetch declaration from the public emitter.
    from gpuwm.fetch import validate_fetch_hints
    validate_fetch_hints(raw.pop('fetch'), source='roundtrip request')
    exp = build_experiment(raw, source='roundtrip request')
    native = render_namelist_input(exp)
    physics = parse_namelist_text(native)['physics']
    assert physics['rdlai2d'] == physics['usemonalb'] == [prescribed]
    text, _ = import_namelists(*_pair(
        tmp_path, wps=render_wps_namelist(exp), inp=native))
    imported = build_experiment(tomllib.loads(text), source='native roundtrip')
    for domain in imported.domains:
        assert domain.run.rdlai2d is domain.run.usemonalb is prescribed
        assert domain.run.ruc_qvg_cold_start == 'wrf'
        assert domain.run.ruc_2m_diagnostic == 'flux'


@pytest.mark.parametrize('field', ('rdlai2d', 'usemonalb'))
@pytest.mark.parametrize('root_prescribed', (True, False))
def test_native_ruc_companion_refuses_mixed_monthly_surface_choices(
        tmp_path, field, root_prescribed):
    text, _ = import_namelists(*_pair(tmp_path, inp=_ruc_input(fork=True)))
    exp = build_experiment(tomllib.loads(text), source='mixed monthly request')
    assert len(exp.domains) > 1
    domains = tuple(
        replace(domain, run=replace(domain.run, **{
            field: root_prescribed if index == 0 else not root_prescribed}))
        for index, domain in enumerate(exp.domains))
    with pytest.raises(ValueError, match=field + ' differs between domains'):
        render_namelist_input(replace(exp, domains=domains))


def test_content_inference_fills_missing_monthly_fields_and_respects_false(tmp_path):
    missing = _ruc_input(fork=True)
    text, report = import_namelists(*_pair(tmp_path, inp=missing))
    inferred = build_experiment(tomllib.loads(text), source='fork inference')
    assert inferred.root.run.rdlai2d is inferred.root.run.usemonalb is True
    applied = {entry.key: entry.value for entry in report.defaults_applied}
    assert applied['rdlai2d'] is applied['usemonalb'] is True
    explicit = missing.replace(
        '&physics\n', '&physics\n rdlai2d = .false.,\n usemonalb = .false.,\n')
    explicit = with_physics_selector_comment(explicit, {
        'ruc_qvg_cold_start': 'wrf', 'ruc_2m_diagnostic': 'flux'})
    text, _ = import_namelists(*_pair(tmp_path, inp=explicit))
    legacy = build_experiment(tomllib.loads(text), source='explicit legacy')
    assert legacy.root.run.rdlai2d is legacy.root.run.usemonalb is False
    assert legacy.root.run.ruc_qvg_cold_start == 'wrf'
    assert legacy.root.run.ruc_2m_diagnostic == 'flux'
