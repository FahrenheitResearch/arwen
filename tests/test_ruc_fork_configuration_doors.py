"""RUC fork choices belong to the requested model configuration.

An operational-fork namelist and the shipped full HRRR recipes select
the fork snow, irrigation, humidity and diagnostic forms. A filename or a meteorological data
source alone does not select them.
"""

from pathlib import Path
import hashlib
import re
import tomllib

import pytest

from gpuwm.experiment import load_experiment
from gpuwm.namelist_import import (
    import_namelists, operational_fork_ruc_defaults, parse_namelist_text)
from test_namelist_import import INPUT_TEXT, _pair


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_HRRR_NAMELIST = ROOT / 'tests' / 'fixtures' / 'hrrr_wrf_v4121.nl'
ACTUAL_HRRR_SHA256 = 'b244df1bff090f6ba7bb43817b91c842ae9b1add1bb6f8214fbb6403382c38c0'
RECIPES = (
    ROOT / 'configs' / 'recipes' / 'conus_hrrr_configuration.toml',
    ROOT / 'configs' / 'recipes' / 'hrrr_configuration_cut.toml',
)


def _actual_hrrr_sections():
    return parse_namelist_text(ACTUAL_HRRR_NAMELIST.read_text(encoding='utf-8'))


def test_actual_v4121_hrrr_wrf_namelist_selects_the_fork_ruc_forms():
    """The pinned public namelist itself identifies the configuration."""
    canonical = ACTUAL_HRRR_NAMELIST.read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(canonical).hexdigest() == ACTUAL_HRRR_SHA256
    sections = _actual_hrrr_sections()
    assert sections['physics']['sf_surface_physics'] == [3, 3, 1]
    assert sections['physics']['num_soil_layers'] == [9]
    assert operational_fork_ruc_defaults(sections) == {
        'ruc_soilprop': 'wrf_45',
        'ruc_irrigation': 'wrf_45', 'ruc_snow': 'wrf_45',
        'ruc_qvg_cold_start': 'air', 'ruc_2m_diagnostic': 'log_profile'}


def _ruc_input(*, fork=False):
    text = INPUT_TEXT.replace('mp_physics = 55, 55', 'mp_physics = 8, 8').replace(
        'sf_sfclay_physics = 91, 91', 'sf_sfclay_physics = 5, 5').replace(
        'sf_surface_physics = 2, 2', 'sf_surface_physics = 3, 3').replace(
        'bl_pbl_physics = 11, 11', 'bl_pbl_physics = 5, 5').replace(
        ' bldt = 0, 0,', ' num_soil_layers = 9,\n bldt = 0, 0,')
    if fork:
        text = text.replace('&time_control\n', '&time_control\n gsd_diagnostics = 0,\n')
    return text


def _supported_actual_ruc_input():
    """Project the source's RUC vectors into the supported two-domain fixture.

    Geometry and the remaining physics come from the importer fixture.
    The source's active additional diagnostics are explicitly disabled;
    this fixture does not claim to import the complete operational run.
    """
    source = _actual_hrrr_sections()
    text = _ruc_input(fork=True)
    for key in ('sf_surface_physics', 'num_soil_layers', 'mosaic_lu', 'mosaic_soil',
                'rdlai2d', 'usemonalb'):
        values = source['physics'][key][:2]
        line = f" {key} = " + ', '.join(
            '.true.' if value is True else '.false.' if value is False else str(value)
            for value in values) + ','
        text, count = re.subn(rf'(?m)^\s*{key}\s*=.*$', line, text)
        if not count:
            text = text.replace('&physics\n', '&physics\n' + line + '\n')
        assert count <= 1
    return text


def test_actual_hrrr_ruc_setting_vectors_import_through_the_supported_door(tmp_path):
    text, report = import_namelists(*_pair(tmp_path, inp=_supported_actual_ruc_input()))
    path = tmp_path / 'resolved-hrrr-ruc.toml'
    path.write_text(text, encoding='utf-8')
    experiment = load_experiment(path)
    source = _actual_hrrr_sections()['physics']
    for index, domain in enumerate(experiment.domains):
        for key in ('sf_surface_physics', 'num_soil_layers', 'mosaic_lu', 'mosaic_soil',
                    'rdlai2d', 'usemonalb'):
            values = source[key]
            assert getattr(domain.run, key) == values[min(index, len(values) - 1)]
        assert domain.run.ruc_irrigation == domain.run.ruc_snow == 'wrf_45'
        assert domain.run.ruc_qvg_cold_start == 'air'
        assert domain.run.ruc_2m_diagnostic == 'log_profile'
    assert any(entry.key == 'gsd_diagnostics' for entry in report.dropped)


def test_actual_hrrr_active_diagnostics_are_not_silently_imported(tmp_path):
    source = _actual_hrrr_sections()
    assert source['time_control']['gsd_diagnostics'] == [1]
    text = _supported_actual_ruc_input().replace('gsd_diagnostics = 0', 'gsd_diagnostics = 1')
    # Output only, so it imports -- but as a declared difference the
    # report and the emitted TOML both print, never silently.
    _assert_declared_diagnostics(*import_namelists(*_pair(tmp_path, inp=text)))


def _assert_declared_diagnostics(toml_text, report):
    declared = {s.key: s for s in report.substitutions if s.reason}
    assert 'not written' in declared['gsd_diagnostics'].reason
    assert '#   gsd_diagnostics 1' in toml_text
    assert 'gsd_diagnostics' in report.format()


def _import(tmp_path, *, fork, filename):
    wps, original = _pair(tmp_path, inp=_ruc_input(fork=fork))
    named = tmp_path / filename
    named.write_text(original.read_text(), encoding='utf-8')
    text, report = import_namelists(wps, named, name='ruc-configuration')
    output = tmp_path / 'imported.toml'
    output.write_text(text, encoding='utf-8')
    return text, report, load_experiment(output)


@pytest.mark.parametrize('filename', ['hrrr_wrf.nl', 'renamed-model.nl'])
def test_operational_fork_import_selects_snow_and_irrigation_by_content(tmp_path, filename):
    text, report, experiment = _import(tmp_path, fork=True, filename=filename)
    for domain in experiment.domains:
        assert domain.run.ruc_irrigation == 'wrf_45'
        assert domain.run.ruc_snow == 'wrf_45'
        assert domain.run.ruc_soilprop == 'wrf_45'
        assert domain.run.ruc_qvg_cold_start == 'air'
        assert domain.run.ruc_2m_diagnostic == 'log_profile'
    assert 'ruc_irrigation = "wrf_45"' in text
    assert 'ruc_snow = "wrf_45"' in text
    assert 'ruc_soilprop = "wrf_45"' in text
    assert 'ruc_qvg_cold_start = "air"' in text
    assert 'ruc_2m_diagnostic = "log_profile"' in text
    defaults = {entry.key: entry.value for entry in report.defaults_applied}
    assert defaults['ruc_irrigation'] == defaults['ruc_snow'] == 'wrf_45'
    assert defaults['ruc_qvg_cold_start'] == 'air'
    assert defaults['ruc_2m_diagnostic'] == 'log_profile'


def test_a_public_wrf_namelist_named_hrrr_keeps_the_generic_lineages(tmp_path):
    text, _, experiment = _import(tmp_path, fork=False, filename='hrrr_wrf.nl')
    assert 'ruc_irrigation =' not in text and 'ruc_snow =' not in text
    for domain in experiment.domains:
        assert domain.run.ruc_irrigation == 'wrf_461'
        assert domain.run.ruc_snow == 'wrf_461'
        assert domain.run.ruc_soilprop == 'wrf_45'
        assert domain.run.ruc_qvg_cold_start == 'wrf'
        assert domain.run.ruc_2m_diagnostic == 'flux'


def test_the_fork_signature_does_not_change_a_non_ruc_configuration():
    sections = parse_namelist_text(INPUT_TEXT.replace(
        '&time_control\n', '&time_control\n gsd_diagnostics = 0,\n'))
    assert operational_fork_ruc_defaults(sections) == {}


def test_enabled_unported_fork_diagnostics_are_declared(tmp_path):
    text = _ruc_input(fork=True).replace('gsd_diagnostics = 0', 'gsd_diagnostics = 1')
    _assert_declared_diagnostics(*import_namelists(*_pair(tmp_path, inp=text)))


@pytest.mark.parametrize('recipe', RECIPES, ids=lambda p: p.stem)
def test_shipped_hrrr_configuration_recipes_load_explicit_fork_ruc_choices(recipe):
    raw = tomllib.loads(recipe.read_text(encoding='utf-8'))
    assert raw['static']['source'] == 'hrrr-conus-v4'
    assert raw['shared']['ruc_irrigation'] == raw['shared']['ruc_snow'] == 'wrf_45'
    assert raw['shared']['ruc_soilprop'] == 'wrf_45'
    assert raw['shared']['ruc_qvg_cold_start'] == 'air'
    assert raw['shared']['ruc_2m_diagnostic'] == 'log_profile'
    assert raw['shared']['rdlai2d'] is raw['shared']['usemonalb'] is True
    experiment = load_experiment(recipe)
    for domain in experiment.domains:
        assert domain.run.sf_surface_physics == 3
        assert domain.run.ruc_irrigation == domain.run.ruc_snow == 'wrf_45'
        assert domain.run.ruc_soilprop == 'wrf_45'
        assert domain.run.ruc_qvg_cold_start == 'air'
        assert domain.run.ruc_2m_diagnostic == 'log_profile'
        assert domain.run.rdlai2d is domain.run.usemonalb is True


@pytest.mark.parametrize('recipe', RECIPES, ids=lambda p: p.stem)
def test_go_front_door_reads_the_shipped_hrrr_configuration_recipe(recipe, tmp_path, capsys, monkeypatch):
    from gpuwm.cli import main
    from gpuwm.ingest.wif_climatology import (
        WIF_CLIMATOLOGY_PATH_ENV, load_wif_climatology)
    from test_wif_climatology import _synthetic_dat

    # The actual GPU-visible planner admits the declared MP28 climatology
    # and analyzed-aerosol emission paths only with a staged WIF dataset.
    # Use the existing complete twelve-month IFV5 fixture and its real
    # reader; keep the shipped recipe's physics and aerosol source intact.
    wif_path, arrays = _synthetic_dat(tmp_path)
    climatology = load_wif_climatology(wif_path)
    import numpy as np
    for actual, expected in ((climatology.qnwfa, arrays['QNWFA']),
                             (climatology.qnifa, arrays['QNIFA']),
                             (climatology.pressure, arrays['P_WIF'])):
        np.testing.assert_array_equal(actual, expected)
    monkeypatch.setenv(WIF_CLIMATOLOGY_PATH_ENV, str(wif_path))
    before = recipe.read_bytes()

    outdir = tmp_path / 'uncreated-forecast'
    assert main(['go', str(recipe), '--dry-run', '--outdir', str(outdir)]) == 0
    output = capsys.readouterr().out
    assert 'Run:' in output
    assert str(recipe) in output
    assert not outdir.exists()
    assert recipe.read_bytes() == before


def test_a_custom_hrrr_data_configuration_keeps_generic_ruc_lineages(tmp_path):
    raw = RECIPES[0].read_text(encoding='utf-8')
    for key, value in (('ruc_soilprop', 'wrf_45'), ('ruc_irrigation', 'wrf_45'), ('ruc_snow', 'wrf_45'),
                       ('ruc_qvg_cold_start', 'air'), ('ruc_2m_diagnostic', 'log_profile')):
        raw = raw.replace(f'{key} = "{value}"\n', '')
    path = tmp_path / 'custom.toml'
    path.write_text(raw, encoding='utf-8')
    cfg = load_experiment(path).root.run
    assert cfg.ruc_irrigation == cfg.ruc_snow == 'wrf_461'
    assert cfg.ruc_soilprop == 'wrf_45'
    assert cfg.ruc_qvg_cold_start == 'wrf'
    assert cfg.ruc_2m_diagnostic == 'flux'


def test_prescribed_surface_selectors_reuse_the_same_prepared_state():
    from dataclasses import asdict, replace
    from gpuwm.ingest.prepared_cache import effective_prepared_domain_config

    before = load_experiment(RECIPES[0]).root
    before = replace(before, run=replace(before.run, rdlai2d=False, usemonalb=False,
                                       ruc_qvg_cold_start='wrf', ruc_2m_diagnostic='flux'))
    after = replace(before, run=replace(before.run, rdlai2d=True, usemonalb=True,
                                      ruc_qvg_cold_start='air', ruc_2m_diagnostic='log_profile'))
    assert effective_prepared_domain_config(asdict(before)) == \
        effective_prepared_domain_config(asdict(after))
    ice_change = replace(after, run=replace(after.run, fractional_seaice=1))
    assert effective_prepared_domain_config(asdict(ice_change)) != \
        effective_prepared_domain_config(asdict(after))
