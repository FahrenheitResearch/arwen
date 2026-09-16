"""Source discovery and CLI handoff; numeric recovery has native producer tests."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.wrf_soil_recovery import authority_from_vtable, recover_supplied_soil


VTABLE = """
85 | 111 | 5 | | SOILT001 | K | Soil temperature | 2 | 3 | 18 | 106 |
85 | 111 | 1458 | | SOILT999 | K | Soil temperature | 2 | 3 | 18 | 106 |
86 | 112 | 0 | 1 | SOILM001 | kg m-2 | Soil water | 2 | 3 | 20 | 106 |
86 | 112 | 729 | 2187 | SOILM999 | kg m-2 | Soil water | 2 | 3 | 20 | 106 |
"""


def _source(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'Vtable').write_text(VTABLE, encoding='utf-8')
    (directory / 'met_em.d01.2026-09-12_00_00_00.nc').write_bytes(b'source')


def _fields():
    return {'SMOIS': np.full((4, 2, 2), 22.),
            'SH2O': np.zeros((4, 2, 2)), 'LANDMASK': np.ones((2, 2))}


def test_vtable_binds_water_layers_to_actual_metgrid_axis_not_temperature_selector(tmp_path):
    _source(tmp_path)
    result = authority_from_vtable(tmp_path / 'Vtable')
    assert result['source_depths_from_metgrid'] is True
    assert 'source_layer_depths_m' not in result
    assert result['source_layer_bounds_m'] == [[0, .01], [7.29, 21.87]]
    assert result['source_quantity'] == 'layer_water_mass'


@pytest.mark.parametrize('edit', [
    lambda s: s + s.splitlines()[3] + '\n',
    lambda s: s.replace('SOILM', 'NOTSOIL'),
    lambda s: s.replace('112 | 0', '111 | 0'),
    lambda s: s.replace('kg m-2', 'K'),
])
def test_incomplete_or_ambiguous_vtable_does_not_guess(tmp_path, edit):
    _source(tmp_path)
    path = tmp_path / 'Vtable'
    path.write_text(edit(VTABLE), encoding='utf-8')
    with pytest.raises(ValueError):
        authority_from_vtable(path)


def test_physical_soil_does_not_need_or_read_original_sources(tmp_path):
    fields = _fields()
    fields['SMOIS'][:] = .23
    assert recover_supplied_soil(tmp_path / 'missing', fields, {}) == ({}, None)


def test_default_source_discovery_records_exact_authority_and_preserves_files(tmp_path, monkeypatch):
    from gpuwm import netcdf_bridge
    _source(tmp_path)
    path = tmp_path / 'wrfinput_d01'
    path.write_bytes(b'wrf')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    calls = []
    def recover(wrfinput, met_em, authority):
        calls.append((wrfinput, met_em, authority))
        return {'SMOIS': np.full((4, 2, 2), .25)}, {'method': 'source-layer-recovery'}
    monkeypatch.setattr(netcdf_bridge, 'recover_wrf_soil', recover)
    fields = _fields()
    result, receipt = recover_supplied_soil(path, fields, {'START_DATE': '2026-09-12_00:00:00'})
    assert calls[0][1].name == 'met_em.d01.2026-09-12_00_00_00.nc'
    assert set(receipt['input_files']) == {'wrfinput', 'met_em', 'authority'}
    assert receipt['authority'] == calls[0][2]
    assert all(len(item['sha256']) == 64 for item in receipt['input_files'].values())
    np.testing.assert_array_equal(result['SMOIS'], .25)
    np.testing.assert_array_equal(fields['SMOIS'], 22.)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_missing_original_source_gives_concrete_recovery_action(tmp_path):
    with pytest.raises(ValueError, match='--soil-source DIR'):
        recover_supplied_soil(tmp_path / 'wrfinput_d01', _fields(),
                              {'START_DATE': '2026-09-12_00:00:00'})


def test_source_change_during_native_recovery_cannot_publish_result(tmp_path, monkeypatch):
    from gpuwm import netcdf_bridge
    _source(tmp_path)
    path = tmp_path / 'wrfinput_d01'
    path.write_bytes(b'wrf')
    def changed(wrfinput, met_em, authority):
        met_em.write_bytes(b'changed')
        return {'SMOIS': np.full((4, 2, 2), .25)}, {}
    monkeypatch.setattr(netcdf_bridge, 'recover_wrf_soil', changed)
    with pytest.raises(ValueError, match='inputs changed'):
        recover_supplied_soil(path, _fields(), {'START_DATE': '2026-09-12_00:00:00'})


def test_source_directory_reaches_public_door_and_worker_parser(tmp_path, monkeypatch):
    from gpuwm import capabilities, cli, provenance_gate, wrfinput_forecast
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    calls = []
    monkeypatch.setattr(wrfinput_forecast, 'run_wrf_forecast',
                        lambda *a, **kw: calls.append(kw) or 0)
    source = tmp_path / 'original source'
    assert cli.main(['run', '--wrfinput', 'wrf', '--soil-source', str(source)]) == 0
    assert calls[0]['soil_source'] == source
    parsed = wrfinput_forecast.build_parser().parse_args(
        ['--wrfinput', 'wrf', '--outdir', 'out', '--soil-source', str(source), '--_worker'])
    assert parsed.soil_source == source


def test_unrelated_run_cannot_silently_ignore_original_soil_source(monkeypatch):
    from gpuwm import capabilities, cli, provenance_gate
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    with pytest.raises(SystemExit) as stopped:
        cli.main(['run', 'case.toml', '--soil-source', 'source'])
    assert stopped.value.code == 2


def test_supervised_worker_keeps_original_source_directory(tmp_path, monkeypatch):
    import subprocess
    from gpuwm import go_cli, supervisor, wrfinput_door, wrfinput_forecast
    from gpuwm.filesystem_paths import canonical_path
    monkeypatch.setattr(wrfinput_door, 'resolve_wrfinput_run', lambda *a, **kw: SimpleNamespace(experiment=None))
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(supervisor, 'select_gpu', lambda *a: SimpleNamespace(uuid='test-device'))
    monkeypatch.setattr(supervisor, 'GPUFileLock', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(supervisor, 'preflight_exclusive_gpu', lambda *a, **kw: None)
    commands = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw:
                        commands.append(command) or SimpleNamespace(returncode=0))
    source = tmp_path / 'original WPS inputs'
    assert wrfinput_forecast.run_wrf_forecast(tmp_path / 'wrf', tmp_path / 'out',
                                             soil_source=source) == 0
    command = commands[0]
    assert Path(command[command.index('--soil-source') + 1]) == canonical_path(source)
    assert '--_worker' in command and '--_output-owner' in command
