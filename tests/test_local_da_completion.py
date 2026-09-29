"""Terminal output and simultaneous resource limits through their real owners."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.local_da import (Card, Request, PlanError, build_plan,
                           REMEDY_CADENCE, REMEDY_CARD_MEMORY, REMEDY_HOST_MEMORY)
from test_local_da_plan import EPOCH, availability, price, request


def test_requested_forcing_horizon_is_not_hidden_by_lower_rung_costs():
    req = Request(epoch=EPOCH, point=(40., -100.),
                  card=Card(vram_gib=10., free_gib=1.5, host_gib=4.),
                  scale=6, cadence_seconds=100000., budget_seconds=1.e8)
    with pytest.raises(PlanError) as caught:
        build_plan(req, availability=availability)
    assert caught.value.code == 'FORCING_HORIZON'
    assert 'Supply the required frames' in str(caught.value)
    assert '--free-gib' not in str(caught.value)
    assert any(row['priced'] and REMEDY_HOST_MEMORY in row['remedies']
               for row in caught.value.details['alternatives'])


def test_cadence_and_both_memory_advisories_are_present_together():
    def cost(req, rung, exp, streams):
        return dict(price(req, rung, exp, streams), forecast_peak_bytes=1 << 45,
                    host_peak_bytes=1 << 46, cycle_seconds=4000.)
    plan = build_plan(request(scale=3), availability=availability, price=cost)
    message = ' '.join(plan['warnings'])
    for remedy in (REMEDY_CADENCE, REMEDY_CARD_MEMORY, REMEDY_HOST_MEMORY):
        assert message.count(remedy.rstrip('.')) == 1


def _member_with_real_output(tmp_path, monkeypatch, stop):
    from gpuwm import runtime
    from gpuwm.core import streaming, health, refl, uh_diag
    from gpuwm.io import wrfout, restart as restart_io
    from gpuwm.ensemble.member import run_member
    from gpuwm import supervisor

    cfg = SimpleNamespace(dt=15., run_seconds=stop, output_interval_s=900.,
                          restart_interval_s=0., mp_physics=10, nz=2, nx=2, ny=2,
                          dx=3000., dy=3000., spec_bdy_width=1)
    state = SimpleNamespace(elapsed_seconds=0., thp=np.zeros((2, 2, 2)),
                            qv=np.full((2, 2, 2), .005), ready=False,
                            physics=SimpleNamespace(fields={'swdown': np.zeros((2, 2))},
                                microphysics=SimpleNamespace(rainnc=np.zeros((2, 2))),
                                rainc=None, ysu_nan_guard_fires=0, call_counts={'radiation': 0}))
    prepared = SimpleNamespace(cfg=cfg, initial_result=SimpleNamespace(state=state, coord=None),
                               grid=None, static_fields={})
    exp = SimpleNamespace(domains=(SimpleNamespace(run=cfg, history_interval_s=900.),),
                          start_time=datetime(2026, 9, 10, 12), run_seconds=stop,
                          restart_interval_s=0., spectral_numerics=None)
    checkpoint = tmp_path / 'analysis.npz'
    np.savez(checkpoint, **{'state/thp': np.ones((2, 2, 2)),
                           'meta/elapsed_seconds': np.asarray(900.)})
    due = []
    def step(current, configuration, *, refl_10cm_due=False):
        current.elapsed_seconds += configuration.dt
        current.thp += .01
        current.ready = refl_10cm_due
        if refl_10cm_due:
            due.append(current.elapsed_seconds)
    def restore(path, current, configuration):
        with np.load(path) as file:
            current.elapsed_seconds = float(file['meta/elapsed_seconds'])
            current.thp[:] = file['state/thp']
        return SimpleNamespace(elapsed_seconds=current.elapsed_seconds, run_trackers=None)
    fake_cp = SimpleNamespace(asnumpy=np.asarray,
        cuda=SimpleNamespace(runtime=SimpleNamespace(deviceSynchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cp)
    monkeypatch.setitem(sys.modules, 'gpuwm.core.dycore', SimpleNamespace(step=step))
    monkeypatch.setattr(health, 'StateHealthValidator', lambda state: SimpleNamespace(require_healthy=lambda **kw: None))
    monkeypatch.setattr(streaming, 'stability_observer', lambda _: lambda *a, **kw: {
        'nan': False, 'w_max': 0., 'w_argmax': 0, 'boundary_w_max': 0., 'interior_w_max': 0., 'cfl': 0.})
    monkeypatch.setattr(streaming, 'domain_field_max', lambda *a: 0.)
    monkeypatch.setattr(runtime, 'apply_single_domain_pbl_cadence', lambda *a: None)
    monkeypatch.setattr(runtime, 'trajectory_digest_enabled', lambda: False)
    monkeypatch.setattr(runtime, '_metadata_frame', lambda *a: {})
    monkeypatch.setattr(runtime, '_global_wrf_attrs', lambda *a, **kw: {})
    monkeypatch.setattr(runtime, 'soil_layer_count', lambda *a: 4)
    monkeypatch.setattr(uh_diag, 'reset_up_heli_max', lambda *a: None)
    monkeypatch.setattr(supervisor, 'validate_manifest_checkpoint', Path)
    monkeypatch.setattr(restart_io, 'restore_restart', restore)
    monkeypatch.setattr(wrfout, 'state_frame', lambda *a, **kw: {'T2': np.full((2, 2), 290.)})
    def consume(current):
        assert current.ready, 'terminal frame missed its microphysics output handshake'
        current.ready = False
        return np.zeros((2, 2, 2))
    monkeypatch.setattr(refl, 'consume_refl_10cm', consume)
    member_dir = tmp_path / 'forecast' / 'member_000'
    outcome = run_member(base_config=tmp_path / 'unused.toml', member_dir=member_dir,
                         index=0, seed=1, perturbation='none', run_seconds=stop,
                         restart=checkpoint, prepare=lambda _: (
                             exp, SimpleNamespace(output_title='output test', output_domain=1), prepared))
    assert outcome.sim_seconds == stop
    return outcome, due, exp.start_time


@pytest.mark.parametrize('stop,expected', [(1500., [1500.]), (1800., [1800.]), (2100., [1800., 2100.])])
def test_resumed_member_terminal_frame_reaches_inventory_and_render_route(tmp_path, monkeypatch, stop, expected):
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm import go_cli
    from gpuwm.ensemble.manifest import write_manifest_atomically, ENSEMBLE_MANIFEST_SCHEMA
    outcome, due, start = _member_with_real_output(tmp_path, monkeypatch, stop)
    assert due == expected
    assert outcome.wrfout_count == len(expected)
    assert [entry['frames'][0]['valid_time'] for entry in outcome.wrfout_inventory] == [
        (start + timedelta(seconds=value)).strftime('%Y-%m-%d_%H:%M:%S') for value in expected]
    root = outcome.member_dir.parent
    manifest = root / 'ensemble-manifest.json'
    write_manifest_atomically(manifest, {'schema': ENSEMBLE_MANIFEST_SCHEMA,
        'members': [{'index': 0, 'member_dir': 'member_000', 'wrfout_inventory': list(outcome.wrfout_inventory)}]})
    backend = PreparedBackend({'analysis_times': []}, tmp_path)
    backend.go = {'render': tmp_path / 'products'}
    commands = []
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(go_cli, '_run_stage', lambda name, command, **kw: commands.append(command))
    result = backend.products(SimpleNamespace(manifest_path=manifest, ens_root=root))
    assert result['members'][0]['status'] == 'complete'
    assert commands[0][1:4] == ['-m', 'gpuwm.cli', 'render']
    assert all(str(outcome.member_dir / entry['path']) in commands[0]
               for entry in outcome.wrfout_inventory)


def test_empty_forecast_inventory_does_not_claim_complete_products(tmp_path):
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm.ensemble.manifest import write_manifest_atomically, ENSEMBLE_MANIFEST_SCHEMA
    manifest = tmp_path / 'ensemble-manifest.json'
    write_manifest_atomically(manifest, {'schema': ENSEMBLE_MANIFEST_SCHEMA,
        'members': [{'index': 0, 'member_dir': 'member_000', 'wrfout_inventory': []}]})
    backend = PreparedBackend({'analysis_times': []}, tmp_path)
    with pytest.raises(RuntimeError, match='products are incomplete'):
        backend.products(SimpleNamespace(manifest_path=manifest, ens_root=tmp_path))


def test_member_pictures_with_no_map_files_are_named_in_the_products_record(tmp_path, monkeypatch, capsys):
    """The member renders print their map-file warning into output that is
    captured and dropped when they succeed; the products record and the
    terminal carry it once instead, and the pictures are still drawn."""
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm import go_cli, rustwx
    from gpuwm.ensemble import wrfout_inventory
    from gpuwm.ensemble.manifest import write_manifest_atomically, ENSEMBLE_MANIFEST_SCHEMA
    from gpuwm.render import BASEMAP_MISSING_CODE
    from test_render_basemap_delivery import wheel_with_companion
    wheel_with_companion(tmp_path, monkeypatch, maps=False)
    root = tmp_path / 'forecast'
    root.mkdir()
    manifest = root / 'ensemble-manifest.json'
    write_manifest_atomically(manifest, {'schema': ENSEMBLE_MANIFEST_SCHEMA, 'members': [
        {'index': index, 'member_dir': f'member_{index:03d}',
         'wrfout_inventory': [{'path': 'wrfout_d01_2026-09-10_12:15:00'}]} for index in range(2)]})
    monkeypatch.setattr(wrfout_inventory, 'verify_entry', lambda *a, **k: [])
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    commands = []
    monkeypatch.setattr(go_cli, '_run_stage', lambda name, command, **kw: commands.append(command))
    backend = PreparedBackend({'analysis_times': []}, tmp_path)
    backend.go = {'render': tmp_path / 'products'}
    forecast = SimpleNamespace(manifest_path=manifest, ens_root=root)
    result = backend.products(forecast)
    assert len(commands) == 2 and [row['status'] for row in result['members']] == ['complete'] * 2
    [warning] = result['warnings']
    assert warning['code'] == BASEMAP_MISSING_CODE
    assert 'no coastlines, borders or state lines' in warning['message']
    assert warning['remedy'].startswith('pip install --force-reinstall gpuwm-data')
    assert capsys.readouterr().err.count('render: warning: no map assets resolve') == 1
    # Asked again, the record says it again: it is a record, not a one-time event.
    assert backend.products(forecast)['warnings'] == [warning]
    # An install whose gpuwm-data has its maps records nothing.
    maps = tmp_path / 'companion' / 'basemap'
    maps.mkdir(parents=True)
    monkeypatch.setattr(rustwx, 'companion_basemap_dir', lambda: maps)
    assert 'warnings' not in backend.products(forecast)


def test_missing_renderer_is_refused_before_configuration_or_gpu_work(tmp_path, monkeypatch):
    from gpuwm.local_da_runtime import PreparedBackend
    from gpuwm import capabilities, go_cli
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: 'Rendering dependency is missing; install it before launch.')
    backend = PreparedBackend({'analysis_times': []}, tmp_path)
    with pytest.raises(RuntimeError, match='Rendering dependency'):
        backend.preflight()
    assert not list(tmp_path.iterdir())


def test_product_failure_is_recorded_as_failed_execution(tmp_path):
    from gpuwm.local_da_runtime import launch, PreparedBackend
    from test_local_da_runtime import saved, Backend
    class FailedRender(Backend):
        def products(self, forecast):
            return PreparedBackend({'analysis_times': []}, tmp_path).products(forecast)
    path, _ = saved(tmp_path)
    with pytest.raises(RuntimeError, match='no output frames'):
        launch(path, backend=FailedRender())
    report = json.loads((path.parent / 'execution.json').read_text())
    assert report['status'] == 'FAILED'
    assert report['products']['members'][0]['status'] == 'unavailable'


def test_launch_recovers_an_old_complete_forecast_with_no_frames(tmp_path):
    from gpuwm.local_da_runtime import launch
    from test_local_da_runtime import saved, Backend
    path, plan = saved(tmp_path)
    launch(path, backend=Backend())
    old = path.parent / 'forecast' / 'ensemble-manifest.json'
    document = json.loads(old.read_text())
    for member in document['members']:
        member['wrfout_inventory'] = []
    old.write_text(json.dumps(document))
    original = old.read_bytes()
    backend = Backend()
    report = launch(path, backend=backend)
    assert report['status'] == 'COMPLETE'
    assert Path(report['forecast_manifest']).parent.name == 'forecast-output-1'
    assert old.read_bytes() == original
    assert not [call for call in backend.calls if call[0] == 'analysis']
    members = [call for call in backend.calls if call[0] == 'member']
    assert len(members) == plan['selected']['members']
    assert all(call[3].name == 'analysis.npz' for call in members)
    again = Backend()
    launch(path, backend=again)
    assert not [call for call in again.calls if call[0] in ('analysis', 'member')]
