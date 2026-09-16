"""Exercise the production controller adapter with CPU forecast substitution."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import numpy as np
import pytest

from gpuwm import local_da_runtime as runtime
from gpuwm.da import radar_assimilation as radar
from gpuwm.ensemble import cycle
from gpuwm.ensemble.analysis_commit import read_record, write_record
from gpuwm.output_identity import file_record
from test_continuous_cycle_window import clocked_runner
from test_local_da_controller import Backend, controller
from test_ensemble_engine import _contract_names


def test_actual_adapter_reuses_owners_clocks_scratch_and_usage(tmp_path, monkeypatch):
    helper = Backend(tmp_path)
    ctl = controller(tmp_path)
    backend = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    backend.controller, backend.cfg = ctl, helper.cfg
    backend.plan = {'selected': {'cadence_seconds': 60.}, 'memory': {'solve_memory_mib': 256}}
    backend._continuous_epoch = datetime(2026, 9, 10, tzinfo=timezone.utc)
    backend._active_index = 0
    backend.scratch_override = 4096
    backend.mp_physics = 6
    backend.member_runner = clocked_runner
    backend.analysis_context = lambda index, states, recovering: {'index': index}
    # Keep the real scheduling calculation; inject only measured capacity.
    resolve = runtime.analysis_execution_budget
    monkeypatch.setattr(runtime, 'analysis_execution_budget', lambda planned_mib, override_mib:
        resolve(planned_mib, override_mib=override_mib, capacity={}))
    calls = []
    def assimilate(index, states):
        budget, progress, receipt = radar._execution_settings(256, None)
        assert budget == 4096 and receipt['planned_scratch_mib'] == 256
        progress({'phase': 'solve', 'gridpoints_done': 6, 'gridpoints_total': 6})
        published = json.loads((ctl.root / 'status.json').read_text())
        assert published['analysis_execution']['memory_budget_mib'] == 4096
        assert published['analysis_progress']['gridpoints_done'] == 6
        calls.append(index)
        return ({i: {_contract_names()[0]: np.full((2, 3), .25)} for i in states},
                {'method': 'cpu-control', 'innovations': [{'name': 'observed', 'observations': index+1}]})
    backend.assimilate = assimilate
    prior = None
    for index in range(2):
        directory = tmp_path / f'window_{index:06d}'
        directory.mkdir()
        outcome = backend.analyze_window(index, directory, prior, {'generation': index})
        assert cycle._restart_elapsed_seconds(outcome['analysis'][0]['path']) == (index+1)*60
        assert outcome['observation_usage'][0]['accepted_for_analysis'] == index+1
        assert runtime._analysis_time(backend, index) == backend._continuous_epoch.replace(minute=index+1)
        # The saved roster is recovered without analysis or new scheduler work.
        recovered = backend.analyze_window(index, directory, prior, {'generation': index})
        assert recovered == outcome
        prior = outcome
    assert calls == [0, 1]
    assert radar._execution_settings(256, None) == (256, None, None)


def test_cumulative_timing_survives_process_restarts_without_counting_downtime(tmp_path):
    clock = [0.]
    backend = Backend(tmp_path)
    prepare, analyze, products = backend.prepare_window, backend.analyze_window, backend.produce_window
    for attempt in range(2):
        ctl = controller(tmp_path, monotonic=lambda: clock[0])
        def prepare_timed(*args, **kwargs):
            ctl.stage('preparation')
            clock[0] += 2
            return prepare(*args, **kwargs)
        def analyze_timed(*args, **kwargs):
            ctl.stage('analysis')
            clock[0] += 3
            return analyze(*args, **kwargs)
        def products_timed(*args, **kwargs):
            ctl.stage('render')
            clock[0] += 5
            return products(*args, **kwargs)
        backend.prepare_window, backend.analyze_window, backend.produce_window = prepare_timed, analyze_timed, products_timed
        result = ctl.step(backend)
        assert result['elapsed_seconds'] == (attempt+1)*10
        assert result['session_elapsed_seconds'] == 10
        assert result['stage_seconds'] == {'preparation': 2., 'analysis': 3., 'render': 5.}
        assert result['total_stage_seconds'] == {'preparation': 2.*(attempt+1), 'analysis': 3.*(attempt+1), 'render': 5.*(attempt+1)}
        clock[0] += 1000  # A stopped process has no measurements during this interval.


def test_committed_product_intent_recovers_without_forecast_or_render(tmp_path):
    backend = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    backend.plan = {'review_sha256': 'review'}
    cfg = tmp_path / 'experiment.toml'
    cfg.write_text('immutable configuration')
    backend.cfg = SimpleNamespace(base_config=cfg, base_config_sha256=file_record(cfg)['sha256'])
    asset = tmp_path / 'published-output'
    asset.write_bytes(b'original published output')
    execution = dict(schema='arwen.local-da-execution.v1', review_sha256='review', cycles=[7], status='COMPLETE',
        base_config=str(cfg), base_config_sha256=backend.cfg.base_config_sha256,
        forecast_manifest='retained-forecast-manifest', cycle_manifest='retained-cycle-manifest')
    write_record(tmp_path / 'products-intent.json', {'execution': execution, 'artifacts': [file_record(asset)]})
    first = backend.produce_window(7, tmp_path, {})
    assert first == backend.produce_window(7, tmp_path, {})
    assert read_record(tmp_path / 'execution.json')['status'] == 'COMPLETE'
    with pytest.raises(ValueError, match='another review or window'):
        backend.produce_window(8, tmp_path, {})
    asset.write_bytes(b'changed published output')
    with pytest.raises(ValueError, match='committed window bytes changed'):
        backend.produce_window(7, tmp_path, {})


def test_previous_forcing_input_decision_is_verified_before_restore(tmp_path):
    backend = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    inputs = tmp_path / 'inputs.json'
    write_record(inputs, {'prepared': {'original': True}})
    decision = tmp_path / 'analysis.json'
    write_record(decision, {'inputs': file_record(inputs)})
    backend.restore_window = lambda *args: pytest.fail('Changed inputs must never reach preparation')
    inputs.write_text('changed')
    with pytest.raises(ValueError, match='committed window bytes changed'):
        backend._restore_previous_generation({'analysis_decision': file_record(decision)})


def test_late_windows_keep_observation_assignment_and_consumption_history(tmp_path):
    from gpuwm.local_da_fetch import assigned_document, _used_cwp_scans, WINDOW_SCHEMA
    from gpuwm.obs.goes_window import ACQUISITION_SCHEMA
    backend = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    backend.plan = {'selected': {'cadence_seconds': 60.}, 'review_sha256': 'original-review'}
    backend._continuous_epoch = datetime(2026, 9, 10, tzinfo=timezone.utc)
    backend.root = tmp_path
    backend.grid = SimpleNamespace(identity_sha256=lambda: 'original-grid')
    backend._schedule(100)
    when = runtime._analysis_time(backend, 100)
    observation = {'valid_time': when.isoformat()}
    assert assigned_document(observation, when, backend.analysis_times, 60.)
    backend._schedule(101)
    assert not assigned_document(observation, when+timedelta(seconds=60), backend.analysis_times, 60.)
    assert len(backend.analysis_times) == 3
    # Satellite consumption remains rooted in the original run, even when
    # windows and forcing generations are separate durable directories.
    previous = tmp_path / 'observations/cycle_099'
    previous.mkdir(parents=True)
    acquisition = previous / 'acquisition.json'
    acquisition.write_text(json.dumps({'schema': ACQUISITION_SCHEMA, 'consumed_scan_ids': ['original-scan']}))
    (previous / 'window.json').write_text(json.dumps(dict(schema=WINDOW_SCHEMA,
        review_sha256='original-review', grid_identity='original-grid',
        analysis_time=(when-timedelta(seconds=60)).isoformat(),
        assets=[dict(kind='cwp-acquisition', path=str(acquisition), sha256=file_record(acquisition)['sha256'])])))
    assert _used_cwp_scans(backend, 100, when, 120.) == {'original-scan'}


def test_continuous_member_failure_retains_started_and_recovery_state(tmp_path, monkeypatch):
    backend = Backend(tmp_path)
    ctl = controller(tmp_path)
    adapter = runtime.ContinuousBackend.__new__(runtime.ContinuousBackend)
    adapter.controller, adapter._active_index = ctl, 0
    def fail(self, **kwargs):
        raise OSError('controlled actual member failure')
    monkeypatch.setattr(runtime.PreparedBackend, 'member_runner', fail)
    backend.preflight = lambda: None
    backend.close = lambda: None
    backend.analyze_window = lambda *args: adapter.member_runner()
    from gpuwm.local_da_controller import WindowFailure
    with pytest.raises(WindowFailure, match='window 0.*actual member failure') as raised:
        ctl.run(backend)
    assert isinstance(raised.value.__cause__, OSError)
    assert raised.value.forecast_started is True
    assert raised.value.details == dict(window=0, stage='forecast', reason='controlled actual member failure')
    assert 'resume its unfinished window' in raised.value.recovery
    status = json.loads((ctl.root / 'status.json').read_text())
    assert status['status'] == 'FAILED' and status['forecast_started'] is True
    assert status['failed_window'] == 0 and status['reason'] == 'controlled actual member failure'
