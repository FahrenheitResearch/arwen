"""Controller recovery with actual cycle and publication owners, CPU forecasts."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from gpuwm.ensemble import cycle
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.local_da_controller import Controller, ForcingUnavailable
from gpuwm.output_identity import file_record
from test_continuous_cycle_window import clocked_runner
from test_ensemble_engine import _contract_names, _write_overlay


class Backend:
    def __init__(self, tmp_path):
        self.cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=1, perturbation='none'))
        self.calls = []
        self.prepares = []
        self.missing = False
        self.product_failure = False
        self.stop = None

    def prepare_window(self, index, when, forecast, directory, *, prior=None):
        if self.missing:
            raise ForcingUnavailable('Required source frame is absent; retry after its arrival')
        path = directory / 'forcing'
        path.write_text(str(index // 2))
        self.prepares.append(index)
        return dict(assets=[file_record(path)], generation=index // 2)

    def restore_window(self, prepared, directory):
        pass

    def assimilate(self, index, states):
        self.calls.append(index)
        return {i: {_contract_names()[0]: np.full((2, 3), .25, np.float32)} for i in states}, {'window': index}

    def analyze_window(self, index, directory, prior, inputs):
        restarts = None if prior is None else {i: row['path'] for i, row in enumerate(prior['analysis'])}
        result = cycle.run_cycles(self.cfg, directory / 'cycles', n_cycles=1, first_cycle=index,
            cycle_seconds=60., initial_restarts=restarts, input_binding=inputs,
            runner=clocked_runner, assimilate=self.assimilate)
        roster = cycle.read_analysis_roster(cycle.cycle_root(directory / 'cycles', index), n_members=1)
        return dict(analysis=[file_record(path) for _, path in sorted(roster.items())],
                    manifest=file_record(result.manifest_path))

    def produce_window(self, index, directory, decision):
        if self.stop:
            self.stop()
        if self.product_failure:
            raise OSError('controlled product interruption')
        from gpuwm.ensemble.analysis_commit import write_record
        path = directory / 'execution.json'
        write_record(path, dict(schema='arwen.local-da-execution.v1', status='COMPLETE', window=index))
        artifact = file_record(path)
        return dict(status='complete', index=index, artifacts=[artifact],
                    execution_path=str(path), execution_sha256=artifact['sha256'])


def controller(tmp_path, *, now=None, windows=1000, **kwargs):
    return Controller(tmp_path / 'continuous', binding={'review': 'fixed'},
        epoch='2026-09-10T00:00:00+00:00', cadence_seconds=60., forecast_seconds=60.,
        members=1, windows=windows, now=now or (lambda: datetime(2026, 9, 11, tzinfo=timezone.utc)), **kwargs)


def test_many_windows_restore_each_analysis_without_resetting_clock(tmp_path):
    backend = Backend(tmp_path)
    for index in range(40):
        # A new controller object simulates process re-entry at every boundary.
        result = controller(tmp_path).step(backend)
        assert result['completed_windows'] == index + 1
    assert backend.calls == list(range(40))
    assert backend.prepares == list(range(40))
    last = json.loads((tmp_path / 'continuous/window_000039/complete.json').read_text())
    assert cycle._restart_elapsed_seconds(last['analysis'][0]['path']) == 2400


def test_product_interruption_does_not_call_assimilation_again(tmp_path):
    backend = Backend(tmp_path)
    backend.product_failure = True
    with pytest.raises(OSError, match='product interruption'):
        controller(tmp_path).step(backend)
    backend.product_failure = False
    result = controller(tmp_path).step(backend)
    assert result['completed_windows'] == 1
    assert backend.calls == [0] and backend.prepares == [0]


def test_lost_head_write_reuses_completed_window(tmp_path, monkeypatch):
    from gpuwm import local_da_controller as owner
    backend = Backend(tmp_path)
    original = owner.write_json_atomically
    def fail(path, payload):
        if path.name == 'head.json':
            raise OSError('head interrupted')
        return original(path, payload)
    monkeypatch.setattr(owner, 'write_json_atomically', fail)
    with pytest.raises(OSError, match='head interrupted'):
        controller(tmp_path).step(backend)
    monkeypatch.setattr(owner, 'write_json_atomically', original)
    assert controller(tmp_path).step(backend)['completed_windows'] == 1
    assert backend.calls == [0]


def test_missing_forcing_retains_prior_state_and_next_window_index(tmp_path):
    backend = Backend(tmp_path)
    # Half a minute after window 1's analysis time, inside the forcing wait.
    ctl = controller(tmp_path, now=lambda: datetime(2026, 9, 10, 0, 2, 30, tzinfo=timezone.utc))
    ctl.step(backend)
    before = (ctl.root / 'head.json').read_bytes()
    backend.missing = True
    result = ctl.step(backend)
    assert result['status'] == 'WAITING_FORCING' and result['completed_windows'] == 1
    assert (ctl.root / 'head.json').read_bytes() == before and backend.calls == [0]
    backend.missing = False
    assert ctl.step(backend)['completed_windows'] == 2


def test_graceful_stop_during_products_commits_then_resumes_next_window(tmp_path):
    backend = Backend(tmp_path)
    ctl = controller(tmp_path)
    backend.stop = ctl.request_stop
    assert ctl.step(backend)['status'] == 'STOPPED'
    backend.stop = None
    assert controller(tmp_path).step(backend)['status'] == 'STOPPED'
    assert backend.calls == [0]
    result = controller(tmp_path).step(backend, resume=True)
    assert result['status'] == 'READY' and backend.calls == [0, 1]


def test_actual_utc_wait_does_not_depend_on_predicted_cost(tmp_path):
    backend = Backend(tmp_path)
    now = datetime(2026, 9, 10, tzinfo=timezone.utc)
    ctl = controller(tmp_path, now=lambda: now)
    assert ctl.step(backend)['status'] == 'WAITING_TIME'
    now += timedelta(seconds=90)
    result = ctl.step(backend)
    assert result['lag_seconds'] == 30
    assert backend.calls == [0]


def test_corrupt_committed_checkpoint_is_not_silently_recomputed(tmp_path):
    backend = Backend(tmp_path)
    ctl = controller(tmp_path)
    result = ctl.step(backend)
    record = json.loads(open(result['latest']['path']).read())
    with open(record['analysis'][0]['path'], 'ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='committed window bytes changed'):
        ctl.step(backend)
    assert backend.calls == [0]


@pytest.mark.parametrize('value', [True, 0, -1, float('inf'), .5, '60'])
def test_invalid_clock_cannot_publish_a_controller(tmp_path, value):
    with pytest.raises(ValueError):
        Controller(tmp_path, binding={}, epoch='2026-09-10T00:00:00Z',
                   cadence_seconds=value, forecast_seconds=60, members=1, windows=1)


@pytest.mark.parametrize('value', [0, -1, True, 2., '3'])
def test_a_continuous_run_needs_a_positive_whole_window_count(tmp_path, value):
    with pytest.raises(ValueError, match='positive whole number of windows'):
        Controller(tmp_path, binding={}, epoch='2026-09-10T00:00:00Z',
                   cadence_seconds=60, forecast_seconds=60, members=1, windows=value)


def test_bounded_window_count_completes_and_stays_complete(tmp_path):
    backend = Backend(tmp_path)
    ctl = controller(tmp_path, windows=3)
    statuses = [ctl.step(backend)['status'] for _ in range(3)]
    assert statuses == ['READY', 'READY', 'COMPLETE']
    again = controller(tmp_path, windows=3).step(backend)
    assert again['status'] == 'COMPLETE' and again['completed_windows'] == 3 and again['remaining_windows'] == 0
    assert backend.calls == [0, 1, 2]
    document = json.loads((tmp_path / 'continuous/status.json').read_text())
    assert document['windows'] == 3 and document['status'] == 'COMPLETE'


def test_run_loop_stops_at_complete_and_reports_it(tmp_path):
    backend = Backend(tmp_path)
    backend.preflight = lambda: None
    backend.close = lambda: None
    result = controller(tmp_path, windows=2).run(backend, sleeper=lambda seconds: pytest.fail('nothing to wait for'))
    assert result['status'] == 'COMPLETE' and result['completed_windows'] == 2
    assert result['controller_alive'] is False and backend.calls == [0, 1]


def test_forcing_unpublished_past_one_interval_fails_naming_the_window(tmp_path):
    backend = Backend(tmp_path)
    now = datetime(2026, 9, 10, 0, 5, tzinfo=timezone.utc)
    ctl = controller(tmp_path, now=lambda: now, forcing_wait_seconds=300.)
    ctl.step(backend)
    backend.missing = True
    waiting = ctl.step(backend)
    assert waiting['status'] == 'WAITING_FORCING' and waiting['active_window'] == 1
    assert waiting['lag_seconds'] == 180. and waiting['wait_remaining_seconds'] == 120.
    now += timedelta(seconds=200)
    backend.preflight = lambda: None
    backend.close = lambda: None
    from gpuwm.local_da_controller import WindowFailure
    with pytest.raises(WindowFailure, match='window 1 during preparation.*still unpublished') as raised:
        ctl.run(backend, sleeper=lambda seconds: None)
    assert raised.value.details['window'] == 1
    status = json.loads((ctl.root / 'status.json').read_text())
    assert status['status'] == 'FAILED' and status['failed_window'] == 1
    assert backend.calls == [0]
