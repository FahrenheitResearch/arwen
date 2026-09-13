"""Cost projections preserve a requested finite run through review and launch."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm.local_da import Card, Request, PlanError, build_plan, derive_rung, publish
from gpuwm.local_da_runtime import PreparedBackend, launch, read_plan
from test_local_da_plan import EPOCH, availability, price, request


def pessimistic(req, rung, exp, streams):
    return dict(price(req, rung, exp, streams), forecast_peak_bytes=1 << 50,
        analysis_peak_bytes=1 << 49, host_peak_bytes=1 << 51,
        cycle_seconds=1.e6, preparation_seconds=2.e6, forecast_seconds=3.e6)


def test_positive_pessimistic_prices_preserve_exact_requested_plan_and_publication(tmp_path):
    req = Request(epoch=EPOCH, region=(-98., 35., -97.5, 35.5), scale=3,
        cadence_seconds=467., forecast_seconds=600., budget_seconds=1.,
        card=Card(vram_gib=2., free_gib=.1, host_gib=1.e-6))
    expected = derive_rung(req, req.scale)
    plan = build_plan(req, availability=availability, price=pessimistic)
    assert plan['status'] == 'ready' and not plan['changed_scale']
    assert plan['selected'] == expected
    assert plan['selected']['cycles'] == 3 and plan['selected']['members'] == 4
    assert plan['selected']['cadence_seconds'] == 467.
    raw = tomllib.loads(plan['configuration']['experiment'])
    assert raw['experiment']['run_seconds'] == 3 * 467 + 600
    assert raw['experiment']['restart_interval_s'] == 467.
    assert plan['clock']['cycle_ticks'] == 467000
    assert not plan['alternatives'][-1]['fits']
    assert plan['memory']['policy'] == plan['wall']['policy'] == 'advisory'
    assert plan['cadence_overrun']['policy'] == 'queue'
    assert plan['cadence_overrun']['outcome'] == 'queued'
    assert plan['cadence_overrun']['cost_basis'] == 'estimated'
    assert any('requested settings are retained' in line for line in plan['warnings'])
    output = publish(plan, tmp_path / 'case')
    assert read_plan(output['plan_path'])['selected'] == expected


def test_positive_budget_changes_do_not_change_finite_configuration():
    req = request(scale=3)
    generous = build_plan(replace(req, budget_seconds=1.e9), availability=availability, price=price)
    tight = build_plan(replace(req, budget_seconds=.01), availability=availability, price=price)
    assert tight['selected'] == generous['selected']
    assert tight['configuration'] == generous['configuration']
    assert tight['wall']['budget_seconds'] == .01
    assert 'not an enforced deadline' in tight['wall']['budget_semantics']
    assert generous['cadence_overrun']['outcome'] == 'clear'


@pytest.mark.parametrize('cadence', [1., 467., 901.])
def test_supported_explicit_cadence_is_exact(cadence):
    plan = build_plan(request(cadence_seconds=cadence), availability=availability, price=price)
    assert plan['selected']['cadence_seconds'] == cadence
    assert plan['clock']['cycle_ticks'] == cadence * 1000


@pytest.mark.parametrize('cadence', [.5, 467.5])
def test_unrepresentable_cadence_is_named_instead_of_silently_floored(cadence):
    with pytest.raises(PlanError, match='whole-second.*timestamps'):
        build_plan(request(cadence_seconds=cadence), availability=availability, price=price)


def test_cli_ready_plan_does_not_require_an_override(tmp_path, monkeypatch, capsys):
    import gpuwm.local_da as module
    parser = argparse.ArgumentParser()
    module.register_cli(parser.add_subparsers(dest='command'))
    original = module.build_plan
    monkeypatch.setattr(module, 'build_plan', lambda req: original(req, availability=availability, price=pessimistic))
    args = parser.parse_args(['local-da', '--point', '40,-100', '--epoch', EPOCH,
        '--scale', '3', '--cadence-seconds', '467', '--forecast-seconds', '600',
        '--vram-gib', '2', '--budget-seconds', '1', '--out', str(tmp_path / 'case')])
    assert module.main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document['status'] == 'ready' and document['selected']['scale'] == 3
    assert document['selected']['cycles'] == 3
    assert Path(document['publication']['plan_path']).is_file()


def test_live_estimate_above_free_memory_is_advisory_before_preparation(tmp_path, monkeypatch):
    from gpuwm import capabilities, go_cli, geog_assets, experiment
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *args: None)
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    exp = SimpleNamespace(domains=[SimpleNamespace(run=SimpleNamespace(mp_physics=6))])
    monkeypatch.setattr(experiment, 'load_experiment', lambda *args: exp)
    monkeypatch.setattr(experiment, 'refuse_unrouted_spectral_numerics', lambda *args: None)
    monkeypatch.setattr(go_cli, 'plan_from_config', lambda *args, **kwargs: {})
    monkeypatch.setattr(go_cli, 'resolve_bridge', lambda: tmp_path)
    monkeypatch.setattr(geog_assets, 'default_geog_root', lambda: tmp_path)
    monkeypatch.setattr(go_cli, 'geography_refusal', lambda *args: None)
    devices = []
    monkeypatch.setattr(go_cli, '_require_forecast_device', lambda: devices.append(True))
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(memGetInfo=lambda: (1, 1024)))))
    backend = PreparedBackend({'analysis_times': [], 'memory': {'peak_bytes': 1 << 50}, 'observations': []}, tmp_path)
    backend.preflight()
    assert devices == [True]
    assert len(backend.warnings) == 1 and 'requested settings are retained' in backend.warnings[0]
    assert not list(tmp_path.iterdir())


def test_actual_allocation_failure_preserves_checkpoint_and_cli_reports_failure(tmp_path, monkeypatch, capsys):
    import gpuwm.local_da as module
    from test_local_da_runtime import saved, Backend
    path, _ = saved(tmp_path)
    sentinel = path.parent / 'prior-analysis.npz'
    sentinel.write_bytes(b'committed checkpoint bytes')
    before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    class Exhausted(Backend):
        def prepare(self):
            raise MemoryError('allocator could not reserve 4096 bytes')
    original = launch
    monkeypatch.setattr('gpuwm.local_da_runtime.launch', lambda path: original(path, backend=Exhausted()))
    parser = argparse.ArgumentParser()
    module.register_cli(parser.add_subparsers(dest='command'))
    assert module.main(parser.parse_args(['local-da', '--launch', str(path)])) == 1
    document = json.loads(capsys.readouterr().out)
    assert 'allocator could not reserve 4096 bytes' in document['error']
    report = json.loads((path.parent / 'execution.json').read_text())
    assert report['status'] == 'FAILED' and not report['forecast_started']
    assert 'retained' in report['recovery'] and 'resume' in report['recovery']
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before


def test_measured_lag_is_recorded_without_shortening_remaining_cycles(tmp_path, monkeypatch):
    from test_local_da_runtime import saved, Backend
    import gpuwm.local_da_runtime as runtime
    path, plan = saved(tmp_path, scale=2)
    clock = [1000.]
    monkeypatch.setattr(runtime, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    snapshots = []
    atomic = runtime._atomic
    def observe(path, value):
        if 'cadence_progress' in value:
            snapshots.append(json.loads(json.dumps(value['cadence_progress'])))
        atomic(path, value)
    monkeypatch.setattr(runtime, '_atomic', observe)
    class Slow(Backend):
        def assimilate(self, cycle_index, members):
            result = super().assimilate(cycle_index, members)
            clock[0] += 1200.
            return result
    result = launch(path, backend=Slow())
    assert result['status'] == 'COMPLETE'
    progress = result['cadence_progress']
    assert progress['measured_cycles_this_launch'] == plan['selected']['cycles'] == 2
    assert progress['cycle_wall_seconds'] == 1200.
    assert progress['lag_seconds'] == 600.
    projected = next(row['projection'] for row in snapshots if row['projection'])
    assert projected['policy'] == 'queue' and projected['cost_basis'] == 'measured'
    assert projected['projected_backlog_seconds'] == 300.
