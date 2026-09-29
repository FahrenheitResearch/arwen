"""Regression coverage for forecast supervision and render ownership."""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from gpuwm import forecast_supervisor as fs, go_cli, supervisor


@pytest.fixture
def render_worker(tmp_path, monkeypatch):
    source = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta
from gpuwm import first_products, go_cli, live_products, render_georef

def main(argv, *, observer):
    out = Path(argv[argv.index('--outdir') + 1])
    out.mkdir(parents=True)
    pictures = out / 'png'
    stop = '--stop-probe' in argv
    def render(command):
        time.sleep(1)
        target = Path(command[command.index('--out') + 1])
        frame = Path(command[command.index('--series') - 1])
        key = 'd01/field/20260928/' + frame.name + '.png'
        (target / key).parent.mkdir(parents=True, exist_ok=True)
        (target / key).write_bytes(b'fixture')
        (target / render_georef.GEOREF_FILENAME).write_text(json.dumps({
            'schema': render_georef.GEOREF_SCHEMA, 'generated_utc': '2026-09-28T00:00:00Z',
            'panels': {key: {'projection': 'lambert'}}, 'without_georeference': []}))
        return subprocess.CompletedProcess(command, 0, '', '')
    if stop:
        child = out / 'renderer.py'
        child.write_text("import os,sys,time\nfrom pathlib import Path\n"
                         "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
                         "Path(sys.argv[2]).write_text('unfinished')\n"
                         "time.sleep(25)\nPath(sys.argv[3]).write_text('orphan')\n")
        go_cli.render_command = lambda plan, frames=None, **kw: [
            sys.executable, str(child), str(out / 'renderer.pid'),
            str(plan['render'] / 'partial.png'), str(out / 'orphan.txt')]
    renders = live_products.LandingRenders(
        {'run': out, 'render': pictures, 'render_products': 'all'},
        report=lambda *a, **k: None, report_live=lambda *a, **k: None,
        warn=lambda *a, **k: None, runner=None if stop else render)
    start = datetime(2026, 9, 28)
    # Two live waves after the exclusive early render exceed the step bound
    # at every worker width, including a host that draws three frames at once.
    frame_count = 1 if stop else 1 + 2 * getattr(renders.live, '_concurrency', 1)
    for index in range(frame_count):
        valid = start + timedelta(hours=index)
        frame = out / ('wrfout_d01_' + valid.strftime('%Y-%m-%d_%H_%M_%S'))
        frame.write_bytes(b'frame fixture')
        renders.frame_committed(domain=1, valid_time=valid, path=frame)
    if stop:
        deadline = time.monotonic() + 10
        while not (out / 'renderer.pid').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (out / 'renderer.pid').exists()
    observer(model_elapsed_seconds=1., outer_step=1,
             last_durable_wrfout=None, last_checkpoint=None)
    try:
        if stop:
            print('worker diagnostic before stall', file=sys.stderr, flush=True)
            observer.finalizing('stalled-output')
            time.sleep(25)
        else:
            observer.finalizing('finish-first-products')
            joined = time.monotonic()
            renders.wait()
            (out / 'drain.json').write_text(json.dumps({'seconds': time.monotonic() - joined}))
    except KeyboardInterrupt:
        renders.halt()
        (out / 'interrupted.txt').write_text('renders halted')
        raise
    return 0
'''
    (tmp_path / 'render_probe.py').write_text(source, encoding='utf-8')
    env = go_cli._stage_env()
    env['PYTHONPATH'] = os.pathsep.join([str(tmp_path), str(Path(go_cli.__file__).resolve().parents[1])])
    monkeypatch.setattr(go_cli, '_stage_env', lambda: env)
    return [sys.executable, '-m', 'render_probe', '--outdir', str(tmp_path / 'run')]


def test_real_landing_drain_outlasts_step_bound(render_worker, monkeypatch):
    # The real LandingRenders drain at a smaller scale: a finalization
    # bound of 0.5 s, a landing wait of 30 s and 1 s per stand-in frame,
    # so draining three frames outlasts the bound alone several times over.
    from gpuwm import live_products

    bound = .5
    monkeypatch.setattr(supervisor, 'finalization_stale_threshold_seconds',
                        lambda *a: bound)
    monkeypatch.setattr(live_products, 'landing_render_wait_seconds', lambda: 30.)
    ends = []
    go_cli._run_stage('forecast', render_worker, explain=False,
                      observer=SimpleNamespace(stage_end=lambda **f: ends.append(f)))
    assert [(end['exit_code'], end['ok']) for end in ends] == [(0, True)]
    out = Path(render_worker[-1])
    assert json.loads((out / 'drain.json').read_text())['seconds'] > 4 * bound
    assert supervisor.read_heartbeat(out / 'run-progress.json').status == 'complete'
    assert len(list((out / 'png').rglob('*.png'))) == len(list(out.glob('wrfout_*')))


@pytest.mark.skipif(os.name != 'posix', reason='POSIX interrupt and renderer ownership')
def test_watchdog_stop_reaps_renderer_and_sweeps_scratch(render_worker, monkeypatch):
    monkeypatch.setattr(supervisor, 'finalization_stale_threshold_seconds', lambda *a: .3)
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage('forecast', [*render_worker, '--stop-probe'], explain=False)
    out = Path(render_worker[-1])
    pid = int((out / 'renderer.pid').read_text())
    try:
        assert caught.value.code == 124
        assert (out / 'interrupted.txt').exists()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not list((out / 'png').glob('.*scratch*'))
        assert not (out / 'orphan.txt').exists()
    finally:
        # The red run must not leave its deliberately orphaned probe alive.
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_watchdog_preserves_worker_stderr(tmp_path, monkeypatch):
    (tmp_path / 'stderr_probe.py').write_text('''
import sys,time
from pathlib import Path
def main(argv, *, observer):
    Path(argv[argv.index('--outdir')+1]).mkdir()
    print('worker diagnostic before stall', file=sys.stderr, flush=True)
    observer(model_elapsed_seconds=1., outer_step=1, last_durable_wrfout=None, last_checkpoint=None)
    observer.finalizing('stalled-output')
    try:
        time.sleep(20)
    except KeyboardInterrupt:
        return 1
''')
    env = go_cli._stage_env()
    env['PYTHONPATH'] = os.pathsep.join([str(tmp_path), str(Path(go_cli.__file__).resolve().parents[1])])
    monkeypatch.setattr(go_cli, '_stage_env', lambda: env)
    monkeypatch.setattr(supervisor, 'finalization_stale_threshold_seconds', lambda *a: .3)
    events = []
    with pytest.raises(go_cli.GoStageFailed) as caught:
        go_cli._run_stage('forecast', [sys.executable, '-m', 'stderr_probe', '--outdir', str(tmp_path/'run')],
                          explain=False, observer=SimpleNamespace(
                              stage_failed=lambda **fields: events.append(fields)))
    assert 'worker diagnostic before stall' in caught.value.diagnostic
    assert 'forecast stalled' in caught.value.diagnostic.splitlines()[-1]
    assert any('worker diagnostic before stall' in row.get('diagnostic', '') for row in events)


def _watch(tmp_path):
    return fs.ForecastWatchdog([sys.executable, '-m', 'probe', '--outdir', str(tmp_path)])


def test_a_restart_s_first_step_is_preparation_not_integration(tmp_path, monkeypatch):
    # A go --restart enters at its checkpoint's step, not step 0, and its
    # first step still compiles every kernel.
    now = [0.]
    monkeypatch.setattr(fs.time, 'monotonic', lambda: now[0])
    watch = _watch(tmp_path)
    beat = fs.ForecastHeartbeat(watch.path, run_id=watch.run_id,
                                config_sha256=watch.digest, started_at_utc=watch.started_at)
    beat(model_elapsed_seconds=300., outer_step=5, last_durable_wrfout=None, last_checkpoint=None)
    record = supervisor.read_heartbeat(watch.path)
    assert (record.status, record.outer_step) == ('preparing:first-step', 5)
    assert watch.check(os.getpid()) is None
    now[0] += 7200.
    assert watch.check(os.getpid()) is None
    beat(model_elapsed_seconds=360., outer_step=6, last_durable_wrfout=None, last_checkpoint=None)
    assert supervisor.read_heartbeat(watch.path).status == 'integrating'
    assert watch.check(os.getpid()) is None
    now[0] += 121.
    assert 'integrating' in watch.check(os.getpid())


def test_preparation_and_cold_step_have_no_default_deadline(tmp_path, monkeypatch):
    now = [0.]
    monkeypatch.setattr(fs.time, 'monotonic', lambda: now[0])
    watch = _watch(tmp_path)
    assert watch.check(os.getpid()) is None
    now[0] = 7200.
    assert watch.check(os.getpid()) is None
    beat = fs.ForecastHeartbeat(watch.path, run_id=watch.run_id,
                                config_sha256=watch.digest, started_at_utc=watch.started_at)
    beat(model_elapsed_seconds=0., outer_step=0, last_durable_wrfout=None, last_checkpoint=None)
    assert supervisor.read_heartbeat(watch.path).status == 'preparing:first-step'
    assert watch.check(os.getpid()) is None
    now[0] += 7200.
    assert watch.check(os.getpid()) is None
    beat(model_elapsed_seconds=60., outer_step=1, last_durable_wrfout=None, last_checkpoint=None)
    assert watch.check(os.getpid()) is None
    now[0] += 121.
    assert 'integrating' in watch.check(os.getpid())


def test_step_history_excludes_preparation(tmp_path):
    watch = _watch(tmp_path)
    beat = supervisor.RuntimeHeartbeat(watch.path, run_id=watch.run_id,
                                      config_sha256=watch.digest, started_at_utc=watch.started_at)
    beat.starting()
    first = supervisor.read_heartbeat(watch.path)
    first = dataclasses.replace(first, updated_at_utc='2026-09-28T00:00:00+00:00')
    supervisor.write_heartbeat(watch.path, first)
    assert watch.check(os.getpid()) is None
    second = dataclasses.replace(first, status='integrating', outer_step=1,
                                 model_elapsed_seconds=60., updated_at_utc='2026-09-28T00:10:00+00:00')
    supervisor.write_heartbeat(watch.path, second)
    assert watch.check(os.getpid()) is None
    assert watch.history.p99 == 0
    supervisor.write_heartbeat(watch.path, dataclasses.replace(second, outer_step=2,
        model_elapsed_seconds=120., updated_at_utc='2026-09-28T00:10:02+00:00'))
    assert watch.check(os.getpid()) is None
    assert watch.history.p99 == 2


def test_preprocessing_package_excludes_forecast_supervisor():
    from tools.build_rw_wps_release import _TOP_LEVEL_EXCLUDES
    assert 'forecast_supervisor.py' in _TOP_LEVEL_EXCLUDES


def test_halted_landing_queue_sweeps_leftover_live_scratch(tmp_path):
    from gpuwm.live_products import LandingRenders
    scratch = tmp_path / 'png/.live-products-scratch'
    scratch.mkdir(parents=True)
    (scratch / 'partial.png').write_bytes(b'fixture')
    retained = tmp_path / 'png/existing.png'
    retained.write_bytes(b'existing content')
    renders = LandingRenders({'run': tmp_path, 'render': tmp_path / 'png'},
        report=lambda *a: None, report_live=lambda *a: None, warn=lambda *a, **k: None)
    assert renders.halt()['ended']
    assert not scratch.exists()
    assert retained.read_bytes() == b'existing content'


@pytest.mark.parametrize('marker,route', [
    ('evidence/progress.json', 'prepared domain-tree forecast'),
    ('progress.json', 'prepared single-domain forecast')])
def test_prepared_progress_identifies_stopped_route(tmp_path, marker, route):
    from gpuwm import report_bundle
    path = tmp_path / marker
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status":"RUNNING"}')
    (tmp_path / 'run-progress.json').write_text('{"status":"failed"}')
    report = report_bundle.build_report(tmp_path, environ={})
    assert report.manifest['route_detected'] == route
