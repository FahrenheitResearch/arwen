"""Independent processes must not adopt one live forecast's empty output."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from queue import Queue
from threading import Event
from types import SimpleNamespace
from datetime import datetime

import pytest

from gpuwm.stage_reuse import claim_run_output
from gpuwm import stage_reuse
from gpuwm.filesystem_paths import canonical_path, io_path


WORKER = """
import json, pathlib, sys, time
from gpuwm.stage_reuse import claim_run_output
root, ready, stop = map(pathlib.Path, sys.argv[1:4])
token = sys.argv[4] or None
restart = pathlib.Path(sys.argv[5]) if sys.argv[5] else None
if len(sys.argv) > 6:
    barrier = pathlib.Path(sys.argv[6])
    ready.with_suffix('.waiting').touch()
    deadline = time.monotonic() + 20
    while not barrier.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('test launch barrier was not released')
        time.sleep(.01)
claim = claim_run_output(root, resume=restart, **({} if token is None else {'owner_token': token}))
try:
    # The pre-fix API returned only a path. Keep this producer usable for the
    # causal baseline control: it must fail on duplicate ownership, not on API.
    path = getattr(claim, 'path', claim)
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(json.dumps({'path': str(path), 'token': getattr(claim, 'token', None)}))
    temporary.replace(ready)
    deadline = time.monotonic() + 30
    while not stop.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('test owner was not released')
        time.sleep(.01)
finally:
    if hasattr(claim, 'close'):
        claim.close()
"""


@contextmanager
def worker(tmp_path, output, name, *, token=None, restart=None, barrier=None):
    ready, stop = tmp_path / (name + '.json'), tmp_path / (name + '.stop')
    process = subprocess.Popen(
        [sys.executable, '-c', WORKER, str(output), str(ready), str(stop), token or '',
         '' if restart is None else str(restart),
         *([] if barrier is None else [str(barrier)])],
        cwd=Path(stage_reuse.__file__).resolve().parents[1],
        env=dict(os.environ, GPUWM_NO_LOCAL_GPU='1', PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while not ready.is_file():
            if process.poll() is not None:
                raise AssertionError(process.communicate())
            if time.monotonic() > deadline:
                raise TimeoutError('test owner did not claim its output')
            time.sleep(.01)
        yield process, json.loads(ready.read_text())
    finally:
        stop.touch()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=10)
            raise AssertionError(('owned test helper did not exit', stdout, stderr))


def test_live_unrelated_launches_receive_distinct_output_directories(tmp_path):
    output = tmp_path / 'forecast'
    output.mkdir()
    with worker(tmp_path, output, 'first') as (first, a):
        with worker(tmp_path, output, 'second') as (second, b):
            assert first.poll() is second.poll() is None
            assert Path(a['path']) == output.resolve()
            assert Path(b['path']) == output.with_name('forecast-attempt-001').resolve()
            assert a['token'] != b['token']
            # Each ordinary writer can retain the same filename independently.
            for row, value in ((a, b'first run'), (b, b'second run')):
                (Path(row['path']) / 'receipt.json').write_bytes(value)
            assert (output / 'receipt.json').read_bytes() == b'first run'
    assert first.returncode == second.returncode == 0


def test_four_simultaneous_contenders_keep_distinct_live_output_owners(tmp_path):
    output = tmp_path / 'forecast'
    barrier = tmp_path / 'start-all'
    rows = Queue()
    release = Event()

    def contender(index):
        with worker(tmp_path, output, f'contender-{index}', barrier=barrier) as pair:
            rows.put(pair)
            assert release.wait(20), 'test contenders were not released'
        assert pair[0].returncode == 0

    with claim_run_output(output) as original, ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(contender, index) for index in range(4)]
        try:
            deadline = time.monotonic() + 20
            while len(list(tmp_path.glob('contender-*.waiting'))) != 4:
                if time.monotonic() > deadline:
                    raise TimeoutError('all four subprocesses did not reach the launch barrier')
                time.sleep(.01)
            barrier.touch()
            claims = [rows.get(timeout=20) for _ in range(4)]
            assert all(process.poll() is None for process, _ in claims)
            paths = {canonical_path(row['path']) for _, row in claims}
            assert len(paths) == 4 and original.path not in paths
            assert {path.name for path in paths} == {f'forecast-attempt-{i:03d}' for i in range(1, 5)}
        finally:
            barrier.touch()
            release.set()
            for future in futures:
                future.result(timeout=30)


def test_worker_keeps_ownership_when_supervisor_exits(tmp_path):
    output = tmp_path / 'forecast'
    parent = claim_run_output(output)
    with worker(tmp_path, output, 'child', token=parent.token) as (child, row):
        assert Path(row['path']) == parent.path
        parent.close()
        with claim_run_output(output) as contender:
            assert contender.path != output
        assert child.poll() is None
    with claim_run_output(output) as recovered:
        assert recovered.path == output
        assert recovered.token != row['token']
        with pytest.raises(ValueError, match='token'):
            claim_run_output(output, owner_token=row['token'])
    assert child.returncode == 0


def test_completed_token_cannot_reopen_an_old_output(tmp_path):
    output = tmp_path / 'forecast'
    with claim_run_output(output) as parent:
        token = parent.token
    with pytest.raises(ValueError, match='no longer active'):
        claim_run_output(output, owner_token=token)


def test_dead_process_releases_an_empty_precreated_output(tmp_path):
    output = tmp_path / 'forecast'
    output.mkdir()
    with worker(tmp_path, output, 'abrupt') as (process, old):
        process.kill()
        process.wait(timeout=10)
        with claim_run_output(output) as recovered:
            assert recovered.path == output
            assert recovered.token != old['token']


def test_independent_checkpoint_replays_preserve_the_parent_and_each_other(tmp_path):
    output = tmp_path / 'forecast'
    output.mkdir()
    checkpoint = output / 'restart.gpuwmrst'
    checkpoint.write_bytes(b'retained checkpoint')
    receipt = output / 'receipt.json'
    receipt.write_bytes(b'retained publication')
    with worker(tmp_path, output, 'resume-one', restart=checkpoint) as (first, a):
        with worker(tmp_path, output, 'resume-two', restart=checkpoint) as (second, b):
            assert first.poll() is second.poll() is None
            assert {Path(a['path']).name, Path(b['path']).name} == {'segment-001', 'segment-002'}
            assert checkpoint.read_bytes() == b'retained checkpoint'
            assert receipt.read_bytes() == b'retained publication'
    assert first.returncode == second.returncode == 0


def test_protected_inputs_are_checked_before_lock_sidecars_are_created(tmp_path):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    with pytest.raises(ValueError, match='protected input tree'):
        claim_run_output(inputs / 'forecast', protected_roots=(inputs,))
    assert list(inputs.iterdir()) == []


@pytest.mark.skipif(os.name != 'nt', reason='Windows MAX_PATH filesystem boundary')
def test_long_output_sidecars_and_worker_handoff_are_usable(tmp_path):
    parent = tmp_path / ('p' * max(1, 200 - len(str(tmp_path)) - 1))
    parent.mkdir()
    inputs = parent / 'inputs'
    inputs.mkdir()
    output = parent / 'forecast'
    assert len(str(parent / '.arwen-output-owners' / ('x' * 64 + '.lease'))) > 260
    with claim_run_output(output, protected_roots=(inputs,)) as owner:
        with claim_run_output(output, owner_token=owner.token, protected_roots=(inputs,)) as child:
            assert child.path == owner.path == output
        with claim_run_output(output, protected_roots=(inputs,)) as other:
            assert other.path != output


@pytest.mark.skipif(os.name != 'nt', reason='Windows raw/extended aliases')
def test_extended_spelling_cannot_claim_a_live_output_again(tmp_path):
    output = tmp_path / 'forecast'
    with claim_run_output(output) as owner:
        with claim_run_output(io_path(output)) as other:
            assert canonical_path(other.path) != canonical_path(owner.path)
        with claim_run_output(io_path(output), owner_token=owner.token) as child:
            assert child.path == owner.path


@pytest.mark.skipif(os.name != 'nt', reason='Windows raw/extended protected-path aliases')
@pytest.mark.parametrize('extended_input', [False, True])
def test_extended_spelling_cannot_bypass_protected_inputs(tmp_path, extended_input):
    inputs = tmp_path / 'input'
    inputs.mkdir()
    output = inputs / 'forecast'
    protected = io_path(inputs) if extended_input else inputs
    requested = output if extended_input else io_path(output)
    with pytest.raises(ValueError, match='protected input tree'):
        with claim_run_output(requested, protected_roots=(protected,)):
            pass
    assert list(inputs.iterdir()) == []


@pytest.mark.skipif(os.name != 'nt', reason='Windows deep worker output I/O')
@pytest.mark.parametrize('door', ['wrfinput', 'metem'])
def test_deep_output_reaches_worker_preparation_and_publication(tmp_path, monkeypatch, door):
    from gpuwm import go_cli, prepared_domain_tree_forecast, supervisor
    from gpuwm import wrfinput_door, wrfinput_forecast, metem_door, metem_forecast
    from test_metem_forecast import _launcher_stub_run

    output = tmp_path / ('a' * 100) / ('b' * 100) / 'forecast'
    assert len(str(output)) > 260
    inputs = SimpleNamespace(experiment=SimpleNamespace(start_time=datetime(2026, 9, 12)))
    run = (_launcher_stub_run() if door == 'metem' else SimpleNamespace())
    run.substitution_report = SimpleNamespace(substitutions=())
    module = metem_forecast if door == 'metem' else wrfinput_forecast
    reader = metem_door if door == 'metem' else wrfinput_door
    monkeypatch.setattr(reader, 'resolve_metem_run' if door == 'metem' else 'resolve_wrfinput_run',
                        lambda *args, **kwargs: run)
    monkeypatch.setattr(go_cli, 'render_extra_missing', lambda: None)
    monkeypatch.setattr(supervisor, 'select_gpu', lambda *args: pytest.fail('GPU selected'))

    def prepare(run, directory, **kwargs):
        # Ordinary Path I/O deliberately uses the exact spelling the actual
        # worker hands preparation. Claiming a deep path alone is insufficient.
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / 'prepared.txt').write_bytes(b'prepared inputs')
        inputs.prepared_root = directory
        return inputs

    def execute(inputs, *, output_directory, **kwargs):
        evidence = Path(output_directory) / 'evidence'
        evidence.mkdir()
        (evidence / 'run.json').write_bytes(b'completed worker fixture')

    monkeypatch.setattr(module, 'prepare_metem_run' if door == 'metem' else 'prepare_wrf_run', prepare)
    monkeypatch.setattr(module, 'MetemInitialization' if door == 'metem' else 'WrfInitialization', lambda _: None)
    monkeypatch.setattr(prepared_domain_tree_forecast, 'run_prepared_tree', execute)
    launch = module.run_metem_forecast if door == 'metem' else module.run_wrf_forecast
    assert launch(tmp_path / 'input', output, exclusive_gpu=False, render_products='none') == 0
    assert io_path(output / 'input/prepared.txt').read_bytes() == b'prepared inputs'
    assert io_path(output / 'evidence/run.json').read_bytes() == b'completed worker fixture'
