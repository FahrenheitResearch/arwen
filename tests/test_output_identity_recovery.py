"""Receipts keep their original addresses across retries and publication."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_cupy
from gpuwm import go_cli, runplan, run_stamp, runtime, wrfinput_forecast
from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename


def _publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with WrfoutWriter(path, nx=1, ny=1, nz=1, dx=1., dy=1.) as writer:
        writer.write_frame('2026-09-12_01:00:00',
                           {'T': np.full((1, 1, 1), value, np.float32)})


def test_retry_preserves_literal_frame_receipt_and_render_addresses(tmp_path, monkeypatch):
    root = tmp_path/'chain'/'run'
    name = wrfout_filename(datetime(2026, 9, 12, 1), 1)
    frame = root/'wrfout'/name
    _publish(frame, 1.)
    record = runtime._frame_records([frame])[0]
    receipt = root/'evidence'/'run-receipt.json'
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({'frames': [record]}))
    picture = root.parent/'png'/'old.png'
    picture.parent.mkdir(); picture.write_bytes(b'previous picture')
    retained = {p: p.read_bytes() for p in (frame, receipt, picture)}
    observer = SimpleNamespace(events=SimpleNamespace(emit=lambda *a, **k: None),
        enter_stage=lambda *a, **k: None, last_model_seconds=120.)
    selected = runplan._clear_forecast_output(root, observer=observer)
    output = selected if isinstance(selected, Path) else root
    _publish(output/'wrfout'/name, 2.)
    (output/'report.json').write_text(json.dumps({'status': 'PASS'}))
    (output/'progress.json').write_text(json.dumps({'model_elapsed_seconds': 120.}))
    seen = []
    monkeypatch.setattr(go_cli, '_render_stage', lambda plan, **kw: seen.append(plan) or True)
    result = runplan._chain_render(SimpleNamespace(run_options={}),
        forecast_dir=output, run_dir=tmp_path, observer=observer)
    assert all(path.read_bytes() == content for path, content in retained.items())
    assert hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() == record['sha256']
    assert seen[0]['run'] == output and seen[0]['render'] == output.parent/'png'
    assert result['report'] == str(output/'report.json')
    assert result['forecast_root'] == str(output)
    assert result['render_root'] == str(output.parent/'png')


@pytest.mark.parametrize('replacement', [b'new bytes', b'new bytes with a different length'])
def test_hashing_rejects_atomic_replacement_even_when_size_matches(tmp_path, monkeypatch, replacement):
    path = tmp_path/'frame'; path.write_bytes(b'old bytes')
    next_path = tmp_path/'next'; next_path.write_bytes(replacement)
    opening = Path.open

    class Reading:
        def __init__(self, stream):
            self.stream, self.fired = stream, False
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, size=-1):
            value = self.stream.read(size)
            if not self.fired:
                self.fired = True
                os.replace(next_path, path)
            return value

    def open_file(candidate, mode='r', *args, **kwargs):
        stream = opening(candidate, mode, *args, **kwargs)
        return Reading(stream) if candidate == path and mode == 'rb' else stream
    monkeypatch.setattr(Path, 'open', open_file)
    with pytest.raises(RuntimeError, match='changed while'):
        runtime._frame_records([path])


def test_stable_file_record_binds_exact_bytes(tmp_path):
    path = tmp_path/'frame'; path.write_bytes(b'stable')
    assert runtime._frame_records([path]) == [{'path': str(path.resolve()), 'bytes': 6,
        'sha256': hashlib.sha256(b'stable').hexdigest()}]


@pytest.mark.parametrize('reason', ['none', 'missing-renderer', 'no-frames'])
def test_skipped_real_render_never_promotes_an_old_image(tmp_path, monkeypatch, reason):
    older = tmp_path/'run-20260912-010000Z'; older.mkdir()
    newer = tmp_path/'run-20260912-020000Z'; newer.mkdir()
    (older/'old.png').write_bytes(b'previous image')
    run_stamp.record_latest(tmp_path, newer)
    pointer = tmp_path/run_stamp.LATEST_POINTER
    before = pointer.read_bytes()
    publications = []
    publish = run_stamp.record_latest
    monkeypatch.setattr(run_stamp, 'record_latest',
        lambda *a: publications.append(a) or publish(*a))
    plan = {'render': older, 'render_products': 'none', 'run': tmp_path/'forecast'}
    if reason != 'none':
        plan['render_products'] = None
        monkeypatch.setattr(go_cli, 'render_extra_missing',
                            lambda: 'missing' if reason == 'missing-renderer' else None)
        monkeypatch.setattr(go_cli, 'wrfout_frames', lambda plan: [])
        monkeypatch.setattr(go_cli, 'render_command', lambda plan: ['render'])
    assert wrfinput_forecast.draw_door_products(plan, door='test') is False
    assert pointer.read_bytes() == before
    assert publications == []


def test_successful_old_completion_cannot_demote_newer_launch(tmp_path, monkeypatch):
    folders = [tmp_path/f'run-20260912-{hour:02d}0000Z' for hour in (1, 2)]
    for folder in folders:
        folder.mkdir(); (folder/'frame.png').write_bytes(b'image')
    monkeypatch.setattr(go_cli, '_render_stage', lambda *a, **k: True)
    for folder in reversed(folders):
        assert wrfinput_forecast.draw_door_products({'render': folder}, door='test')
    assert run_stamp.latest(tmp_path) == folders[-1]


def test_concurrent_publications_keep_the_newest_launch_and_complete_pointer(tmp_path):
    folders = [tmp_path/f'run-20260912-0100{second:02d}Z' for second in range(12)]
    for folder in folders: folder.mkdir()
    run_stamp.record_latest(tmp_path, folders[-1])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda path: run_stamp.record_latest(tmp_path, path),
                      list(reversed(folders))*3))
    assert (tmp_path/run_stamp.LATEST_POINTER).read_bytes() == (folders[-1].name+'\n').encode()
    assert run_stamp.latest(tmp_path) == folders[-1]


def test_competing_newer_publication_cannot_be_overwritten_after_selection(tmp_path, monkeypatch):
    folders = [tmp_path/f'run-20260912-{hour:02d}0000Z' for hour in (1, 2)]
    for folder in folders:
        folder.mkdir()
    selected, release, newer_done = Event(), Event(), Event()
    create_temporary = run_stamp.tempfile.NamedTemporaryFile

    def delayed_temporary(*args, **kwargs):
        if current_thread().name == 'publication_0':
            selected.set()
            assert release.wait(2)
        return create_temporary(*args, **kwargs)

    monkeypatch.setattr(run_stamp.tempfile, 'NamedTemporaryFile', delayed_temporary)

    def newer_publication():
        try:
            run_stamp.record_latest(tmp_path, folders[-1])
        finally:
            newer_done.set()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='publication') as pool:
        older = pool.submit(run_stamp.record_latest, tmp_path, folders[0])
        assert selected.wait(2)
        newer = pool.submit(newer_publication)
        # A lockless read/compare/replace lets the newer publication finish
        # here and then overwrites it when the older selection is released.
        newer_done.wait(.25)
        release.set()
        older.result()
        newer.result()
    assert run_stamp.latest(tmp_path) == folders[-1]


def test_nested_chain_relays_the_reserved_generation_to_runner_and_early_render(tmp_path, monkeypatch):
    from test_runplan_hrrr_tree import _drive
    original = tmp_path/'run'/'chain'/'run'
    frame = original/'wrfout'/wrfout_filename(datetime(2026,9,12,1), 1)
    _publish(frame, 1.)
    before = frame.read_bytes()
    plans = []
    build_plan = runplan._chain_render_plan
    monkeypatch.setattr(runplan, '_chain_render_plan',
        lambda *a, **k: plans.append(build_plan(*a, **k)) or plans[-1])
    _, captured, _, _ = _drive(tmp_path, monkeypatch)
    argv = captured['argv']
    selected = Path(argv[argv.index('--outdir')+1])
    assert selected != original
    assert frame.read_bytes() == before
    assert plans[0]['run'] == selected
    assert plans[0]['render'] == selected.parent/'png'


@requires_cupy
def test_table_chain_relays_the_reserved_generation_to_runner_and_final_render(tmp_path, monkeypatch):
    from test_runplan import _executed_staged_chain
    original = tmp_path/'run'/'chain'/'run'
    frame = original/'wrfout'/wrfout_filename(datetime(2026,9,12,1), 1)
    _publish(frame, 1.)
    before = frame.read_bytes()
    staged, _, _, _ = _executed_staged_chain(tmp_path, monkeypatch)
    command = next(row[1] for row in staged if row[0] == 'forecast')
    selected = Path(command[command.index('--outdir')+1])
    rendered = next(row[1] for row in staged if row[0] == 'render')
    assert selected != original
    assert frame.read_bytes() == before
    assert rendered['run'] == selected and rendered['render'] == selected.parent/'png'
