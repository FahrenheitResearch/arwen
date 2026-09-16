"""A status read observes a matching OS owner, independent of heartbeat age."""
from dataclasses import replace
import json
import os
import subprocess
import sys
import time

import pytest

from gpuwm.local_da import build_plan, publish
from gpuwm.local_da_controller import status_for_plan, STATUS_SCHEMA
from test_local_da_plan import request, availability, price


def saved(tmp_path):
    plan = build_plan(replace(request(), continuous_windows=3), availability=availability, price=price)
    root = tmp_path / 'case'
    publish(plan, root)
    return root / 'local-da.json', plan


def test_new_saved_continuous_review_is_not_started_and_reads_cheaply(tmp_path, monkeypatch):
    path, plan = saved(tmp_path)
    monkeypatch.setattr('gpuwm.output_identity.file_record', lambda *args: pytest.fail('status polling must not hash products'))
    value = status_for_plan(path)
    assert value['status'] == 'NOT_STARTED' and value['controller_alive'] is False
    assert value['windows'] == 3 and value['remaining_windows'] == 3 and value['completed_windows'] == 0
    assert value['stop_requested'] is False
    assert value['review_sha256'] == plan['review_sha256'] and value['plan_path'] == str(path.resolve())
    assert plan['continuous']['windows'] == 3


def test_stop_request_on_a_saved_plan_is_durable_and_visible(tmp_path):
    from gpuwm.local_da_controller import stop_plan
    path, plan = saved(tmp_path)
    value = stop_plan(path)
    assert value['stop_requested'] is True and value['status'] == 'NOT_STARTED'
    requests = list((path.parent / 'continuous' / 'stop').glob('*.json'))
    assert len(requests) == 1 and json.loads(requests[0].read_text())['review_sha256'] == plan['review_sha256']
    assert status_for_plan(path)['stop_requested'] is True


def test_actual_held_owner_is_live_and_becomes_interrupted_after_exit(tmp_path):
    path, plan = saved(tmp_path)
    root = path.parent / 'continuous'
    root.mkdir()
    program = r'''
import json,os,sys,time
from pathlib import Path
from gpuwm.supervisor import GPUFileLock
root=Path(sys.argv[1]); review=sys.argv[2]
with GPUFileLock(review,path=root/'.controller.lock',run_id='unique-launch'):
    (root/'status.json').write_text(json.dumps(dict(schema='arwen.local-da-continuous-status.v1',
        review_sha256=review,status='ANALYZING',updated_utc='2000-01-01T00:00:00Z',
        controller_owner={'pid':os.getpid(),'nonce':'unique-launch'})))
    print('ready',flush=True)
    sys.stdin.readline()
'''
    process = subprocess.Popen([sys.executable, '-c', program, str(root), plan['review_sha256']],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        status = status_for_plan(path)
        assert status['controller_alive'] and status['status'] == 'ANALYZING'
        document = json.loads((root / 'status.json').read_text())
        document['controller_owner']['nonce'] = 'unrelated-launch'
        (root / 'status.json').write_text(json.dumps(document))
        assert not status_for_plan(path)['controller_alive']
        document['controller_owner']['nonce'] = 'unique-launch'
        (root / 'status.json').write_text(json.dumps(document))
    finally:
        process.communicate('\n', timeout=10)
    assert process.returncode == 0
    status = status_for_plan(path)
    assert not status['controller_alive'] and status['status'] == 'INTERRUPTED'


def test_stop_during_preflight_is_not_acknowledged_by_earlier_resume(tmp_path):
    from test_local_da_controller import Backend, controller
    backend = Backend(tmp_path)
    ctl = controller(tmp_path)
    backend.preflight = ctl.request_stop
    backend.close = lambda: None
    result = ctl.run(backend)
    assert result['status'] == 'STOPPED' and backend.calls == []


def test_corrupt_committed_product_blocks_next_analysis(tmp_path):
    from test_local_da_controller import Backend, controller
    backend = Backend(tmp_path)
    ctl = controller(tmp_path)
    ctl.step(backend)
    (ctl.root / 'window_000000/execution.json').write_text('changed')
    with pytest.raises(ValueError, match='committed window bytes changed'):
        controller(tmp_path).step(backend)
    assert backend.calls == [0]


@pytest.mark.parametrize('failing', [False, True])
def test_the_document_left_behind_by_an_exited_controller_is_not_claimed(tmp_path, failing):
    from test_local_da_controller import Backend, controller
    from gpuwm.local_da_controller import WindowFailure
    backend = Backend(tmp_path)
    backend.preflight = lambda: None
    backend.close = lambda: None
    backend.product_failure = failing
    ctl = controller(tmp_path, windows=1)
    if failing:
        with pytest.raises(WindowFailure, match='controlled product interruption'):
            ctl.run(backend, sleeper=lambda seconds: None)
    else:
        assert ctl.run(backend, sleeper=lambda seconds: None)['status'] == 'COMPLETE'
    document = json.loads((ctl.root / 'status.json').read_text())
    assert document['status'] == ('FAILED' if failing else 'COMPLETE')
    assert document['controller_alive'] is False
    assert document['controller_owner']['pid'] == os.getpid()
