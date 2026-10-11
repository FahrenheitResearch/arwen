"""The CPU scheduler runs the real native CLI and keeps telemetry out of scores."""
import importlib.util
from pathlib import Path
import json
import os
import pytest


def driver():
    path=Path(__file__).parents[1]/'tools/scoreboard/nightly_job.py'
    spec=importlib.util.spec_from_file_location('nightly_job',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_native_refusal_is_receipted_and_request_is_removed(tmp_path,capsys):
    binary=os.environ.get('GPUWM_RW_SCOREBOARD')
    if not binary:
        pytest.skip('GPUWM_RW_SCOREBOARD must name the newly built native executable')
    catalog=tmp_path/'catalog.json'
    catalog.write_text(json.dumps({'schema':'woof.scoreboard.nightly-catalog.v1','complete':False,'window_start':'2026-09-01T00:00:00Z','window_end':'2026-10-08T00:00:00Z','runs':[]}))
    code=driver().main(['--binary',binary,'--catalog',str(catalog),'--out',str(tmp_path/'out'),'--as-of','2026-10-08T00:00:00Z'])
    result=capsys.readouterr();receipt=json.loads(result.out)
    assert code==1 and receipt['exit_code']==1 and receipt['gpu_seconds']==0
    assert receipt['wall_seconds']>=0
    assert 'every public run' in result.err
    assert not list(tmp_path.glob('nightly-request-*'))
    assert not (tmp_path/'out/nightly.json').exists()
