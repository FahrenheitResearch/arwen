"""The frozen-record owner rebuilds the published sweep without external inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def test_committed_coverage_rebuild_keeps_the_exact_chain_case_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    here = root / 'tools/combo_sweep'
    output, receipt = tmp_path / 'combos.json', tmp_path / 'receipt.json'
    result = subprocess.run([sys.executable, str(here / 'refresh_committed.py'), '--out', str(output),
                             '--receipt', str(receipt)], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == (here / 'combos.json').read_bytes()
    doc = json.loads(receipt.read_text())
    assert doc['inputs']['external_templates_read'] is False
    assert doc['inputs']['current_sha256'] == hashlib.sha256((here / 'combos-v2.json').read_bytes()).hexdigest()
    assert all(c['covered'] == c['required'] for c in doc['coverage'].values())
    before = {r['id']: r for r in json.loads((here / 'combos-v2.json').read_text())['combos']}
    after = {r['id']: r for r in json.loads(output.read_text())['combos']}
    for cid in ('A071', 'A195'):
        assert before[cid] == after[cid]
