"""Byte gates for the sequential reference and the level-group arm."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('arm', ['levels', 'sedlevels'])
@pytest.mark.parametrize('predict_nc', [False, True])
@pytest.mark.parametrize('block', [4, 32, 128])
def test_level_groups_match_sequential_reference(predict_nc, block, arm):
    cp = pytest.importorskip('cupy')
    from gpuwm.core import p3_device as PD

    spec = importlib.util.spec_from_file_location(
        'p3_speed_fixture', Path(__file__).with_name('test_p3_cuda_gpu.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    host, nk, ncol = fixture._tiny_column_case(nk=49, ncol=137)
    # A dry column exercises the whole-column early exit alongside wet ones.
    for name in ('qc', 'qr', 'qi', 'qir', 'qib', 'ni', 'nr'):
        host[name][:, 0] = 0.0
    host['qv'][:, 0] *= np.float32(0.1)
    host['qv_old'][:, 0] = host['qv'][:, 0]
    host['nc'][:] = np.float32(1.0e8)
    # Thin lower layers exercise adaptive sedimentation substeps.
    host['dz'][:4, :] = np.float32(25.0)

    def run(arm):
        fields = {k: cp.asarray(v) for k, v in host.items()}
        diag = {k: cp.zeros((nk, ncol), dtype=cp.float32)
                for k in PD.DIAG_SLOTS}
        surf = {k: cp.zeros(ncol, dtype=cp.float32) for k in PD.SURF_SLOTS}
        ws = PD.make_workspace(ncol, nk)
        for it in (1, 2, 3):
            PD.run_p3_device(fields, diag, surf, workspace=ws, dt=15.0,
                             it=it, arm=arm, block=block,
                             log_predictNc=predict_nc)
        out = {}
        for group, arrays in (('fields', fields), ('diag', diag),
                              ('surf', surf), ('carriers', ws.carriers)):
            out.update({(group, k): v.get().tobytes()
                        for k, v in arrays.items()})
        out['flags'] = ws.flags.get().tobytes()
        return out

    reference = run('unfused')
    candidate = run(arm)
    assert reference.keys() == candidate.keys()
    assert not [k for k in reference if reference[k] != candidate[k]]
