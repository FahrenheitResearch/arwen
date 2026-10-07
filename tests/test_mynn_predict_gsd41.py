"""Unmodified GSD v4.1 TKE solve, including its binding edge cases."""
from pathlib import Path
import csv
import numpy as np
import pytest
from test_mynn_gsd41 import requires_gpu


def _predict_fixture():
    path = Path(__file__).resolve().parents[1] / 'tests/data/oracles/mynn/predict-gsd41.csv'
    with path.open(newline='', encoding='ascii') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8 * 12
    return {key: np.array([np.float32(row[key]) for row in rows]).reshape(8, 12)
            for key in rows[0] if key not in ('case', 'k')}


def _predict_inputs(fields):
    values = {name: fields[name] for name in ('dz', 'el', 'dfq', 'pdk', 'pdt', 'pdq', 'pdc')}
    values.update(qke=fields['qke0'], rho=np.ones((8, 12), np.float32))
    for name in ('tsq', 'qsq', 'cov'):
        values[name] = np.zeros((8, 12), np.float32)
    for name, source in (('s_aw', 's_aw'), ('s_awqke', 'awqke')):
        values[name] = np.concatenate((fields[source][:, :1], fields[source + '_next']), axis=1)
    for name in ('delt', 'ust', 'flt', 'flq', 'pmz', 'phh'):
        values[name] = fields[name][:, 0]
    return values


@requires_gpu
@pytest.mark.gpu
def test_gsd41_predict_matches_unmodified_fork_tke():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_predict_default_cuda
    fields = _predict_fixture()
    actual = mynn_predict_default_cuda(_predict_inputs(fields), bl_mynn_version='gsd_41')
    np.testing.assert_array_equal(cp.asnumpy(actual.qke).view(np.uint32),
                                  fields['qke'].view(np.uint32))


@requires_gpu
@pytest.mark.gpu
def test_gsd41_floor_cap_and_top_boundary_differ_from_generic():
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_predict_default_cuda
    fields = _predict_fixture()
    actual = cp.asnumpy(mynn_predict_default_cuda(
        _predict_inputs(fields), bl_mynn_version='gsd_41').qke)
    generic = cp.asnumpy(mynn_predict_default_cuda(_predict_inputs(fields)).qke)
    assert actual[1].min() == np.float32(1.e-4)
    assert generic[1].min() == np.float32(1.e-3)
    assert actual[2].max() > 190 and generic[2].max() == np.float32(150)
    assert abs(float(actual[3, -1] - actual[3, -2])) < 1.e-7
    assert generic[3, -1] == fields['qke0'][3, -1] == np.float32(4)