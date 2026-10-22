"""The CUDA column door must preserve absent and unknown species semantics."""
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")
from gpuwm.ensemble.visibility import diagnose, INPUT_NAMES


def test_resident_and_streamed_visibility_use_same_columns_and_mask_bad_present_species():
    shape = (128, 160)
    volume = (4,) + shape
    p = cp.full(volume, 100000, cp.float32)
    theta = cp.full(volume, 280, cp.float32)
    qv = cp.full(volume, .01, cp.float32)
    qc = cp.zeros(volume, cp.float32)
    qc[1, :, :80] = .0001
    qc[3, :, 80:] = .1  # Above the three sampled levels, no surface attenuation.
    state = SimpleNamespace(p=p, thp=theta, thb=cp.zeros(4, cp.float32), qv=qv, qc=qc)
    vis = cp.empty(shape, cp.float32)
    flag = cp.empty_like(vis)
    workspace = {f"headline:visibility:{name}": cp.empty((3,) + shape, cp.float32) for name in INPUT_NAMES}
    assert diagnose(state=state, fields={}, streamed=False, visibility=vis,
                    fog_flag=flag, workspace=workspace, array_module=cp)
    resident = vis.copy()
    assert bool(cp.all(vis[:, :80] < 400))
    assert bool(cp.all(flag[:, :80] == 1))
    cp.testing.assert_allclose(vis[:, 80:], np.float32(12180.17549129514), rtol=1.e-6)
    fields = {"P": cp.asnumpy(p), "PB": np.zeros(volume, np.float32),
              "T": cp.asnumpy(theta - 300), "QVAPOR": cp.asnumpy(qv), "QCLOUD": cp.asnumpy(qc)}
    assert diagnose(state=None, fields=fields, streamed=True, visibility=vis,
                    fog_flag=flag, workspace=workspace, array_module=cp)
    cp.testing.assert_array_equal(vis, resident)
    fields["QRAIN"] = np.zeros(volume, np.float32)
    fields["QRAIN"][2, 4, 5] = np.nan
    assert diagnose(state=None, fields=fields, streamed=True, visibility=vis,
                    fog_flag=flag, workspace=workspace, array_module=cp)
    assert bool(cp.isnan(vis[4, 5])) and bool(cp.isnan(flag[4, 5]))
    del fields["QCLOUD"]
    assert not diagnose(state=None, fields=fields, streamed=True, visibility=vis,
                        fog_flag=flag, workspace=workspace, array_module=cp)
