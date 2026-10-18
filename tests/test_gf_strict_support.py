"""The strict publishing boundary preserves every active word."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.core.gf import _mask_output_support
from gpuwm import wrf_exact


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("boundary", ["specified", "nested", "open_x", "open_y", "periodic"])
@pytest.mark.parametrize("shape", [(2,12,13), (12,13), (2,5,6)])
def test_publication_words(monkeypatch, strict, boundary, shape):
    monkeypatch.setattr(wrf_exact, "ENABLED", strict)
    cfg=SimpleNamespace(**{k:k==boundary for k in ("specified","nested","open_x","open_y")})
    a=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)+np.float32(1)
    before=a.copy(); active=np.ones(shape[-2:],bool)
    if strict and boundary in ("specified","nested","open_x"):
        active[:,:4]=active[:,-4:]=False
    if strict and boundary in ("specified","nested","open_y"):
        active[:4,:]=active[-4:,:]=False
    assert _mask_output_support(a,cfg) is a
    np.testing.assert_array_equal(a[...,active].view(np.uint32),before[...,active].view(np.uint32))
    assert np.count_nonzero(a[...,~active].view(np.uint32))==0
