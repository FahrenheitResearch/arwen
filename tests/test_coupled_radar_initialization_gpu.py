"""The production forecast radar initializer on a captured whole-model call."""
import os
from pathlib import Path
import sys
import numpy as np
import pytest
cp=pytest.importorskip("cupy")

from gpuwm.core.refl import launch_refl10cm_wsm6
from gpuwm.verify_exact.recording import _load_raw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from coupled_reflectivity_probe import capture


def test_scalar_initializer_clears_the_captured_full_active_radar_field():
    root=os.environ.get("GPUWM_COUPLED_RADAR_CAPTURE")
    reference=os.environ.get("GPUWM_COUPLED_RADAR_REFERENCE")
    if not root or not reference:pytest.skip("captured call and its stock recording are required")
    shape=(50,40,40)
    fields=capture(Path(root),"woof",shape)
    expected=_load_raw([Path(reference)],1)["REFL_10CM"][:,1:-1,1:-1]
    args=[cp.asarray(v) for v in fields]
    out=cp.empty(shape,cp.float32)
    launch_refl10cm_wsm6(*args,out)
    old=cp.asnumpy(out)[:,1:-1,1:-1]
    assert np.count_nonzero(old.view(np.uint32)!=expected.view(np.uint32))==3
    launch_refl10cm_wsm6(*args,out,radar_initialization="scalar_build")
    actual=cp.asnumpy(out)[:,1:-1,1:-1]
    assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32))
